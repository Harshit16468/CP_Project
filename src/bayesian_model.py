"""
Step 6 – Bayesian Hierarchical Modeling
=========================================
Fits a Bayesian linear mixed-effects model using PyMC:

    log_RT ~ 1 + Σ β_k * predictor_k
           + (1 + β_surprisal + β_ic | subject)

This allows us to:
  - Estimate full posterior distributions over predictor coefficients
  - Quantify per-reader random slopes (Hypothesis 6)
  - Compare models with/without specific predictors (Hypotheses 1–4)
  - Use LOO-CV (leave-one-out cross-validation) for model comparison

Model variants (selected by ``predictors`` in config)
------------------------------------------------------
  baseline        : ngram_surprisal only
  deep_surprisal  : gpt2_surprisal only
  full_neural     : gpt2 + bert + t5 surprisal + entropy + integration_cost
  comparison      : all metrics simultaneously

Public API
----------
    BayesianHierarchicalModel(cfg)
    model.fit(df)          -> arviz.InferenceData
    model.compare_models(results_dict) -> pd.DataFrame  (LOO-CV table)
    model.plot_posteriors(idata)
    model.save(idata, path)
    model.load(path)        -> arviz.InferenceData
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import zscore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class BayesianHierarchicalModel:
    """
    Bayesian hierarchical linear model for reading-time data.

    Parameters
    ----------
    cfg : dict
        The ``bayesian`` sub-dict from config.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg         = cfg
        self.predictors  = cfg["predictors"]
        self.rand_slopes = cfg.get("random_slopes", [])
        self.draws       = cfg.get("draws", 2000)
        self.tune        = cfg.get("tune", 1000)
        self.chains      = cfg.get("chains", 4)
        self.target_acc  = cfg.get("target_accept", 0.9)
        self.seed        = cfg.get("random_seed", 42)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame) -> dict:
        """
        Extract and z-score predictors, encode subject index.
        Drops rows with any NaN in predictors or log_rt.
        Subsamples to max_obs rows (balanced across subjects) if configured.
        """
        available = [p for p in self.predictors if p in df.columns]
        missing   = set(self.predictors) - set(available)
        if missing:
            logger.warning("Predictors missing from DataFrame (will skip): %s", missing)

        cols = ["subject", "log_rt"] + available
        clean = df[cols].dropna().copy()
        logger.info("Fitting on %d observations after dropping NaN rows.", len(clean))

        # Optional subsample (set max_obs in config to enable)
        max_obs = self.cfg.get("max_obs", None)
        if max_obs and len(clean) > max_obs:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(clean), size=max_obs, replace=False)
            clean = clean.iloc[idx].copy()
            logger.info("Subsampled to %d observations (max_obs=%d).", len(clean), max_obs)

        subjects, subject_idx = np.unique(clean["subject"].values, return_inverse=True)
        n_subj = len(subjects)

        # Z-score each predictor (helps MCMC mixing)
        X = {}
        for pred in available:
            vals = clean[pred].values.astype(float)
            mu, sd = vals.mean(), vals.std()
            X[pred] = (vals - mu) / (sd if sd > 0 else 1.0)

        return {
            "log_rt":      clean["log_rt"].values.astype(float),
            "X":           X,
            "subject_idx": subject_idx.astype(int),
            "n_subj":      n_subj,
            "subjects":    subjects,
            "n_obs":       len(clean),
            "predictors":  available,
        }

    # ------------------------------------------------------------------
    # Model specification
    # ------------------------------------------------------------------

    def _build_model(self, data: dict) -> pm.Model:
        """
        Construct the PyMC model.

        Fixed effects: intercept + all predictors
        Random effects: per-subject intercept + random slopes for configured predictors
        """
        log_rt      = data["log_rt"]
        subject_idx = data["subject_idx"]
        n_subj      = data["n_subj"]
        predictors  = data["predictors"]
        rand_slopes = [p for p in self.rand_slopes if p in predictors]

        with pm.Model() as model:
            # ---- Fixed effects priors ----------------------------------------
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            beta = {
                pred: pm.Normal(f"beta_{pred}", mu=0, sigma=1)
                for pred in predictors
            }

            # ---- Random intercept per subject (non-centered) ------------------
            sigma_u0 = pm.HalfNormal("sigma_u0", sigma=0.5)
            u0_raw   = pm.Normal("u0_raw", mu=0, sigma=1, shape=n_subj)
            u0       = pm.Deterministic("u0", u0_raw * sigma_u0)

            # ---- Random slopes per subject (non-centered) ---------------------
            u_slopes = {}
            for pred in rand_slopes:
                sigma_slope = pm.HalfNormal(f"sigma_slope_{pred}", sigma=0.3)
                u_raw = pm.Normal(f"u_slope_raw_{pred}", mu=0, sigma=1, shape=n_subj)
                u_slopes[pred] = pm.Deterministic(
                    f"u_slope_{pred}", u_raw * sigma_slope
                )

            # ---- Linear predictor --------------------------------------------
            mu = intercept + u0[subject_idx]
            for pred in predictors:
                X_pred = data["X"][pred]
                fixed  = beta[pred] * X_pred
                if pred in u_slopes:
                    random = u_slopes[pred][subject_idx] * X_pred
                    mu = mu + fixed + random
                else:
                    mu = mu + fixed

            # ---- Likelihood --------------------------------------------------
            sigma_eps = pm.HalfNormal("sigma_eps", sigma=0.5)
            pm.Normal("log_rt_obs", mu=mu, sigma=sigma_eps, observed=log_rt)

        return model

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> az.InferenceData:
        """
        Fit the model and return an ArviZ InferenceData object.
        """
        data       = self._prepare_data(df)
        self._last_data = data          # needed for subsampled LOO
        pymc_model = self._build_model(data)

        logger.info(
            "Sampling: %d draws, %d tune, %d chains (target_accept=%.2f) …",
            self.draws, self.tune, self.chains, self.target_acc,
        )
        with pymc_model:
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.chains,   # one core per chain — parallel sampling
                target_accept=self.target_acc,
                random_seed=self.seed,
                progressbar=True,
                return_inferencedata=True,
            )
            pass  # log_likelihood computed via subsampling below — no OOM

        if self.cfg.get("compute_loo", False):
            logger.info("Computing subsampled LOO …")
            self.last_loo = self._compute_loo_subsampled(
                idata, self._last_data,
                n_loo=self.cfg.get("loo_subsample", 20_000),
            )
            logger.info("LOO ELPD=%.2f (SE=%.2f)", self.last_loo.elpd_loo, self.last_loo.se)
        else:
            self.last_loo = None

        logger.info("Sampling complete.")
        return idata

    def _compute_loo_subsampled(
        self, idata: az.InferenceData, data: dict, n_loo: int = 20_000
    ) -> az.ELPDData:
        """
        Compute LOO-CV on a random subsample of observations to avoid OOM.

        Uses the already-sampled posterior to evaluate the normal log-likelihood
        at n_loo randomly selected observations. Memory: chains×draws×n_loo×8B
        (~1.3 GB for n_loo=20K, 4 chains, 2K draws).
        """
        import xarray as xr
        from scipy.stats import norm as sp_norm

        n_obs = data["n_obs"]
        n_loo = min(n_loo, n_obs)
        rng   = np.random.default_rng(self.seed + 99)
        idx   = rng.choice(n_obs, size=n_loo, replace=False)

        post       = idata.posterior
        intercept  = post["intercept"].values          # (chains, draws)
        sigma_eps  = post["sigma_eps"].values          # (chains, draws)
        u0         = post["u0"].values                 # (chains, draws, n_subj)
        chains, draws = intercept.shape

        subj_sub   = data["subject_idx"][idx]          # (n_loo,)
        log_rt_sub = data["log_rt"][idx]               # (n_loo,)

        # mu: (chains, draws, n_loo)
        mu = intercept[:, :, None] + u0[:, :, subj_sub]

        rand_slopes = [p for p in self.rand_slopes if p in data["predictors"]]
        for pred in data["predictors"]:
            X_sub = data["X"][pred][idx]               # (n_loo,)
            beta  = post[f"beta_{pred}"].values        # (chains, draws)
            mu    = mu + beta[:, :, None] * X_sub[None, None, :]
            if pred in rand_slopes:
                u_sl = post[f"u_slope_{pred}"].values  # (chains, draws, n_subj)
                mu   = mu + u_sl[:, :, subj_sub] * X_sub[None, None, :]

        log_lik = sp_norm.logpdf(
            log_rt_sub[None, None, :],
            loc=mu,
            scale=sigma_eps[:, :, None],
        )  # (chains, draws, n_loo)

        logger.info(
            "LOO subsample: %d / %d obs, log_lik array %.1f MB",
            n_loo, n_obs, log_lik.nbytes / 1e6,
        )

        # Compute reff from actual posterior (needed by az.loo when no
        # posterior group is present in the subsampled idata)
        n_samples = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
        try:
            ess = az.ess(idata, var_names=["intercept"], method="bulk")
            reff = float(float(ess["intercept"].values.mean()) / n_samples)
            reff = float(np.clip(reff, 0.01, 1.0))
        except Exception:
            reff = 1.0
        logger.info("reff=%.3f", reff)

        idata_sub = az.from_dict(log_likelihood={"log_rt_obs": log_lik})
        return az.loo(idata_sub, var_name="log_rt_obs", reff=reff)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self, loo_results: dict[str, az.ELPDData]
    ) -> pd.DataFrame:
        """
        Run LOO-CV comparison across model variants using pre-computed ELPDData.

        Parameters
        ----------
        loo_results : dict mapping model_name → ELPDData (from az.loo)

        Returns
        -------
        pd.DataFrame   ArviZ compare table (sorted by ELPD)
        """
        comparison = az.compare(loo_results, ic="loo")
        logger.info("Model comparison:\n%s", comparison.to_string())
        return comparison

    # ------------------------------------------------------------------
    # Summary & diagnostics
    # ------------------------------------------------------------------

    def summary(self, idata: az.InferenceData) -> pd.DataFrame:
        """Return an ArviZ summary DataFrame."""
        return az.summary(idata, round_to=3)

    def plot_posteriors(
        self,
        idata: az.InferenceData,
        save_dir: str | Path | None = None,
    ) -> None:
        """Plot posterior distributions for fixed-effect beta coefficients."""
        import matplotlib.pyplot as plt

        var_names = [f"beta_{p}" for p in self.cfg["predictors"]]
        existing  = [v for v in var_names if v in idata.posterior]

        axes = az.plot_posterior(idata, var_names=existing, hdi_prob=0.95)
        fig  = plt.gcf()
        fig.suptitle("Posterior Distributions of Predictor Coefficients", y=1.02)
        plt.tight_layout()

        if save_dir:
            out = Path(save_dir) / "posterior_betas.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            logger.info("Saved posterior plot to %s", out)
        plt.show()

    def plot_forest(
        self,
        idata: az.InferenceData,
        save_dir: str | Path | None = None,
    ) -> None:
        """Forest plot of fixed effects with 95% HDI."""
        import matplotlib.pyplot as plt

        var_names = [f"beta_{p}" for p in self.cfg["predictors"]]
        existing  = [v for v in var_names if v in idata.posterior]

        axes = az.plot_forest(idata, var_names=existing, combined=True,
                              hdi_prob=0.95, r_hat=True)
        fig = plt.gcf()
        fig.suptitle("Forest Plot: Fixed Effects (95% HDI)", y=1.01)
        plt.tight_layout()

        if save_dir:
            out = Path(save_dir) / "forest_plot.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            logger.info("Saved forest plot to %s", out)
        plt.show()

    def plot_trace(
        self,
        idata: az.InferenceData,
        save_dir: str | Path | None = None,
    ) -> None:
        """Trace plot for MCMC diagnostics."""
        import matplotlib.pyplot as plt

        az.plot_trace(idata)
        fig = plt.gcf()
        if save_dir:
            out = Path(save_dir) / "trace_plot.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, idata: az.InferenceData, path: str | Path) -> None:
        """Save InferenceData to NetCDF."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(path))
        logger.info("Saved InferenceData to %s", path)

    def load(self, path: str | Path) -> az.InferenceData:
        """Load InferenceData from NetCDF."""
        return az.from_netcdf(str(path))

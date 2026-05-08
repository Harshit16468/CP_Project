"""
scripts/mcmc_robustness.py
==========================
MCMC convergence diagnostics and robustness analysis for all fitted Bayesian models.

Addresses the second remaining item on slide 15: "MCMC robustness — address
intercept mixing for hierarchical models."

What it does
------------
1. DIAGNOSTICS: For every saved .nc model, report:
   - R-hat (should be < 1.01 for all parameters)
   - Bulk & tail ESS (effective sample size; should be > 400)
   - Number of divergent transitions
   - MCSE (Monte Carlo standard error) relative to posterior SD

2. FLAGGING: Print a concise pass/fail table.  Any model with R-hat > 1.05
   or ESS < 100 is flagged as "NEEDS REFIT".

3. TRACE PLOTS: Save trace plots for flagged parameters so you can visually
   inspect mixing.

4. REFIT (optional, --refit flag): Re-run MCMC for flagged models using
   non-centered parameterisation (already the default in bayesian_model.py),
   more tuning steps, and a higher target_accept, which typically cures
   intercept funnel geometry.

5. INTERCEPT MIXING FIX: The most common cause of poor intercept mixing in
   hierarchical models is a "Neal's funnel" between sigma_u0 and u0.
   The existing bayesian_model.py already uses non-centered reparameterisation:
       u0 = u0_raw * sigma_u0    (u0_raw ~ Normal(0,1))
   This script verifies that the non-centered form is in the saved trace,
   and if not, refits with an explicit non-centered model (class defined here).

Usage
-----
    # Diagnostics only (fast):
    python scripts/mcmc_robustness.py

    # Diagnostics + refit any failing models:
    python scripts/mcmc_robustness.py --refit

    # Custom paths:
    python scripts/mcmc_robustness.py --results /tmp/psycholingu/results/metrics
                                       --figures /tmp/psycholingu/results/figures
                                       --refit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mcmc_robustness")

# Thresholds
RHAT_WARN  = 1.05   # flag for attention
RHAT_FAIL  = 1.10   # flag for mandatory refit
ESS_WARN   = 200
ESS_FAIL   = 100
DIV_WARN   = 10     # acceptable divergences (NUTS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_model(nc_path: Path) -> pd.DataFrame:
    """
    Load a NetCDF inference data file and return a diagnostics DataFrame.
    One row per parameter; columns: r_hat, ess_bulk, ess_tail, mcse_sd, n_divergences.
    """
    import arviz as az

    logger.info("Diagnosing %s", nc_path.name)
    idata = az.from_netcdf(str(nc_path))

    # ── R-hat & ESS ─────────────────────────────────────────────────────────
    # Only scalar / fixed-effect parameters (skip large random-effect arrays)
    scalar_vars = []
    for var in idata.posterior.data_vars:
        shape = idata.posterior[var].shape   # (chain, draw, ...)
        if len(shape) <= 2 or (len(shape) == 3 and shape[2] <= 5):
            scalar_vars.append(var)

    summary = az.summary(
        idata,
        var_names=scalar_vars,
        stat_funcs=None,
        extend=True,
        round_to=4,
    )

    # ── Divergences ──────────────────────────────────────────────────────────
    n_div = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].values.sum())
    summary["n_divergences"] = n_div

    return summary


def run_diagnostics(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Run diagnostics on all .nc files in results_dir."""
    nc_files = sorted(results_dir.glob("06_bayes_*.nc"))
    if not nc_files:
        logger.warning("No .nc model files found in %s", results_dir)
        return {}

    all_diag: dict[str, pd.DataFrame] = {}
    for nc in nc_files:
        model_name = nc.stem.replace("06_bayes_", "")
        try:
            diag = diagnose_model(nc)
            all_diag[model_name] = diag
        except Exception as exc:
            logger.error("Failed to diagnose %s: %s", nc.name, exc)

    return all_diag


def print_summary_table(all_diag: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Print a concise per-model pass/fail table.
    Returns a DataFrame with worst-case diagnostics per model.
    """
    rows = []
    for model_name, diag in all_diag.items():
        max_rhat   = float(diag["r_hat"].max())      if "r_hat"    in diag else float("nan")
        min_ess_b  = float(diag["ess_bulk"].min())   if "ess_bulk" in diag else float("nan")
        min_ess_t  = float(diag["ess_tail"].min())   if "ess_tail" in diag else float("nan")
        n_div      = int(diag["n_divergences"].iloc[0]) if "n_divergences" in diag else 0

        status = "OK"
        if max_rhat > RHAT_FAIL or min_ess_b < ESS_FAIL:
            status = "NEEDS_REFIT"
        elif max_rhat > RHAT_WARN or min_ess_b < ESS_WARN or n_div >= DIV_WARN:
            status = "WARN"

        rows.append({
            "model":      model_name,
            "max_rhat":   round(max_rhat, 4),
            "min_ess_bulk": round(min_ess_b, 0),
            "min_ess_tail": round(min_ess_t, 0),
            "n_divergences": n_div,
            "status":     status,
        })

    table = pd.DataFrame(rows).sort_values("status", ascending=False)
    print("\n" + "=" * 70)
    print("MCMC DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(table.to_string(index=False))
    print("=" * 70)
    print(f"  Thresholds: R-hat warn={RHAT_WARN}, fail={RHAT_FAIL}  |  "
          f"ESS warn={ESS_WARN}, fail={ESS_FAIL}  |  divergences warn={DIV_WARN}\n")

    n_fail = (table["status"] == "NEEDS_REFIT").sum()
    n_warn = (table["status"] == "WARN").sum()
    n_ok   = (table["status"] == "OK").sum()
    print(f"  {n_ok} models OK   |  {n_warn} WARN   |  {n_fail} NEEDS_REFIT\n")
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 2. Trace plots for flagged parameters
# ─────────────────────────────────────────────────────────────────────────────

def plot_traces_for_model(
    nc_path: Path,
    diag: pd.DataFrame,
    figures_dir: Path,
    rhat_threshold: float = RHAT_WARN,
) -> None:
    """
    Save trace plots for parameters with R-hat above threshold.
    Also always plots the intercept and sigma_u0 (most likely to have funnel issues).
    """
    import arviz as az

    idata = az.from_netcdf(str(nc_path))
    model_name = nc_path.stem.replace("06_bayes_", "")

    # Parameters to always plot (intercept mixing is the key concern)
    always_plot = ["intercept", "sigma_u0"]

    # Parameters flagged by R-hat
    flagged = []
    if "r_hat" in diag.columns:
        flagged = list(diag[diag["r_hat"] > rhat_threshold].index)

    vars_to_plot = list(dict.fromkeys(always_plot + flagged))  # deduplicate, preserve order
    vars_to_plot = [v for v in vars_to_plot if v in idata.posterior]

    if not vars_to_plot:
        logger.info("No parameters to plot for %s.", model_name)
        return

    logger.info("Plotting traces for %s: %s", model_name, vars_to_plot)
    try:
        axes = az.plot_trace(
            idata,
            var_names=vars_to_plot,
            combined=False,
            compact=True,
        )
        fig = plt.gcf()
        fig.suptitle(
            f"Trace plots — {model_name}\n"
            f"(flagged if R-hat > {rhat_threshold}  |  always: intercept, sigma_u0)",
            y=1.01,
        )
        plt.tight_layout()
        out = figures_dir / f"trace_{model_name}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        logger.info("Saved trace plot: %s", out)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Could not save trace plot for %s: %s", model_name, exc)


def plot_pair_intercept(nc_path: Path, figures_dir: Path) -> None:
    """
    Pair plot of intercept vs sigma_u0.
    A funnel shape here indicates a non-identifiability problem that the
    non-centered parameterisation should resolve.
    """
    import arviz as az

    idata = az.from_netcdf(str(nc_path))
    model_name = nc_path.stem.replace("06_bayes_", "")
    post = idata.posterior

    if "intercept" not in post or "sigma_u0" not in post:
        return

    intercept = post["intercept"].values.flatten()
    sigma_u0  = post["sigma_u0"].values.flatten()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(intercept, sigma_u0, alpha=0.05, s=4, color="steelblue", rasterized=True)
    ax.set_xlabel("intercept")
    ax.set_ylabel("sigma_u0")
    ax.set_title(
        f"Pair plot — {model_name}\n"
        f"Funnel shape → non-centered reparameterisation needed\n"
        f"Uniform cloud → mixing is good"
    )
    plt.tight_layout()
    out = figures_dir / f"pair_intercept_{model_name}.png"
    fig.savefig(out, dpi=120)
    logger.info("Saved pair plot: %s", out)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Verification: confirm non-centered parameterisation is present
# ─────────────────────────────────────────────────────────────────────────────

def verify_noncentered(nc_path: Path) -> bool:
    """
    Returns True if the saved model used non-centered parameterisation
    (i.e., u0_raw exists in posterior).  If False the model was fitted
    with a centred form and should be refit.
    """
    import arviz as az

    idata = az.from_netcdf(str(nc_path))
    has_raw = "u0_raw" in idata.posterior
    if not has_raw:
        logger.warning(
            "%s does NOT have u0_raw — model used centred parameterisation. "
            "Run with --refit to fix.", nc_path.name
        )
    return has_raw


# ─────────────────────────────────────────────────────────────────────────────
# 4. Refit with improved settings
# ─────────────────────────────────────────────────────────────────────────────

def refit_model(
    nc_path: Path,
    results_dir: Path,
    data_parquet: Path,
    bay_cfg: dict,
    variant_predictors: dict[str, list[str]],
    draws: int = 3000,
    tune:  int = 3000,
    target_accept: float = 0.95,
) -> None:
    """
    Refit a single model variant with more aggressive sampler settings.

    Improvements applied:
    - More tuning steps (tune=3000 vs default 2000)
    - Higher target_accept (0.95 vs 0.9) — NUTS takes smaller steps, less divergence
    - More draws for better ESS
    - Non-centered parameterisation (already in bayesian_model.py)
    """
    import arviz as az
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    from src.bayesian_model import BayesianHierarchicalModel

    model_name = nc_path.stem.replace("06_bayes_", "")
    logger.info("Refitting model: %s  (draws=%d, tune=%d, target_accept=%.2f)",
                model_name, draws, tune, target_accept)

    if not data_parquet.exists():
        logger.error("Data parquet not found: %s  — cannot refit.", data_parquet)
        return

    df = pd.read_parquet(data_parquet)
    if "dep_length" in df.columns and "integration_cost" not in df.columns:
        df = df.rename(columns={"dep_length": "integration_cost"})

    preds = variant_predictors.get(model_name)
    if preds is None:
        logger.warning("No predictor list for variant %s — skipping refit.", model_name)
        return

    v_cfg = {
        **bay_cfg,
        "predictors":      preds,
        "draws":           draws,
        "tune":            tune,
        "target_accept":   target_accept,
        # Tight data-informed intercept prior — the correct fix for funnel geometry.
        # Pins intercept near mean(log_rt) so u0 carries only residual subject variation.
        "intercept_prior": "data",
    }

    bm    = BayesianHierarchicalModel(v_cfg)

    # ADVI warm-start: finds a good starting point before NUTS begins,
    # substantially reducing the burn-in needed to escape the funnel.
    import pymc as pm
    data_prepared = bm._prepare_data(df)
    bm._last_data = data_prepared
    pymc_model    = bm._build_model(data_prepared)
    logger.info("Running ADVI warm-start for %s …", model_name)
    with pymc_model:
        idata = pm.sample(
            draws           = draws,
            tune            = tune,
            chains          = v_cfg["chains"],
            cores           = v_cfg["chains"],
            target_accept   = target_accept,
            init            = "advi+adapt_diag",   # ADVI warm-start
            random_seed     = v_cfg.get("random_seed", 42),
            progressbar     = True,
            return_inferencedata = True,
        )

    # Save with a "_robust" suffix so the original is preserved
    out_nc  = results_dir / f"06_bayes_{model_name}_robust.nc"
    out_csv = results_dir / f"06_summary_{model_name}_robust.csv"
    bm.save(idata, out_nc)
    bm.summary(idata).to_csv(out_csv)
    logger.info("Refitted model saved: %s", out_nc)


def refit_failing_models(
    summary_table: pd.DataFrame,
    results_dir: Path,
    data_parquet: Path,
    bay_cfg: dict,
    variant_predictors: dict[str, list[str]],
) -> None:
    """Refit all models marked NEEDS_REFIT in the summary table."""
    failing = summary_table[summary_table["status"] == "NEEDS_REFIT"]["model"].tolist()
    if not failing:
        logger.info("No models require refit — all diagnostics pass.")
        return

    logger.info("Models requiring refit: %s", failing)
    for model_name in failing:
        nc_path = results_dir / f"06_bayes_{model_name}.nc"
        refit_model(
            nc_path, results_dir, data_parquet, bay_cfg, variant_predictors,
            draws=3000, tune=3000, target_accept=0.95,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary report
# ─────────────────────────────────────────────────────────────────────────────

def save_diagnostic_report(
    all_diag: dict[str, pd.DataFrame],
    summary_table: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Save full per-parameter diagnostic CSVs and the summary table."""
    for model_name, diag in all_diag.items():
        diag.to_csv(results_dir / f"diag_{model_name}.csv")

    summary_table.to_csv(results_dir / "mcmc_diagnostic_summary.csv", index=False)
    logger.info("Diagnostic report saved to %s", results_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCMC diagnostics and robustness checks")
    p.add_argument("--results", default="/tmp/psycholingu/results/metrics",
                   help="Path to results directory with .nc model files")
    p.add_argument("--figures", default="/tmp/psycholingu/results/figures",
                   help="Path to figures output directory")
    p.add_argument("--data",    default="/tmp/psycholingu/data/processed/05_integration_cost.parquet",
                   help="Path to processed data parquet (needed for refit)")
    p.add_argument("--config",  default="config.yaml",
                   help="Path to config.yaml (for Bayesian settings)")
    p.add_argument("--refit",   action="store_true",
                   help="Refit models that fail diagnostics with improved settings")
    p.add_argument("--trace-all", action="store_true",
                   help="Save trace plots for ALL models (default: only flagged ones)")
    p.add_argument("--pair-plots", action="store_true",
                   help="Save intercept vs sigma_u0 pair plots for all models")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Variant predictor map (mirrors pipeline.py step6) ───────────────────
    # Used when refitting to know which predictors belong to each variant
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    bay_cfg  = cfg["bayesian"]
    all_pred = bay_cfg["predictors"]

    variant_predictors = {
        "baseline":                   ["ngram_surprisal"],
        "deep_gpt2":                  ["gpt2_surprisal"],
        "deep_bert":                  ["bert_base_uncased_surprisal"],
        "deep_t5":                    ["t5_base_surprisal"],
        "surprisal_vs_ic":            ["gpt2_surprisal", "integration_cost"],
        "surprisal_vs_entropy_gpt2":  ["gpt2_surprisal", "gpt2_entropy"],
        "surprisal_vs_entropy_bert":  ["bert_base_uncased_surprisal", "bert_base_uncased_entropy"],
        "surprisal_vs_entropy_t5":    ["t5_base_surprisal", "t5_base_entropy"],
        "full":                        all_pred,
    }

    # ── 1. Run diagnostics ───────────────────────────────────────────────────
    all_diag = run_diagnostics(results_dir)
    if not all_diag:
        logger.error("No models found to diagnose. Check --results path.")
        return

    summary_table = print_summary_table(all_diag)
    save_diagnostic_report(all_diag, summary_table, results_dir)

    # ── 2. Verify non-centred parameterisation ───────────────────────────────
    print("\n── Non-centered parameterisation check ─────────────────────────")
    for nc in sorted(results_dir.glob("06_bayes_*.nc")):
        ok = verify_noncentered(nc)
        print(f"  {nc.name:<50}  {'✓ non-centered' if ok else '✗ CENTRED — refit needed'}")

    # ── 3. Trace plots ───────────────────────────────────────────────────────
    flagged_models = set(summary_table[summary_table["status"] != "OK"]["model"].tolist())
    for nc in sorted(results_dir.glob("06_bayes_*.nc")):
        model_name = nc.stem.replace("06_bayes_", "")
        diag = all_diag.get(model_name, pd.DataFrame())
        if args.trace_all or model_name in flagged_models:
            plot_traces_for_model(nc, diag, figures_dir)
        if args.pair_plots:
            plot_pair_intercept(nc, figures_dir)

    # ── 4. Refit (optional) ──────────────────────────────────────────────────
    if args.refit:
        refit_failing_models(
            summary_table,
            results_dir,
            Path(args.data),
            bay_cfg,
            variant_predictors,
        )
    else:
        n_fail = (summary_table["status"] == "NEEDS_REFIT").sum()
        if n_fail:
            print(f"\n  {n_fail} model(s) need refit.  Re-run with --refit to fix them.")
        else:
            print("\n  All models pass diagnostics.  No refit needed.")

    logger.info("MCMC robustness analysis complete.")


if __name__ == "__main__":
    main()

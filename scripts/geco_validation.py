"""
scripts/geco_validation.py
==========================
Cross-dataset validation: run the full pipeline on the GECO eye-tracking corpus
and compare key findings against the Natural Stories (self-paced reading) results.

Purpose
-------
Slide 15 of the mid-project presentation lists GECO validation as the first
remaining task for the final report.  This script:

  1. Runs pipeline steps 1-6 on GECO (GPT-2 surprisal, integration cost,
     Bayesian modeling with the same 9 model variants).
  2. Loads the already-computed Natural Stories results.
  3. Produces a side-by-side comparison table and plots for:
       - H1: does deep > shallow surprisal replicate on eye-tracking data?
       - H2: is integration cost still explained away by neural surprisal?
       - H3: are surprisal & entropy still independent?
       - H4: is GPT-2 still the best architecture?

Usage
-----
    # Download GECO first (see instructions below), then:
    python scripts/geco_validation.py [--config config.yaml] [--steps all|1,2,3,5,6]
                                       [--out-dir /tmp/geco_results]

GECO download
-------------
    1. Visit https://expsy.ugent.be/downloads/geco/
    2. Download EnglishMaterial.xlsx  → place at path in config.yaml: geco.words_file
    3. Download L1ReadingData.xlsx    → place at path in config.yaml: geco.rt_file
    Update config.yaml:  dataset: geco
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME",            "/tmp/psycholingu/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/psycholingu/hf_cache/hub")
os.environ.setdefault("STANZA_RESOURCES_DIR", "/tmp/psycholingu/stanza_resources")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_prep          import load_dataset
from src.ngram_surprisal    import build_ngram_model, compute_ngram_surprisal
from src.neural_metrics     import NeuralMetricsExtractor
from src.integration_cost   import build_parser, compute_integration_cost
from src.bayesian_model     import BayesianHierarchicalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("geco_validation")


# ─────────────────────────────────────────────────────────────────────────────
# Config & helpers
# ─────────────────────────────────────────────────────────────────────────────

GECO_VARIANTS = {
    "baseline":                   ["ngram_surprisal"],
    "deep_gpt2":                  ["gpt2_surprisal"],
    "deep_bert":                  ["bert_base_uncased_surprisal"],
    "deep_t5":                    ["t5_base_surprisal"],
    "surprisal_vs_ic":            ["gpt2_surprisal", "integration_cost"],
    "surprisal_vs_entropy_gpt2":  ["gpt2_surprisal", "gpt2_entropy"],
    "surprisal_vs_entropy_bert":  ["bert_base_uncased_surprisal", "bert_base_uncased_entropy"],
    "surprisal_vs_entropy_t5":    ["t5_base_surprisal", "t5_base_entropy"],
    "full": None,   # filled from config at runtime
}


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def cache(out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / name


def load_or_run_parquet(path: Path, fn, *args, **kwargs) -> pd.DataFrame:
    if path.exists():
        logger.info("Loading cached parquet: %s", path)
        return pd.read_parquet(path)
    result = fn(*args, **kwargs)
    result.to_parquet(path, index=True)
    logger.info("Cached to %s", path)
    return result


def posterior_summary(idata, var: str) -> tuple[float, float, float]:
    """Return (mean, 2.5-pct, 97.5-pct) for a posterior variable."""
    import arviz as az
    if var not in idata.posterior:
        return float("nan"), float("nan"), float("nan")
    s = idata.posterior[var].values.flatten()
    return float(s.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps (GECO-specific output directories)
# ─────────────────────────────────────────────────────────────────────────────

def run_step1(cfg: dict, proc_dir: Path) -> pd.DataFrame:
    logger.info("Step 1 — Loading GECO data")
    p = cache(proc_dir, "01_geco_reading_times.parquet")
    return load_or_run_parquet(p, load_dataset, "geco", cfg)


def run_step2(cfg: dict, df: pd.DataFrame, proc_dir: Path) -> pd.DataFrame:
    logger.info("Step 2 — N-gram surprisal")
    p = cache(proc_dir, "02_geco_ngram_surprisal.parquet")
    if p.exists():
        return pd.read_parquet(p)
    corpus_path = Path(cfg["paths"]["ngram_corpus"])
    model_cache = cache(proc_dir, "ngram_model.pkl")
    model = build_ngram_model(corpus_path, cfg["ngram"], cache_path=model_cache)
    df    = compute_ngram_surprisal(df, model)
    df.to_parquet(p, index=True)
    return df


def run_step3(cfg: dict, df: pd.DataFrame, proc_dir: Path) -> pd.DataFrame:
    logger.info("Step 3 — Neural surprisal & entropy")
    p = cache(proc_dir, "03_geco_neural_metrics.parquet")
    if p.exists():
        return pd.read_parquet(p)
    for label, name, mtype, device in [
        ("gpt2", cfg["models"]["gpt2"]["name"], "causal",  cfg["models"]["gpt2"]["device"]),
        ("bert", cfg["models"]["bert"]["name"], "masked",  cfg["models"]["bert"]["device"]),
        ("t5",   cfg["models"]["t5"]["name"],   "seq2seq", cfg["models"]["t5"]["device"]),
    ]:
        logger.info("  model: %s", name)
        extractor = NeuralMetricsExtractor(name, mtype, device)
        df        = extractor.compute_metrics(df)
        del extractor
    df.to_parquet(p, index=True)
    return df


def run_step5(cfg: dict, df: pd.DataFrame, proc_dir: Path) -> pd.DataFrame:
    logger.info("Step 5 — Integration cost (UD parsing)")
    p = cache(proc_dir, "05_geco_integration_cost.parquet")
    if p.exists():
        return pd.read_parquet(p)
    parser = build_parser(cfg)
    df     = compute_integration_cost(df, parser)
    df.to_parquet(p, index=True)
    return df


def run_step6(
    cfg: dict,
    df: pd.DataFrame,
    metrics_dir: Path,
    figures_dir: Path,
) -> dict:
    """Fit all Bayesian variants on GECO. Returns {variant: idata}."""
    import arviz as az

    bay_cfg  = cfg["bayesian"].copy()
    all_pred = bay_cfg["predictors"]

    variants = {k: (v if v is not None else all_pred) for k, v in GECO_VARIANTS.items()}

    idata_results: dict = {}
    loo_results:   dict = {}

    for name, preds in variants.items():
        nc_path  = metrics_dir / f"geco_bayes_{name}.nc"
        loo_pkl  = metrics_dir / f"geco_loo_{name}.pkl"

        if nc_path.exists():
            logger.info("Loading cached GECO model: %s", name)
            idata_results[name] = az.from_netcdf(str(nc_path))
            if loo_pkl.exists():
                with open(loo_pkl, "rb") as fh:
                    loo_results[name] = pickle.load(fh)
            continue

        v_cfg = {**bay_cfg, "predictors": preds}
        bm    = BayesianHierarchicalModel(v_cfg)
        logger.info("Fitting GECO model: %s  predictors=%s", name, preds)
        idata = bm.fit(df)

        if bm.last_loo is not None:
            with open(loo_pkl, "wb") as fh:
                pickle.dump(bm.last_loo, fh)
            loo_results[name] = bm.last_loo

        bm.save(idata, nc_path)
        bm.summary(idata).to_csv(metrics_dir / f"geco_summary_{name}.csv")
        idata_results[name] = idata

    if len(loo_results) > 1:
        bm_ref = BayesianHierarchicalModel(bay_cfg)
        comp   = bm_ref.compare_models(loo_results)
        comp.to_csv(metrics_dir / "geco_model_comparison.csv")
        logger.info("GECO model comparison saved.")

    return idata_results, loo_results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def _load_ns_comparison(ns_metrics: Path) -> pd.DataFrame | None:
    """Load the Natural Stories model comparison table if it exists."""
    p = ns_metrics / "06_model_comparison.csv"
    if not p.exists():
        logger.warning("Natural Stories comparison table not found at %s", p)
        return None
    return pd.read_csv(p, index_col=0)


def plot_elpd_comparison(
    geco_comp_path: Path,
    ns_metrics: Path,
    figures_dir: Path,
) -> None:
    """
    Side-by-side ELPD bar chart: Natural Stories vs GECO.
    Shows whether the architecture ranking (GPT-2 > T5 > BERT ≈ trigram) replicates.
    """
    geco_comp = pd.read_csv(geco_comp_path, index_col=0) if geco_comp_path.exists() else None
    ns_comp   = _load_ns_comparison(ns_metrics)

    arch_rows = ["baseline", "deep_gpt2", "deep_bert", "deep_t5"]
    label_map = {
        "baseline":  "Trigram",
        "deep_gpt2": "GPT-2",
        "deep_bert": "BERT",
        "deep_t5":   "T5",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, comp, title in zip(
        axes,
        [ns_comp, geco_comp],
        ["Natural Stories (self-paced RT)", "GECO (eye-tracking)"],
    ):
        if comp is None or comp.empty:
            ax.set_title(f"{title}\n(no data)")
            continue
        rows = comp[comp.index.isin(arch_rows)].copy()
        rows.index = [label_map.get(i, i) for i in rows.index]
        rows = rows.sort_values("elpd_loo")
        colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"][: len(rows)]
        ax.barh(rows.index, rows["elpd_loo"], color=colors, edgecolor="white", height=0.5)
        ax.set_xlabel("ELPD (LOO-CV)  ← worse   better →")
        ax.set_title(title)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Cross-Dataset Architecture Comparison (H4 replication)", fontsize=13)
    plt.tight_layout()
    out = figures_dir / "geco_vs_ns_elpd.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


def plot_beta_comparison(
    geco_idata: dict,
    ns_metrics: Path,
    figures_dir: Path,
) -> None:
    """
    Forest plot comparing key β coefficients across both corpora.
    Checks that surprisal > 0 and IC ≈ 0 replicate on GECO.
    """
    import arviz as az

    # Collect betas from GECO full model
    geco_full = geco_idata.get("full")
    if geco_full is None:
        logger.warning("GECO 'full' model not available for beta comparison.")
        return

    # Load NS full model
    ns_full_path = ns_metrics / "06_bayes_full.nc"
    if not ns_full_path.exists():
        logger.warning("NS full model not found at %s", ns_full_path)
        return
    ns_full = az.from_netcdf(str(ns_full_path))

    interest_vars = [
        ("beta_gpt2_surprisal",            "GPT-2 surprisal"),
        ("beta_integration_cost",          "Integration cost"),
        ("beta_gpt2_entropy",              "GPT-2 entropy"),
        ("beta_bert_base_uncased_surprisal", "BERT surprisal"),
        ("beta_t5_base_surprisal",         "T5 surprisal"),
    ]

    rows = []
    for var, label in interest_vars:
        for corpus_name, idata in [("Natural Stories", ns_full), ("GECO", geco_full)]:
            m, lo, hi = posterior_summary(idata, var)
            if not np.isnan(m):
                rows.append({"corpus": corpus_name, "predictor": label,
                             "mean": m, "lo": lo, "hi": hi})

    if not rows:
        logger.warning("No matching beta variables found for comparison.")
        return

    df_plot = pd.DataFrame(rows)
    predictors = df_plot["predictor"].unique()
    n = len(predictors)
    y_ns   = np.arange(n) * 2.5
    y_geco = y_ns + 0.8

    fig, ax = plt.subplots(figsize=(9, max(4, n * 1.5)))
    for y_pos, df_sub, color, label in [
        (y_ns,   df_plot[df_plot["corpus"] == "Natural Stories"], "#2980b9", "Natural Stories"),
        (y_geco, df_plot[df_plot["corpus"] == "GECO"],            "#e67e22", "GECO"),
    ]:
        df_sub = df_sub.set_index("predictor").reindex(predictors)
        means  = df_sub["mean"].values
        lo     = df_sub["lo"].values
        hi     = df_sub["hi"].values
        ax.scatter(means, y_pos, color=color, zorder=3, s=60, label=label)
        for i, (m, l, h) in enumerate(zip(means, lo, hi)):
            if not np.isnan(m):
                ax.plot([l, h], [y_pos[i], y_pos[i]], color=color, linewidth=2, alpha=0.7)

    ax.set_yticks((y_ns + y_geco) / 2)
    ax.set_yticklabels(predictors)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Posterior β (z-scored predictors)")
    ax.set_title("Cross-Corpus Beta Comparison: Natural Stories vs GECO\n(95% HDI error bars)")
    ax.legend()
    plt.tight_layout()
    out = figures_dir / "geco_vs_ns_betas.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


def plot_ic_replication(geco_idata: dict, ns_metrics: Path, figures_dir: Path) -> None:
    """
    H2 replication: posterior of β_IC on GECO (should still span zero).
    Side-by-side with Natural Stories posterior.
    """
    import arviz as az

    geco_vic = geco_idata.get("surprisal_vs_ic")
    ns_vic_path = ns_metrics / "06_bayes_surprisal_vs_ic.nc"

    if geco_vic is None:
        logger.warning("GECO surprisal_vs_ic model not available for H2 replication.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, label, path_or_idata in [
        (axes[0], "Natural Stories", ns_vic_path),
        (axes[1], "GECO",           geco_vic),
    ]:
        if isinstance(path_or_idata, Path):
            if not path_or_idata.exists():
                ax.set_title(f"{label}\n(not found)")
                continue
            idata = az.from_netcdf(str(path_or_idata))
        else:
            idata = path_or_idata

        m, lo, hi = posterior_summary(idata, "beta_integration_cost")
        if np.isnan(m):
            ax.set_title(f"{label}\n(beta_IC not in model)")
            continue

        samples = idata.posterior["beta_integration_cost"].values.flatten()
        sig     = not (lo < 0 < hi)
        ax.hist(samples, bins=60, color="tomato" if not sig else "steelblue",
                alpha=0.75, edgecolor="white")
        ax.axvline(0,  color="black", linestyle="--", linewidth=1.2, label="zero")
        ax.axvline(lo, color="gray",  linestyle=":",  linewidth=0.9)
        ax.axvline(hi, color="gray",  linestyle=":",  linewidth=0.9, label="95% HDI")
        ax.set_xlabel("β Integration Cost (z-scored)")
        ax.set_title(
            f"H2 replication — {label}\n"
            f"mean={m:.3f}  HDI=[{lo:.3f},{hi:.3f}]  "
            f"{'significant ✓' if sig else 'n.s. ✓ (expected)'}"
        )
        ax.legend(fontsize=9)

    fig.suptitle("H2: Is Integration Cost Still Explained Away on GECO?", fontsize=12)
    plt.tight_layout()
    out = figures_dir / "geco_h2_ic_replication.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


def write_cross_dataset_table(
    geco_idata: dict,
    ns_metrics: Path,
    metrics_dir: Path,
) -> None:
    """
    Produce a CSV summarising key β coefficients and significance across both corpora.
    """
    import arviz as az

    ns_models = {
        "surprisal_vs_ic":           ns_metrics / "06_bayes_surprisal_vs_ic.nc",
        "surprisal_vs_entropy_gpt2": ns_metrics / "06_bayes_surprisal_vs_entropy_gpt2.nc",
        "full":                      ns_metrics / "06_bayes_full.nc",
    }

    rows = []
    for variant in ("surprisal_vs_ic", "surprisal_vs_entropy_gpt2", "full"):
        # Natural Stories
        ns_path = ns_models[variant]
        ns_id   = az.from_netcdf(str(ns_path)) if ns_path.exists() else None
        # GECO
        geco_id = geco_idata.get(variant)

        for corpus, idata in [("Natural Stories", ns_id), ("GECO", geco_id)]:
            if idata is None:
                continue
            for var in idata.posterior.data_vars:
                if not var.startswith("beta_"):
                    continue
                m, lo, hi = posterior_summary(idata, var)
                rows.append({
                    "corpus":    corpus,
                    "variant":   variant,
                    "predictor": var.replace("beta_", ""),
                    "mean":      round(m, 4),
                    "hdi_2.5":   round(lo, 4),
                    "hdi_97.5":  round(hi, 4),
                    "sig":       not (lo < 0 < hi),
                })

    if rows:
        out_df = pd.DataFrame(rows)
        out    = metrics_dir / "cross_dataset_comparison.csv"
        out_df.to_csv(out, index=False)
        logger.info("Cross-dataset comparison table saved to %s", out)
        print("\n── Cross-dataset coefficient comparison ──────────────")
        print(out_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GECO cross-dataset validation")
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--steps", default="all",
        help="Steps to run: 'all' or comma-separated subset of 1,2,3,5,6 "
             "(step 4 / attention not included — already done on NS)."
    )
    p.add_argument(
        "--out-dir", default="/tmp/psycholingu_geco",
        help="Root output directory for GECO results."
    )
    p.add_argument(
        "--ns-metrics", default="/tmp/psycholingu/results/metrics",
        help="Path to existing Natural Stories metrics directory."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Point config at GECO ────────────────────────────────────────────────
    cfg["dataset"] = "geco"

    proc_dir    = Path(args.out_dir) / "processed"
    metrics_dir = Path(args.out_dir) / "metrics"
    figures_dir = Path(args.out_dir) / "figures"
    for d in (proc_dir, metrics_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    ns_metrics = Path(args.ns_metrics)

    steps_str = args.steps.strip().lower()
    if steps_str == "all":
        steps = {1, 2, 3, 5, 6}
    elif steps_str == "":
        steps = set()  # plots-only mode — load cached results, skip compute
    else:
        steps = {int(s) for s in steps_str.split(",") if s.strip()}

    logger.info("Running GECO validation — steps: %s", sorted(steps))

    # ── Step 1: load GECO ───────────────────────────────────────────────────
    df = run_step1(cfg, proc_dir)

    # ── Step 2: n-gram surprisal ────────────────────────────────────────────
    if 2 in steps:
        df = run_step2(cfg, df, proc_dir)
    else:
        p = proc_dir / "02_geco_ngram_surprisal.parquet"
        if p.exists():
            df = pd.read_parquet(p)

    # ── Step 3: neural metrics ──────────────────────────────────────────────
    if 3 in steps:
        df = run_step3(cfg, df, proc_dir)
    else:
        p = proc_dir / "03_geco_neural_metrics.parquet"
        if p.exists():
            df = pd.read_parquet(p)

    # ── Step 5: integration cost ─────────────────────────────────────────────
    if 5 in steps:
        df = run_step5(cfg, df, proc_dir)
    else:
        p = proc_dir / "05_geco_integration_cost.parquet"
        if p.exists():
            df = pd.read_parquet(p)

    # Normalise column name (integration_cost.py writes dep_length)
    if "dep_length" in df.columns and "integration_cost" not in df.columns:
        df = df.rename(columns={"dep_length": "integration_cost"})

    # ── Step 6: Bayesian modeling ────────────────────────────────────────────
    if 6 in steps:
        geco_idata, geco_loo = run_step6(cfg, df, metrics_dir, figures_dir)
    else:
        # Load any already-fitted models
        import arviz as az
        geco_idata = {}
        for name in GECO_VARIANTS:
            nc = metrics_dir / f"geco_bayes_{name}.nc"
            if nc.exists():
                geco_idata[name] = az.from_netcdf(str(nc))

    # ── Comparison plots ─────────────────────────────────────────────────────
    logger.info("Generating cross-dataset comparison plots …")

    geco_comp_path = metrics_dir / "geco_model_comparison.csv"
    plot_elpd_comparison(geco_comp_path, ns_metrics, figures_dir)
    plot_beta_comparison(geco_idata, ns_metrics, figures_dir)
    plot_ic_replication(geco_idata, ns_metrics, figures_dir)
    write_cross_dataset_table(geco_idata, ns_metrics, metrics_dir)

    logger.info("GECO validation complete.  Outputs: %s", Path(args.out_dir))
    print(f"\nOutputs saved to: {args.out_dir}")
    print(f"  Figures : {figures_dir}")
    print(f"  Metrics : {metrics_dir}")


if __name__ == "__main__":
    main()

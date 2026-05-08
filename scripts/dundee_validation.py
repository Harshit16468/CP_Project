"""
scripts/dundee_validation.py
============================
Cross-dataset validation on the Dundee Corpus eye-tracking data.

Why Dundee matters for this project
-------------------------------------
Reference #3 in the proposal — Demberg & Keller (2008) "Data from eye-tracking
corpora as evidence for theories of syntactic processing complexity" — used the
Dundee Corpus *specifically* to test surprisal vs. integration cost.
Running H2 (Prediction vs. Memory) on Dundee is therefore a **direct
replication** of the benchmark the project is measured against.

Access note
-----------
Dundee is NOT freely downloadable.  Obtain it via:
  - Original authors: Alan Kennedy (Univ. of Dundee) & Joel Pynte (Univ. of Aix)
  - Or an institutional linguistic data consortium (LDC / ELRA)

Once you have the .dat files, put them at the path in config.yaml dundee.data_dir
and run:
    python scripts/dundee_validation.py --config config.yaml

What it produces
----------------
  /tmp/psycholingu_dundee/
    figures/
      dundee_vs_ns_elpd.png        – architecture ELPD ranking (H4 replication)
      dundee_vs_ns_betas.png       – β coefficient comparison (Natural Stories vs Dundee)
      dundee_h2_ic_replication.png – H2: is β_IC still n.s. on Dundee?
    metrics/
      dundee_model_comparison.csv
      dundee_summary_<variant>.csv
      cross_dataset_comparison_dundee.csv

Usage
-----
    python scripts/dundee_validation.py [--config config.yaml]
                                         [--steps all|1,2,3,5,6]
                                         [--out-dir /tmp/psycholingu_dundee]
                                         [--ns-metrics /tmp/psycholingu/results/metrics]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME",              "/tmp/psycholingu/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE",   "/tmp/psycholingu/hf_cache/hub")
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

from src.data_prep        import load_dataset
from src.ngram_surprisal  import build_ngram_model, compute_ngram_surprisal
from src.neural_metrics   import NeuralMetricsExtractor
from src.integration_cost import build_parser, compute_integration_cost
from src.bayesian_model   import BayesianHierarchicalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dundee_validation")

DUNDEE_VARIANTS = {
    "baseline":                   ["ngram_surprisal"],
    "deep_gpt2":                  ["gpt2_surprisal"],
    "deep_bert":                  ["bert_base_uncased_surprisal"],
    "deep_t5":                    ["t5_base_surprisal"],
    "surprisal_vs_ic":            ["gpt2_surprisal", "integration_cost"],
    "surprisal_vs_entropy_gpt2":  ["gpt2_surprisal", "gpt2_entropy"],
    "surprisal_vs_entropy_bert":  ["bert_base_uncased_surprisal", "bert_base_uncased_entropy"],
    "surprisal_vs_entropy_t5":    ["t5_base_surprisal", "t5_base_entropy"],
    "full": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def parquet_cache(path: Path, fn, *args, **kwargs) -> pd.DataFrame:
    if path.exists():
        logger.info("Loading cached: %s", path)
        return pd.read_parquet(path)
    df = fn(*args, **kwargs)
    df.to_parquet(path, index=True)
    return df


def posterior_summary(idata, var: str) -> tuple[float, float, float]:
    if var not in idata.posterior:
        return float("nan"), float("nan"), float("nan")
    s = idata.posterior[var].values.flatten()
    return float(s.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def run_step1(cfg: dict, proc_dir: Path) -> pd.DataFrame:
    logger.info("Step 1 — Loading Dundee data")
    return parquet_cache(
        proc_dir / "01_dundee_reading_times.parquet",
        load_dataset, "dundee", cfg,
    )


def run_step2(cfg: dict, df: pd.DataFrame, proc_dir: Path) -> pd.DataFrame:
    logger.info("Step 2 — N-gram surprisal")
    p = proc_dir / "02_dundee_ngram_surprisal.parquet"
    if p.exists():
        return pd.read_parquet(p)
    corpus_path = Path(cfg["paths"]["ngram_corpus"])
    model_cache = proc_dir / "ngram_model.pkl"
    model = build_ngram_model(corpus_path, cfg["ngram"], cache_path=model_cache)
    df    = compute_ngram_surprisal(df, model)
    df.to_parquet(p, index=True)
    return df


def run_step3(cfg: dict, df: pd.DataFrame, proc_dir: Path) -> pd.DataFrame:
    logger.info("Step 3 — Neural surprisal & entropy")
    p = proc_dir / "03_dundee_neural_metrics.parquet"
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
    p = proc_dir / "05_dundee_integration_cost.parquet"
    if p.exists():
        return pd.read_parquet(p)
    parser = build_parser(cfg)
    df     = compute_integration_cost(df, parser)
    df.to_parquet(p, index=True)
    return df


def run_step6(cfg: dict, df: pd.DataFrame, metrics_dir: Path, figures_dir: Path) -> tuple:
    import arviz as az

    bay_cfg  = cfg["bayesian"].copy()
    all_pred = bay_cfg["predictors"]
    variants = {k: (v if v is not None else all_pred) for k, v in DUNDEE_VARIANTS.items()}

    idata_results: dict = {}
    loo_results:   dict = {}

    for name, preds in variants.items():
        nc_path  = metrics_dir / f"dundee_bayes_{name}.nc"
        loo_pkl  = metrics_dir / f"dundee_loo_{name}.pkl"

        if nc_path.exists():
            logger.info("Loading cached Dundee model: %s", name)
            idata_results[name] = az.from_netcdf(str(nc_path))
            if loo_pkl.exists():
                with open(loo_pkl, "rb") as fh:
                    loo_results[name] = pickle.load(fh)
            continue

        v_cfg = {**bay_cfg, "predictors": preds}
        bm    = BayesianHierarchicalModel(v_cfg)
        logger.info("Fitting Dundee model: %s  predictors=%s", name, preds)
        idata = bm.fit(df)

        if bm.last_loo is not None:
            with open(loo_pkl, "wb") as fh:
                pickle.dump(bm.last_loo, fh)
            loo_results[name] = bm.last_loo

        bm.save(idata, nc_path)
        bm.summary(idata).to_csv(metrics_dir / f"dundee_summary_{name}.csv")
        idata_results[name] = idata

    if len(loo_results) > 1:
        bm_ref = BayesianHierarchicalModel(bay_cfg)
        comp   = bm_ref.compare_models(loo_results)
        comp.to_csv(metrics_dir / "dundee_model_comparison.csv")

    return idata_results, loo_results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset comparison plots  (Dundee vs Natural Stories)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ns_comp(ns_metrics: Path) -> pd.DataFrame | None:
    p = ns_metrics / "06_model_comparison.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


def plot_elpd_comparison(dundee_metrics: Path, ns_metrics: Path, figures_dir: Path) -> None:
    dundee_comp = (pd.read_csv(dundee_metrics / "dundee_model_comparison.csv", index_col=0)
                   if (dundee_metrics / "dundee_model_comparison.csv").exists() else None)
    ns_comp = _load_ns_comp(ns_metrics)

    arch_rows = ["baseline", "deep_gpt2", "deep_bert", "deep_t5"]
    label_map = {"baseline": "Trigram", "deep_gpt2": "GPT-2",
                 "deep_bert": "BERT",   "deep_t5":   "T5"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, comp, title in zip(
        axes,
        [ns_comp,     dundee_comp],
        ["Natural Stories (self-paced RT)", "Dundee (eye-tracking, Demberg & Keller 2008)"],
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

    fig.suptitle("Cross-Dataset Architecture Comparison — H4 replication on Dundee", fontsize=13)
    plt.tight_layout()
    out = figures_dir / "dundee_vs_ns_elpd.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


def plot_h2_replication(dundee_idata: dict, ns_metrics: Path, figures_dir: Path) -> None:
    """
    Core Dundee contribution: H2 replication on the Demberg & Keller (2008) corpus.
    Shows whether β_IC still becomes non-significant once GPT-2 surprisal is controlled.
    """
    import arviz as az

    gd_vic = dundee_idata.get("surprisal_vs_ic")
    ns_path = ns_metrics / "06_bayes_surprisal_vs_ic.nc"

    if gd_vic is None:
        logger.warning("Dundee surprisal_vs_ic model not available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, label, src in [
        (axes[0], "Natural Stories", ns_path),
        (axes[1], "Dundee (Demberg & Keller 2008 corpus)", gd_vic),
    ]:
        idata = (az.from_netcdf(str(src)) if isinstance(src, Path) and src.exists()
                 else src if not isinstance(src, Path) else None)
        if idata is None:
            ax.set_title(f"{label}\n(not found)")
            continue

        m, lo, hi = posterior_summary(idata, "beta_integration_cost")
        if np.isnan(m):
            ax.set_title(f"{label}\nbeta_IC not in model")
            continue

        samples = idata.posterior["beta_integration_cost"].values.flatten()
        sig = not (lo < 0 < hi)
        ax.hist(samples, bins=60,
                color="steelblue" if sig else "tomato", alpha=0.75, edgecolor="white")
        ax.axvline(0,  color="black", linestyle="--", linewidth=1.2)
        ax.axvline(lo, color="gray",  linestyle=":", linewidth=0.9)
        ax.axvline(hi, color="gray",  linestyle=":", linewidth=0.9, label="95% HDI")
        ax.set_xlabel("β Integration Cost (z-scored)")
        ax.set_title(
            f"H2 on {label}\n"
            f"mean={m:.3f}  HDI=[{lo:.3f},{hi:.3f}]  "
            f"{'SIGNIFICANT (unexpected)' if sig else 'n.s. ✓ (expected)'}"
        )
        ax.legend(fontsize=9)

    fig.suptitle(
        "H2 Replication on Dundee: Is Integration Cost Still Explained Away?\n"
        "(Demberg & Keller 2008 originally found IC significant — we predict neural surprisal absorbs it)",
        fontsize=11,
    )
    plt.tight_layout()
    out = figures_dir / "dundee_h2_ic_replication.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


def plot_beta_comparison(dundee_idata: dict, ns_metrics: Path, figures_dir: Path) -> None:
    import arviz as az

    d_full = dundee_idata.get("full")
    ns_path = ns_metrics / "06_bayes_full.nc"
    if d_full is None or not ns_path.exists():
        logger.warning("Cannot produce beta comparison — missing full models.")
        return
    ns_full = az.from_netcdf(str(ns_path))

    interest = [
        ("beta_gpt2_surprisal",              "GPT-2 surprisal"),
        ("beta_integration_cost",            "Integration cost"),
        ("beta_gpt2_entropy",                "GPT-2 entropy"),
        ("beta_bert_base_uncased_surprisal", "BERT surprisal"),
        ("beta_t5_base_surprisal",           "T5 surprisal"),
    ]

    rows = []
    for var, label in interest:
        for cname, idata in [("Natural Stories", ns_full), ("Dundee", d_full)]:
            m, lo, hi = posterior_summary(idata, var)
            if not np.isnan(m):
                rows.append({"corpus": cname, "predictor": label,
                             "mean": m, "lo": lo, "hi": hi})
    if not rows:
        return

    df_plot    = pd.DataFrame(rows)
    predictors = df_plot["predictor"].unique()
    n          = len(predictors)
    y_ns       = np.arange(n) * 2.5
    y_dundee   = y_ns + 0.8

    fig, ax = plt.subplots(figsize=(9, max(4, n * 1.5)))
    for y_pos, grp, color, lab in [
        (y_ns,     df_plot[df_plot["corpus"] == "Natural Stories"], "#2980b9", "Natural Stories"),
        (y_dundee, df_plot[df_plot["corpus"] == "Dundee"],          "#8e44ad", "Dundee"),
    ]:
        sub = grp.set_index("predictor").reindex(predictors)
        ax.scatter(sub["mean"], y_pos, color=color, zorder=3, s=60, label=lab)
        for i, (m, lo, hi) in enumerate(zip(sub["mean"], sub["lo"], sub["hi"])):
            if not np.isnan(m):
                ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=color, linewidth=2, alpha=0.7)

    ax.set_yticks((y_ns + y_dundee) / 2)
    ax.set_yticklabels(predictors)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Posterior β  (z-scored predictors)")
    ax.set_title("Cross-Corpus β Comparison: Natural Stories vs Dundee\n(95% HDI error bars)")
    ax.legend()
    plt.tight_layout()
    out = figures_dir / "dundee_vs_ns_betas.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


def write_cross_dataset_table(dundee_idata: dict, ns_metrics: Path, metrics_dir: Path) -> None:
    import arviz as az

    ns_models = {
        "surprisal_vs_ic":           ns_metrics / "06_bayes_surprisal_vs_ic.nc",
        "surprisal_vs_entropy_gpt2": ns_metrics / "06_bayes_surprisal_vs_entropy_gpt2.nc",
        "full":                      ns_metrics / "06_bayes_full.nc",
    }
    rows = []
    for variant in ("surprisal_vs_ic", "surprisal_vs_entropy_gpt2", "full"):
        for corpus, src in [
            ("Natural Stories", ns_models[variant]),
            ("Dundee",          dundee_idata.get(variant)),
        ]:
            if src is None:
                continue
            idata = az.from_netcdf(str(src)) if isinstance(src, Path) and src.exists() else (
                src if not isinstance(src, Path) else None
            )
            if idata is None:
                continue
            for var in idata.posterior.data_vars:
                if not var.startswith("beta_"):
                    continue
                m, lo, hi = posterior_summary(idata, var)
                rows.append({
                    "corpus": corpus, "variant": variant,
                    "predictor": var.replace("beta_", ""),
                    "mean": round(m, 4), "hdi_2.5": round(lo, 4), "hdi_97.5": round(hi, 4),
                    "sig": not (lo < 0 < hi),
                })

    if rows:
        out = metrics_dir / "cross_dataset_comparison_dundee.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Cross-dataset table saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dundee cross-dataset validation")
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--steps",      default="all",
                   help="Steps: 'all' or comma-separated subset of 1,2,3,5,6")
    p.add_argument("--out-dir",    default="/tmp/psycholingu_dundee")
    p.add_argument("--ns-metrics", default="/tmp/psycholingu/results/metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    cfg["dataset"] = "dundee"

    proc_dir    = ensure_dir(Path(args.out_dir) / "processed")
    metrics_dir = ensure_dir(Path(args.out_dir) / "metrics")
    figures_dir = ensure_dir(Path(args.out_dir) / "figures")
    ns_metrics  = Path(args.ns_metrics)

    steps_str = args.steps.strip().lower()
    if steps_str == "all":
        steps = {1, 2, 3, 5, 6}
    elif steps_str == "":
        steps = set()  # plots-only mode — load cached results, skip compute
    else:
        steps = {int(s) for s in steps_str.split(",") if s.strip()}
    logger.info("Running Dundee validation — steps: %s", sorted(steps))

    df = run_step1(cfg, proc_dir)

    if 2 in steps:
        df = run_step2(cfg, df, proc_dir)
    else:
        p = proc_dir / "02_dundee_ngram_surprisal.parquet"
        if p.exists(): df = pd.read_parquet(p)

    if 3 in steps:
        df = run_step3(cfg, df, proc_dir)
    else:
        p = proc_dir / "03_dundee_neural_metrics.parquet"
        if p.exists(): df = pd.read_parquet(p)

    if 5 in steps:
        df = run_step5(cfg, df, proc_dir)
    else:
        p = proc_dir / "05_dundee_integration_cost.parquet"
        if p.exists(): df = pd.read_parquet(p)

    if "dep_length" in df.columns and "integration_cost" not in df.columns:
        df = df.rename(columns={"dep_length": "integration_cost"})

    if 6 in steps:
        dundee_idata, _ = run_step6(cfg, df, metrics_dir, figures_dir)
    else:
        import arviz as az
        dundee_idata = {}
        for name in DUNDEE_VARIANTS:
            nc = metrics_dir / f"dundee_bayes_{name}.nc"
            if nc.exists():
                dundee_idata[name] = az.from_netcdf(str(nc))

    logger.info("Generating comparison plots …")
    plot_elpd_comparison(metrics_dir, ns_metrics, figures_dir)
    plot_h2_replication(dundee_idata, ns_metrics, figures_dir)
    plot_beta_comparison(dundee_idata, ns_metrics, figures_dir)
    write_cross_dataset_table(dundee_idata, ns_metrics, metrics_dir)

    logger.info("Dundee validation complete.  Outputs: %s", args.out_dir)
    print(f"\nOutputs saved to: {args.out_dir}")
    print(f"  Figures : {figures_dir}")
    print(f"  Metrics : {metrics_dir}")


if __name__ == "__main__":
    main()

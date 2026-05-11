"""
scripts/llama_validation.py
============================
Scale-invariance replication using LLaMA 3 8B.

Runs LLaMA 3 8B surprisal and entropy extraction on Natural Stories and
Dundee, then fits four Bayesian model variants per corpus:
  - deep_llama           : LLaMA surprisal only
  - surprisal_vs_ic_llama: LLaMA surprisal + integration cost
  - surprisal_vs_entropy_llama: LLaMA surprisal + entropy
  - spillover_llama      : LLaMA current + lag-1 + controls

Results are compared against GPT-2 / BERT / T5 to test whether the
processing-directionality effects hold at modern LLM scale.

Prerequisites
-------------
  1. Obtain HuggingFace access to meta-llama/Meta-Llama-3-8B
       huggingface-cli login
  2. At least 16 GB GPU VRAM (fp16) or 32 GB RAM (CPU, slow)

Usage
-----
    python scripts/llama_validation.py --config config.yaml
    python scripts/llama_validation.py --config config.yaml --corpus dundee
    python scripts/llama_validation.py --config config.yaml --steps 3,6
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

from src.neural_metrics  import NeuralMetricsExtractor
from src.bayesian_model  import BayesianHierarchicalModel
from src.lexical_features import add_lexical_controls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llama_validation")

# Model variants to fit for each corpus
LLAMA_VARIANTS = {
    "deep_llama":                 ["llama3_surprisal"],
    "surprisal_vs_ic_llama":      ["llama3_surprisal", "integration_cost"],
    "surprisal_vs_entropy_llama": ["llama3_surprisal", "llama3_entropy"],
    "spillover_llama":            ["llama3_surprisal", "llama3_surprisal_lag1",
                                   "log_freq", "word_length"],
}

# GPT-2 counterparts for comparison (must already exist in results dir)
GPT2_COUNTERPARTS = {
    "deep_llama":                 "deep_gpt2",
    "surprisal_vs_ic_llama":      "surprisal_vs_ic",
    "surprisal_vs_entropy_llama": "surprisal_vs_entropy_gpt2",
    "spillover_llama":            "spillover",
}


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


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: LLaMA 3 extraction
# ─────────────────────────────────────────────────────────────────────────────

def run_llama_extraction(cfg: dict, df: pd.DataFrame, proc_dir: Path,
                         corpus_tag: str) -> pd.DataFrame:
    """Extract LLaMA 3 surprisal and entropy; cache result."""
    cache = proc_dir / f"03_llama3_{corpus_tag}_metrics.parquet"
    if cache.exists():
        logger.info("Loading cached LLaMA 3 metrics from %s", cache)
        existing = pd.read_parquet(cache)
        # Merge new columns onto df if they aren't already there
        new_cols = [c for c in existing.columns
                    if c.startswith("llama3_") and c not in df.columns]
        if new_cols:
            df = df.merge(existing[["story_id", "sentence_id",
                                    "word_position", "subject"] + new_cols],
                          on=["story_id", "sentence_id", "word_position", "subject"],
                          how="left")
        return df

    lc = cfg["models"]["llama3"]
    logger.info("Loading LLaMA 3 8B (fp16=%s) …", lc.get("use_half_precision", True))
    extractor = NeuralMetricsExtractor(
        lc["name"],
        lc.get("type", "causal"),
        lc.get("device", "auto"),
        use_half_precision=lc.get("use_half_precision", True),
        column_prefix=lc.get("column_prefix", "llama3"),
    )
    df = extractor.compute_metrics(df)
    del extractor

    df = add_lexical_controls(df, cfg)
    df.to_parquet(cache, index=True)
    logger.info("Cached LLaMA 3 metrics to %s", cache)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Bayesian fitting
# ─────────────────────────────────────────────────────────────────────────────

def run_bayesian_variants(cfg: dict, df: pd.DataFrame,
                          metrics_dir: Path, prefix: str) -> dict:
    """Fit the four LLaMA Bayesian variants; return {name: ELPDData}."""
    import arviz as az
    bay_cfg  = cfg["bayesian"].copy()
    loo_dict = {}

    for name, preds in LLAMA_VARIANTS.items():
        nc_path  = metrics_dir / f"{prefix}_bayes_{name}.nc"
        loo_pkl  = metrics_dir / f"{prefix}_loo_{name}.pkl"

        if nc_path.exists():
            logger.info("Loading cached model: %s", name)
            idata = az.from_netcdf(str(nc_path))
        else:
            available = [p for p in preds if p in df.columns]
            missing   = set(preds) - set(available)
            if missing:
                logger.warning("Skipping %s — missing columns: %s", name, missing)
                continue

            v_cfg = {
                **bay_cfg,
                "predictors":   available,
                "random_slopes": [s for s in bay_cfg.get("random_slopes", [])
                                  if s in available],
            }
            bm    = BayesianHierarchicalModel(v_cfg)
            logger.info("Fitting %s  predictors=%s", name, available)
            idata = bm.fit(df)

            if bm.last_loo is not None:
                with open(loo_pkl, "wb") as fh:
                    pickle.dump(bm.last_loo, fh)
            bm.save(idata, nc_path)
            bm.summary(idata).to_csv(
                metrics_dir / f"{prefix}_summary_{name}.csv"
            )

        if loo_pkl.exists():
            with open(loo_pkl, "rb") as fh:
                loo_dict[name] = pickle.load(fh)

    return loo_dict


# ─────────────────────────────────────────────────────────────────────────────
# Comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_scale_invariance(cfg: dict, metrics_dir: Path,
                          figures_dir: Path, prefix: str) -> None:
    """
    Figure: LLaMA 3 vs GPT-2 β coefficients side-by-side.
    Tests whether effects replicate at modern model scale.
    """
    import arviz as az

    rows = []
    for llama_name, gpt2_name in GPT2_COUNTERPARTS.items():
        for model_tag, nc_name in [(f"llama3", llama_name), ("gpt2", gpt2_name)]:
            nc = metrics_dir / f"{prefix}_bayes_{nc_name}.nc"
            if not nc.exists():
                # Try the NS pipeline output path for GPT-2 (06_bayes_*)
                nc = metrics_dir.parent / "metrics" / f"06_bayes_{nc_name}.nc"
            if not nc.exists():
                continue
            idata = az.from_netcdf(str(nc))
            for var in idata.posterior.data_vars:
                if not var.startswith("beta_"):
                    continue
                s = idata.posterior[var].values.flatten()
                m, lo, hi = float(s.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
                rows.append({
                    "model":     model_tag,
                    "variant":   llama_name,
                    "predictor": var.replace("beta_", ""),
                    "mean": m, "lo": lo, "hi": hi,
                    "sig": not (lo < 0 < hi),
                })

    if not rows:
        logger.warning("No model results found for scale-invariance plot.")
        return

    df_plot = pd.DataFrame(rows)
    predictors = sorted(df_plot["predictor"].unique())
    n = len(predictors)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 1.2)))
    colors = {"llama3": "#e67e22", "gpt2": "#2980b9"}
    offsets = {"llama3": 0.3, "gpt2": -0.3}

    for model_tag, color in colors.items():
        sub = df_plot[df_plot["model"] == model_tag]
        for i, pred in enumerate(predictors):
            row = sub[sub["predictor"] == pred]
            if row.empty:
                continue
            row = row.iloc[0]
            y = i + offsets[model_tag]
            ax.scatter(row["mean"], y, color=color, s=60, zorder=3,
                       label=model_tag if i == 0 else "")
            ax.plot([row["lo"], row["hi"]], [y, y], color=color, linewidth=2, alpha=0.7)

    ax.set_yticks(range(n))
    ax.set_yticklabels(predictors, fontsize=9)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Posterior β (z-scored predictors, 95% HDI)")
    ax.set_title(f"Scale-Invariance Check: LLaMA 3 8B vs GPT-2\n"
                 f"({prefix.upper()} corpus — effects should replicate)")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, markersize=8, label=t)
               for t, c in colors.items()]
    ax.legend(handles=handles)
    plt.tight_layout()
    out = figures_dir / f"{prefix}_llama_vs_gpt2_scale_invariance.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


def plot_architecture_ranking(metrics_dir: Path, figures_dir: Path,
                               prefix: str) -> None:
    """Extended Figure 1 including LLaMA 3 in the architecture comparison."""
    import pickle

    elpd_vals = {}
    # Collect ELPD for all architecture variants (including LLaMA)
    arch_variants = {
        "Trigram (baseline)": "baseline",
        "GPT-2 (causal)":     "deep_gpt2",
        "BERT (masked)":      "deep_bert",
        "T5 (enc-dec)":       "deep_t5",
        "LLaMA 3 8B (causal)": "deep_llama",
    }
    for label, name in arch_variants.items():
        # Look in prefix dir first, then fallback NS metrics dir
        for stem in [f"{prefix}_loo_{name}.pkl", f"06_loo_{name}.pkl"]:
            pkl = metrics_dir / stem
            if not pkl.exists():
                pkl = metrics_dir.parent / "metrics" / f"06_loo_{name}.pkl"
            if pkl.exists():
                with open(pkl, "rb") as fh:
                    loo = pickle.load(fh)
                elpd_vals[label] = float(loo.elpd_loo)
                break

    if not elpd_vals:
        logger.warning("No LOO results found for architecture ranking plot.")
        return

    labels = list(elpd_vals.keys())
    vals   = [elpd_vals[l] for l in labels]
    colors = ["#c0392b" if v == min(vals) else "#27ae60" if v == max(vals)
              else "#2980b9" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(labels, vals, color=colors, edgecolor="white", height=0.5)
    ax.set_xlabel("ELPD (LOO-CV) ← worse   better →")
    ax.set_title(f"Architecture Ranking Including LLaMA 3 8B\n"
                 f"({prefix.upper()} corpus)")
    ax.axvline(min(vals) * 0.98, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out = figures_dir / f"{prefix}_architecture_ranking_with_llama.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLaMA 3 scale-invariance validation")
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--corpus",  default="both",
                   help="'ns', 'dundee', or 'both' (default: both)")
    p.add_argument("--steps",   default="3,6",
                   help="Steps to run: 3=extraction, 6=bayesian (default: 3,6)")
    p.add_argument("--out-dir", default="/tmp/psycholingu_llama")
    p.add_argument("--ns-processed",
                   default="/tmp/psycholingu/data/processed",
                   help="Path to cached NS processed parquets")
    p.add_argument("--dundee-processed",
                   default="/tmp/psycholingu_dundee/processed",
                   help="Path to cached Dundee processed parquets")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    cfg     = load_config(args.config)
    steps   = {int(s) for s in args.steps.split(",") if s.strip()}
    out_dir = Path(args.out_dir)

    corpora = []
    if args.corpus in ("ns", "both"):
        corpora.append(("ns", Path(args.ns_processed)))
    if args.corpus in ("dundee", "both"):
        corpora.append(("dundee", Path(args.dundee_processed)))

    if "llama3" not in cfg.get("models", {}):
        logger.error("No 'llama3' section in config.yaml models. "
                     "Add it before running this script.")
        sys.exit(1)

    for corpus_tag, proc_dir in corpora:
        logger.info("=" * 60)
        logger.info("LLaMA 3 validation — corpus: %s", corpus_tag.upper())
        logger.info("=" * 60)

        metrics_dir = ensure_dir(out_dir / corpus_tag / "metrics")
        figures_dir = ensure_dir(out_dir / corpus_tag / "figures")

        # Load the most complete processed parquet available
        # (after step 5/lexical — produced by main pipeline or dundee_validation.py)
        candidate_caches = [
            proc_dir / "03b_lexical_derived.parquet",
            proc_dir / f"03b_{corpus_tag}_lexical_derived.parquet",
            proc_dir / "05_integration_cost.parquet",
            proc_dir / f"05_{corpus_tag}_integration_cost.parquet",
            proc_dir / "03_neural_metrics.parquet",
            proc_dir / f"03_{corpus_tag}_neural_metrics.parquet",
        ]
        df = None
        for c in candidate_caches:
            if c.exists():
                logger.info("Loading base DataFrame from %s", c)
                df = pd.read_parquet(c)
                break
        if df is None:
            logger.error(
                "No processed parquet found in %s. "
                "Run the main pipeline (or dundee_validation.py) first.",
                proc_dir,
            )
            sys.exit(1)

        # ── Step 3: LLaMA extraction ─────────────────────────────────────────
        if 3 in steps:
            df = run_llama_extraction(cfg, df, metrics_dir, corpus_tag)
        else:
            cache = metrics_dir / f"03_llama3_{corpus_tag}_metrics.parquet"
            if cache.exists():
                existing = pd.read_parquet(cache)
                new_cols = [c for c in existing.columns
                            if c.startswith("llama3_") and c not in df.columns]
                if new_cols:
                    df = df.merge(
                        existing[["story_id", "sentence_id",
                                  "word_position", "subject"] + new_cols],
                        on=["story_id", "sentence_id", "word_position", "subject"],
                        how="left",
                    )

        # Check LLaMA columns are present
        if "llama3_surprisal" not in df.columns:
            logger.error("llama3_surprisal column not found. Run step 3 first.")
            sys.exit(1)

        # ── Step 6: Bayesian fitting ─────────────────────────────────────────
        if 6 in steps:
            loo_dict = run_bayesian_variants(
                cfg, df, metrics_dir, prefix=corpus_tag
            )
            if len(loo_dict) > 1:
                import arviz as az
                comp = az.compare(loo_dict, ic="loo")
                comp.to_csv(metrics_dir / f"{corpus_tag}_llama_model_comparison.csv")
                logger.info("LLaMA model comparison saved.")

        # ── Plots ────────────────────────────────────────────────────────────
        plot_scale_invariance(cfg, metrics_dir, figures_dir, corpus_tag)
        plot_architecture_ranking(metrics_dir, figures_dir, corpus_tag)

        logger.info("LLaMA validation complete for %s.  Outputs: %s",
                    corpus_tag, out_dir / corpus_tag)

    print(f"\nOutputs saved to: {out_dir}")
    print("Next step: run scripts/garden_path_analysis.py for the garden-path dissociation.")


if __name__ == "__main__":
    main()

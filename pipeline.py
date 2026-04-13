"""
pipeline.py
===========
Main entry point for the psycholinguistics project pipeline.

Disentangling Predictive Surprisal, Entropy, and Syntactic Integration Cost
in Human Sentence Processing.

Usage
-----
    python pipeline.py [--config config.yaml] [--steps all|1,2,3,4,5,6]

Steps
-----
  1  Behavioral data preparation
  2  N-gram surprisal (trigram baseline)
  3  Neural surprisal & entropy  (GPT-2, BERT, T5)
  4  Attention head analysis
  5  Integration cost  (UD dependency length)
  6  Bayesian hierarchical modeling

Intermediate results are cached to disk (data/processed/) so individual
steps can be re-run without recomputing upstream work.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Redirect all HuggingFace / torch model caches to /tmp
os.environ.setdefault("HF_HOME",              "/tmp/psycholingu/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE",   "/tmp/psycholingu/hf_cache/hub")
os.environ.setdefault("HF_DATASETS_CACHE",    "/tmp/psycholingu/hf_cache/datasets")
os.environ.setdefault("TORCH_HOME",           "/tmp/psycholingu/torch_cache")
os.environ.setdefault("XDG_CACHE_HOME",       "/tmp/psycholingu/xdg_cache")
# Stanza models
os.environ.setdefault("STANZA_RESOURCES_DIR", "/tmp/psycholingu/stanza_resources")

import pandas as pd
import yaml

# ─── ensure project root is on PYTHONPATH ────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_prep       import load_dataset
from src.ngram_surprisal import build_ngram_model, compute_ngram_surprisal
from src.neural_metrics  import NeuralMetricsExtractor
from src.attention_analysis import AttentionAnalyzer
from src.integration_cost   import build_parser, compute_integration_cost
from src.bayesian_model      import BayesianHierarchicalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def cache_path(cfg: dict, name: str) -> Path:
    base = Path(cfg["paths"]["processed_data"])
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def load_or_run(path: Path, fn, *args, **kwargs) -> pd.DataFrame:
    """Load cached parquet if it exists, otherwise run fn and cache."""
    if path.exists():
        logger.info("Loading cached data from %s", path)
        return pd.read_parquet(path)
    result = fn(*args, **kwargs)
    result.to_parquet(path, index=True)
    logger.info("Cached result to %s", path)
    return result


def makedirs(cfg: dict) -> None:
    for key in ("raw_data", "processed_data", "results", "figures"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step1_data_prep(cfg: dict) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 1 — Behavioral Data Preparation")
    logger.info("=" * 60)
    cache = cache_path(cfg, "01_reading_times.parquet")
    return load_or_run(cache, load_dataset, cfg["dataset"], cfg)


def step2_ngram(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 2 — N-Gram Surprisal (Trigram Kneser-Ney)")
    logger.info("=" * 60)
    cache = cache_path(cfg, "02_ngram_surprisal.parquet")
    if cache.exists():
        logger.info("Loading cached n-gram surprisal from %s", cache)
        return pd.read_parquet(cache)

    corpus_path     = Path(cfg["paths"]["ngram_corpus"])
    model_cache     = cache_path(cfg, "ngram_model.pkl")
    model           = build_ngram_model(corpus_path, cfg["ngram"], cache_path=model_cache)
    df              = compute_ngram_surprisal(df, model)
    df.to_parquet(cache, index=True)
    return df


def step3_neural(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 3 — Neural Surprisal & Entropy (GPT-2, BERT, T5)")
    logger.info("=" * 60)
    cache = cache_path(cfg, "03_neural_metrics.parquet")
    if cache.exists():
        logger.info("Loading cached neural metrics from %s", cache)
        return pd.read_parquet(cache)

    model_cfgs = [
        ("gpt2",        cfg["models"]["gpt2"]["name"],  "causal",  cfg["models"]["gpt2"]["device"]),
        ("bert",        cfg["models"]["bert"]["name"],  "masked",  cfg["models"]["bert"]["device"]),
        ("t5",          cfg["models"]["t5"]["name"],    "seq2seq", cfg["models"]["t5"]["device"]),
    ]

    for label, name, mtype, device in model_cfgs:
        logger.info("Processing model: %s (%s)", name, mtype)
        extractor = NeuralMetricsExtractor(name, mtype, device)
        df        = extractor.compute_metrics(df)
        del extractor   # free GPU memory between models

    df.to_parquet(cache, index=True)
    return df


def step4_attention(
    cfg: dict,
    df: pd.DataFrame,
    extractors: dict[str, NeuralMetricsExtractor] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run attention head analysis for each configured model.
    Returns a dict of {model_name: correlation_DataFrame}.
    """
    logger.info("=" * 60)
    logger.info("STEP 4 — Attention Head Analysis")
    logger.info("=" * 60)

    results_dir  = Path(cfg["paths"]["results"])
    figures_dir  = Path(cfg["paths"]["figures"])
    attn_models  = cfg["attention"]["models"]
    results: dict[str, pd.DataFrame] = {}

    for model_label in attn_models:
        model_cfg = cfg["models"][model_label]
        name      = model_cfg["name"]
        mtype     = model_cfg.get("type", "causal")
        device    = model_cfg.get("device", "auto")

        out_csv = results_dir / f"04_attention_{model_label}.csv"
        if out_csv.exists():
            logger.info("Loading cached attention results from %s", out_csv)
            results[model_label] = pd.read_csv(out_csv)
            continue

        extractor = NeuralMetricsExtractor(name, mtype, device)
        analyzer  = AttentionAnalyzer(extractor, df)
        corr_df   = analyzer.run()

        corr_df.to_csv(out_csv, index=False)
        logger.info("Saved attention correlations to %s", out_csv)

        if not corr_df.empty:
            analyzer.plot_top_heads(
                k=10,
                save_path=figures_dir / f"04_top_heads_{model_label}.png",
            )
        else:
            logger.warning("Skipping plot — no attention correlations collected "
                           "(dep_length column missing? run step 5 first).")
        results[model_label] = corr_df

    return results


def step5_integration_cost(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 5 — Integration Cost (UD Dependency Parsing)")
    logger.info("=" * 60)
    cache = cache_path(cfg, "05_integration_cost.parquet")
    if cache.exists():
        logger.info("Loading cached integration cost from %s", cache)
        return pd.read_parquet(cache)

    parser = build_parser(cfg)
    df     = compute_integration_cost(df, parser)
    df.to_parquet(cache, index=True)
    return df


def step6_bayesian(cfg: dict, df: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info("STEP 6 — Bayesian Hierarchical Modeling")
    logger.info("=" * 60)

    results_dir = Path(cfg["paths"]["results"])
    figures_dir = Path(cfg["paths"]["figures"])
    bay_cfg     = cfg["bayesian"]
    bm          = BayesianHierarchicalModel(bay_cfg)

    # ── Model variants (Hypotheses 1-4) ──────────────────────────────────────
    model_variants = {
        # H1: shallow baseline
        "baseline": {
            **bay_cfg,
            "predictors": ["ngram_surprisal"],
        },
        # H4: architecture comparison — one model per architecture
        "deep_gpt2": {
            **bay_cfg,
            "predictors": ["gpt2_surprisal"],
        },
        "deep_bert": {
            **bay_cfg,
            "predictors": ["bert_surprisal"],
        },
        "deep_t5": {                                   # H4: was missing
            **bay_cfg,
            "predictors": ["t5_surprisal"],
        },
        # H2: does IC explain variance beyond deep surprisal?
        "surprisal_vs_ic": {
            **bay_cfg,
            "predictors": ["gpt2_surprisal", "integration_cost"],
        },
        # H3: surprisal vs entropy — one variant per architecture
        "surprisal_vs_entropy_gpt2": {
            **bay_cfg,
            "predictors": ["gpt2_surprisal", "gpt2_entropy"],
        },
        "surprisal_vs_entropy_bert": {
            **bay_cfg,
            "predictors": ["bert_surprisal", "bert_entropy"],
        },
        "surprisal_vs_entropy_t5": {
            **bay_cfg,
            "predictors": ["t5_surprisal", "t5_entropy"],
        },
        # H6 + full comparison
        "full": bay_cfg,
    }

    idata_results = {}
    for variant_name, v_cfg in model_variants.items():
        nc_path = results_dir / f"06_bayes_{variant_name}.nc"
        if nc_path.exists():
            logger.info("Loading cached Bayesian model: %s", variant_name)
            idata_results[variant_name] = bm.load(nc_path)
            continue

        logger.info("Fitting Bayesian model: %s  predictors=%s",
                    variant_name, v_cfg["predictors"])
        variant_bm = BayesianHierarchicalModel(v_cfg)
        idata      = variant_bm.fit(df)
        variant_bm.save(idata, nc_path)

        # Diagnostics & plots
        summary = variant_bm.summary(idata)
        summary.to_csv(results_dir / f"06_summary_{variant_name}.csv")

        variant_bm.plot_posteriors(
            idata, save_dir=figures_dir
        )
        variant_bm.plot_forest(
            idata, save_dir=figures_dir
        )
        idata_results[variant_name] = idata

    # ── Model comparison (only when log-likelihood was computed) ─────────────
    if len(idata_results) > 1 and bay_cfg.get("compute_loo", False):
        logger.info("Running LOO-CV model comparison …")
        comparison = bm.compare_models(idata_results)
        comparison.to_csv(results_dir / "06_model_comparison.csv")
        logger.info("Model comparison saved.")
    elif len(idata_results) > 1:
        logger.info("Skipping LOO comparison (compute_loo=False in config).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Psycholinguistics pipeline: surprisal, entropy, integration cost"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    parser.add_argument(
        "--steps", default="all",
        help=(
            "Comma-separated list of steps to run, e.g. '1,2,5,6'. "
            "Use 'all' to run everything (default: all)."
        ),
    )
    return parser.parse_args()


def parse_steps(steps_str: str) -> set[int]:
    if steps_str.strip().lower() == "all":
        return {1, 2, 3, 4, 5, 6}
    return {int(s.strip()) for s in steps_str.split(",")}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args  = parse_args()
    cfg   = load_config(args.config)
    steps = parse_steps(args.steps)

    makedirs(cfg)
    logger.info("Running steps: %s", sorted(steps))

    df: pd.DataFrame | None = None

    # ── Step 1: Data prep ────────────────────────────────────────────────────
    if 1 in steps:
        df = step1_data_prep(cfg)
    else:
        # Try loading from cache so downstream steps still work
        cache = cache_path(cfg, "01_reading_times.parquet")
        if cache.exists():
            df = pd.read_parquet(cache)
        else:
            logger.error("Step 1 must be run at least once before skipping it.")
            sys.exit(1)

    # ── Step 2: N-gram surprisal ─────────────────────────────────────────────
    if 2 in steps:
        df = step2_ngram(cfg, df)
    else:
        cache = cache_path(cfg, "02_ngram_surprisal.parquet")
        if cache.exists():
            df = pd.read_parquet(cache)

    # ── Step 3: Neural metrics ───────────────────────────────────────────────
    if 3 in steps:
        df = step3_neural(cfg, df)
    else:
        cache = cache_path(cfg, "03_neural_metrics.parquet")
        if cache.exists():
            df = pd.read_parquet(cache)

    # ── Step 5: Integration cost (runs before step 4 — 4 needs dep_length) ───
    if 5 in steps:
        df = step5_integration_cost(cfg, df)
    else:
        cache = cache_path(cfg, "05_integration_cost.parquet")
        if cache.exists():
            df = pd.read_parquet(cache)

    # ── Step 4: Attention analysis (needs dep_length from step 5) ────────────
    if 4 in steps:
        if "dep_length" not in df.columns:
            logger.warning("dep_length not in DataFrame — run step 5 first. "
                           "Skipping attention analysis.")
        else:
            step4_attention(cfg, df)

    # ── Step 6: Bayesian modeling ────────────────────────────────────────────
    if 6 in steps:
        step6_bayesian(cfg, df)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()

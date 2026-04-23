"""
scripts/extended_analysis.py
=============================
Extended analyses beyond the main pipeline:

  1. H6 Individual Differences
     Extract per-reader surprisal sensitivity (random slopes) from the
     fitted Bayesian model and plot their distribution.

  2. BERT L0H10 Attention Analysis
     A. Layer-wise mean attention weight to syntactic head (Head 10)
     B. Head-finding accuracy by dependency length bucket

Usage
-----
    python scripts/extended_analysis.py [--model-path PATH] [--data-path PATH]
                                         [--out-dir DIR] [--n-sentences N]

All paths default to the /tmp/psycholingu layout used by the main pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── ensure project root is on PYTHONPATH ────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extended_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NC  = "/tmp/psycholingu/results/metrics/06_bayes_full.nc"
DEFAULT_DATA_PAR  = "/tmp/psycholingu/data/processed/05_integration_cost.parquet"
DEFAULT_OUT_DIR   = "/tmp/psycholingu/results/figures"
DEFAULT_N_SENT    = 150   # sentences to process for BERT analysis


# ─────────────────────────────────────────────────────────────────────────────
# 1. H6 — Per-reader surprisal sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def analysis_h6_individual_slopes(nc_path: str, out_dir: Path) -> None:
    """
    Extract posterior mean of (beta_gpt2_surprisal + u_slope_gpt2_surprisal[i])
    for every subject and plot the distribution.
    """
    import arviz as az

    logger.info("Loading InferenceData from %s", nc_path)
    idata = az.from_netcdf(nc_path)
    post  = idata.posterior

    if "beta_gpt2_surprisal" not in post or "u_slope_gpt2_surprisal" not in post:
        logger.error("beta_gpt2_surprisal or u_slope_gpt2_surprisal not in posterior. "
                     "Was the full model fitted with gpt2_surprisal random slope?")
        return

    beta_fixed  = float(post["beta_gpt2_surprisal"].values.mean())
    u_slopes    = post["u_slope_gpt2_surprisal"].values.mean(axis=(0, 1))  # (n_subj,)
    total_slope = beta_fixed + u_slopes

    n_subj = len(total_slope)
    logger.info("Subjects: %d  |  group mean β=%.4f  |  SD(slopes)=%.4f",
                n_subj, beta_fixed, total_slope.std())

    # ── also extract IC slopes if present ───────────────────────────────────
    ic_slopes = None
    if "u_slope_integration_cost" in post:
        beta_ic   = float(post["beta_integration_cost"].values.mean())
        ic_slopes = beta_ic + post["u_slope_integration_cost"].values.mean(axis=(0, 1))

    # ── Figure 1: histogram of individual surprisal slopes ──────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(total_slope, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(beta_fixed, color="red", linewidth=2, linestyle="--",
               label=f"Group mean β = {beta_fixed:.4f}")
    ax.set_xlabel("Individual surprisal sensitivity  (β + u_slope)", fontsize=12)
    ax.set_ylabel("Number of readers", fontsize=12)
    ax.set_title("H6: Distribution of Per-Reader GPT-2 Surprisal Sensitivity", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = out_dir / "H6_surprisal_slopes_hist.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved %s", out)
    plt.close(fig)

    # ── Figure 2: scatter surprisal vs IC slopes (if both available) ────────
    if ic_slopes is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(total_slope, ic_slopes, alpha=0.5, s=20, color="steelblue")
        ax.axvline(beta_fixed, color="red",   linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(float(post["beta_integration_cost"].values.mean()),
                   color="orange", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Surprisal sensitivity (β_surprisal + u)", fontsize=11)
        ax.set_ylabel("IC sensitivity (β_IC + u)", fontsize=11)
        ax.set_title("H6: Individual Differences — Surprisal vs IC Sensitivity", fontsize=12)
        plt.tight_layout()
        out2 = out_dir / "H6_surprisal_vs_ic_slopes.png"
        fig.savefig(out2, dpi=150)
        logger.info("Saved %s", out2)
        plt.close(fig)

    # ── Print extremes ───────────────────────────────────────────────────────
    df_slopes = pd.DataFrame({
        "subject_idx": range(n_subj),
        "surprisal_slope": total_slope,
    })
    print("\n── Most surprisal-sensitive readers ──────────────────")
    print(df_slopes.nlargest(5, "surprisal_slope").to_string(index=False))
    print("\n── Least surprisal-sensitive readers ─────────────────")
    print(df_slopes.nsmallest(5, "surprisal_slope").to_string(index=False))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. BERT L0H10 — attention to syntactic head
# ─────────────────────────────────────────────────────────────────────────────

def analysis_bert_l0h10(data_path: str, out_dir: Path, n_sentences: int) -> None:
    """
    Two sub-analyses for BERT L0H10 (ρ = −0.624 with dep_length):

    A. Layer-wise mean attention weight paid to the syntactic head (head 10).
    B. Head-finding accuracy of L0H10 binned by dependency length.
    """
    from src.neural_metrics      import NeuralMetricsExtractor
    from src.attention_analysis  import _word_to_first_token

    logger.info("Loading dependency data from %s", data_path)
    df = pd.read_parquet(data_path)

    # Drop rows without dep annotation and deduplicate to one row per word
    df = df.dropna(subset=["dep_length", "dep_head_position"])
    df = df.drop_duplicates(subset=["story_id", "sentence_id", "word_position"])

    sentences = (
        df[["story_id", "sentence_id", "sentence_text"]]
        .drop_duplicates()
        .head(n_sentences)
    )
    logger.info("Processing %d sentences for BERT L0H10 analysis …", len(sentences))

    extractor = NeuralMetricsExtractor("bert-base-uncased", "masked", "cpu")

    layer_records: list[dict] = []
    acc_records:   list[dict] = []

    for _, row in sentences.iterrows():
        sent      = row["sentence_text"]
        sent_rows = df[
            (df["story_id"]   == row["story_id"]) &
            (df["sentence_id"] == row["sentence_id"])
        ]
        try:
            data = extractor.get_attention_weights(sent)
        except Exception as exc:
            logger.debug("Skipping sentence (attention failed): %s", exc)
            continue

        attn     = data["attentions"]   # (n_layers, n_heads, T, T)
        tokens   = data["tokens"]
        n_layers, n_heads, n_tok, _ = attn.shape
        words    = sent.split()
        w2t      = _word_to_first_token(tokens, words)

        for _, r in sent_rows.iterrows():
            wpos = int(r["word_position"])
            hpos = r.get("dep_head_position")
            dlen = r.get("dep_length")
            if pd.isna(hpos) or pd.isna(dlen):
                continue
            dep_tok  = w2t.get(wpos)
            head_tok = w2t.get(int(hpos))
            if dep_tok is None or head_tok is None:
                continue
            if dep_tok >= n_tok or head_tok >= n_tok:
                continue

            # A. Attention weight dep→head for every layer, head=10
            for l_idx in range(n_layers):
                h_idx = min(10, n_heads - 1)   # head 10 (or last head)
                layer_records.append({
                    "layer":        l_idx,
                    "attn_weight":  float(attn[l_idx, h_idx, dep_tok, head_tok]),
                    "dep_length":   int(dlen),
                })

            # B. Head-finding accuracy for L0H10
            h_idx    = min(10, n_heads - 1)
            pred_tok = int(np.argmax(attn[0, h_idx, dep_tok]))
            acc_records.append({
                "dep_length": int(dlen),
                "correct":    pred_tok == head_tok,
            })

    if not layer_records:
        logger.error("No layer records collected — check dep_length coverage.")
        return

    lm_df  = pd.DataFrame(layer_records)
    acc_df = pd.DataFrame(acc_records)

    # ── Figure A: layer-wise mean attention to syntactic head ────────────────
    layer_avg = lm_df.groupby("layer")["attn_weight"].agg(["mean", "sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_avg["layer"], layer_avg["mean"], marker="o",
            color="steelblue", linewidth=2, label="Head 10")
    ax.fill_between(
        layer_avg["layer"],
        layer_avg["mean"] - layer_avg["sem"],
        layer_avg["mean"] + layer_avg["sem"],
        alpha=0.25, color="steelblue",
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5,
               label="L0H10  (ρ = −0.624)")
    ax.set_xlabel("BERT Layer", fontsize=12)
    ax.set_ylabel("Mean attention weight → syntactic head", fontsize=12)
    ax.set_title("BERT Head 10: Attention to Syntactic Head Across Layers", fontsize=13)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=11)
    plt.tight_layout()
    out_a = out_dir / "bert_L0H10_layer_curve.png"
    fig.savefig(out_a, dpi=150)
    logger.info("Saved %s", out_a)
    plt.close(fig)

    # ── Figure B: head-finding accuracy by dep_length bucket ─────────────────
    bins   = [0, 1, 2, 3, 5, 10, 999]
    labels = ["1", "2", "3", "4–5", "6–10", "11+"]
    acc_df["dep_bucket"] = pd.cut(
        acc_df["dep_length"], bins=bins, labels=labels, right=True
    )
    accuracy = acc_df.groupby("dep_bucket", observed=True)["correct"].agg(
        ["mean", "count"]
    )
    chance = 1.0 / n_tok if n_tok else 0.05

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        accuracy.index.astype(str),
        accuracy["mean"],
        color="tomato", edgecolor="black", alpha=0.85,
    )
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1.5,
               label=f"Chance (~{chance:.3f})")
    # Annotate count above each bar
    for bar, (_, row_a) in zip(bars, accuracy.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"n={int(row_a['count'])}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xlabel("Dependency Length (words)", fontsize=12)
    ax.set_ylabel("Head-finding accuracy", fontsize=12)
    ax.set_title("BERT L0H10: Syntactic Head-Finding Accuracy\nby Dependency Length",
                 fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out_b = out_dir / "bert_L0H10_headfinding_accuracy.png"
    fig.savefig(out_b, dpi=150)
    logger.info("Saved %s", out_b)
    plt.close(fig)

    print("\n── Head-finding accuracy by dep_length ──────────────")
    print(accuracy.to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended H6 + BERT L0H10 analyses")
    p.add_argument("--model-path", default=DEFAULT_MODEL_NC,
                   help="Path to 06_bayes_full.nc (default: %(default)s)")
    p.add_argument("--data-path",  default=DEFAULT_DATA_PAR,
                   help="Path to 05_integration_cost.parquet (default: %(default)s)")
    p.add_argument("--out-dir",    default=DEFAULT_OUT_DIR,
                   help="Output directory for figures (default: %(default)s)")
    p.add_argument("--n-sentences", type=int, default=DEFAULT_N_SENT,
                   help="Number of sentences for BERT analysis (default: %(default)s)")
    p.add_argument("--skip-h6",   action="store_true", help="Skip H6 analysis")
    p.add_argument("--skip-bert", action="store_true", help="Skip BERT L0H10 analysis")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_h6:
        logger.info("── Analysis 1: H6 Individual Differences ──────────────")
        analysis_h6_individual_slopes(args.model_path, out_dir)

    if not args.skip_bert:
        logger.info("── Analysis 2: BERT L0H10 ─────────────────────────────")
        analysis_bert_l0h10(args.data_path, out_dir, args.n_sentences)

    logger.info("Extended analysis complete. Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()

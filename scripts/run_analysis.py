"""
scripts/run_analysis.py
=======================
Post-hoc analysis and hypothesis testing after the pipeline has run.
Generates summary tables and visualisations for each hypothesis.

Usage
-----
    python scripts/run_analysis.py [--results results/metrics]

Output
------
  results/figures/hyp1_deep_vs_ngram.png
  results/figures/hyp2_ic_variance_explained.png
  results/figures/hyp3_surprisal_vs_entropy.png
  results/figures/hyp4_architecture_comparison.png
  results/figures/hyp6_random_slopes.png
  results/metrics/hypothesis_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

RESULTS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_idata(variant: str) -> az.InferenceData | None:
    path = RESULTS_DIR / f"06_bayes_{variant}.nc"
    if not path.exists():
        logger.warning("Not found (run pipeline first): %s", path)
        return None
    return az.from_netcdf(str(path))


def posterior_mean_ci(idata: az.InferenceData, var: str) -> tuple[float, float, float]:
    """Return (mean, 2.5%, 97.5%) for a scalar variable."""
    samples = idata.posterior[var].values.flatten()
    return float(samples.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ---------------------------------------------------------------------------
# Hypothesis 1: Deep vs. Shallow surprisal
# ---------------------------------------------------------------------------

def hyp1_deep_vs_ngram() -> None:
    """Compare baseline (trigram) vs. GPT-2 model fit via LOO-CV."""
    comparison_path = RESULTS_DIR / "06_model_comparison.csv"
    if not comparison_path.exists():
        logger.warning("Model comparison table not found. Run step 6 first.")
        return

    comp = pd.read_csv(comparison_path, index_col=0)
    logger.info("Model comparison (LOO-CV):\n%s", comp.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    elpd = comp["elpd_loo"]
    elpd.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_ylabel("ELPD (LOO-CV)")
    ax.set_title("Hyp 1 & 4: Model Comparison by Architecture")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hyp1_deep_vs_ngram.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Hypothesis 2: Does IC explain variance beyond deep surprisal?
# ---------------------------------------------------------------------------

def hyp2_ic_variance() -> None:
    """
    Compare 'deep_gpt2' vs 'surprisal_vs_ic' model:
    if beta_integration_cost posterior mass crosses zero → IC adds nothing.
    """
    idata = load_idata("surprisal_vs_ic")
    if idata is None:
        return

    mean, lo, hi = posterior_mean_ci(idata, "beta_integration_cost")
    significant  = not (lo < 0 < hi)

    logger.info(
        "Hyp 2 — beta_IC: mean=%.3f  95%%HDI=[%.3f, %.3f]  significant=%s",
        mean, lo, hi, significant
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    samples = idata.posterior["beta_integration_cost"].values.flatten()
    ax.hist(samples, bins=60, color="tomato", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", label="zero")
    ax.axvline(lo, color="gray",  linestyle=":", label="95% HDI")
    ax.axvline(hi, color="gray",  linestyle=":")
    ax.set_xlabel("β Integration Cost (z-scored)")
    ax.set_title("Hyp 2: Posterior of β_IC controlling for GPT-2 surprisal")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hyp2_ic_variance_explained.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Hypothesis 3: Independent contribution of Surprisal vs. Entropy
# ---------------------------------------------------------------------------

def hyp3_surprisal_vs_entropy() -> None:
    """
    For each architecture, show that both surprisal AND entropy have
    posteriors credibly different from zero when modelled together.
    Tests: are they providing *independent* predictive power?
    """
    arch_variants = [
        ("gpt2", "surprisal_vs_entropy_gpt2",
         "beta_gpt2_surprisal", "beta_gpt2_entropy"),
        ("bert", "surprisal_vs_entropy_bert",
         "beta_bert_surprisal", "beta_bert_entropy"),
        ("t5",   "surprisal_vs_entropy_t5",
         "beta_t5_surprisal",  "beta_t5_entropy"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    rows_written = 0

    for row_idx, (arch, variant, surp_var, entr_var) in enumerate(arch_variants):
        idata = load_idata(variant)
        for col_idx, (var, label) in enumerate([
            (surp_var, f"{arch.upper()} Surprisal"),
            (entr_var, f"{arch.upper()} Entropy"),
        ]):
            ax = axes[row_idx, col_idx]
            if idata is None or var not in idata.posterior:
                ax.set_title(f"{label}\n(not available)")
                ax.set_visible(True)
                continue
            samples = idata.posterior[var].values.flatten()
            m  = float(samples.mean())
            lo = float(np.percentile(samples, 2.5))
            hi = float(np.percentile(samples, 97.5))
            sig = not (lo < 0 < hi)
            color = "steelblue" if sig else "lightgray"
            ax.hist(samples, bins=60, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(0,  color="black", linestyle="--", linewidth=1.0)
            ax.axvline(lo, color="gray",  linestyle=":", linewidth=0.8)
            ax.axvline(hi, color="gray",  linestyle=":", linewidth=0.8)
            ax.set_title(
                f"β {label}  {'★ sig' if sig else '✗ n.s.'}\n"
                f"mean={m:.3f}  95%HDI=[{lo:.3f},{hi:.3f}]"
            )
            ax.set_xlabel("Coefficient (z-scored)")
            rows_written += 1

    fig.suptitle("Hyp 3: Surprisal vs. Entropy — independent contributions per architecture",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hyp3_surprisal_vs_entropy.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Hypothesis 4: Architecture comparison (GPT-2 vs. BERT vs. T5)
# ---------------------------------------------------------------------------

def hyp4_architecture() -> None:
    """
    Dedicated ELPD-LOO comparison: baseline (trigram), GPT-2, BERT, T5.
    Tests: does autoregressive (GPT-2) fit reading times better than
    bidirectional (BERT) or encoder-decoder (T5)?
    """
    comparison_path = RESULTS_DIR / "06_model_comparison.csv"
    if not comparison_path.exists():
        logger.warning("Model comparison table not found. Run step 6 first.")
        return

    comp = pd.read_csv(comparison_path, index_col=0)

    # Keep only the architecture-relevant rows
    arch_rows = ["baseline", "deep_gpt2", "deep_bert", "deep_t5"]
    arch_comp = comp[comp.index.isin(arch_rows)].copy()

    if arch_comp.empty:
        logger.warning("Architecture model variants not found in comparison table.")
        return

    # Rename for readability
    label_map = {
        "baseline":  "Trigram (baseline)",
        "deep_gpt2": "GPT-2 (autoregressive)",
        "deep_bert": "BERT (bidirectional)",
        "deep_t5":   "T5 (encoder-decoder)",
    }
    arch_comp.index = [label_map.get(i, i) for i in arch_comp.index]
    arch_comp = arch_comp.sort_values("elpd_loo", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
    bars = ax.barh(arch_comp.index, arch_comp["elpd_loo"],
                   xerr=arch_comp.get("se", None),
                   color=colors[:len(arch_comp)], edgecolor="white", height=0.5)
    ax.set_xlabel("ELPD (LOO-CV)  ← worse   better →")
    ax.set_title("Hyp 4: Architecture Comparison\n(higher ELPD = better fit to reading times)")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "hyp4_architecture_comparison.png", dpi=150)
    logger.info("Architecture comparison:\n%s", arch_comp[["elpd_loo", "p_loo"]].to_string())
    plt.show()


# ---------------------------------------------------------------------------
# Hypothesis 5: Attention head analysis
# ---------------------------------------------------------------------------

def hyp5_attention() -> None:
    """
    Visualise top attention heads correlated with dependency length.
    Tests: do specific heads implicitly track syntactic structure?
    """
    for model_label in ("gpt2", "bert"):
        path = RESULTS_DIR / f"04_attention_{model_label}.csv"
        if not path.exists():
            logger.warning("Attention results not found for %s (run step 4).", model_label)
            continue

        df = pd.read_csv(path)
        top_k = df.head(10).copy()
        top_k["label"] = top_k.apply(
            lambda r: f"L{int(r.layer)}H{int(r.head)}", axis=1
        )
        logger.info("Top 5 heads (%s):\n%s", model_label, df.head(5).to_string())

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["steelblue" if r >= 0 else "tomato" for r in top_k["rho"]]
        ax.barh(top_k["label"][::-1], top_k["rho"][::-1].abs(),
                color=colors[::-1], edgecolor="white")
        ax.set_xlabel("|Spearman ρ| with Dependency Length")
        ax.set_title(
            f"Hyp 5: Top Attention Heads — {model_label.upper()}\n"
            f"(positive ρ = head attends more to syntactically distant words)"
        )
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"hyp5_attention_{model_label}.png", dpi=150)
        plt.show()


# ---------------------------------------------------------------------------
# Hypothesis 6: Random slopes (individual reader strategies)
# ---------------------------------------------------------------------------

def hyp6_random_slopes() -> None:
    """
    Two-part test for significant per-reader variance:
      (a) Variance component: posterior of sigma_slope — is it credibly > 0?
      (b) Per-subject slopes: sorted bar chart showing reader heterogeneity.
    Both parts are required to support Hypothesis 6.
    """
    idata = load_idata("full")
    if idata is None:
        return

    slope_vars = [
        ("u_slope_gpt2_surprisal",  "sigma_slope_gpt2_surprisal",  "GPT-2 Surprisal"),
        ("u_slope_integration_cost", "sigma_slope_integration_cost", "Integration Cost"),
    ]

    for u_var, sigma_var, label in slope_vars:
        if u_var not in idata.posterior:
            logger.warning("Random slopes %s not found in full model.", u_var)
            continue

        fig, (ax_sigma, ax_slopes) = plt.subplots(1, 2, figsize=(12, 4))

        # ── (a) Variance component posterior ─────────────────────────────────
        if sigma_var in idata.posterior:
            sigma_samples = idata.posterior[sigma_var].values.flatten()
            s_mean = float(sigma_samples.mean())
            s_lo   = float(np.percentile(sigma_samples, 2.5))
            s_hi   = float(np.percentile(sigma_samples, 97.5))
            # sigma is strictly positive; HDI entirely above zero = significant variance
            sig = s_lo > 0
            ax_sigma.hist(sigma_samples, bins=60,
                          color="darkorange" if sig else "lightgray",
                          alpha=0.8, edgecolor="white")
            ax_sigma.axvline(0,    color="black", linestyle="--", linewidth=1.0)
            ax_sigma.axvline(s_lo, color="gray",  linestyle=":", linewidth=0.8)
            ax_sigma.axvline(s_hi, color="gray",  linestyle=":", linewidth=0.8)
            ax_sigma.set_xlabel("σ_slope (must be > 0 to be meaningful)")
            ax_sigma.set_title(
                f"Hyp 6: Variance component — {label}\n"
                f"σ mean={s_mean:.3f}  95%HDI=[{s_lo:.3f},{s_hi:.3f}]  "
                f"{'★ sig' if sig else '✗ n.s.'}"
            )
            logger.info(
                "Hyp 6 — sigma_slope_%s: mean=%.3f  95%%HDI=[%.3f, %.3f]  significant=%s",
                label, s_mean, s_lo, s_hi, sig
            )
        else:
            ax_sigma.set_title(f"σ_slope for {label}\n(not found in model)")

        # ── (b) Per-subject slope distribution ────────────────────────────────
        slopes      = idata.posterior[u_var].values              # (chains, draws, n_subj)
        slopes_flat = slopes.reshape(-1, slopes.shape[-1])
        means       = slopes_flat.mean(axis=0)
        hdi_lo      = np.percentile(slopes_flat, 2.5, axis=0)
        hdi_hi      = np.percentile(slopes_flat, 97.5, axis=0)
        order       = np.argsort(means)

        x = np.arange(len(means))
        ax_slopes.bar(x, means[order], color="steelblue", alpha=0.7, edgecolor="white")
        ax_slopes.vlines(x, hdi_lo[order], hdi_hi[order],
                         color="black", linewidth=0.6, alpha=0.5)
        ax_slopes.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax_slopes.set_xlabel("Subject (sorted by mean slope)")
        ax_slopes.set_ylabel(f"Random slope: {label}")
        ax_slopes.set_title(
            f"Hyp 6: Per-reader slopes — {label}\n"
            f"(bars = posterior mean, whiskers = 95% HDI)"
        )

        plt.tight_layout()
        safe_label = label.lower().replace(" ", "_")
        fig.savefig(FIGURES_DIR / f"hyp6_random_slopes_{safe_label}.png", dpi=150)
        plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary_table() -> None:
    rows = []
    for variant in ("baseline", "deep_gpt2", "deep_bert", "deep_t5",
                    "surprisal_vs_ic",
                    "surprisal_vs_entropy_gpt2", "surprisal_vs_entropy_bert",
                    "surprisal_vs_entropy_t5", "full"):
        idata = load_idata(variant)
        if idata is None:
            continue
        for var in idata.posterior.data_vars:
            if var.startswith("beta_"):
                m, lo, hi = posterior_mean_ci(idata, var)
                rows.append({
                    "model":     variant,
                    "predictor": var.replace("beta_", ""),
                    "mean":      round(m, 4),
                    "hdi_2.5":   round(lo, 4),
                    "hdi_97.5":  round(hi, 4),
                    "sig":       not (lo < 0 < hi),
                })

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(RESULTS_DIR / "hypothesis_summary.csv", index=False)
        logger.info("Saved hypothesis summary to %s",
                    RESULTS_DIR / "hypothesis_summary.csv")
        print(out.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/metrics",
                        help="Path to results directory")
    args = parser.parse_args()

    global RESULTS_DIR, FIGURES_DIR
    RESULTS_DIR = Path(args.results)
    FIGURES_DIR = RESULTS_DIR.parent / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Running hypothesis analyses …")
    hyp1_deep_vs_ngram()           # H1: deep > shallow surprisal
    hyp2_ic_variance()             # H2: IC explained away by surprisal?
    hyp3_surprisal_vs_entropy()    # H3: surprisal & entropy independent (all 3 archs)
    hyp4_architecture()            # H4: GPT-2 vs BERT vs T5
    hyp5_attention()               # H5: attention heads ↔ dep length
    hyp6_random_slopes()           # H6: per-reader variance (sigma + slopes)
    write_summary_table()
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()

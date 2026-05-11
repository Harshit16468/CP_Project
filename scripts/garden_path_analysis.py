"""
scripts/garden_path_analysis.py
================================
Empirical test of the entropy/surprisal prospective–retrospective dissociation.

Hypothesis (§6.3 in paper):
  - Entropy is PROSPECTIVE: indexes maintenance of multiple parses at the
    point where ambiguity is introduced (ambiguity onset).
  - Surprisal is RETROSPECTIVE: indexes expectation violation at the word
    that forces the reader to abandon the initial parse (disambiguation).

Test:
  Identify structurally ambiguous regions in the Dundee corpus (reduced
  relative clauses, NP/S complements) using the Stanza dependency parse.
  Fit two Bayesian models on first-fixation duration (FDUR):
    Model A — at DISAMBIGUATION words: β_surprisal >> β_entropy expected
    Model B — at AMBIGUITY ONSET words:  β_entropy >> β_surprisal expected

  A credible double-dissociation (A vs B) constitutes the novel finding
  that converts §6.3 from a proposal to an empirical result.

Usage
-----
    python scripts/garden_path_analysis.py --config config.yaml
    python scripts/garden_path_analysis.py --config config.yaml --min-regions 20
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stanza
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.bayesian_model import BayesianHierarchicalModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("garden_path")


# ─────────────────────────────────────────────────────────────────────────────
# Garden-path region detection
# ─────────────────────────────────────────────────────────────────────────────

def _load_stanza() -> stanza.Pipeline:
    stanza.download("en", verbose=False)
    return stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=False,
        verbose=False,
    )


def detect_gp_regions(sentence: str, nlp: stanza.Pipeline) -> list[dict]:
    """
    Detect structurally ambiguous regions using Stanza dependency parse.

    Returns list of {onset_pos, disambig_pos, gp_type} dicts.
    Positions are 0-based word indices within the sentence.

    Patterns detected
    -----------------
    1. Reduced relative clause (RRC):
       "The horse raced past the barn fell."
       - onset    : head noun of the RRC (e.g. 'horse')
       - disambig : root verb after the RRC region (e.g. 'fell')

    2. Temporary NP/S complement ambiguity:
       "While Anna dressed the baby spit up."
       - onset    : first NP following the subordinator
       - disambig : main clause root verb
    """
    regions = []

    try:
        doc = nlp(sentence)
    except Exception as exc:
        logger.debug("Parse failed: %s", exc)
        return regions

    for sent in doc.sentences:
        words     = sent.words
        n         = len(words)
        pos_list  = [w.upos  for w in words]
        dep_list  = [w.deprel for w in words]
        head_list = [w.head  for w in words]   # 1-based; 0 = root

        # 1. Reduced relative clauses ─────────────────────────────────────
        # Look for VBN (past participle) with acl/acl:relcl dependency and
        # no explicit auxiliary (was/were) immediately before it.
        for i, word in enumerate(words):
            if word.upos != "VERB":
                continue
            if word.deprel not in ("acl", "acl:relcl"):
                continue
            # Check xpos tag for VBN (past participle)
            if word.xpos not in ("VBN", "VBD"):
                continue

            # Check no auxiliary in [head_pos..i] range
            head_0 = word.head - 1  # 0-based position of head noun
            if head_0 < 0 or head_0 >= n:
                continue
            aux_present = any(
                words[j].upos == "AUX"
                for j in range(max(0, head_0), i)
            )
            if aux_present:
                continue

            # Find root verb that comes AFTER word i
            disambig_pos = None
            for j in range(i + 1, n):
                if words[j].deprel == "root" and words[j].upos == "VERB":
                    disambig_pos = j
                    break
            if disambig_pos is None:
                continue

            regions.append({
                "onset_pos":    head_0,
                "disambig_pos": disambig_pos,
                "gp_type":      "reduced_relative",
                "sentence":     sentence,
            })

        # 2. Temporary NP/S complement ambiguity ──────────────────────────
        # "While/After/Since NP V(main) NP V(clause)"
        subordinators = {"while", "after", "since", "until", "when", "once"}
        for i, word in enumerate(words):
            if word.text.lower() not in subordinators:
                continue
            if word.deprel not in ("mark", "advmod"):
                continue

            # Find first NP head after subordinator
            onset_pos = None
            for j in range(i + 1, n):
                if words[j].upos == "NOUN" and words[j].deprel in ("nsubj", "dobj", "obj"):
                    onset_pos = j
                    break
            if onset_pos is None:
                continue

            # Find the main clause root (comes after the subordinate clause)
            disambig_pos = None
            for j in range(onset_pos + 1, n):
                if words[j].deprel == "root" and words[j].upos == "VERB":
                    disambig_pos = j
                    break
            if disambig_pos is None:
                continue

            # Sanity check: onset and disambig must not be adjacent (trivial)
            if disambig_pos <= onset_pos + 1:
                continue

            regions.append({
                "onset_pos":    onset_pos,
                "disambig_pos": disambig_pos,
                "gp_type":      "np_s_complement",
                "sentence":     sentence,
            })

    return regions


def tag_gp_words(df: pd.DataFrame, nlp: stanza.Pipeline) -> pd.DataFrame:
    """
    Add 'gp_role' column: 'onset', 'disambig', or None.
    Processes per unique sentence to avoid repeated parsing.
    """
    df = df.copy()
    df["gp_role"] = None

    unique_sents = (
        df[["story_id", "sentence_id", "sentence_text"]]
        .drop_duplicates(subset=["story_id", "sentence_id"])
    )

    n_onset, n_disambig = 0, 0
    for _, row in unique_sents.iterrows():
        regions = detect_gp_regions(row["sentence_text"], nlp)
        for r in regions:
            mask = (
                (df["story_id"]    == row["story_id"])
                & (df["sentence_id"] == row["sentence_id"])
            )
            onset_mask    = mask & (df["word_position"] == r["onset_pos"])
            disambig_mask = mask & (df["word_position"] == r["disambig_pos"])
            df.loc[onset_mask,    "gp_role"] = "onset"
            df.loc[disambig_mask, "gp_role"] = "disambig"
            n_onset    += int(onset_mask.sum())
            n_disambig += int(disambig_mask.sum())

    logger.info(
        "GP tagging: %d onset observations, %d disambiguation observations "
        "across %d unique sentences.",
        n_onset, n_disambig, len(unique_sents),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian models for each role
# ─────────────────────────────────────────────────────────────────────────────

GP_PREDICTORS = [
    "gpt2_surprisal",
    "gpt2_entropy",
    "word_length",
    "log_freq",
]

def fit_gp_model(df_role: pd.DataFrame, cfg: dict,
                 metrics_dir: Path, tag: str) -> dict:
    """
    Fit a Bayesian model on df_role using GP_PREDICTORS.
    Returns {'idata': ..., 'summary': pd.DataFrame, 'loo': ELPDData|None}.
    """
    import arviz as az
    nc_path  = metrics_dir / f"gp_{tag}.nc"
    loo_pkl  = metrics_dir / f"gp_{tag}_loo.pkl"
    sum_path = metrics_dir / f"gp_{tag}_summary.csv"

    bay_cfg = cfg["bayesian"].copy()
    v_cfg   = {
        **bay_cfg,
        "predictors":   [p for p in GP_PREDICTORS if p in df_role.columns],
        "random_slopes": [],    # keep it simple — only fixed effects for subset
        "intercept_prior": "data",
        "compute_loo": True,
        "draws": 2000,
        "tune": 1000,
    }

    if nc_path.exists():
        logger.info("Loading cached GP model: %s", tag)
        idata = az.from_netcdf(str(nc_path))
        summary = pd.read_csv(sum_path) if sum_path.exists() else None
        loo = None
        if loo_pkl.exists():
            with open(loo_pkl, "rb") as fh:
                loo = pickle.load(fh)
        return {"idata": idata, "summary": summary, "loo": loo}

    bm    = BayesianHierarchicalModel(v_cfg)
    idata = bm.fit(df_role)
    bm.save(idata, nc_path)
    summary = bm.summary(idata)
    summary.to_csv(sum_path)

    loo = None
    if bm.last_loo is not None:
        with open(loo_pkl, "wb") as fh:
            pickle.dump(bm.last_loo, fh)
        loo = bm.last_loo

    return {"idata": idata, "summary": summary, "loo": loo}


# ─────────────────────────────────────────────────────────────────────────────
# Dissociation figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_dissociation(onset_res: dict, disambig_res: dict,
                      figures_dir: Path) -> None:
    """
    Forest plot comparing β_entropy and β_surprisal at onset vs. disambiguation.
    The prospective/retrospective dissociation is visible if:
      β_entropy[onset] > β_entropy[disambig]
      β_surprisal[disambig] > β_surprisal[onset]
    """
    import arviz as az

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    predictors_of_interest = ["gpt2_surprisal", "gpt2_entropy"]
    labels_map = {"gpt2_surprisal": "GPT-2 Surprisal (retrospective)",
                  "gpt2_entropy":   "GPT-2 Entropy (prospective)"}
    colors = {"onset": "#8e44ad", "disambig": "#c0392b"}

    for ax, (role, res) in zip(axes, [("onset", onset_res),
                                       ("disambig", disambig_res)]):
        if res["idata"] is None:
            ax.set_title(f"GP role: {role}\n(no data)")
            continue

        betas, lows, highs = [], [], []
        preds_present = []
        for pred in predictors_of_interest:
            var = f"beta_{pred}"
            if var not in res["idata"].posterior:
                continue
            s  = res["idata"].posterior[var].values.flatten()
            betas.append(float(s.mean()))
            lows.append(float(np.percentile(s, 2.5)))
            highs.append(float(np.percentile(s, 97.5)))
            preds_present.append(labels_map.get(pred, pred))

        y_pos = np.arange(len(preds_present))
        ax.scatter(betas, y_pos, color=colors[role], s=80, zorder=3)
        for i, (m, lo, hi) in enumerate(zip(betas, lows, highs)):
            ax.plot([lo, hi], [i, i], color=colors[role], linewidth=2.5, alpha=0.8)
            sig_marker = "★" if not (lo < 0 < hi) else ""
            ax.text(hi + 0.001, i, sig_marker, va="center",
                    color=colors[role], fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(preds_present, fontsize=9)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
        n_obs = len(res.get("df", pd.DataFrame()))
        ax.set_xlabel("Posterior β (z-scored, 95% HDI)")
        ax.set_title(
            f"GP role: {'Ambiguity ONSET' if role == 'onset' else 'DISAMBIGUATION'}\n"
            f"({'β_entropy dominant expected' if role == 'onset' else 'β_surprisal dominant expected'})"
        )

    fig.suptitle(
        "Garden-Path Dissociation: Prospective Entropy vs. Retrospective Surprisal\n"
        "(Dundee Corpus, FDUR — credible β marked ★)",
        fontsize=12,
    )
    plt.tight_layout()
    out = figures_dir / "garden_path_dissociation.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved garden-path dissociation figure to %s", out)
    plt.close(fig)


def write_dissociation_table(onset_res: dict, disambig_res: dict,
                              metrics_dir: Path) -> None:
    """Write a CSV table of the key coefficients for the paper."""
    rows = []
    for role, res in [("onset", onset_res), ("disambig", disambig_res)]:
        if res["idata"] is None:
            continue
        for pred in GP_PREDICTORS:
            var = f"beta_{pred}"
            if var not in res["idata"].posterior:
                continue
            s  = res["idata"].posterior[var].values.flatten()
            m  = float(s.mean())
            lo = float(np.percentile(s, 2.5))
            hi = float(np.percentile(s, 97.5))
            rows.append({
                "gp_role":    role,
                "predictor":  pred,
                "beta_mean":  round(m, 4),
                "hdi_2.5":    round(lo, 4),
                "hdi_97.5":   round(hi, 4),
                "significant": not (lo < 0 < hi),
            })

    if rows:
        out = metrics_dir / "garden_path_dissociation_table.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info("Garden-path table saved to %s", out)

        # Print summary to log
        df_tab = pd.DataFrame(rows)
        for role in ("onset", "disambig"):
            sub = df_tab[df_tab["gp_role"] == role]
            logger.info("\nGP role: %s", role.upper())
            for _, r in sub.iterrows():
                sig = "* (credible)" if r["significant"] else "n.s."
                logger.info("  β_%s = %.4f  HDI [%.4f, %.4f]  %s",
                            r["predictor"], r["beta_mean"],
                            r["hdi_2.5"], r["hdi_97.5"], sig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Garden-path dissociation analysis")
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--out-dir",     default="/tmp/psycholingu_gp")
    p.add_argument("--dundee-processed",
                   default="/tmp/psycholingu_dundee/processed",
                   help="Path to Dundee processed parquets (from dundee_validation.py)")
    p.add_argument("--min-regions", type=int, default=15,
                   help="Minimum GP regions to proceed with Bayesian fitting (default 15)")
    p.add_argument("--skip-parse",  action="store_true",
                   help="Load cached GP tags instead of re-running Stanza")
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    cfg         = yaml.safe_load(open(args.config))
    out_dir     = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load processed Dundee data ────────────────────────────────────────────
    proc_dir = Path(args.dundee_processed)
    candidates = [
        proc_dir / "03b_dundee_lexical_derived.parquet",
        proc_dir / "05_dundee_integration_cost.parquet",
        proc_dir / "03_dundee_neural_metrics.parquet",
    ]
    df = None
    for c in candidates:
        if c.exists():
            logger.info("Loading Dundee DataFrame from %s", c)
            df = pd.read_parquet(c)
            break
    if df is None:
        logger.error(
            "No processed Dundee parquet found in %s.\n"
            "Run: python scripts/dundee_validation.py --config config.yaml",
            proc_dir,
        )
        sys.exit(1)

    # ── Check required LM columns ─────────────────────────────────────────────
    required = ["gpt2_surprisal", "gpt2_entropy"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.error(
            "Missing columns in Dundee data: %s\n"
            "Ensure dundee_validation.py has run steps 3 and lexical.", missing
        )
        sys.exit(1)

    # ── Detect / load garden-path regions ─────────────────────────────────────
    gp_cache = metrics_dir / "gp_tagged_dundee.parquet"
    if args.skip_parse and gp_cache.exists():
        logger.info("Loading cached GP tags from %s", gp_cache)
        df = pd.read_parquet(gp_cache)
    else:
        logger.info("Loading Stanza parser for GP detection …")
        nlp = _load_stanza()
        df  = tag_gp_words(df, nlp)
        df.to_parquet(gp_cache, index=True)
        logger.info("Cached GP-tagged DataFrame to %s", gp_cache)

    # ── Summary of GP regions ─────────────────────────────────────────────────
    n_onset    = (df["gp_role"] == "onset").sum()
    n_disambig = (df["gp_role"] == "disambig").sum()
    logger.info("GP regions found: %d onset observations, %d disambiguation observations",
                n_onset, n_disambig)

    if min(n_onset, n_disambig) < args.min_regions:
        logger.warning(
            "Too few GP regions (onset=%d, disambig=%d, min_required=%d).\n"
            "Garden-path analysis will proceed but results may be underpowered.",
            n_onset, n_disambig, args.min_regions,
        )

    # ── Per-role subsets ──────────────────────────────────────────────────────
    df_onset    = df[df["gp_role"] == "onset"].copy()
    df_disambig = df[df["gp_role"] == "disambig"].copy()

    logger.info("Onset subset: %d rows, %d unique subjects, %d unique sentences",
                len(df_onset), df_onset["subject"].nunique(),
                df_onset["sentence_id"].nunique())
    logger.info("Disambig subset: %d rows, %d unique subjects, %d unique sentences",
                len(df_disambig), df_disambig["subject"].nunique(),
                df_disambig["sentence_id"].nunique())

    # ── Fit Bayesian models ───────────────────────────────────────────────────
    onset_res    = {"idata": None, "summary": None, "loo": None, "df": df_onset}
    disambig_res = {"idata": None, "summary": None, "loo": None, "df": df_disambig}

    if len(df_onset) >= args.min_regions:
        logger.info("Fitting Bayesian model for ONSET positions …")
        onset_res.update(fit_gp_model(df_onset, cfg, metrics_dir, "onset"))
    else:
        logger.warning("Skipping onset model (too few observations).")

    if len(df_disambig) >= args.min_regions:
        logger.info("Fitting Bayesian model for DISAMBIGUATION positions …")
        disambig_res.update(fit_gp_model(df_disambig, cfg, metrics_dir, "disambig"))
    else:
        logger.warning("Skipping disambiguation model (too few observations).")

    # ── Plots and tables ──────────────────────────────────────────────────────
    plot_dissociation(onset_res, disambig_res, figures_dir)
    write_dissociation_table(onset_res, disambig_res, metrics_dir)

    # ── Verbal summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GARDEN-PATH DISSOCIATION SUMMARY")
    print("=" * 60)
    print(f"Dundee corpus: {n_onset} onset obs / {n_disambig} disambiguation obs")
    print("\nExpected dissociation pattern:")
    print("  β_entropy[onset]    >> β_entropy[disambig]   (prospective uncertainty)")
    print("  β_surprisal[disambig] >> β_surprisal[onset]  (retrospective violation)")
    print(f"\nResults in: {out_dir}")
    print(f"  {metrics_dir / 'garden_path_dissociation_table.csv'}")
    print(f"  {figures_dir / 'garden_path_dissociation.png'}")


if __name__ == "__main__":
    main()

"""
Lexical Features for Psycholinguistic Controls
===============================================
Adds word-level controls required for any psycholinguistics reading-time study:

  word_length      : number of characters (surface form)
  log_freq         : Zipf-scale log word frequency from SUBTLEX-US
                     (falls back to corpus frequency if file not found)
  *_lag1           : spillover — predictor value at word N-1 within sentence
  *_entropy_reduction : ΔH = H_{n-1} − H_n  (entropy reduction account)

Public API
----------
    add_lexical_controls(df, cfg)   -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spillover columns: surprisal at N-1
SPILLOVER_COLS = [
    "gpt2_surprisal",
    "bert_base_uncased_surprisal",
    "t5_base_surprisal",
    "ngram_surprisal",
]

# Entropy reduction: ΔH = H_{n-1} − H_n
ENTROPY_COLS = [
    "gpt2_entropy",
    "bert_base_uncased_entropy",
    "t5_base_entropy",
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_lexical_controls(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add all psycholinguistic control variables to df in-place (returns copy).

    Parameters
    ----------
    df  : DataFrame from neural_metrics step (has word, subject, sentence cols)
    cfg : top-level config dict

    Added columns
    -------------
    word_length, log_freq,
    {col}_lag1 for each surprisal column present,
    {col}_reduction for each entropy column present.
    """
    df = df.copy()

    df = _add_word_length(df)
    df = _add_log_frequency(df, cfg)
    df = _add_spillover(df)
    df = _add_entropy_reduction(df)

    n_lag = sum(f"{c}_lag1" in df.columns for c in SPILLOVER_COLS)
    n_red = sum(f"{c}_reduction" in df.columns for c in ENTROPY_COLS)
    logger.info(
        "Lexical controls added: word_length, log_freq, "
        "%d spillover columns, %d entropy-reduction columns.",
        n_lag, n_red,
    )
    return df


# ---------------------------------------------------------------------------
# Word length
# ---------------------------------------------------------------------------

def _add_word_length(df: pd.DataFrame) -> pd.DataFrame:
    df["word_length"] = df["word"].str.len().astype(float)
    return df


# ---------------------------------------------------------------------------
# Log word frequency (SUBTLEX-US Zipf values, with corpus fallback)
# ---------------------------------------------------------------------------

def _add_log_frequency(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    subtlex_path = cfg.get("paths", {}).get("subtlex_file", "")

    if subtlex_path and Path(subtlex_path).exists():
        logger.info("Loading SUBTLEX-US from %s", subtlex_path)
        freq_map = _load_subtlex(subtlex_path)
        df["log_freq"] = df["word"].str.lower().map(freq_map)
        missing = df["log_freq"].isna().sum()
        logger.info(
            "SUBTLEX-US: %d / %d words matched (missing=%.1f%%)",
            len(df) - missing, len(df), 100 * missing / len(df),
        )
        # Fill missing with median Zipf (≈ 3.0 for English)
        df["log_freq"] = df["log_freq"].fillna(df["log_freq"].median())
    else:
        logger.warning(
            "SUBTLEX-US file not found at '%s'. "
            "Falling back to corpus frequency. "
            "Set paths.subtlex_file in config.yaml for proper frequency norms.",
            subtlex_path,
        )
        df = _corpus_frequency_fallback(df)

    return df


def _load_subtlex(path: str) -> dict[str, float]:
    """
    Load SUBTLEX-US and return word → Zipf-value mapping.

    SUBTLEX-US TSV columns include: Word, Zipf-value (log10 freq per million).
    Accepts both .xlsx and tab-separated .txt / .csv formats.
    """
    p = Path(path)
    if p.suffix in (".xlsx", ".xls"):
        sub = pd.read_excel(p)
    else:
        sub = pd.read_csv(p, sep="\t", encoding="latin-1")

    # Normalise column names (different SUBTLEX releases vary)
    sub.columns = [c.strip() for c in sub.columns]
    word_col = next((c for c in sub.columns if c.lower() == "word"), None)
    freq_col = next(
        (c for c in sub.columns if "zipf" in c.lower()),
        next((c for c in sub.columns if "lg10wf" in c.lower()), None),
    )
    if word_col is None or freq_col is None:
        raise ValueError(
            f"Cannot find Word / Zipf columns in SUBTLEX file. "
            f"Columns found: {list(sub.columns)}"
        )

    sub[word_col] = sub[word_col].str.lower().str.strip()
    return dict(zip(sub[word_col], pd.to_numeric(sub[freq_col], errors="coerce")))


def _corpus_frequency_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute corpus Zipf frequency from the reading-time data itself.
    Uses unique (story_id, word_position, word) to avoid inflating counts
    by the number of subjects.
    """
    unique_words = (
        df[["story_id", "word_position", "word"]]
        .drop_duplicates()
        ["word"]
        .str.lower()
    )
    counts    = unique_words.value_counts()
    total     = counts.sum()
    zipf_vals = np.log10(counts / total * 1e6 + 1)   # +1 for Laplace smoothing
    freq_map  = zipf_vals.to_dict()
    df["log_freq"] = df["word"].str.lower().map(freq_map).fillna(zipf_vals.min())
    return df


# ---------------------------------------------------------------------------
# Spillover: surprisal at word N-1
# ---------------------------------------------------------------------------

def _add_spillover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *_lag1 columns: predictor value at word N-1 within each
    (story_id, sentence_id, subject) group.
    First word of each sentence gets NaN (dropped in Bayesian model).
    """
    group_cols = ["story_id", "sentence_id", "subject"]
    for col in SPILLOVER_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = (
            df.sort_values(group_cols + ["word_position"])
              .groupby(group_cols)[col]
              .shift(1)
        )
    return df


# ---------------------------------------------------------------------------
# Entropy reduction: ΔH = H_{n-1} − H_n
# ---------------------------------------------------------------------------

def _add_entropy_reduction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *_reduction columns: entropy at word N-1 minus entropy at word N.

    Positive ΔH means uncertainty decreased (informative word).
    Negative ΔH means uncertainty increased (rare structural choice).
    """
    group_cols = ["story_id", "sentence_id", "subject"]
    for col in ENTROPY_COLS:
        if col not in df.columns:
            continue
        shifted = (
            df.sort_values(group_cols + ["word_position"])
              .groupby(group_cols)[col]
              .shift(1)
        )
        df[f"{col}_reduction"] = shifted - df[col]
    return df

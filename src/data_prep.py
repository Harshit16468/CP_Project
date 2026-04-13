"""
Step 1 – Behavioral Data Preparation
=====================================
Loads a psycholinguistic dataset (Natural Stories Corpus or GECO),
aligns word-by-word reading times with the corresponding token text,
applies exclusion criteria (min/max RT, per-subject SD cutoff),
and returns a tidy DataFrame ready for feature extraction.

Columns in the returned DataFrame
----------------------------------
  subject       : participant identifier (str)
  story_id      : passage / story identifier
  sentence_id   : sentence index within story
  word_position : 0-based token index within sentence
  word          : surface word form (lowercased)
  rt_ms         : raw reading time in milliseconds
  log_rt        : natural-log reading time (primary DV)
  sentence_text : full sentence string (for parsing downstream)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(
    dataset: Literal["natural_stories", "geco"],
    cfg: dict,
) -> pd.DataFrame:
    """
    Load and preprocess the chosen psycholinguistic dataset.

    Parameters
    ----------
    dataset : str
        One of ``"natural_stories"`` or ``"geco"``.
    cfg : dict
        Top-level config dict (loaded from config.yaml).

    Returns
    -------
    pd.DataFrame
        Tidy, cleaned reading-time data.
    """
    if dataset == "natural_stories":
        df = _load_natural_stories(cfg)
    elif dataset == "geco":
        df = _load_geco(cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    df = _apply_exclusions(df, cfg["bayesian"])
    df["log_rt"] = np.log(df["rt_ms"])
    logger.info("Dataset ready: %d word tokens, %d subjects",
                len(df), df["subject"].nunique())
    return df


# ---------------------------------------------------------------------------
# Natural Stories Corpus
# ---------------------------------------------------------------------------

def _load_natural_stories(cfg: dict) -> pd.DataFrame:
    """
    Parse the Natural Stories Corpus layout (all files from naturalstories_RTS/):

      processed_RTs.tsv      : WorkerId, item, zone, RT, ...
      processed_wordinfo.tsv : item, zone, word, sentence, ...
      all_stories.tok        : fallback plain-text token list

    ``item``  maps to story_id  (1-10)
    ``zone``  maps to word_position (1-based in the corpus → converted to 0-based)
    """
    ns_cfg       = cfg["natural_stories"]
    rt_path      = Path(ns_cfg["rt_file"])
    wordinfo_path = Path(ns_cfg.get("wordinfo_file",
                         str(Path(ns_cfg["rt_file"]).parent / "processed_wordinfo.tsv")))

    _check_exists(rt_path, "Natural Stories RT file (processed_RTs.tsv)")

    # --- reading times -------------------------------------------------------
    rts = pd.read_csv(rt_path, sep="\t")
    rts = rts.rename(columns={
        "WorkerId": "subject",
        "item":     "story_id",
        "zone":     "word_position",
        "RT":       "rt_ms",
    })
    rts = rts[["subject", "story_id", "word_position", "rt_ms"]].copy()
    rts["rt_ms"]        = pd.to_numeric(rts["rt_ms"], errors="coerce")
    rts["story_id"]     = pd.to_numeric(rts["story_id"], errors="coerce")
    rts["word_position"] = pd.to_numeric(rts["word_position"], errors="coerce")
    rts.dropna(subset=["rt_ms", "story_id", "word_position"], inplace=True)
    rts["story_id"]      = rts["story_id"].astype(int)
    rts["word_position"] = rts["word_position"].astype(int)

    # --- word info (preferred: processed_wordinfo.tsv) -----------------------
    if wordinfo_path.exists():
        logger.info("Loading word info from %s", wordinfo_path)
        wordinfo = pd.read_csv(wordinfo_path, sep="\t")
        # Canonical columns: item, zone, word  (plus optional sentence, etc.)
        wordinfo = wordinfo.rename(columns={
            "item": "story_id",
            "zone": "word_position",
        })
        if "word" not in wordinfo.columns:
            # Some releases call it "Word"
            wordinfo = wordinfo.rename(columns={"Word": "word"})
        wordinfo["story_id"]      = wordinfo["story_id"].astype(int)
        wordinfo["word_position"] = wordinfo["word_position"].astype(int)
        wordinfo["word"]          = wordinfo["word"].str.lower().str.strip()
        cols = ["story_id", "word_position", "word"]
        if "sentence" in wordinfo.columns:
            cols.append("sentence")
        tokens = wordinfo[cols].drop_duplicates(subset=["story_id", "word_position"])
    else:
        # Fallback: parse all_stories.tok
        stories_path = Path(ns_cfg["stories_file"])
        _check_exists(stories_path, "Natural Stories token file (all_stories.tok)")
        logger.info("processed_wordinfo.tsv not found, falling back to %s", stories_path)
        tokens = pd.read_csv(
            stories_path, sep="\t", header=None,
            names=["story_id", "word_position", "word"],
            dtype={"story_id": int, "word_position": int, "word": str},
        )
        tokens["word"] = tokens["word"].str.lower().str.strip()

    # --- merge ---------------------------------------------------------------
    df = pd.merge(rts, tokens, on=["story_id", "word_position"], how="inner")
    logger.info("Merged RT+tokens: %d rows", len(df))

    # --- sentence boundaries -------------------------------------------------
    if "sentence" in df.columns:
        # `sentence` is an INTEGER sentence number, not sentence text.
        # Use it directly as sentence_id, then reconstruct sentence_text
        # by joining the words for each (story_id, sentence_id) group.
        df = df.rename(columns={"sentence": "sentence_id"})
        df["sentence_id"] = pd.to_numeric(df["sentence_id"], errors="coerce").astype("Int64")

        # Build sentence_text: join unique words per (story_id, sentence_id)
        # Use one representative row per subject to avoid duplicates
        word_order = (
            df[["story_id", "sentence_id", "word_position", "word"]]
            .drop_duplicates(subset=["story_id", "sentence_id", "word_position"])
            .sort_values(["story_id", "sentence_id", "word_position"])
        )
        sent_texts = (
            word_order
            .groupby(["story_id", "sentence_id"])["word"]
            .apply(lambda ws: " ".join(ws))
            .reset_index()
            .rename(columns={"word": "sentence_text"})
        )
        df = pd.merge(df, sent_texts, on=["story_id", "sentence_id"], how="left")

        # Reset word_position to 0-based within each sentence per subject
        df["word_position"] = df.groupby(
            ["story_id", "sentence_id", "subject"]
        ).cumcount()
    else:
        df = _assign_sentence_ids(df, group_cols=["story_id"])

    return df


# ---------------------------------------------------------------------------
# GECO (Ghent Eye-Tracking Corpus)
# ---------------------------------------------------------------------------

def _load_geco(cfg: dict) -> pd.DataFrame:
    """
    Parse the GECO corpus Excel files.
    EnglishMaterial.xlsx  – word text + positions
    L1ReadingData.xlsx    – first fixation / total reading time per participant
    """
    geco_cfg    = cfg["geco"]
    words_path  = Path(geco_cfg["words_file"])
    rt_path     = Path(geco_cfg["rt_file"])

    _check_exists(words_path, "GECO material file")
    _check_exists(rt_path,    "GECO RT file")

    words = pd.read_excel(words_path)
    rts   = pd.read_excel(rt_path)

    # GECO canonical columns
    words = words.rename(columns={
        "WORD":         "word",
        "WORD_ID":      "word_id",
        "SENTENCE_ID":  "sentence_id",
        "PART":         "story_id",
    })
    words["word"] = words["word"].str.lower().str.strip()

    rts = rts.rename(columns={
        "PP_NR":         "subject",
        "WORD_ID":       "word_id",
        "TOTAL_READING_TIME": "rt_ms",
    })
    rts = rts[["subject", "word_id", "rt_ms"]].copy()
    rts["rt_ms"] = pd.to_numeric(rts["rt_ms"], errors="coerce")
    rts.dropna(subset=["rt_ms"], inplace=True)

    df = pd.merge(rts, words, on="word_id", how="inner")
    df["word_position"] = df.groupby(["subject", "sentence_id"]).cumcount()

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_sentence_ids(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Assign sentence_id, sentence_text, and within-sentence word_position.
    Operates on UNIQUE words (one row per story × word_position) then merges
    back — avoids iterating over all 800K subject-level rows.
    """
    # Step 1: unique words per story position
    unique_words = (
        df[group_cols + ["word_position", "word"]]
        .drop_duplicates(subset=group_cols + ["word_position"])
        .sort_values(group_cols + ["word_position"])
        .reset_index(drop=True)
    )

    # Step 2: assign sentence boundaries on unique words
    sent_records = []
    for key, grp in unique_words.groupby(group_cols, sort=True):
        grp = grp.sort_values("word_position").reset_index(drop=True)
        sent_id   = 0
        sent_pos  = 0
        sent_words: list[str] = []
        rows: list[dict] = []

        for _, row in grp.iterrows():
            rows.append({
                **{c: (key if len(group_cols) == 1 else key[i])
                   for i, c in enumerate(group_cols)},
                "story_word_pos": int(row["word_position"]),  # original position
                "sentence_id":    sent_id,
                "sent_word_pos":  sent_pos,
            })
            sent_words.append(str(row["word"]))
            sent_pos += 1
            if str(row["word"]).rstrip("\"'").endswith((".", "?", "!")):
                sent_text = " ".join(sent_words)
                for r in rows:
                    r["sentence_text"] = sent_text
                sent_records.extend(rows)
                sent_id   += 1
                sent_pos   = 0
                sent_words = []
                rows       = []

        if rows:
            sent_text = " ".join(sent_words)
            for r in rows:
                r["sentence_text"] = sent_text
            sent_records.extend(rows)

    sent_df = pd.DataFrame(sent_records)
    # Ensure key columns are int64 (dict-built DataFrames can infer object)
    for c in group_cols:
        sent_df[c] = pd.to_numeric(sent_df[c], errors="coerce").astype(int)
    sent_df["story_word_pos"] = sent_df["story_word_pos"].astype(int)
    sent_df["sentence_id"]   = sent_df["sentence_id"].astype(int)

    # Step 3: merge sentence info back using original word_position as key
    merge_cols = group_cols + ["story_word_pos"]
    df = df.copy()
    for c in group_cols:
        df[c] = df[c].astype(int)
    df["story_word_pos"] = df["word_position"].astype(int)
    df = df.merge(
        sent_df[merge_cols + ["sentence_id", "sentence_text"]],
        on=merge_cols,
        how="left",
    )
    df = df.drop(columns=["story_word_pos"])

    # Step 4: reset word_position to 0-based within each sentence per subject
    df["word_position"] = df.groupby(
        group_cols + ["sentence_id", "subject"]
    ).cumcount()

    return df


def _apply_exclusions(df: pd.DataFrame, bay_cfg: dict) -> pd.DataFrame:
    """Remove outlier RTs per the config thresholds."""
    n0 = len(df)
    min_rt = bay_cfg.get("min_rt_ms", 100)
    max_rt = bay_cfg.get("max_rt_ms", 3000)
    sd_cut = bay_cfg.get("outlier_sd_cutoff", 3.0)

    df = df[(df["rt_ms"] >= min_rt) & (df["rt_ms"] <= max_rt)].copy()

    # per-subject SD cutoff on log RT
    df["_log_rt"] = np.log(df["rt_ms"])
    sub_stats = (
        df.groupby("subject")["_log_rt"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "_mu", "std": "_sd"})
    )
    df = df.join(sub_stats, on="subject")
    df = df[np.abs(df["_log_rt"] - df["_mu"]) <= sd_cut * df["_sd"]]
    df = df.drop(columns=["_log_rt", "_mu", "_sd"])

    logger.info("Exclusions: %d → %d rows (removed %d)", n0, len(df), n0 - len(df))
    return df.reset_index(drop=True)


def _check_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at '{path}'.\n"
            "Please download the corpus and place it at the expected path "
            "(see config.yaml)."
        )


def build_sentence_corpus(df: pd.DataFrame) -> list[str]:
    """Return a deduplicated list of sentence strings for UD parsing."""
    return (
        df[["story_id", "sentence_id", "sentence_text"]]
        .drop_duplicates()
        ["sentence_text"]
        .tolist()
    )

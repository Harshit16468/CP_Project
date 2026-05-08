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
    dataset: Literal["natural_stories", "geco", "dundee"],
    cfg: dict,
) -> pd.DataFrame:
    """
    Load and preprocess the chosen psycholinguistic dataset.

    Parameters
    ----------
    dataset : str
        One of ``"natural_stories"``, ``"geco"``, or ``"dundee"``.
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
    elif dataset == "dundee":
        df = _load_dundee(cfg)
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

    Columns produced match the Natural Stories schema:
      subject, story_id, sentence_id, word_position (0-based within sentence),
      word, rt_ms, sentence_text
    """
    geco_cfg    = cfg["geco"]
    words_path  = Path(geco_cfg["words_file"])
    rt_path     = Path(geco_cfg["rt_file"])

    _check_exists(words_path, "GECO material file")
    _check_exists(rt_path,    "GECO RT file")

    words = pd.read_excel(words_path)
    rts   = pd.read_excel(rt_path)

    # GECO release format:
    #   WORD_ID     = "PART-SENTENCE-WORDPOSITION"  (e.g. "1-5-1")
    #   SENTENCE_ID = "PART-SENTENCE"               (e.g. "1-1")
    #   PP_NR       = "pp01" ... (string)
    #   RT column   = WORD_TOTAL_READING_TIME       (object dtype; "." marks skipped)

    # ── Material: parse compound WORD_ID into numeric story / sentence / position ─
    if "WORD_ID" not in words.columns:
        raise ValueError(f"GECO material missing WORD_ID. Cols: {list(words.columns)}")

    parts = words["WORD_ID"].astype(str).str.split("-", expand=True)
    if parts.shape[1] < 3:
        raise ValueError(
            f"Expected WORD_ID format 'PART-SENTENCE-POSITION', got: {words['WORD_ID'].head().tolist()}"
        )
    words["story_id"]      = pd.to_numeric(parts[0], errors="coerce")
    words["sentence_in_p"] = pd.to_numeric(parts[1], errors="coerce")
    words["word_position"] = pd.to_numeric(parts[2], errors="coerce") - 1   # 0-based

    # Build a globally-unique numeric sentence_id (story * 1e6 + sentence)
    words["sentence_id"] = (
        words["story_id"].astype("Int64") * 1_000_000
        + words["sentence_in_p"].astype("Int64")
    )

    words["word"]    = words["WORD"].astype(str).str.lower().str.strip()
    words["word_id"] = words["WORD_ID"].astype(str)            # keep string key for merge
    words = words.dropna(subset=["story_id", "sentence_id", "word_position"]).copy()
    words["story_id"]      = words["story_id"].astype(int)
    words["sentence_id"]   = words["sentence_id"].astype("Int64").astype(int)
    words["word_position"] = words["word_position"].astype(int)

    # Build sentence_text per sentence_id (Stanza needs raw text in step 5)
    word_order = (
        words[["sentence_id", "word_position", "word"]]
        .sort_values(["sentence_id", "word_position"])
        .drop_duplicates(subset=["sentence_id", "word_position"])
    )
    sent_texts = (
        word_order.groupby("sentence_id")["word"]
        .apply(lambda ws: " ".join(ws))
        .reset_index()
        .rename(columns={"word": "sentence_text"})
    )
    words = words.merge(sent_texts, on="sentence_id", how="left")

    # ── Reading times: pick the right RT column and clean ────────────────────
    rt_candidates = ["WORD_TOTAL_READING_TIME", "TOTAL_READING_TIME",
                     "WORD_GAZE_DURATION", "WORD_FIRST_FIXATION_DURATION"]
    rt_col = next((c for c in rt_candidates if c in rts.columns), None)
    if rt_col is None:
        raise ValueError(
            f"GECO RT file has no recognised RT column. Cols: {list(rts.columns)}"
        )
    logger.info("GECO using RT column: %s", rt_col)

    if "PP_NR" not in rts.columns or "WORD_ID" not in rts.columns:
        raise ValueError(
            f"GECO RT file missing PP_NR / WORD_ID. Cols: {list(rts.columns)}"
        )

    rts = rts.rename(columns={"PP_NR": "subject", "WORD_ID": "word_id", rt_col: "rt_ms"})
    rts = rts[["subject", "word_id", "rt_ms"]].copy()
    rts["word_id"] = rts["word_id"].astype(str)

    # GECO uses "." for skipped/missing fixations — coerce to NaN, drop
    rts["rt_ms"] = pd.to_numeric(rts["rt_ms"], errors="coerce")
    rts.dropna(subset=["rt_ms"], inplace=True)
    rts = rts[rts["rt_ms"] > 0]

    # ── Merge RTs with material on string WORD_ID ────────────────────────────
    keep_cols = ["word_id", "sentence_id", "story_id",
                 "word", "word_position", "sentence_text"]
    df = pd.merge(rts, words[keep_cols].drop_duplicates(subset=["word_id"]),
                  on="word_id", how="inner")

    logger.info(
        "GECO loaded: %d observations, %d subjects, %d sentences",
        len(df), df["subject"].nunique(), df["sentence_id"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Dundee Corpus
# ---------------------------------------------------------------------------

def _load_dundee(cfg: dict) -> pd.DataFrame:
    """
    Parse the Dundee Corpus eye-tracking data.

    ⚠  ACCESS: Dundee requires institutional licensing.
       Request from the original authors or an LDC consortium.
       Reference: Kennedy, A. & Pynte, J. (2005). Parafoveal-on-foveal effects
       in normal reading. Vision Research, 45(2), 153-168.

    Expected files (set paths in config.yaml under dundee:)
      data_dir   : directory containing the per-article .dat files (sa_ukb*.dat)
                   OR a single merged TSV with all data

    Standard Dundee column names (tab-separated per-word lines):
      WORD        – word surface form
      WNUM        – word index within text (1-based)
      SNUM        – sentence index within text (1-based)
      TNUM        – text (article) number 1-20
      FDURP       – first-fixation duration (prior fixation; often skipped)
      FDURS       – first-fixation duration (single fixation only)
      GDUR        – gaze duration  ← primary RT measure used here
      RPURT       – right-pass (re-reading) duration
      Subject is encoded in filename (sa, sb, …, sj for 10 subjects)
      or in a SUBJ column if you have a merged file.

    The loader is flexible: it accepts both the original per-subject files
    and a single pre-merged TSV (produced by, e.g., dundee_merge.py).

    Columns produced match the pipeline schema:
      subject, story_id (=TNUM), sentence_id (=SNUM), word_position (0-based),
      word, rt_ms (=GDUR), sentence_text
    """
    dundee_cfg = cfg.get("dundee", {})
    data_dir   = Path(dundee_cfg.get("data_dir", ""))
    merged_tsv = dundee_cfg.get("merged_file", "")
    rt_measure = dundee_cfg.get("rt_measure", "FDUR")   # FDUR | GDUR | FPRT | RPURT
    treebank_p = Path(dundee_cfg.get("treebank_file", "")) if dundee_cfg.get("treebank_file") else None

    # ── Load data ────────────────────────────────────────────────────────────
    if merged_tsv and Path(merged_tsv).exists():
        logger.info("Loading Dundee from merged TSV: %s", merged_tsv)
        df_raw = pd.read_csv(merged_tsv, sep="\t", low_memory=False)
    elif data_dir.exists():
        dfs = []
        # The released Dundee per-fixation .dat files are whitespace-aligned,
        # NOT tab-separated. Use a regex separator.
        # Per-fixation files follow pattern "sXNNma{1|2}p.dat" (subject s[a-j],
        # text 01..20, pass 1|2). Filter out corpus / treebank text files like
        # tx01wrdp.dat which are not eye-tracking data.
        dat_files = sorted(p for p in data_dir.glob("*.dat")
                           if p.stem.startswith("s") and "ma" in p.stem)
        if not dat_files:
            raise FileNotFoundError(
                f"No subject .dat files found in Dundee data_dir: {data_dir}"
            )
        for fpath in dat_files:
            # subject = first 2 chars of filename (e.g. "sa", "sb", ..., "sj")
            subject_id = fpath.stem[:2]
            tmp = None
            for enc in ("utf-8", "latin-1"):
                try:
                    tmp = pd.read_csv(fpath, sep=r"\s+", engine="python",
                                      on_bad_lines="skip", encoding=enc)
                    break
                except (UnicodeDecodeError, Exception) as e:
                    if enc == "latin-1":
                        logger.warning("Skipping unreadable file %s: %s", fpath.name, e)
                    continue
            if tmp is None:
                continue
            # Drop *Blink and other comment / sentinel rows
            if "WORD" in tmp.columns:
                tmp = tmp[~tmp["WORD"].astype(str).str.startswith("*")]
            tmp["subject"] = subject_id
            tmp["__src_file"] = fpath.name
            dfs.append(tmp)
        df_raw = pd.concat(dfs, ignore_index=True)
        logger.info("Loaded %d Dundee files, %d raw fixation rows total",
                    len(dat_files), len(df_raw))
    else:
        raise FileNotFoundError(
            "Dundee data not found.  Set dundee.data_dir or dundee.merged_file "
            "in config.yaml.  Obtain the corpus via institutional licensing."
        )

    # ── Normalise column names ────────────────────────────────────────────────
    col_map = {
        "WORD": "word",
        "WNUM": "word_id",
        "SNUM": "sentence_id",   # only present in some Dundee releases
        "TNUM": "story_id",
        "TEXT": "story_id",      # released per-fixation files use TEXT
        "SUBJ": "subject",
        rt_measure: "rt_ms",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    if "rt_ms" not in df_raw.columns:
        # Fallback: any duration-looking column
        for cand in ("FDUR", "GDUR", "FPRT", "RPURT", "TFD"):
            if cand in df_raw.columns:
                df_raw = df_raw.rename(columns={cand: "rt_ms"})
                logger.info("Dundee using fallback RT column: %s", cand)
                break

    # ── Per-fixation → per-word: aggregate fixations to total fixation time ──
    # Released .dat files are PER FIXATION; the same word can appear many times.
    # We aggregate sum(rt_ms) across all fixations per (subject, story, word) =
    # total fixation duration (TFD), the standard eye-tracking RT measure that
    # is most comparable to GECO's WORD_TOTAL_READING_TIME and to self-paced RTs.
    needed = {"subject", "story_id", "word_id", "word", "rt_ms"}
    missing = needed - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"Dundee loader: missing columns after rename: {missing}. "
            f"Available columns: {list(df_raw.columns)}"
        )

    for col in ("word_id", "story_id"):
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    df_raw["rt_ms"] = pd.to_numeric(df_raw["rt_ms"], errors="coerce")
    df_raw = df_raw[(df_raw["rt_ms"] > 0)
                    & df_raw["word_id"].notna()
                    & df_raw["story_id"].notna()
                    & (df_raw["word_id"] > 0)
                    & (df_raw["story_id"] > 0)].copy()
    df_raw["word_id"]  = df_raw["word_id"].astype(int)
    df_raw["story_id"] = df_raw["story_id"].astype(int)
    df_raw["word"]     = df_raw["word"].astype(str).str.lower().str.strip()

    # Aggregate fixations → per-word total fixation duration
    df_raw = (
        df_raw.groupby(["subject", "story_id", "word_id"], as_index=False)
              .agg(rt_ms=("rt_ms", "sum"),
                   word=("word", "first"))
    )
    logger.info("After per-word aggregation: %d observations, %d subjects",
                len(df_raw), df_raw["subject"].nunique())

    # ── Sentence ID assignment ──────────────────────────────────────────────
    # If the loaded files don't have SNUM, look up sentence IDs from the
    # Dundee Treebank (TLT2015) which provides (Itemno, WNUM) → SentenceID.
    if "sentence_id" not in df_raw.columns or df_raw["sentence_id"].isna().any():
        # Try to find the treebank file automatically if not configured
        if treebank_p is None or not treebank_p.exists():
            auto = data_dir / "TLT2015" / "TheDundeeTreebank_v1-0.csv"
            if auto.exists():
                treebank_p = auto
        if treebank_p is not None and treebank_p.exists():
            logger.info("Joining sentence IDs from treebank: %s", treebank_p)
            tb = pd.read_csv(treebank_p, sep="\t")
            tb = tb.rename(columns={"Itemno": "story_id",
                                    "WNUM":   "word_id",
                                    "SentenceID": "_sent"})
            tb = tb[["story_id", "word_id", "_sent"]].drop_duplicates(
                subset=["story_id", "word_id"]
            )
            df_raw = df_raw.merge(tb, on=["story_id", "word_id"], how="left")
            # Globally unique sentence_id = story * 1e6 + sentence
            df_raw["sentence_id"] = (
                df_raw["story_id"].astype("Int64") * 1_000_000
                + df_raw["_sent"].astype("Int64")
            )
            df_raw = df_raw.drop(columns=["_sent"])
        else:
            # Fallback: each text becomes one giant sentence
            logger.warning("No SNUM and no treebank — each text treated as a single sentence.")
            df_raw["sentence_id"] = df_raw["story_id"] * 1_000_000

    df_raw = df_raw.dropna(subset=["sentence_id"]).copy()
    df_raw["sentence_id"] = df_raw["sentence_id"].astype("Int64").astype(int)

    # ── Build sentence_text ──────────────────────────────────────────────────
    word_order = (
        df_raw[["story_id", "sentence_id", "word_id", "word"]]
        .drop_duplicates(subset=["story_id", "sentence_id", "word_id"])
        .sort_values(["story_id", "sentence_id", "word_id"])
    )
    sent_texts = (
        word_order
        .groupby(["story_id", "sentence_id"])["word"]
        .apply(lambda ws: " ".join(ws))
        .reset_index()
        .rename(columns={"word": "sentence_text"})
    )
    df_raw = df_raw.merge(sent_texts, on=["story_id", "sentence_id"], how="left")

    # ── word_position: 0-based rank of word_id within sentence ──────────────
    # Must rank-by-word_id (NOT cumcount of rows) so the same word gets the same
    # position across all subjects — neural_metrics aligns metrics to
    # sentence_text by word_position, and sentence_text contains one entry per
    # unique word.
    df_raw = df_raw.sort_values(["story_id", "sentence_id", "word_id"])
    df_raw["word_position"] = (
        df_raw.groupby(["story_id", "sentence_id"])["word_id"]
              .rank(method="dense").astype(int) - 1
    )

    logger.info(
        "Dundee loaded: %d observations, %d subjects, %d texts, %d sentences",
        len(df_raw),
        df_raw["subject"].nunique(),
        df_raw["story_id"].nunique(),
        df_raw["sentence_id"].nunique(),
    )
    return df_raw


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
            # groupby with a list always returns a tuple key, even for 1 column
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_vals = {c: int(key_tuple[i]) for i, c in enumerate(group_cols)}
            rows.append({
                **key_vals,
                "story_word_pos": int(row["word_position"]),
                "sentence_id":    int(sent_id),
                "sent_word_pos":  int(sent_pos),
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
    # Values are already Python ints from the record-building loop above

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

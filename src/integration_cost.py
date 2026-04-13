"""
Step 5 – Integration Cost (Syntactic Memory Load)
===================================================
Uses Universal Dependencies (UD) parsing via Stanza to compute the linear
dependency length for each word:

    DL(w_i) = |position(w_i) − position(syntactic_head(w_i))|

This is the standard operationalisation of Dependency Locality Theory (Gibson
2000) and integration cost in the psycholinguistics literature.

For root tokens (which have no head), DL is set to 0.

Public API
----------
    build_parser(cfg)                      -> UDParser
    compute_integration_cost(df, parser)   -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import stanza
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser wrapper
# ---------------------------------------------------------------------------

class UDParser:
    """Thin wrapper around a Stanza NLP pipeline for UD dependency parsing."""

    def __init__(self, language: str = "en") -> None:
        logger.info("Loading Stanza UD parser (lang=%s) …", language)
        # Download models if not present
        stanza.download(language, verbose=False)
        self._nlp = stanza.Pipeline(
            lang=language,
            processors="tokenize,pos,lemma,depparse",
            tokenize_pretokenized=False,
            verbose=False,
        )

    def parse(self, sentence: str) -> list[dict]:
        """
        Parse a sentence and return per-token dependency info.

        Returns
        -------
        list of dicts with keys:
            id           : 1-based token index (Stanza convention)
            word         : surface form
            head_id      : 1-based head index (0 = root)
            dep_rel      : dependency relation label
            dep_length   : |id − head_id|  (0 for root)
        """
        doc = self._nlp(sentence)
        results = []
        for sent in doc.sentences:
            for token in sent.words:
                dep_len = (
                    0 if token.head == 0
                    else abs(token.id - token.head)
                )
                results.append({
                    "id":            token.id,
                    "word":          token.text.lower(),
                    "head_id":       token.head,
                    "dep_rel":       token.deprel,
                    "dep_length":    dep_len,
                })
        return results


# ---------------------------------------------------------------------------
# Build parser from config
# ---------------------------------------------------------------------------

def build_parser(cfg: dict) -> UDParser:
    ic_cfg   = cfg.get("integration_cost", {})
    language = ic_cfg.get("language", "en")
    return UDParser(language=language)


# ---------------------------------------------------------------------------
# Attach integration cost to the DataFrame
# ---------------------------------------------------------------------------

def compute_integration_cost(
    df: pd.DataFrame,
    parser: UDParser,
) -> pd.DataFrame:
    """
    Add columns to *df*:
        dep_length        : linear distance to syntactic head
        dep_head_position : 0-based position of the syntactic head in sentence
        dep_rel           : dependency relation label

    Parameters
    ----------
    df     : reading-time DataFrame (from data_prep)
    parser : UDParser instance
    """
    df = df.copy()
    dep_len_vals:  dict[int, float] = {}
    dep_head_vals: dict[int, float] = {}
    dep_rel_vals:  dict[int, str]   = {}

    for (sid, sent_id), grp in tqdm(
        df.groupby(["story_id", "sentence_id"]),
        desc="UD parsing",
        unit="sent",
    ):
        grp_sorted = grp.sort_values("word_position")
        sent_text  = grp_sorted["sentence_text"].iloc[0]
        df_words   = grp_sorted["word"].str.lower().tolist()

        try:
            parse_results = parser.parse(sent_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parsing failed for (%s,%s): %s", sid, sent_id, exc)
            for idx in grp_sorted.index:
                dep_len_vals[idx]  = float("nan")
                dep_head_vals[idx] = float("nan")
                dep_rel_vals[idx]  = ""
            continue

        # Align parser tokens (1-based) with DataFrame word_position (0-based)
        # Use greedy surface match to handle minor tokenisation differences.
        aligned = _align_tokens_to_words(df_words, parse_results)

        for row_idx, (_, row) in zip(grp_sorted.index, grp_sorted.iterrows()):
            wpos  = int(row["word_position"])
            match = aligned.get(wpos)
            if match is not None:
                dep_len_vals[row_idx]  = float(match["dep_length"])
                # Convert head_id (1-based) to 0-based word_position
                dep_head_vals[row_idx] = float(match["head_id"] - 1) if match["head_id"] > 0 else float("nan")
                dep_rel_vals[row_idx]  = match["dep_rel"]
            else:
                dep_len_vals[row_idx]  = float("nan")
                dep_head_vals[row_idx] = float("nan")
                dep_rel_vals[row_idx]  = ""

    df["dep_length"]        = df.index.map(dep_len_vals)
    df["dep_head_position"] = df.index.map(dep_head_vals)
    df["dep_rel"]           = df.index.map(dep_rel_vals)

    logger.info(
        "Integration cost computed: mean dep_length=%.2f, missing=%d",
        df["dep_length"].mean(),
        df["dep_length"].isna().sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Token alignment helper
# ---------------------------------------------------------------------------

def _align_tokens_to_words(
    words: list[str],
    parse_tokens: list[dict],
) -> dict[int, dict]:
    """
    Map 0-based word_position → parse_token dict.

    Handles slight tokenisation mismatches by matching on lowercased surface
    forms; falls back to positional alignment when counts match.
    """
    if len(words) == len(parse_tokens):
        return {i: parse_tokens[i] for i in range(len(words))}

    # Build surface→parse index lookup
    mapping: dict[int, dict] = {}
    tok_idx = 0
    for word_pos, word in enumerate(words):
        while tok_idx < len(parse_tokens):
            pt = parse_tokens[tok_idx]
            if pt["word"].startswith(word[:3]) or word.startswith(pt["word"][:3]):
                mapping[word_pos] = pt
                tok_idx += 1
                break
            tok_idx += 1
        else:
            break
    return mapping

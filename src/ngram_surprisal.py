"""
Step 2 – N-Gram Surprisal (Baseline Predictability)
=====================================================
Trains a trigram language model with Kneser-Ney smoothing on a background
corpus and computes per-word surprisal:

    S_ngram(w_i) = -log2 P(w_i | w_{i-2}, w_{i-1})

The model is trained on a background corpus (e.g., Wikipedia / BookCorpus
excerpt) that is *separate* from the reading-time stimulus text to avoid
in-sample inflation.

Public API
----------
    build_ngram_model(corpus_path, cfg)  -> NgramSurprisalModel
    compute_ngram_surprisal(df, model)   -> pd.DataFrame  (adds column)
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Generator, Sequence

import nltk
import numpy as np
import pandas as pd
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class NgramSurprisalModel:
    """Thin wrapper around NLTK's KneserNeyInterpolated LM."""

    def __init__(self, order: int = 3):
        self.order = order
        self._lm: KneserNeyInterpolated | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sentences: list[list[str]]) -> None:
        """
        Train on a list of tokenised sentences (each a list of strings).
        """
        logger.info("Training %d-gram Kneser-Ney model on %d sentences …",
                    self.order, len(sentences))
        train_data, vocab = padded_everygram_pipeline(self.order, sentences)
        self._lm = KneserNeyInterpolated(self.order)
        self._lm.fit(train_data, vocab)
        logger.info("N-gram model trained (vocab size ≈ %d)", len(self._lm.vocab))

    # ------------------------------------------------------------------
    # Surprisal
    # ------------------------------------------------------------------

    def surprisal(self, word: str, context: tuple[str, ...]) -> float:
        """
        Return -log2 P(word | context).
        ``context`` should have length == order-1 (padded as needed).
        """
        if self._lm is None:
            raise RuntimeError("Model has not been trained yet.")
        # KneserNeyInterpolated.score returns log10; convert to log2 bits
        # Actually NLTK .score() returns the actual probability (0-1).
        prob = self._lm.score(word, list(context))
        if prob <= 0:
            prob = 1e-10   # floor for OOV
        return -math.log2(prob)

    def surprisal_sentence(self, tokens: list[str]) -> list[float]:
        """Surprisal for each token in a tokenised sentence."""
        pad = ["<s>"] * (self.order - 1)
        padded = pad + [t.lower() for t in tokens]
        surprisals = []
        for i in range(len(tokens)):
            ctx_start = i                           # already offset by pad
            ctx = tuple(padded[ctx_start: ctx_start + self.order - 1])
            w   = padded[ctx_start + self.order - 1]
            surprisals.append(self.surprisal(w, ctx))
        return surprisals


# ---------------------------------------------------------------------------
# Build from corpus file
# ---------------------------------------------------------------------------

def build_ngram_model(corpus_path: str | Path, cfg: dict) -> NgramSurprisalModel:
    """
    Load background corpus, tokenise, and train a KN n-gram model.

    Parameters
    ----------
    corpus_path : path to plain-text background corpus (one paragraph / line).
    cfg         : ``cfg["ngram"]`` sub-dict from config.yaml.
    """
    corpus_path = Path(corpus_path)
    order = cfg.get("order", 3)

    logger.info("Loading background corpus from %s …", corpus_path)
    text = corpus_path.read_text(encoding="utf-8", errors="replace")

    sentences = [
        [w.lower() for w in word_tokenize(sent)]
        for sent in sent_tokenize(text)
        if len(sent.split()) >= 2
    ]

    model = NgramSurprisalModel(order=order)
    model.train(sentences)
    return model


# ---------------------------------------------------------------------------
# Attach surprisal to the reading-time DataFrame
# ---------------------------------------------------------------------------

def compute_ngram_surprisal(
    df: pd.DataFrame,
    model: NgramSurprisalModel,
) -> pd.DataFrame:
    """
    Add column ``ngram_surprisal`` to *df* (bits, -log2).

    The function groups by (story_id, sentence_id) to reconstruct sentence
    context for each word.
    """
    df = df.copy()
    surprisal_values: dict[int, float] = {}

    group_cols = ["story_id", "sentence_id"]
    for _, grp in df.groupby(group_cols):
        grp_sorted = grp.sort_values("word_position")
        tokens  = grp_sorted["word"].str.lower().tolist()
        surprs  = model.surprisal_sentence(tokens)
        for idx, surp in zip(grp_sorted.index, surprs):
            surprisal_values[idx] = surp

    df["ngram_surprisal"] = df.index.map(surprisal_values)
    logger.info("N-gram surprisal computed (mean=%.3f bits)",
                df["ngram_surprisal"].mean())
    return df

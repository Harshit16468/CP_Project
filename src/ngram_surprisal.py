"""
Step 2 – N-Gram Surprisal (Baseline Predictability)
=====================================================
Trains a trigram language model with Kneser-Ney smoothing on a background
corpus and computes per-word surprisal:

    S_ngram(w_i) = -log2 P(w_i | w_{i-2}, w_{i-1})

Speed optimisations
-------------------
- Trained model is pickled to disk and reloaded on subsequent runs
  (training the KN model on 2MB corpus takes ~2 min; loading pickle ~2 sec)
- Fast whitespace tokenisation for background corpus (no NLTK punkt overhead)
- Surprisal computed only for unique (story_id, sentence_id, word_position)
  triples, then broadcast back to all subjects via a merge

Public API
----------
    build_ngram_model(corpus_path, cfg, cache_path)  -> NgramSurprisalModel
    compute_ngram_surprisal(df, model)               -> pd.DataFrame
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

logger = logging.getLogger(__name__)

for _pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class NgramSurprisalModel:
    def __init__(self, order: int = 3):
        self.order = order
        self._lm: KneserNeyInterpolated | None = None

    def train(self, sentences: list[list[str]]) -> None:
        logger.info("Training %d-gram KN model on %d sentences …",
                    self.order, len(sentences))
        train_data, vocab = padded_everygram_pipeline(self.order, sentences)
        self._lm = KneserNeyInterpolated(self.order)
        self._lm.fit(train_data, vocab)
        logger.info("N-gram model trained (vocab size ≈ %d)", len(self._lm.vocab))

    def surprisal(self, word: str, context: tuple[str, ...]) -> float:
        if self._lm is None:
            raise RuntimeError("Model not trained.")
        prob = self._lm.score(word, list(context))
        return -math.log2(prob) if prob > 0 else -math.log2(1e-10)

    def surprisal_sentence(self, tokens: list[str]) -> list[float]:
        pad    = ["<s>"] * (self.order - 1)
        padded = pad + [t.lower() for t in tokens]
        return [
            self.surprisal(
                padded[i + self.order - 1],
                tuple(padded[i: i + self.order - 1]),
            )
            for i in range(len(tokens))
        ]


# ---------------------------------------------------------------------------
# Build / load model
# ---------------------------------------------------------------------------

def build_ngram_model(
    corpus_path: str | Path,
    cfg: dict,
    cache_path: str | Path | None = None,
) -> NgramSurprisalModel:
    """
    Load from pickle cache if available, otherwise train and cache.

    Parameters
    ----------
    corpus_path : plain-text background corpus
    cfg         : cfg["ngram"] sub-dict
    cache_path  : where to pickle the trained model
                  (defaults to <corpus_path>.ngram.pkl)
    """
    corpus_path = Path(corpus_path)
    order       = cfg.get("order", 3)

    if cache_path is None:
        cache_path = corpus_path.with_suffix(".ngram.pkl")
    cache_path = Path(cache_path)

    # ── Load from cache ───────────────────────────────────────────────────
    if cache_path.exists():
        logger.info("Loading cached n-gram model from %s …", cache_path)
        with cache_path.open("rb") as fh:
            model = pickle.load(fh)
        logger.info("N-gram model loaded (vocab size ≈ %d)", len(model._lm.vocab))
        return model

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Loading background corpus from %s …", corpus_path)
    text = corpus_path.read_text(encoding="utf-8", errors="replace")

    # Fast tokenisation: split on whitespace / punctuation boundaries.
    # No punkt overhead — sufficient for a background LM.
    sentences = _fast_tokenise(text)
    logger.info("Tokenised %d sentences from background corpus.", len(sentences))

    model = NgramSurprisalModel(order=order)
    model.train(sentences)

    # ── Cache ─────────────────────────────────────────────────────────────
    with cache_path.open("wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cached n-gram model to %s", cache_path)

    return model


def _fast_tokenise(text: str) -> list[list[str]]:
    """
    Split text into sentences then words using simple regex rules.
    ~10-20× faster than NLTK punkt for a background corpus.
    """
    # Split on sentence-ending punctuation
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for sent in raw_sentences:
        words = re.findall(r"\b\w+\b", sent.lower())
        if len(words) >= 2:
            sentences.append(words)
    return sentences


# ---------------------------------------------------------------------------
# Attach surprisal to DataFrame
# ---------------------------------------------------------------------------

def compute_ngram_surprisal(
    df: pd.DataFrame,
    model: NgramSurprisalModel,
) -> pd.DataFrame:
    """
    Add column ``ngram_surprisal`` to *df*.

    Computes surprisal only for unique (story_id, sentence_id, word_position)
    triples, then merges back — avoids redundant work across subjects.
    """
    # Unique word rows (one per story/sentence/position)
    unique_words = (
        df[["story_id", "sentence_id", "word_position", "word"]]
        .drop_duplicates(subset=["story_id", "sentence_id", "word_position"])
        .sort_values(["story_id", "sentence_id", "word_position"])
    )

    records = []
    for (sid, sent_id), grp in unique_words.groupby(["story_id", "sentence_id"]):
        tokens = grp["word"].str.lower().tolist()
        surprs = model.surprisal_sentence(tokens)
        for (_, row), surp in zip(grp.iterrows(), surprs):
            records.append({
                "story_id":      sid,
                "sentence_id":   sent_id,
                "word_position": row["word_position"],
                "ngram_surprisal": surp,
            })

    surp_df = pd.DataFrame(records)

    df = df.merge(
        surp_df[["story_id", "sentence_id", "word_position", "ngram_surprisal"]],
        on=["story_id", "sentence_id", "word_position"],
        how="left",
    )

    logger.info("N-gram surprisal computed (mean=%.3f bits, missing=%d)",
                df["ngram_surprisal"].mean(), df["ngram_surprisal"].isna().sum())
    return df

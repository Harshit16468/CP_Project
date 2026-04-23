"""
Step 3 – Neural Surprisal & Contextual Entropy
================================================
Extracts per-word metrics from three Transformer architectures:

  GPT-2  (causal / autoregressive)  → true next-word surprisal
  BERT   (masked LM)                → pseudo-surprisal via one word at a time masking
  T5     (encoder-decoder)          → decoder surprisal conditioned on encoder

For each model and each word we compute:
  - Surprisal   S = -log P(w_i | context)          [nats, converted to bits]
  - Entropy     H = -Σ P(x) log P(x)  over vocab   [bits]

Attention weights are also returned (raw tensors per layer/head) for Step 4.

Public API
----------
    NeuralMetricsExtractor(model_name, model_type, device)
    extractor.compute_metrics(df)  -> pd.DataFrame   (adds surprisal/entropy cols)
    extractor.get_attention_weights(sentence)  -> dict
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

logger = logging.getLogger(__name__)

ModelType = Literal["causal", "masked", "seq2seq"]


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class NeuralMetricsExtractor:
    """
    Wraps a HuggingFace model and provides per-word surprisal / entropy.

    Parameters
    ----------
    model_name : HuggingFace model identifier (e.g. ``"gpt2"``)
    model_type : ``"causal"`` | ``"masked"`` | ``"seq2seq"``
    device     : ``"cpu"`` | ``"cuda"`` | ``"auto"``
    """

    def __init__(
        self,
        model_name: str,
        model_type: ModelType,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.device = _resolve_device(device)

        logger.info("Loading %s (%s) on %s …", model_name, model_type, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = _load_model(model_name, model_type)
        self._model.to(self.device)
        self._model.eval()

        # GPT-2 has no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Per-sentence metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sentence_metrics(self, sentence: str) -> list[dict]:
        """
        Return a list of dicts, one per *whitespace token* in ``sentence``:
            {word, token_ids, surprisal_bits, entropy_bits}

        The sentence is re-tokenised with the model's sub-word tokenizer;
        surprisal/entropy for multi-token words are summed / averaged.
        """
        if self.model_type == "causal":
            return self._causal_metrics(sentence)
        elif self.model_type == "masked":
            return self._masked_metrics(sentence)
        elif self.model_type == "seq2seq":
            return self._seq2seq_metrics(sentence)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

    # ------------------------------------------------------------------
    # Attach metrics to DataFrame
    # ------------------------------------------------------------------

    def compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ``{prefix}_surprisal`` and ``{prefix}_entropy`` columns to *df*.

        Groups by (story_id, sentence_id) → calls sentence_metrics once per
        unique sentence, then aligns back by word_position.
        """
        prefix = self.model_name.replace("-", "_").replace("/", "_")
        surp_col = f"{prefix}_surprisal"
        entr_col = f"{prefix}_entropy"

        df = df.copy()
        surp_vals: dict[int, float] = {}
        entr_vals: dict[int, float] = {}

        sentences = df.groupby(["story_id", "sentence_id"])

        for (sid, sent_id), grp in tqdm(
            sentences, desc=f"{self.model_name} metrics", unit="sent"
        ):
            grp_sorted = grp.sort_values("word_position")
            sent_text  = grp_sorted["sentence_text"].iloc[0]

            try:
                results = self.sentence_metrics(sent_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sentence metrics failed for (%s,%s): %s",
                               sid, sent_id, exc)
                for idx in grp_sorted.index:
                    surp_vals[idx] = float("nan")
                    entr_vals[idx] = float("nan")
                continue

            # Align model output (word-level) with DataFrame rows
            # The model splits on whitespace; we trust sentence_text reflects the
            # same tokenisation as word_position order.
            for row_idx, (_, row) in zip(grp_sorted.index, grp_sorted.iterrows()):
                word_pos = int(row["word_position"])
                if word_pos < len(results):
                    surp_vals[row_idx] = results[word_pos]["surprisal_bits"]
                    entr_vals[row_idx] = results[word_pos]["entropy_bits"]
                else:
                    surp_vals[row_idx] = float("nan")
                    entr_vals[row_idx] = float("nan")

        df[surp_col] = df.index.map(surp_vals)
        df[entr_col] = df.index.map(entr_vals)
        logger.info("%s: surprisal mean=%.3f bits, entropy mean=%.3f bits",
                    self.model_name,
                    df[surp_col].mean(), df[entr_col].mean())
        return df

    # ------------------------------------------------------------------
    # Attention weights (for Step 4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_attention_weights(self, sentence: str) -> dict:
        """
        Return raw attention matrices for all layers/heads.

        Returns
        -------
        dict with keys:
            "words"     : list[str]  whitespace-tokenised words
            "tokens"    : list[str]  sub-word tokens
            "attentions": np.ndarray shape (n_layers, n_heads, n_tok, n_tok)
        """
        inputs = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        if self.model_type == "seq2seq":
            # T5: use a minimal decoder so encoder self-attention is available
            bos = torch.tensor(
                [[self._model.config.decoder_start_token_id]], device=self.device
            )
            outputs = self._model(
                **inputs, decoder_input_ids=bos, output_attentions=True
            )
            # encoder_attentions: tuple of (1, n_heads, seq, seq) per encoder layer
            attn_source = outputs.encoder_attentions
        else:
            outputs = self._model(**inputs, output_attentions=True)
            attn_source = outputs.attentions

        attn = np.stack([a.squeeze(0).cpu().numpy()
                         for a in attn_source])   # (L, H, T, T)
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze().tolist()
        )
        return {
            "words":      sentence.split(),
            "tokens":     tokens,
            "attentions": attn,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _causal_metrics(self, sentence: str) -> list[dict]:
        """GPT-2 style left-to-right surprisal."""
        words  = sentence.split()
        max_len = getattr(self._model.config, "n_positions", 1024)
        inputs = self.tokenizer(
            sentence, return_tensors="pt",
            truncation=True, max_length=max_len,
        ).to(self.device)
        input_ids = inputs["input_ids"]                          # (1, T)

        outputs  = self._model(**inputs, output_attentions=False)
        logits   = outputs.logits.squeeze(0)                    # (T, V)

        # shift: logits[i] predicts token[i+1]
        log_probs = F.log_softmax(logits[:-1], dim=-1)           # (T-1, V)
        target_ids = input_ids[0, 1:]                            # (T-1,)
        token_surprisals = (
            -log_probs[torch.arange(len(target_ids)), target_ids]
        ).cpu().tolist()                                         # nats
        token_entropies  = _entropy_from_logits(logits[:-1])    # (T-1,) bits

        # Map sub-word tokens back to whitespace words
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        # tokens[0] is BOS / first token; targets start at tokens[1]
        return _aggregate_to_words(
            words, tokens[1:], token_surprisals, token_entropies
        )

    def _masked_metrics(self, sentence: str) -> list[dict]:
        """BERT pseudo-surprisal: mask one token at a time."""
        inputs    = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        input_ids = inputs["input_ids"].squeeze(0)   # (T,)
        tokens    = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # Identify non-special token positions
        special_ids = set(self.tokenizer.all_special_ids)
        target_positions = [
            i for i, tid in enumerate(input_ids.tolist())
            if tid not in special_ids
        ]

        token_surprisals: list[float] = []
        token_entropies:  list[float] = []

        for pos in target_positions:
            masked = input_ids.clone()
            masked[pos] = self.tokenizer.mask_token_id
            with torch.no_grad():
                out = self._model(
                    input_ids=masked.unsqueeze(0).to(self.device)
                )
            logits_pos = out.logits[0, pos]                  # (V,)
            lp         = F.log_softmax(logits_pos, dim=-1)
            surp       = -lp[input_ids[pos]].item()          # nats
            entr       = _entropy_from_logits(logits_pos.unsqueeze(0))[0]
            token_surprisals.append(surp)
            token_entropies.append(entr)

        words = sentence.split()
        return _aggregate_to_words(
            words,
            [tokens[p] for p in target_positions],
            token_surprisals,
            token_entropies,
        )

    def _seq2seq_metrics(self, sentence: str) -> list[dict]:
        """
        T5 decoder surprisal: encode the sentence, then decode it
        token-by-token using teacher forcing.
        """
        inputs = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        # Decoder input: shift right (T5 uses </s> as BOS)
        dec_inputs = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        dec_ids = dec_inputs["input_ids"]

        # Prepend decoder start token
        bos   = torch.tensor([[self._model.config.decoder_start_token_id]],
                              device=self.device)
        dec_in = torch.cat([bos, dec_ids[:, :-1]], dim=1)

        out    = self._model(**inputs, decoder_input_ids=dec_in,
                             output_attentions=False)
        logits = out.logits.squeeze(0)                       # (T, V)

        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = dec_ids.squeeze(0)
        token_surprisals = (
            -log_probs[torch.arange(len(target_ids)), target_ids]
        ).cpu().tolist()
        token_entropies  = _entropy_from_logits(logits)         # bits

        tokens = self.tokenizer.convert_ids_to_tokens(target_ids.tolist())
        words  = sentence.split()
        return _aggregate_to_words(words, tokens, token_surprisals, token_entropies)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _entropy_from_logits(logits: torch.Tensor) -> list[float]:
    """
    Compute token-level Shannon entropy (bits) from a (T, V) or (V,) logit tensor.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = F.softmax(logits, dim=-1)                 # (T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy_nats = -(probs * log_probs).sum(dim=-1)   # (T,)
    return (entropy_nats / math.log(2)).cpu().tolist()  # convert to bits


def _aggregate_to_words(
    words: list[str],
    sub_tokens: list[str],
    surprisals: list[float],
    entropies: list[float],
) -> list[dict]:
    """
    Re-align sub-word token metrics to whitespace-delimited words.
    Surprisal is *summed* (chain rule); entropy is *averaged*.
    """
    results: list[dict] = []
    tok_idx = 0

    for word in words:
        word_lower = word.lower()
        # Greedily consume sub-tokens that reconstruct this word
        w_surp: list[float] = []
        w_entr: list[float] = []
        accumulated = ""

        while tok_idx < len(sub_tokens) and len(accumulated) < len(word_lower):
            tok = sub_tokens[tok_idx]
            # Strip GPT-2 Ġ / BERT ## prefixes
            clean_tok = tok.lstrip("Ġ▁").replace("##", "")
            accumulated += clean_tok
            w_surp.append(surprisals[tok_idx] if tok_idx < len(surprisals) else 0.0)
            w_entr.append(entropies[tok_idx]  if tok_idx < len(entropies)  else 0.0)
            tok_idx += 1

        results.append({
            "word":           word,
            # surprisals are in nats → convert to bits
            "surprisal_bits": (sum(w_surp) / math.log(2)) if w_surp else float("nan"),
            # _entropy_from_logits already returns bits → no conversion needed
            "entropy_bits":   float(np.mean(w_entr)) if w_entr else float("nan"),
        })

    return results


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_model(name: str, model_type: ModelType):
    if model_type == "causal":
        return AutoModelForCausalLM.from_pretrained(name)
    elif model_type == "masked":
        return AutoModelForMaskedLM.from_pretrained(name)
    elif model_type == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(name)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

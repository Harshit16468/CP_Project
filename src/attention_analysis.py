"""
Step 4 – Attention Head Analysis
==================================
Extracts self-attention matrices from Transformer models and identifies
heads that correlate with syntactic dependency structure (Integration Cost).

Hypothesis 5: Specific attention heads will correlate with dependency length,
suggesting the models implicitly track structural working memory constraints.

Methodology
-----------
For each sentence:
  1. Extract attention weights A^(l,h) of shape (T, T) for each layer l, head h.
  2. For each dependency arc (dep_pos → head_pos), compute the attention weight
     the dependent token pays to its syntactic head: A^(l,h)[dep, head].
  3. Also compute the *maximum attention target* for each token and compare to
     the UD-annotated syntactic head (syntactic head-finding accuracy).
  4. Aggregate across sentences via Spearman correlation:
       corr( A^(l,h)[dep, head_token],  dependency_length )
  5. Report top-K heads ranked by correlation with dependency length.

Public API
----------
    AttentionAnalyzer(extractor, dep_df)
    analyzer.run()  -> pd.DataFrame  (layer × head correlation table)
    analyzer.plot_top_heads(k)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from src.neural_metrics import NeuralMetricsExtractor

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Correlates Transformer attention heads with syntactic dependency length.

    Parameters
    ----------
    extractor : NeuralMetricsExtractor
        A loaded model (causal or masked) with ``get_attention_weights``.
    dep_df : pd.DataFrame
        Must contain columns: story_id, sentence_id, word_position,
        sentence_text, dep_head_position, dep_length.
        (Produced by ``integration_cost.compute_integration_cost``.)
    """

    def __init__(
        self,
        extractor: NeuralMetricsExtractor,
        dep_df: pd.DataFrame,
    ) -> None:
        self.extractor = extractor
        self.dep_df    = dep_df
        self._results: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Compute per-(layer, head) Spearman correlations with dep_length.

        Returns a DataFrame with columns:
            layer, head, rho, pvalue, n_arcs
        sorted by descending |rho|.
        """
        records: list[dict] = []   # (layer, head, dep_len, attn_weight)

        sentences_seen = set()
        for _, sent_grp in tqdm(
            self.dep_df.groupby(["story_id", "sentence_id"]),
            desc=f"Attention ({self.extractor.model_name})",
            unit="sent",
        ):
            sent_text = sent_grp["sentence_text"].iloc[0]
            if sent_text in sentences_seen:
                continue
            sentences_seen.add(sent_text)

            try:
                attn_data = self.extractor.get_attention_weights(sent_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Attention extraction failed: %s", exc)
                continue

            attn       = attn_data["attentions"]    # (L, H, T, T)
            tokens     = attn_data["tokens"]
            n_layers, n_heads, n_tok, _ = attn.shape

            # Map word_position → sub-word token index (first sub-token)
            word_to_tok = _word_to_first_token(tokens, sent_text.split())

            for _, row in sent_grp.iterrows():
                wpos  = int(row["word_position"])
                hpos  = row.get("dep_head_position")
                dlen  = row.get("dep_length", float("nan"))

                if pd.isna(dlen) or pd.isna(hpos):
                    continue

                hpos = int(hpos)
                dep_tok  = word_to_tok.get(wpos)
                head_tok = word_to_tok.get(hpos)
                if dep_tok is None or head_tok is None:
                    continue
                if dep_tok >= n_tok or head_tok >= n_tok:
                    continue

                # attn weight the dependent pays to its syntactic head
                for l_idx in range(n_layers):
                    for h_idx in range(n_heads):
                        records.append({
                            "layer":      l_idx,
                            "head":       h_idx,
                            "dep_length": float(dlen),
                            "attn_weight": float(attn[l_idx, h_idx, dep_tok, head_tok]),
                        })

        if not records:
            logger.warning("No attention records collected.")
            return pd.DataFrame()

        rec_df = pd.DataFrame(records)

        # Correlate per (layer, head)
        corr_rows = []
        for (layer, head), grp in rec_df.groupby(["layer", "head"]):
            rho, pval = spearmanr(grp["dep_length"], grp["attn_weight"])
            corr_rows.append({
                "layer":  layer,
                "head":   head,
                "rho":    rho,
                "pvalue": pval,
                "n_arcs": len(grp),
            })

        result = (
            pd.DataFrame(corr_rows)
            .sort_values("rho", key=abs, ascending=False)
            .reset_index(drop=True)
        )
        self._results = result
        logger.info(
            "Top head: layer=%d head=%d rho=%.3f",
            int(result.iloc[0]["layer"]),
            int(result.iloc[0]["head"]),
            result.iloc[0]["rho"],
        )
        return result

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_top_heads(
        self,
        k: int = 10,
        save_path: str | Path | None = None,
    ) -> None:
        """Bar-chart of top-K heads by |rho| with dependency length."""
        import matplotlib.pyplot as plt

        if self._results is None:
            raise RuntimeError("Call run() first.")

        top = self._results.head(k).copy()
        top["label"] = top.apply(lambda r: f"L{int(r['layer'])}H{int(r['head'])}", axis=1)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["steelblue" if r > 0 else "tomato" for r in top["rho"]]
        ax.barh(top["label"][::-1], top["rho"][::-1].abs(), color=colors[::-1])
        ax.set_xlabel("|Spearman ρ| with Dependency Length")
        ax.set_title(f"Top-{k} Attention Heads ({self.extractor.model_name})")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Saved head plot to %s", save_path)
        plt.show()

    def plot_head_heatmap(
        self,
        layer: int,
        head: int,
        sentence: str,
        save_path: str | Path | None = None,
    ) -> None:
        """Visualise a single attention head's weight matrix."""
        import matplotlib.pyplot as plt

        data   = self.extractor.get_attention_weights(sentence)
        attn   = data["attentions"]     # (L, H, T, T)
        tokens = data["tokens"]

        weights = attn[layer, head]
        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.4),
                                        max(6, len(tokens) * 0.4)))
        im = ax.imshow(weights, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"Layer {layer}, Head {head}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _word_to_first_token(
    tokens: list[str],
    words: list[str],
) -> dict[int, int]:
    """
    Map word index → index of its first sub-word token.
    Handles GPT-2 (Ġ prefix), BERT (## continuation), T5 (▁ prefix).
    """
    mapping: dict[int, int] = {}
    word_idx = 0
    accumulated = ""
    first_tok_of_word = 0

    for tok_idx, tok in enumerate(tokens):
        clean = tok.lstrip("Ġ▁").replace("##", "")
        if not clean or tok in ("<s>", "</s>", "[CLS]", "[SEP]", "<pad>"):
            continue

        if not accumulated:
            first_tok_of_word = tok_idx
            mapping[word_idx] = first_tok_of_word

        accumulated += clean

        if word_idx < len(words) and accumulated >= words[word_idx].lower():
            accumulated = ""
            word_idx += 1
            if word_idx >= len(words):
                break

    return mapping

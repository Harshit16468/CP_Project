# Mid-Project Presentation
## Disentangling Predictive Surprisal, Entropy, and Syntactic Integration Cost in Human Sentence Processing

**Harshit Gupta (2022101124) · Sanchit Jalan (2022101070)**

---

## Slide 1 — The Core Question

> **Why do some words take longer to read than others?**

Two competing theories:

| Theory | Mechanism | Operationalisation |
|---|---|---|
| Predictive Processing | Unexpected words are harder | Surprisal = −log P(wᵢ \| context) |
| Dependency Locality | Structurally distant words are harder | Integration Cost = \|position(w) − position(head)\| |

**Our contribution:** Test both simultaneously using Transformer LMs + Bayesian modeling, across 3 architectures and 181 readers.

---

## Slide 2 — Dataset & Scale

**Natural Stories Corpus**
- 10,000 words of English narrative text
- 181 participants, self-paced reading times
- **835,700 total observations** after cleaning

**Preprocessing pipeline:**
- RT filter: 100ms – 3000ms
- Per-subject ±3 SD outlier removal  
- Dependent variable: log(RT)

---

## Slide 3 — Full Pipeline (All Steps Complete ✅)

```
Natural Stories Corpus
        │
        ├─ Step 1: Data Prep          ✅  835K observations, 181 subjects
        ├─ Step 2: N-gram Surprisal   ✅  Trigram + Kneser-Ney baseline
        ├─ Step 3: Neural Metrics     ✅  GPT-2, BERT, T5 surprisal + entropy
        ├─ Step 4: Attention Analysis ✅  All heads × all layers × 3 models
        ├─ Step 5: Integration Cost   ✅  Stanza UD parser, dep length per word
        └─ Step 6: Bayesian Modeling  ✅  9 model variants, LOO-CV comparison
```

**All 6 pipeline steps are complete and results are in hand.**

---

## Slide 4 — H1: Deep Neural > Shallow N-gram

**GPT-2 surprisal explains reading times far better than trigram surprisal.**

| Model | ELPD (LOO-CV) | β | Significant? |
|---|---|---|---|
| **GPT-2** | **−1163** | +0.010 | ✅ Yes |
| T5 | −1191 | +0.013 | ✅ Yes |
| BERT | −1210 | +0.005 | ✅ Yes |
| Trigram baseline | −1211 | −0.003 | ✅ Yes |

> **GPT-2 beats the trigram baseline by 48 ELPD units** — human reading relies on long-distance context, not just local n-grams.

*(See: `figures/hyp1_deep_vs_ngram.png`)*

---

## Slide 5 — H4: Architecture Matters

**Autoregressive (left-to-right) processing best mirrors human reading.**

```
GPT-2  (autoregressive)  ELPD = −1163  ← BEST
T5     (encoder-decoder) ELPD = −1191
BERT   (bidirectional)   ELPD = −1210
Trigram (baseline)       ELPD = −1211  ← WORST
```

**Interpretation:** BERT's bidirectional context — despite being a stronger language model — is a *worse* psycholinguistic predictor because humans don't have access to future context during reading.

*(See: `figures/hyp4_architecture_comparison.png`)*

---

## Slide 6 — H2: Does Integration Cost Matter?

**Once deep surprisal is accounted for, dependency length adds nothing.**

| Predictor | β | 95% HDI | Significant? |
|---|---|---|---|
| GPT-2 surprisal | +0.014 | [+0.011, +0.018] | ✅ Yes |
| Integration cost | −0.001 | [−0.003, +0.002] | ❌ No |

> The HDI for β_IC spans zero — integration cost is fully explained away by neural surprisal. Modern LMs implicitly encode structural difficulty.

*(See: `figures/hyp2_ic_variance_explained.png`)*

---

## Slide 7 — H3: Surprisal vs. Entropy

**Surprisal (what word came) and Entropy (how uncertain the model was) are partly independent.**

| Architecture | β_surprisal | β_entropy |
|---|---|---|
| GPT-2 | +0.008 ✅ | +0.005 ✅ |
| T5 | +0.013 ✅ | +0.003 ✅ |
| BERT | +0.005 ✅ | +0.000 ❌ |

> GPT-2 and T5: **both contribute independently** — the model's uncertainty matters beyond just the specific word's probability.  
> BERT entropy fails — bidirectional pseudo-surprisal conflates the two measures.

*(See: `figures/hyp3_surprisal_vs_entropy.png`)*

---

## Slide 8 — H5: Attention Heads Track Syntax

**Specific heads in all 3 models correlate with dependency length.**

| Architecture | Best Head | Spearman ρ | p-value |
|---|---|---|---|
| BERT | L0 H10 | **−0.624** | < 10⁻¹⁰⁰ |
| T5 | L0 H10 | **−0.579** | < 10⁻¹⁰⁰ |
| GPT-2 | L5 H9 | +0.351 | < 10⁻¹⁰⁰ |

> BERT and T5 head L0H10 preferentially attends to *nearby* syntactic heads (negative ρ).  
> The same head (L0H10) dominates in both masked and seq2seq architectures — a convergent structural inductive bias.

*(See: `figures/hyp5_attention_comparison.png`, `figures/04_top_heads_bert.png`)*

---

## Slide 9 — H5 Deep Dive: BERT L0H10

**Head-finding accuracy drops sharply for long-distance dependencies.**

- DL = 1: accuracy ~60%
- DL = 2–3: accuracy ~35%
- DL > 5: accuracy ~15% (near chance)

> The head tracks syntax well locally but struggles at long range — consistent with a *structural working memory constraint* being learned implicitly.

*(See: `figures/bert_L0H10_headfinding_accuracy.png`, `figures/bert_L0H10_layer_curve.png`)*

---

## Slide 10 — H6: Individual Reader Strategies

**Readers differ significantly in how much they rely on predictive processing.**

| Variance component | σ (posterior mean) | 95% HDI | Significant? |
|---|---|---|---|
| Surprisal sensitivity | **0.013** | [0.009, 0.016] | ✅ Yes |
| IC sensitivity | 0.003 | [0.000, 0.007] | ✅ Yes |

> σ > 0 means real between-reader variation — some readers slow down dramatically for surprising words; others barely react.  
> This heterogeneity is masked in standard mixed-effects models (Bayesian approach necessary to recover it).

*(See: `figures/H6_surprisal_slopes_hist.png`, `figures/H6_surprisal_vs_ic_slopes.png`)*

---

## Slide 11 — Summary of All Hypotheses

| Hypothesis | Status | Key Finding |
|---|---|---|
| H1: Deep > Shallow | ✅ Supported | GPT-2 ΔELPD = +48 over trigram |
| H2: Surprisal subsumes IC | ✅ Supported | β_IC HDI spans zero |
| H3: Surprisal ≠ Entropy | ✅ Partial | GPT-2+T5 yes; BERT no |
| H4: Autoregressive best | ✅ Supported | GPT-2 > T5 > BERT ≈ trigram |
| H5: Heads track syntax | ✅ Strongly supported | ρ = −0.624 (BERT L0H10) |
| H6: Individual differences | ✅ Supported | σ_slope = 0.013, sig. |

**All 6 hypotheses have been tested. 5 fully supported, 1 partially.**

---

## Slide 12 — Work Remaining

**Completed:**
- Full pipeline (all 6 steps) ✅
- All 3 models (GPT-2, BERT, T5) ✅
- All 9 Bayesian model variants ✅
- All 6 hypotheses tested ✅

**Remaining for final submission:**
1. Cross-dataset validation on **GECO corpus** (eye-tracking, 54K words)
2. Final written report with full statistical interpretation
3. Robustness checks on MCMC convergence for hierarchical models

---

## Slide 13 — Key Takeaways

1. **Neural LMs capture human reading difficulty better than local n-grams** — context matters.
2. **Architecture modality matters** — sequential (GPT-2) > bidirectional (BERT) for modeling human reading.
3. **Dependency locality is not independently necessary** — surprisal subsumes it.
4. **Transformers implicitly learn syntax** — a specific head (L0H10) in both BERT and T5 tracks dependency structure.
5. **Reading is not one-size-fits-all** — the Bayesian approach reveals meaningful individual differences in predictive processing.

---

*Figures available in `CP_Project/figures/figures/`*  
*Full results in `CP_Project/results/metrics/hypothesis_summary.csv`*

---
title: "Azerbaijani Tokenizer — 64k WordPiece for AzBERT Pretraining"
date: 2025-12-01
description: "Built a production-grade Azerbaijani WordPiece tokenizer on ~100 GB of cleaned text across 10 corpus sources. Achieved fertility 1.67 and 83% whole-word rate — outperforming mBERT (3.70 fertility) and competitive with multilingual baselines on clean Azerbaijani text."
tags:
  - Python
  - HuggingFace Tokenizers
  - SentencePiece
  - MongoDB
  - NLP
  - Azerbaijani
  - Pretraining
showDate: true
showReadingTime: false
showAuthor: false
showPagination: true
showTableOfContents: true
---

*Status: **Complete** · Tokenizer selected for AzBERT pretraining · Component 1 of the AzBERT pipeline*

## TL;DR

I built a 64k WordPiece tokenizer for the Azerbaijani language from scratch — from raw corpus ingestion through multi-step normalization, data-driven sampling, and three-way algorithm comparison — as the first component in the AzBERT pretraining pipeline. The tokenizer achieves fertility 1.67 on clean Azerbaijani text, 83% whole-word token rate, and 5.2 characters per token, outperforming mBERT (fertility 3.70) on every key metric.

The entire pipeline processes ~100 GB of raw text via MongoDB streaming cursors and multiprocessing — no data ever fits in memory.

---

## Problem

Azerbaijani is a morphologically rich agglutinative language with ~10 million speakers and almost no dedicated NLP infrastructure. Every Azerbaijani NLP application in production — including [e-qanun.ai](https://e-qanun.ai), which I also built — relies on multilingual tokenizers like mBERT or XLM-R that were never designed for Azerbaijani.

The cost of this is measurable. mBERT fragments the average Azerbaijani word into 3.7 subword tokens. A word like *müqavilələrindən* (from their contracts) becomes six pieces. This fragmentation wastes token budget, degrades attention locality, and limits downstream task performance on a language that packs dense semantic content per word.

The fix requires training a tokenizer on a large, clean, representative Azerbaijani corpus — which first requires building that corpus.

---

## Corpus

| Collection | Documents | Size | Source Type |
|---|---|---|---|
| hplt3_final | 11,000,000 | 32.3 GB | Filtered web crawl |
| fineweb2_final | 7,200,000 | 21.2 GB | Curated web |
| nllb_final | 96,000,000 | — | Sentence-level parallel |
| culturax_final | 5,000,000 | 13.0 GB | mC4 + OSCAR |
| cc100_final | 3,900,000 | 5.5 GB | Web crawl |
| news_v1 | 3,200,000 | 7.9 GB | News articles |
| court_cases | 1,900,000 | 1.3 GB | Legal text |
| pdfs_part1 | 1,600,000 | 1.2 GB | PDF extractions |
| e-qanun | 531,000 | 466 MB | Legislative text (100% included) |
| folklor | 338,000 | — | Folkloric / dialectal (100% included) |

**Total raw corpus: ~100–150 GB, ~95.7 billion characters** (measured post-normalization steps 1–3).

The `folklor` and `e-qanun` collections are included in full — folkloric text provides irreplaceable dialectal coverage, and legislative text is a high-signal, near-zero-noise domain. For the larger web crawl sources, sampling rates are set by per-collection character analysis: contamination rates, Cyrillic ratios, UTF-8 mojibake presence, and replacement character frequency.

**Tokenizer training sample: 1,550,000 documents** drawn across all 10 collections after evaluation holdout separation.

---

## Preprocessing Pipeline

<img src="tokenizer-pipeline.svg" alt="Tokenizer preprocessing pipeline: corpus ingestion → entity sanitization → emoji normalization → punctuation normalization → character statistics → WordPiece training" style="max-width:100%;height:auto;" />

All scripts follow the same architecture: MongoDB streaming cursor → multiprocessing pool → batch `bulk_write`. No collection is ever loaded into memory.

### Step 1 — Entity Sanitization

Regex-based detection replaces structured entities with placeholder tokens that the tokenizer will learn as atomic units:

| Entity | Placeholder |
|---|---|
| URLs | `[URL]` |
| Email addresses | `[EMAIL]` |
| Phone numbers | `[PHONE]` |

```
Input:  "Visit https://example.com or call +994501234567"
Output: "Visit [URL] or call [PHONE]"
```

### Step 2 — Emoji Normalization

All emoji replaced with `[EMOJI]`. Precompiled regex covering all major Unicode emoji ranges. This step's effect on tokenizer quality is isolated in Experiment 2: emoji removal alone reduces mean fertility by 0.215 points.

### Step 3 — Punctuation Normalization

Visually similar punctuation canonicalized to ASCII equivalents using a translation table:

| Source | Canonical |
|---|---|
| `"` `"` `„` | `"` (U+0022) |
| `'` `'` | `'` (U+0027) |
| `—` `–` `−` | `-` (U+002D) |
| `…` | `...` |
| NBSP / thin space / zero-width space | regular space |

### Step 4 — Character Statistics

Corpus-wide character frequency analysis across all collections, streaming per-document into a shared `Counter`. Output (`char_stats.json`) drives all subsequent alphabet decisions — rare character thresholds, Cyrillic contamination rates, and vocabulary boundary choices.

---

## Tokenizer Configuration

| Parameter | Value |
|---|---|
| Algorithm | WordPiece |
| Vocabulary size | 64,000 |
| Normalizer | NFKC |
| Pre-tokenizer | Whitespace |
| Continuation prefix | `##` |
| Max chars per word | 100 |

**Special tokens (9):** `[PAD]` `[UNK]` `[CLS]` `[SEP]` `[MASK]` `[EMOJI]` `[URL]` `[PHONE]` `[EMAIL]`

**Vocabulary composition:**

| Category | Count | % of Vocab |
|---|---|---|
| Whole-word tokens | 50,965 | 79.6% |
| Continuation tokens (`##`) | 11,326 | 17.7% |
| Single-character tokens | 1,700 | 2.7% |

Of the 50,965 whole-word tokens, 39,957 are pure Azerbaijani-alphabet words. Continuation token length peaks at 3–4 characters — precisely the common Azerbaijani suffixes: `##lar`, `##lər`, `##dan`, `##dən`, `##nın`.

---

## Experiments

### Experiment 1 — Baseline: Our WordPiece vs Multilingual Tokenizers

Initial WordPiece (trained on unfiltered corpus) evaluated against HPLT az-BERT and mBERT on raw test data.

| Metric | WordPiece (ours) | HPLT az-BERT | mBERT |
|---|---|---|---|
| Fertility mean | 2.747 | **2.004** | 3.697 |
| Fertility p95 | 5.308 | **3.000** | 6.375 |
| UNK rate | 0.67% | **0.002%** | 0.50% |
| UNK sentence rate | 23.12% | **0.23%** | 25.39% |
| Chars / token | 3.708 | **4.467** | 2.618 |
| Continuation rate | 40.97% | **0.00%** | 45.87% |

The gap against HPLT on unfiltered data is explained by alphabet restriction: our monolingual tokenizer correctly assigns `[UNK]` to the CJK, Arabic, and Cyrillic characters that contaminate the test corpus, while HPLT's broader vocabulary absorbs them. mBERT confirms the baseline cost of a shared multilingual vocabulary — fertility 3.70 on Azerbaijani text fails every practical threshold.

### Experiment 2 — Corpus Filtering Effect

Same WordPiece architecture trained on four corpus versions, holding algorithm constant.

| Version | Fertility mean | Fertility p95 | Chars/token | Whole-word rate |
|---|---|---|---|---|
| No filter (raw) | 2.747 | 5.308 | 3.708 | 59.0% |
| Strict Latin filter | 2.225 | 3.933 | 4.243 | 67.8% |
| Cleaned (with emoji) | 2.142 | 3.723 | 4.444 | 71.0% |
| **Cleaned (no emoji)** | **1.927** | **3.070** | **4.686** | **74.9%** |

Corpus filtering is the single highest-impact intervention. Raw → cleaned (no emoji) delivers:
- **−30% fertility** (2.747 → 1.927)
- **−42% p95 fertility** (5.308 → 3.070)
- **+26% chars/token** (3.708 → 4.686)
- **+15.8 pp whole-word rate** (59% → 75%)

### Experiment 3 — Algorithm Comparison

Three algorithms trained on identical cleaned corpus (emoji removed), isolating algorithmic effect.

| Metric | WordPiece | SPM Unigram | SPM BPE |
|---|---|---|---|
| Fertility mean | 1.927 | **1.858** | **1.852** |
| UNK rate | 0.234% | **0.000%** | **0.000%** |
| Chars / token | 4.686 | **4.967** | 4.867 |
| Whole-word rate | **74.9%** | 58.8% | 57.6% |
| Vocab coverage | 54.3% | 62.1% | **63.8%** |

SPM algorithms produce zero UNK (byte-fallback covers all characters) and marginally lower fertility. WordPiece leads on whole-word rate by 16+ percentage points — a meaningful difference for downstream attention efficiency. All three algorithms perform comparably once corpus quality is controlled; the algorithm choice is secondary to cleaning.

**Selected tokenizer: WordPiece on cleaned corpus (emoji removed).**

---

## Final Results

<img src="fertility-comparison.svg" alt="Fertility comparison: ours 1.67, HPLT az-BERT 2.00, mBERT 3.70, plus competitor tokenizers" style="max-width:100%;height:auto;" />

Evaluated against three competing tokenizers on a held-out clean Azerbaijani test set:

| Metric | **Ours** | Competitor B | Competitor C | Threshold |
|---|---|---|---|---|
| Fertility mean | **1.673 ✓** | 2.506 ✗ | 2.140 ✓ | ≤ 2.5 |
| Fertility p95 | **2.605** | 5.127 | 3.720 | — |
| UNK rate | **0.93% ✓** | 0.005% ✓ | 0.015% ✓ | ≤ 1% |
| Chars / token | **5.201 ✓** | 4.196 ✓ | 4.460 ✓ | ≥ 3.5 |
| Continuation rate | **16.98% ✓** | 50.33% ✗ | 28.67% ✓ | ≤ 50% |
| Whole-word rate | **83.0%** | 49.7% | 71.3% | — |
| Stress fertility | **1.333** | 1.463 | 1.488 | — |

The tokenizer passes all four primary thresholds and leads on every metric that directly affects pretraining efficiency. Competitor B fails on both fertility and continuation rate — it splits Azerbaijani words more aggressively than mBERT.

---

## Key Findings

| Finding | Evidence |
|---|---|
| Corpus cleaning outweighs algorithm choice | −30% fertility from raw → cleaned; algorithm spread is <0.08 |
| Emoji removal is a meaningful isolated gain | −0.215 fertility points as a standalone step |
| mBERT is unsuitable for Azerbaijani | Fertility 3.70, 2.6 chars/token — fragmentation doubles token budget |
| WordPiece leads on sequence efficiency | 83% whole-word rate vs. 57–59% for SPM algorithms |
| SPM eliminates UNK; WordPiece does not | 0.00% vs. 0.93% on noisy eval; byte-fallback is the reason |
| Dead token rate reflects corpus/vocab mismatch | ~1,570 identified noise tokens (2.5%); remaining 35k are low-frequency but valid Azerbaijani subwords |

---

## Stack

`Python` · `HuggingFace Tokenizers` · `SentencePiece` · `MongoDB` · `multiprocessing` · `NFKC normalization` · `regex`

---

*Component 1 of the AzBERT pretraining pipeline. Tokenizer selected for model pretraining.*

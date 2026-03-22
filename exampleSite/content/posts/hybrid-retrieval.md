---
title: "Hybrid Retrieval: Combining BM25 and Dense Vectors for Production Search"
date: 2025-01-15T10:00:00+00:00
draft: false
tags:
  - NLP
  - Information Retrieval
  - SBERT
  - BM25
  - RAG
description: "How I built a hybrid retrieval pipeline (BM25 + SBERT/DPR) over 500k+ documents, the trade-offs involved, and what the benchmarks actually showed."
showTableOfContents: true
---

## Why Pure BM25 (or Pure Dense) Isn't Enough

BM25 is fast, interpretable, and great at exact keyword matching. Dense retrieval (SBERT, DPR) captures
semantic similarity but misses precise term overlap. In practice, neither alone covers the full
distribution of user queries — especially in a domain-specific corpus.

This post walks through the hybrid system I built at NAIC (National AI Center), fine-tuned on **500k+
domain-specific documents**, and how we evaluated it.

---

## Architecture Overview

```
Query
  │
  ├─► BM25 (Elasticsearch)  ──► top-k lexical candidates
  │
  └─► Dense Encoder (SBERT) ──► top-k semantic candidates
              │
              └─► Reciprocal Rank Fusion (RRF) / score fusion
                          │
                          └─► Re-ranker (Cross-Encoder)
                                      │
                                      └─► Final ranked list
```

### Stage 1 — Lexical retrieval with BM25

BM25 remains the default first stage for most production systems because it:
- Scales to hundreds of millions of documents without GPU
- Handles rare tokens and out-of-vocabulary terms well
- Is easy to explain to stakeholders

Key tuning knobs: `k1` and `b` parameters, custom analyzers, field-level boosting.

### Stage 2 — Dense retrieval with SBERT / DPR

We fine-tuned `sentence-transformers/all-mpnet-base-v2` using contrastive loss on
domain-specific query-passage pairs. Key decisions:

- **Pooling**: mean-pooling over token embeddings
- **Index**: FAISS `IVF1024,PQ64` for ~500k vectors (~2 GB RAM)
- **Batch inference**: ONNX export + dynamic quantization for 2× speedup at index time

### Stage 3 — Score fusion

We compared two fusion strategies:

| Strategy | MRR@10 | Notes |
|---|---|---|
| Linear interpolation (α·BM25 + (1-α)·Dense) | 0.74 | Sensitive to α tuning |
| Reciprocal Rank Fusion (RRF) | **0.81** | Parameter-free, robust |

RRF won. Formula: `score(d) = Σ 1/(k + rank_i(d))` where `k=60`.

---

## Evaluation Metrics

We used three metrics across a held-out test set of ~2k annotated queries:

- **MRR@10** — mean reciprocal rank of the first relevant result in top 10
- **NDCG@5** — normalized discounted cumulative gain at cutoff 5
- **Precision-Recall** — full curve to understand coverage vs. noise trade-off

Final hybrid system reached **85% accuracy** in user acceptance testing vs. ~62% for BM25-only.

---

## Lessons Learned

1. **Domain fine-tuning matters more than model size.** A fine-tuned `mpnet-base` beat a
   vanilla `e5-large` on our domain.
2. **RRF is underrated.** Zero hyperparameters, surprisingly good fusion.
3. **Build your eval harness first.** We spent two weeks building the annotation pipeline —
   worth every hour.
4. **Quantized ONNX for the encoder saves ~40% latency** at inference time with <1% MRR drop.

---

## Next Steps

- Explore late interaction models (ColBERT) for better semantic granularity
- Online learning: re-rank using implicit click feedback
- Distill the cross-encoder into a smaller bi-encoder for cheaper re-ranking

---

*Code for this pipeline is on [GitHub](https://github.com/ibishovrauf). Questions? [Email me](mailto:info@raufibishov.com).*

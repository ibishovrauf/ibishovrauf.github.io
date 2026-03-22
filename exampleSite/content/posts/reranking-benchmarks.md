---
title: "Re-ranking LLMs in Production: Benchmarking Latency vs. Precision"
date: 2025-03-05T10:00:00+00:00
draft: false
tags:
  - Re-ranking
  - Cross-Encoder
  - LLM
  - Benchmarking
  - RAG
description: "When does adding a re-ranker actually help? A practical benchmark of cross-encoder LLMs and LRT models against a BM25+dense first stage."
showTableOfContents: true
---

## Context: Why Re-rank at All?

First-stage retrieval (BM25 or dense vectors) is built for speed: you need top-1000 candidates
fast. Precision at the very top (position 1–5) is a secondary concern.

Re-ranking flips the priority: given a small candidate set (~50–200 docs), score them accurately
using a more expensive model. The question is: **how much precision do you gain, and at what
latency cost?**

This post documents what we measured at NAIC (National AI Center).

---

## Models Under Test

We evaluated three re-ranking approaches on top of our hybrid BM25+dense first stage:

| Model | Type | Size | Notes |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder | ~22M params | Fast, widely used |
| `cross-encoder/ms-marco-electra-base` | Cross-Encoder | ~110M params | Higher quality |
| LRT (Lightweight Re-ranking Transformer) | Custom | ~14M params | Fine-tuned in-house |

The LRT is a distilled cross-encoder trained on our domain data using knowledge distillation
from the `electra-base` teacher.

---

## Benchmark Setup

- **Corpus**: 500k+ domain-specific documents
- **Test queries**: 2,000 annotated queries (held-out)
- **First stage**: BM25+dense hybrid, top-100 candidates per query
- **Re-ranking input**: top-50 from first stage
- **Hardware**: 4-core CPU (no GPU in production)
- **Metric**: MRR@10, NDCG@5, P95 latency per query

---

## Results

### Quality (MRR@10)

| System | MRR@10 | NDCG@5 | Δ vs. First Stage |
|---|---|---|---|
| First stage only (BM25+dense) | 0.81 | 0.77 | baseline |
| + MiniLM-L6 cross-encoder | 0.86 | 0.83 | +6.2% |
| + Electra-base cross-encoder | **0.89** | **0.86** | **+9.9%** |
| + LRT (in-house distilled) | 0.87 | 0.84 | +7.4% |

### Latency (P95, ms per query, CPU, re-rank top-50)

| System | P95 Latency | Notes |
|---|---|---|
| First stage only | 12 ms | ONNX quantized encoders |
| + MiniLM-L6 | 31 ms | Acceptable for async use cases |
| + Electra-base | 148 ms | Too slow for <100ms SLA |
| + LRT (ONNX INT8) | **44 ms** | Best quality/latency trade-off |

---

## The Trade-off Curve

```
Quality (MRR@10)
  0.89 │              ● Electra-base
  0.87 │         ● LRT
  0.86 │    ● MiniLM
  0.81 │ ● First stage
       └────────────────────────────── Latency (ms)
           12   31   44              148
```

For our **100ms end-to-end SLA**, the LRT model was the only option that delivered meaningful
quality improvement while staying within budget.

---

## Key Findings

1. **The first stage is more important than the re-ranker.** Improving hybrid retrieval MRR
   from 0.74 → 0.81 (by switching to RRF) gave more gains than adding any re-ranker.
2. **LRT > MiniLM at similar latency** when fine-tuned on in-domain data. Domain adaptation
   matters more than parameter count.
3. **Electra-base is a research tool**, not a production re-ranker on CPU. It belongs behind
   a GPU, or as a teacher for distillation.
4. **Quantize your re-ranker too.** LRT INT8 (via ONNX) cut latency from ~70ms to 44ms
   with <0.5% MRR drop.

---

## Deployment Decision

We deployed the LRT re-ranker as a FastAPI microservice:
- Input: `(query, [candidate_passages])`
- Output: re-scored list, sorted descending
- Batch size: 50 passages / request
- P95 latency: 44ms; P99: 61ms
- Horizontal scaling: stateless, behind a load balancer

---

## What I'd Do Differently

- **Collect click logs earlier.** Implicit feedback is a goldmine for re-ranker fine-tuning.
- **Test ColBERT-style late interaction** as an alternative to full cross-attention — better
  precision/latency balance than bi-encoders at the cost of a larger index.
- **Cache re-ranker scores** for repeated queries. A simple Redis cache cut our average
  latency by 20% in production.

---

*Repo on [GitHub](https://github.com/ibishovrauf). Questions? [Email me](mailto:info@raufibishov.com).*

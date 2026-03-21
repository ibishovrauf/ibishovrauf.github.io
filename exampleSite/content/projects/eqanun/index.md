---
title: "e-qanun.ai — AI Legal Search Platform"
date: 2025-09-01
description: "State-commissioned AI legal search platform built at NAIC under Azerbaijan's AI Strategy 2025–2028. Multi-stage retrieval with BM25, SBERT/DPR, and structural ranking over 500k+ legal documents."
tags:
  - Python
  - SBERT
  - DPR
  - DistilBERT
  - Elasticsearch
  - ONNX
showDate: false
showReadingTime: false
showAuthor: false
showPagination: true
showTableOfContents: true
---

State-commissioned AI legal search platform built at NAIC under Azerbaijan's AI Strategy 2025–2028, officially launched September 2025.

## What It Does

Operates across three simultaneous search spaces:
- **Lexical** — BM25 for exact-match and rare-term recall
- **Semantic** — SBERT/DPR fine-tuned on 500k+ legal documents
- **Structural** — legal-hierarchy-aware ranking (chapter, article, clause)

## Pipeline

1. Query normalization
2. Azerbaijani spell correction
3. NER (quantized DistilBERT)
4. Intent classification
5. Hybrid retrieval (BM25 + dense)
6. Cross-encoder + LRT re-ranking
7. Result fusion and ranking

All stages run under a **200ms end-to-end latency budget**.

## Key Numbers

| Metric | Value |
|---|---|
| Model size reduction | 39% (ONNX INT8) |
| End-to-end latency | < 200ms |
| Accuracy (MRR@10 / NDCG@5) | 85% |
| Corpus size | 500k+ documents |

## Links

- [Live Platform](https://e-qanun.ai)
- [Research](https://research.e-qanun.ai)

---
title: "About Me"
date: 2025-01-01
showDate: false
showReadingTime: false
showAuthor: false
showPagination: false
showTableOfContents: true
showBreadcrumbs: false
---

I'm an ML Engineer with three years of production NLP experience, joining TUM's MSc in Information Engineering in 2025.

My work centers on two problems: making search relevant and making models fast. At NAIC (National AI Center) I built a hybrid retrieval pipeline — BM25 for exact-match recall and SBERT/DPR for semantic coverage — over a 500k-document domain corpus. Evaluated against MRR@10, NDCG@5, and precision-recall curves; user testing put accuracy at 85%.

On the efficiency side: I exported DistilBERT and SBERT to ONNX, combined dynamic quantization with mixed-precision training, and reduced model size by **39%** with under 1% accuracy loss. The same pipeline handles query-time NER and query classification.

I track numbers. If I can't measure it, I don't claim it improved.

---

## What I Build

- Hybrid retrieval pipelines combining BM25 with SBERT/DPR dense vectors — evaluated against MRR@10 and NDCG@5 across 500k+ document corpora.
- Re-ranking stages using cross-encoder LLMs and lightweight re-ranking models, trading precision against P95 latency at each stage.
- Transformer models (DistilBERT, SBERT) exported to ONNX with dynamic INT8 quantization — 39% size reduction, under 1% accuracy loss.
- Mixed-precision training (torch.cuda.amp) to cut fine-tuning wall-clock time ~1.8× without benchmark regression.
- Query understanding with quantized DistilBERT — NER and classification in a single inference pass — deployed on AWS.
- Evaluation frameworks (MRR@10, NDCG@5, precision-recall) defined before building models, so every optimization is grounded in measurable outcomes.

---

## Core Stack

| Area | Tools |
|---|---|
| ML / DL | Python · PyTorch · Hugging Face Transformers |
| Optimization | ONNX · Dynamic Quantization · Mixed Precision |
| Retrieval | BM25 · SBERT · DPR · Cross-Encoders |
| Vector DBs | Elasticsearch · FAISS · Qdrant |
| Infrastructure | FastAPI · Docker · AWS |
| Evaluation | MRR · NDCG · Precision-Recall |

---

## Education

**M.Sc. in Information Engineering** — Technical University of Munich (TUM)
*2026 – Expected 2028*
Incoming graduate student. Focus: machine learning systems, information retrieval, efficient deep learning.

**B.Sc. in Computer Science** — UFAZ – French-Azerbaijani University (University of Strasbourg)
*2021 – 2025*
Core coursework: Linear Algebra, Statistics, Calculus, Algorithms & Data Structures, Artificial Intelligence, Software Engineering.

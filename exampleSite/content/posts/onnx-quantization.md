---
title: "Shrinking Transformers for Production: ONNX Export + Dynamic Quantization"
date: 2025-02-10T10:00:00+00:00
draft: false
tags:
  - ONNX
  - Quantization
  - Model Optimization
  - DistilBERT
  - Inference
description: "A practical guide to exporting DistilBERT and SBERT to ONNX, applying dynamic quantization, and measuring the real-world latency vs. accuracy trade-off."
showTableOfContents: true
---

## The Problem: Transformers Are Slow at Inference

A full BERT-base model costs ~110M parameters and ~440 MB on disk. In a retrieval pipeline
where the encoder runs on every query (and every document at index time), that adds up fast.

At NAIC (National AI Center) we needed to cut inference latency while keeping NER and retrieval
quality above a business-defined threshold. Here's what we did.

---

## Export to ONNX

ONNX (Open Neural Network Exchange) lets you decouple the trained model from PyTorch's
runtime overhead. The ONNX Runtime (ORT) executor applies graph-level optimizations automatically.

```python
from transformers import AutoTokenizer, AutoModel
from torch.onnx import export
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).eval()

dummy = tokenizer("example input", return_tensors="pt")

export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    "distilbert.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    },
    opset_version=14,
)
```

---

## Dynamic Quantization

Dynamic quantization converts FP32 weights to INT8 at load time. Unlike static quantization,
no calibration dataset is needed — weights are quantized once, activations are quantized on-the-fly.

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="distilbert.onnx",
    model_output="distilbert_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

That's it. The resulting model is typically **~3–4× smaller** with the `INT8` weight representation.

---

## Results

Tested on a batch of 512 queries (seq len 64) on a single CPU core:

| Model | Size (MB) | Latency (ms/query) | F1 (NER) |
|---|---|---|---|
| DistilBERT FP32 (PyTorch) | 268 | 38 | 91.2 |
| DistilBERT FP32 (ONNX) | 268 | 22 | 91.2 |
| DistilBERT INT8 (ONNX) | **163** | **14** | 90.7 |

Key takeaways:
- **39% size reduction** (268 → 163 MB)
- **~63% latency reduction** vs. native PyTorch
- **<1% F1 drop** — within acceptable range for our use case

---

## Mixed Precision Training (Bonus)

For fine-tuning we applied `torch.cuda.amp` (automatic mixed precision) to speed up
training without quality loss:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Training wall-clock time dropped ~1.8× on a single A100 with no measurable benchmark regression.

---

## Pitfalls to Watch

1. **Operator coverage**: Not all PyTorch ops have ONNX equivalents. Use `opset_version ≥ 14`
   and check the export warnings carefully.
2. **Dynamic shapes**: Always export with `dynamic_axes` if batch size or sequence length varies.
3. **Accuracy drop on edge cases**: INT8 quantization can hurt models fine-tuned on very short
   sequences. Always benchmark on your actual data distribution.
4. **Thread contention**: ONNX Runtime uses its own thread pool. Set `intra_op_num_threads`
   to match your CPU topology.

---

## Should You Use Static Quantization Instead?

Static quantization (quantizing activations too, not just weights) can give another 10–15%
speedup but requires a calibration dataset and more careful validation. For our NER + retrieval
pipeline the dynamic approach hit the latency target, so we stopped there.

---

*Code on [GitHub](https://github.com/ibishovrauf). Questions? [Email me](mailto:info@raufibishov.com).*

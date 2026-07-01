# Benchmark Results

Full run log for all compression techniques. Regenerated automatically each time a script in scripts/ is run.

| Model | Compression | Accuracy | F1 | Size (MB) | Latency (ms/img) | Throughput (img/s) |
|---|---|---:|---:|---:|---:|---:|
| PyTorch CNN (FP32 baseline) | Baseline | 99.92% | 0.9991 | 74.72 | 4.074 | 245.5 |
| PyTorch CNN-ViT (FP32 baseline) | Baseline | 99.83% | 0.9983 | 150.95 | 8.814 | 113.5 |
| StudentCNN (distilled from CNN-ViT) | Distillation | 99.92% | 0.9991 | 1.00 | 0.591 | 1691.6 |
| StudentCNN (hard labels only) | Distillation | 100.00% | 1.0000 | 1.00 | 0.612 | 1633.9 |
| PyTorch CNN — pruned 20% (structured) | Pruning | 77.67% | 0.6968 | 49.16 | 3.146 | 317.9 |
| PyTorch CNN — pruned 20% (unstructured) | Pruning | 99.92% | 0.9991 | 74.72 | 4.143 | 241.3 |
| PyTorch CNN — pruned 40% (structured) | Pruning | 52.00% | 0.0000 | 28.86 | 2.126 | 470.3 |
| PyTorch CNN — pruned 40% (unstructured) | Pruning | 99.83% | 0.9983 | 74.72 | 4.226 | 236.6 |
| PyTorch CNN — pruned 60% (structured) | Pruning | 52.00% | 0.0000 | 13.96 | 1.039 | 962.3 |
| PyTorch CNN — pruned 60% (unstructured) | Pruning | 99.08% | 0.9904 | 74.72 | 4.116 | 242.9 |
| PyTorch CNN — pruned 80% (structured) | Pruning | 52.00% | 0.0000 | 4.34 | 0.506 | 1974.9 |
| PyTorch CNN — pruned 80% (unstructured) | Pruning | 55.25% | 0.1268 | 74.72 | 4.121 | 242.7 |
| PyTorch CNN — INT8 static PTQ | Quantization | 100.00% | 1.0000 | 18.76 | 2.234 | 447.6 |
| PyTorch CNN-ViT — INT8 dynamic PTQ | Quantization | 99.83% | 0.9983 | 90.21 | 5.656 | 176.8 |

## Notes

- Latency: mean of 500 single-image CPU forward passes after 50 warmup steps (`time.perf_counter`).
- Accuracy/F1: fixed SEED=42 held-out validation split (1,200 images), identical for every row.

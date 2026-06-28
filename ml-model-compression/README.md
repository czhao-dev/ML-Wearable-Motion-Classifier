# Model Compression: Pruning, Quantization, and Knowledge Distillation

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)

A focused study in production-oriented model compression applied to the four trained models from [`ml-satellite-image-classifier`](../ml-satellite-image-classifier). The project benchmarks three orthogonal compression families — **magnitude and structured pruning**, **post-training quantization (PTQ)**, and **knowledge distillation** — measuring the tradeoff between model size, inference latency, and classification accuracy on a held-out satellite image test set.

The goal is to answer a concrete engineering question: _for a fixed accuracy budget, what is the smallest, fastest model we can ship?_

## Table of Contents

- [Highlights](#highlights)
- [Background](#background)
- [Techniques](#techniques)
  - [Pruning](#pruning)
  - [Post-Training Quantization](#post-training-quantization)
  - [Knowledge Distillation](#knowledge-distillation)
- [Benchmark Results](#benchmark-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Design Notes](#design-notes)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Highlights

- Three compression techniques applied to the same base architecture and dataset, enabling a controlled, apples-to-apples benchmark
- Unstructured magnitude pruning and structured L1 channel pruning sweeping sparsity from 20% to 80%; structured pruning produces real latency gains without sparse BLAS support
- Static INT8 quantization via `torch.quantization` (observer calibration on 200 training images) and dynamic INT8 for the ViT's linear layers
- Knowledge distillation: PyTorch CNN-ViT (99.67% accuracy, teacher) → lightweight 3-block student CNN, trained with a temperature-scaled soft-label loss; student is ~12× smaller than the teacher
- Unified benchmark table: every model variant is scored on accuracy, F1, model size (MB), CPU inference latency (ms/image), and throughput (images/s)
- Compressed models are drop-in replacements for the inference server in `ml-satellite-image-classifier/serve/` — `ModelRegistry` requires no changes

## Background

`ml-satellite-image-classifier` trains and evaluates four models for binary land-use classification (agricultural vs. non-agricultural, 64×64 satellite tiles):

| Model | Accuracy | F1 | Size (FP32) |
|---|---:|---:|---:|
| PyTorch CNN | 99.83% | 0.9983 | ~34 MB |
| PyTorch CNN-ViT | 99.67% | 0.9967 | ~90 MB |
| Keras CNN | 99.33% | 0.9933 | ~35 MB |
| Keras CNN-ViT | 99.42% | 0.9942 | ~91 MB |

The PyTorch models are used as compression targets throughout this project. Keras models are included in the benchmark table for reference but are not the focus of the compression scripts.

The dataset, data pipeline, and evaluation split are identical to the parent project: 6,000 JPG satellite tiles (3,000 per class), evaluated on a fixed 1,200-image held-out validation split that no model saw during training. The trained FP32 weights (`models/trained/*.pth`) are loaded from `ml-satellite-image-classifier/models/trained/` via a relative path; see [Getting Started](#getting-started).

## Techniques

### Pruning

**Script:** [`scripts/01_pruning.py`](scripts/01_pruning.py)

Pruning removes weights from a trained network, reducing parameter count and (for structured pruning) the number of active channels.

**Unstructured magnitude pruning** (`torch.nn.utils.prune.l1_unstructured`) zeroes out the lowest-magnitude weights globally across all convolutional and linear layers. Because the surviving weights are distributed sparsely across the weight tensors, this does not reduce wall-clock latency on standard hardware without sparse BLAS support — but it reduces storage size after compression and is the canonical first step before fine-tuning recovery.

**Structured L1 channel pruning** (`torch.nn.utils.prune.ln_structured`, `n=1`, `dim=0`) removes entire output channels (filters) from convolutional layers ranked by L1 norm, then rebuilds the network with the surviving channels hardcoded. This produces a physically smaller model with real latency improvements on any CPU or GPU.

Both variants are swept across sparsity levels {20%, 40%, 60%, 80%}. For each level:
1. Prune the FP32 PyTorch CNN
2. Evaluate on the 1,200-image held-out split (no fine-tuning) to measure zero-shot accuracy drop
3. Apply `prune.remove()` to make masks permanent, then save the pruned checkpoint
4. Record model size and CPU latency

The structured variant additionally rebuilds the model with pruned dimensions so the saved checkpoint requires no sparsity mask and can be loaded with a standard `nn.Sequential`.

### Post-Training Quantization

**Script:** [`scripts/02_quantization.py`](scripts/02_quantization.py)

Quantization maps 32-bit floating-point weights and activations to lower-precision integers, reducing memory footprint and enabling integer arithmetic on CPUs that support it.

**Static INT8 PTQ** uses `torch.quantization.prepare` + `torch.quantization.convert` with `torch.quantization.get_default_qconfig('x86')`. A calibration pass of 200 training images (never seen at test time) is run through the prepared model to collect activation statistics for scale and zero-point computation. The quantized model uses `torch.float32` inputs but internally operates in INT8.

**Dynamic INT8 PTQ** (`torch.quantization.quantize_dynamic`) quantizes only the linear layers at runtime, without a calibration pass. It is applied to the ViT's `nn.Linear` modules and is appropriate when activation ranges are input-dependent.

For each quantized model:
1. Load the FP32 PyTorch checkpoint
2. Apply the quantization scheme
3. Evaluate accuracy on the held-out split
4. Measure model size (serialized) and CPU inference latency

No fine-tuning or quantization-aware training (QAT) is applied. PTQ-only results reflect what can be achieved at deployment time with a trained checkpoint and no access to GPU training infrastructure.

### Knowledge Distillation

**Script:** [`scripts/03_distillation.py`](scripts/03_distillation.py)

Knowledge distillation trains a smaller **student** network to match the output distribution of a larger, pre-trained **teacher**, rather than training from hard (one-hot) labels alone. The teacher's soft probability vectors carry more information than a binary ground-truth label — they encode the model's uncertainty across classes and serve as a richer training signal for the student.

**Teacher:** PyTorch CNN-ViT (99.67% accuracy, ~90 MB). The teacher is frozen throughout distillation; only the student's weights are updated.

**Student:** A lightweight 3-block CNN (`StudentCNN`), roughly following the first three blocks of the satellite CNN backbone (32 → 64 → 128 channels), followed by global average pooling and a two-class head. This is approximately 12× fewer parameters than the teacher.

**Loss function:**

```
L = α · CE(student_logits, hard_labels) + (1 - α) · KL(softmax(student_logits / T), softmax(teacher_logits / T))
```

where `T` is the temperature (default 4.0) and `α` balances the hard-label cross-entropy against the soft-label KL divergence (default 0.3). Higher temperature softens the teacher's distribution, exposing more inter-class similarity signal.

Training runs on the 4,800-image training split (the same split used in `ml-satellite-image-classifier`) for 30 epochs with Adam and cosine learning-rate decay. The best checkpoint by validation accuracy is saved.

A **baseline student** (same `StudentCNN` architecture, trained from scratch on hard labels only) is included as a control to isolate the distillation benefit.

## Benchmark Results

> Results are populated after running the compression scripts. See [`reports/results_summary.md`](reports/results_summary.md) for the full run log.

| Model | Compression | Accuracy | F1 | Size (MB) | Latency (ms/img) | Throughput (img/s) |
|---|---|---:|---:|---:|---:|---:|
| PyTorch CNN (FP32 baseline) | — | 99.83% | 0.9983 | — | — | — |
| PyTorch CNN-ViT (FP32 baseline) | — | 99.67% | 0.9967 | — | — | — |
| PyTorch CNN — pruned 20% (structured) | Pruning | | | | | |
| PyTorch CNN — pruned 40% (structured) | Pruning | | | | | |
| PyTorch CNN — pruned 60% (structured) | Pruning | | | | | |
| PyTorch CNN — pruned 80% (structured) | Pruning | | | | | |
| PyTorch CNN — INT8 static PTQ | Quantization | | | | | |
| PyTorch CNN-ViT — INT8 dynamic PTQ | Quantization | | | | | |
| StudentCNN (hard labels only) | Distillation | | | | | |
| StudentCNN (distilled from CNN-ViT) | Distillation | | | | | |

Latency is measured as the mean of 500 single-image CPU inference calls (batch size 1, no GPU) after 50 warmup steps, using `time.perf_counter`. All measurements on the same machine; see `reports/results_summary.md` for hardware spec.

## Repository Structure

```text
ml-model-compression/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_pruning.py              # Magnitude and structured pruning sweep
│   ├── 02_quantization.py         # Static and dynamic INT8 PTQ
│   └── 03_distillation.py         # Teacher–student knowledge distillation
├── src/
│   ├── __init__.py
│   ├── student_model.py           # StudentCNN architecture definition
│   ├── benchmark.py               # Latency/throughput measurement utilities
│   ├── eval_utils.py              # Accuracy/F1 evaluation on the held-out split
│   └── paths.py                   # Shared path constants (points into satellite classifier)
└── reports/
    ├── results_summary.md         # Full benchmark table and commentary
    └── figures/
        ├── accuracy_vs_sparsity.png    # Pruning accuracy curve
        ├── size_vs_latency.png         # Pareto plot: size vs. latency across all variants
        └── distillation_curves.png     # Train/val loss for student and baseline
```

## Getting Started

### Prerequisites

- The trained FP32 PyTorch checkpoints from `ml-satellite-image-classifier` must exist at:
  ```
  ../ml-satellite-image-classifier/models/trained/ai_capstone_pytorch_state_dict.pth
  ../ml-satellite-image-classifier/models/trained/pytorch_cnn_vit_ai_capstone_model_state_dict.pth
  ```
  See [`ml-satellite-image-classifier/models/models.md`](../ml-satellite-image-classifier/models/models.md) for how to produce them.

- The satellite image dataset must be downloaded. Running any script in `ml-satellite-image-classifier/scripts/` on first run will download it automatically.

### Install

```bash
cd ml-model-compression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Compression Scripts

Each script is self-contained and writes its outputs to `reports/`.

```bash
# Pruning sweep (unstructured + structured, 4 sparsity levels each)
python scripts/01_pruning.py

# Post-training quantization (static CNN + dynamic ViT)
python scripts/02_quantization.py

# Knowledge distillation (trains student for ~30 epochs on CPU)
python scripts/03_distillation.py
```

Results are appended to `reports/results_summary.md` and figures saved to `reports/figures/`.

### requirements.txt

```
torch>=2.3.0
torchvision>=0.18.0
pillow>=10.0.0
numpy>=1.26.0
matplotlib>=3.8.0
```

## Design Notes

**Why PyTorch only?** TensorFlow's quantization API (`tf.lite.TFLiteConverter`) targets mobile/edge deployment and outputs a `.tflite` binary rather than a Keras model — it would require a separate serving path and complicates the benchmark. `torch.quantization` produces a standard `nn.Module` that integrates cleanly with the existing `ModelRegistry`.

**Why no QAT?** Quantization-aware training requires a full training loop with fake-quantization nodes inserted. PTQ is the realistic baseline for teams deploying a model they did not train themselves. QAT is flagged in [Future Work](#future-work) as the natural next step.

**Why evaluate without fine-tuning after pruning?** Fine-tuning a pruned model typically recovers most accuracy loss, but it obscures how much damage pruning itself does. Evaluating the pruned-but-not-recovered model shows the raw accuracy cliff, which is the more honest engineering baseline. Fine-tuning recovery is flagged in [Future Work](#future-work).

**Inference server compatibility.** The compressed PyTorch models expose the same `forward(x: Tensor) -> Tensor` interface as the originals. Swapping them into `ml-satellite-image-classifier/serve/model_registry.py` requires only pointing `_MODEL_FILES` at the new checkpoint paths.

## Future Work

- **Fine-tuning recovery after pruning**: a short 5-epoch fine-tune pass typically recovers 1–3% accuracy at 60% sparsity; adds a more complete picture of the pruning budget
- **Quantization-aware training (QAT)**: `torch.quantization.prepare_qat` + training loop with fake-quantization nodes; expected to close the accuracy gap vs. PTQ at high compression ratios
- **TorchScript / ONNX export**: export the best compressed model to ONNX and benchmark with ONNX Runtime (`onnxruntime-cpu`), which often yields additional latency gains through graph-level optimizations
- **4-bit NF4 quantization with `bitsandbytes`**: relevant for the ViT variant; extends the benchmark to sub-8-bit regimes
- **End-to-end inference server benchmark**: replace the FP32 model in `ml-satellite-image-classifier/serve/` with each compressed variant and report `/predict` endpoint latency under load with `locust` or `wrk`

## References

- Han, S., Pool, J., Tran, J., and Dally, W.J. "Learning Both Weights and Connections for Efficient Neural Networks." *NeurIPS*, 2015. [arxiv.org/abs/1506.02626](https://arxiv.org/abs/1506.02626)
- Molchanov, P., Tyree, S., Karras, T., Aila, T., and Kautz, J. "Pruning Convolutional Neural Networks for Resource Efficient Inference." *ICLR*, 2017. [arxiv.org/abs/1611.06440](https://arxiv.org/abs/1611.06440)
- Jacob, B., et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*, 2018. [arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)
- Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." *NeurIPS Workshop*, 2015. [arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
- PyTorch documentation: [Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html), [Static Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

## License

Apache License 2.0. See [LICENSE](../LICENSE) for the full license text.

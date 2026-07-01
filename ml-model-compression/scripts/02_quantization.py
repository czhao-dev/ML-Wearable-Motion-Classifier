#!/usr/bin/env python3
"""Static INT8 PTQ (standalone CNN) and dynamic INT8 PTQ (CNN-ViT linear
layers).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.ao.quantization as tq
import torch.nn as nn

from src.benchmark import measure_latency, measure_model_size, record_result, write_results_summary
from src.eval_utils import build_canonical_split, evaluate_model, load_fp32_cnn, load_fp32_cnn_vit
from src.paths import TRAINED_MODELS_DIR

DEVICE = "cpu"  # quantized INT8 kernels are CPU-only in eager mode
CALIBRATION_IMAGES = 200

# torch.ao.quantization.get_default_qconfig("x86") requires the fbgemm
# backend, which is x86-only. On ARM (e.g. Apple Silicon dev machines), only
# "qnnpack" is available -- pick whichever the current machine supports.
_QENGINE = "x86" if "fbgemm" in torch.backends.quantized.supported_engines else "qnnpack"
torch.backends.quantized.engine = _QENGINE


class _QuantWrapper(nn.Module):
    """Wraps a bare model with QuantStub/DeQuantStub boundaries.

    build_satellite_cnn returns a plain nn.Sequential with no quant stubs --
    eager-mode static quantization (torch.ao.quantization.prepare/convert)
    requires them at the input/output boundary to actually insert INT8 ops,
    otherwise the model just runs FP32 arithmetic internally.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


class _FloatBatchNorm(nn.Module):
    """Runs a BatchNorm1d/2d in FP32, sandwiched between DeQuantStub/QuantStub.

    build_satellite_cnn's conv blocks are Conv2d -> ReLU -> MaxPool2d ->
    BatchNorm2d (BN after pooling, not the usual Conv->BN->ReLU), and the
    classifier head has a BatchNorm1d after its first Linear -- neither
    matches torch.ao.quantization.fuse_modules's recognized adjacent
    Conv+BN[+ReLU] / Linear+ReLU patterns, and there is no standalone
    quantized BatchNorm kernel, so leaving either in the quantized region
    crashes at convert()/inference time. Wrapping them like this keeps
    everything else genuinely INT8 while letting BN run in float, which is
    the standard eager-mode workaround for ops without a quantized kernel.
    """

    def __init__(self, bn: nn.BatchNorm1d | nn.BatchNorm2d) -> None:
        super().__init__()
        self.dequant = tq.DeQuantStub()
        self.bn = bn
        self.bn.qconfig = None  # exclude from quantization -- runs in FP32
        self.quant = tq.QuantStub()

    def forward(self, x):
        return self.quant(self.bn(self.dequant(x)))


def _wrap_batchnorms_as_float(model: nn.Sequential) -> nn.Sequential:
    return nn.Sequential(*[
        _FloatBatchNorm(layer) if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)) else layer
        for layer in model
    ])


def run_static_ptq_cnn(train_loader, val_loader) -> None:
    cnn = _wrap_batchnorms_as_float(load_fp32_cnn(DEVICE))
    model = _QuantWrapper(cnn)
    model.eval()
    model.qconfig = tq.get_default_qconfig(_QENGINE)

    prepared = tq.prepare(model, inplace=False)

    # Calibration pass: 200 training images, never seen at test time.
    seen = 0
    with torch.no_grad():
        for images, _ in train_loader:
            prepared(images)
            seen += images.size(0)
            if seen >= CALIBRATION_IMAGES:
                break

    quantized = tq.convert(prepared, inplace=False)

    metrics = evaluate_model(quantized, val_loader, DEVICE)
    size_mb = measure_model_size(quantized)
    bench = measure_latency(quantized, DEVICE)

    torch.save(quantized.state_dict(), TRAINED_MODELS_DIR / "cnn_int8_static.pth")
    record_result({
        "variant": "PyTorch CNN — INT8 static PTQ",
        "technique": "Quantization",
        "qscheme": "static",
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "size_mb": size_mb,
        **bench,
    })
    print(f"static PTQ CNN: accuracy={metrics['accuracy']:.4f} size={size_mb:.2f}MB latency={bench['latency_ms']:.3f}ms")


def run_dynamic_ptq_cnn_vit(val_loader) -> None:
    model = load_fp32_cnn_vit(DEVICE)
    model.eval()

    quantized = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    metrics = evaluate_model(quantized, val_loader, DEVICE)
    size_mb = measure_model_size(quantized)
    bench = measure_latency(quantized, DEVICE)

    torch.save(quantized.state_dict(), TRAINED_MODELS_DIR / "cnn_vit_int8_dynamic.pth")
    record_result({
        "variant": "PyTorch CNN-ViT — INT8 dynamic PTQ",
        "technique": "Quantization",
        "qscheme": "dynamic",
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "size_mb": size_mb,
        **bench,
    })
    print(f"dynamic PTQ CNN-ViT: accuracy={metrics['accuracy']:.4f} size={size_mb:.2f}MB latency={bench['latency_ms']:.3f}ms")


def main() -> None:
    train_loader, val_loader = build_canonical_split()

    fp32_vit = load_fp32_cnn_vit(DEVICE)
    fp32_metrics = evaluate_model(fp32_vit, val_loader, DEVICE)
    record_result({
        "variant": "PyTorch CNN-ViT (FP32 baseline)",
        "technique": "Baseline",
        "accuracy": fp32_metrics["accuracy"],
        "f1": fp32_metrics["f1"],
        "size_mb": measure_model_size(fp32_vit),
        **measure_latency(fp32_vit, DEVICE),
    })

    run_static_ptq_cnn(train_loader, val_loader)
    run_dynamic_ptq_cnn_vit(val_loader)
    write_results_summary()


if __name__ == "__main__":
    main()

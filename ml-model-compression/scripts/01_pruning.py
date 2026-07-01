#!/usr/bin/env python3
"""Magnitude (unstructured) and structured L1 channel pruning sweep on the
PyTorch CNN baseline, at sparsity levels {20%, 40%, 60%, 80%}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.benchmark import measure_latency, measure_model_size, record_result, write_results_summary
from src.eval_utils import build_canonical_split, evaluate_model, load_fp32_cnn
from src.paths import TRAINED_MODELS_DIR

SPARSITY_LEVELS = [0.2, 0.4, 0.6, 0.8]
DEVICE = "cpu"


def _conv_and_linear_layers(model: nn.Sequential) -> list[tuple[nn.Module, str]]:
    """Return [(module, 'weight'), ...] for every Conv2d/Linear layer, in order."""
    return [(m, "weight") for m in model if isinstance(m, (nn.Conv2d, nn.Linear))]


def run_unstructured_sweep(val_loader) -> None:
    for sparsity in SPARSITY_LEVELS:
        model = load_fp32_cnn(DEVICE)
        parameters_to_prune = _conv_and_linear_layers(model)

        # global_unstructured ranks ALL target weights together, matching
        # the README's "zeroes out the lowest-magnitude weights globally
        # across all convolutional and linear layers" -- looping a per-layer
        # l1_unstructured call instead would apply the percentage
        # independently per layer, a weaker guarantee.
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        for module, name in parameters_to_prune:
            prune.remove(module, name)  # bake in permanent zeros

        metrics = evaluate_model(model, val_loader, DEVICE)
        size_mb = measure_model_size(model)
        bench = measure_latency(model, DEVICE)

        variant = f"PyTorch CNN — pruned {int(sparsity * 100)}% (unstructured)"
        torch.save(model.state_dict(), TRAINED_MODELS_DIR / f"cnn_pruned_unstructured_{int(sparsity * 100)}.pth")
        record_result({
            "variant": variant,
            "technique": "Pruning",
            "prune_type": "unstructured",
            "sparsity": sparsity,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "size_mb": size_mb,
            **bench,
        })
        print(f"{variant}: accuracy={metrics['accuracy']:.4f} size={size_mb:.2f}MB latency={bench['latency_ms']:.3f}ms")


def rebuild_pruned_channels(model: nn.Sequential, channel_masks: dict[int, torch.Tensor]) -> nn.Sequential:
    """Physically rebuild `model` (an nn.Sequential matching build_satellite_cnn's
    layout) with structurally-pruned output channels removed from each Conv2d
    layer, cascading the reduced channel count into the next Conv2d's input
    channels and the paired BatchNorm2d's num_features.

    channel_masks: {conv_layer_index_in_sequential: boolean_keep_mask} where
    True entries mark output channels that survive ln_structured pruning.
    Only conv layers appear as keys; the two trailing Linear layers are left
    untouched EXCEPT the first Linear's in_features, which must shrink to
    match the final conv block's surviving channel count (since
    AdaptiveAvgPool2d(1) passes channel count straight through into Flatten).

    Returns a new nn.Sequential with real (smaller) Conv2d/BatchNorm2d/Linear
    layers -- no pruning masks, loadable as a standard checkpoint.
    """
    new_layers = []
    prev_keep_mask: torch.Tensor | None = None
    last_conv_out_channels: int | None = None
    first_linear_rewritten = False
    pending_bn_rebuild = False  # True right after emitting a pruned conv, until its paired BN is consumed

    for i, layer in enumerate(model):
        if isinstance(layer, nn.Conv2d) and i in channel_masks:
            keep_out = channel_masks[i]
            if keep_out.sum().item() == 0:
                # Safety floor: never prune an entire layer to zero channels.
                keep_out = keep_out.clone()
                keep_out[layer.weight.data.abs().sum(dim=(1, 2, 3)).argmax()] = True

            in_channels = layer.in_channels if prev_keep_mask is None else int(prev_keep_mask.sum())
            new_conv = nn.Conv2d(
                in_channels, int(keep_out.sum()), kernel_size=layer.kernel_size, padding=layer.padding
            )
            weight = layer.weight.data
            if prev_keep_mask is not None:
                weight = weight[:, prev_keep_mask, :, :]
            new_conv.weight.data = weight[keep_out, :, :, :].clone()
            new_conv.bias.data = layer.bias.data[keep_out].clone()
            new_layers.append(new_conv)
            prev_keep_mask = keep_out
            last_conv_out_channels = int(keep_out.sum())
            pending_bn_rebuild = True

        elif isinstance(layer, nn.BatchNorm2d) and pending_bn_rebuild:
            new_bn = nn.BatchNorm2d(int(prev_keep_mask.sum()))
            new_bn.weight.data = layer.weight.data[prev_keep_mask].clone()
            new_bn.bias.data = layer.bias.data[prev_keep_mask].clone()
            new_bn.running_mean = layer.running_mean[prev_keep_mask].clone()
            new_bn.running_var = layer.running_var[prev_keep_mask].clone()
            new_layers.append(new_bn)
            pending_bn_rebuild = False

        elif isinstance(layer, nn.Linear) and not first_linear_rewritten and last_conv_out_channels is not None:
            new_linear = nn.Linear(last_conv_out_channels, layer.out_features)
            new_linear.weight.data = layer.weight.data[:, prev_keep_mask].clone()
            new_linear.bias.data = layer.bias.data.clone()
            new_layers.append(new_linear)
            first_linear_rewritten = True

        else:
            new_layers.append(copy.deepcopy(layer))

    return nn.Sequential(*new_layers)


def run_structured_sweep(val_loader) -> None:
    for sparsity in SPARSITY_LEVELS:
        model = load_fp32_cnn(DEVICE)
        conv_layers = [(i, m) for i, m in enumerate(model) if isinstance(m, nn.Conv2d)]

        channel_masks: dict[int, torch.Tensor] = {}
        for idx, layer in conv_layers:
            prune.ln_structured(layer, name="weight", amount=sparsity, n=1, dim=0)
            keep = layer.weight_mask.abs().sum(dim=(1, 2, 3)) > 0
            channel_masks[idx] = keep
            prune.remove(layer, "weight")

        rebuilt = rebuild_pruned_channels(model, channel_masks)
        metrics = evaluate_model(rebuilt, val_loader, DEVICE)
        size_mb = measure_model_size(rebuilt)
        bench = measure_latency(rebuilt, DEVICE)

        variant = f"PyTorch CNN — pruned {int(sparsity * 100)}% (structured)"
        torch.save(rebuilt.state_dict(), TRAINED_MODELS_DIR / f"cnn_pruned_structured_{int(sparsity * 100)}.pth")
        record_result({
            "variant": variant,
            "technique": "Pruning",
            "prune_type": "structured",
            "sparsity": sparsity,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "size_mb": size_mb,
            **bench,
        })
        print(f"{variant}: accuracy={metrics['accuracy']:.4f} size={size_mb:.2f}MB latency={bench['latency_ms']:.3f}ms")


def main() -> None:
    _, val_loader = build_canonical_split()

    fp32_model = load_fp32_cnn(DEVICE)
    fp32_metrics = evaluate_model(fp32_model, val_loader, DEVICE)
    record_result({
        "variant": "PyTorch CNN (FP32 baseline)",
        "technique": "Baseline",
        "accuracy": fp32_metrics["accuracy"],
        "f1": fp32_metrics["f1"],
        "size_mb": measure_model_size(fp32_model),
        **measure_latency(fp32_model, DEVICE),
    })

    run_unstructured_sweep(val_loader)
    run_structured_sweep(val_loader)
    write_results_summary()


if __name__ == "__main__":
    main()

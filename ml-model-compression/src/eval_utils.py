"""Dataset split construction, FP32 checkpoint loading, and accuracy/F1
evaluation on the canonical held-out split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from src.paths import (
    CNN_VIT_DEPTH,
    CNN_VIT_EMBED_DIM,
    CNN_VIT_HEADS,
    EXTRACTED_DATASET_DIR,
    PYTORCH_CNN_CHECKPOINT,
    PYTORCH_CNN_VIT_CHECKPOINT,
    SPLIT_SEED,
    TRAIN_FRACTION,
    CNN_ViT_Hybrid,
    binary_classification_metrics,
    build_satellite_cnn,
)

IMG_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_canonical_split(
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Reconstruct the SEED=42 80/20 split from the satellite classifier's
    original CNN training run (scripts/05_pytorch_cnn_classifier.py).

    Both loaders use the eval transform (no augmentation), since this
    project only evaluates and distills, never trains with augmentation.
    Returns (train_loader, val_loader): train_loader yields the 4,800-image
    split used both as the calibration pool (quantization) and the
    student's training set (distillation); val_loader yields the fixed
    1,200-image held-out split used for every accuracy/F1 number here.
    """
    transform = _eval_transform()
    base_dataset = datasets.ImageFolder(str(EXTRACTED_DATASET_DIR))
    train_size = int(TRAIN_FRACTION * len(base_dataset))
    val_size = len(base_dataset) - train_size
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

    full_dataset = datasets.ImageFolder(str(EXTRACTED_DATASET_DIR), transform=transform)
    train_dataset = Subset(full_dataset, train_subset.indices)
    val_dataset = Subset(full_dataset, val_subset.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def load_fp32_cnn(device: torch.device | str = "cpu") -> nn.Sequential:
    """Load the standalone PyTorch CNN's FP32 checkpoint (strict load; keys
    are plain nn.Sequential integers, no remapping needed)."""
    model = build_satellite_cnn(num_classes=2)
    state_dict = torch.load(str(PYTORCH_CNN_CHECKPOINT), map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _load_cnn_backbone_weights(cnn_submodule: nn.Module, state_dict_path: Path, map_location) -> None:
    """Reproduction of ml-satellite-image-classifier's
    scripts/09_final_cnn_vit_evaluation.py::load_cnn_backbone_weights: loads
    the standalone CNN checkpoint's conv-block weights into a
    CNN_ViT_Hybrid.cnn submodule, handling "module."/"cnn."-stripped and
    "features."-prefixed key variants.
    """
    state_dict = torch.load(str(state_dict_path), map_location=map_location)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cnn_state = cnn_submodule.state_dict()
    mapped_state = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.").removeprefix("cnn.")
        for candidate in (normalized_key, f"features.{normalized_key}"):
            if candidate in cnn_state and cnn_state[candidate].shape == value.shape:
                mapped_state[candidate] = value
                break
    cnn_submodule.load_state_dict(mapped_state, strict=False)


def load_fp32_cnn_vit(device: torch.device | str = "cpu") -> CNN_ViT_Hybrid:
    """Load the CNN-ViT hybrid's FP32 checkpoint using the two-step sequence
    proven in scripts/09_final_cnn_vit_evaluation.py (NOT the simplified
    strict=False + naive prefix-strip in serve/model_registry.py, which can
    silently leave backbone weights at random init if key prefixes don't
    line up):

      1. Load the standalone CNN checkpoint's conv weights into model.cnn.
      2. Load the full hybrid checkpoint over the top with strict=False.

    depth/heads are passed explicitly (3, 6) -- CNN_ViT_Hybrid's class
    defaults (6, 8) do not match the trained checkpoint's architecture.

    Design caveat: the CNN-ViT checkpoint was originally trained on a
    DIFFERENT split (SEED=7331) than this project's canonical SEED=42 split.
    When this teacher is used in 03_distillation.py to generate soft labels
    over the SEED=42 "training" 4,800 images, a small number of those images
    may have already been seen by the teacher during its own training. This
    is acceptable because the teacher is a frozen label source here, not the
    object being evaluated -- see reports/results_summary.md commentary.
    """
    model = CNN_ViT_Hybrid(num_classes=2, embed_dim=CNN_VIT_EMBED_DIM, depth=CNN_VIT_DEPTH, heads=CNN_VIT_HEADS)
    map_location = torch.device(device) if isinstance(device, str) else device

    _load_cnn_backbone_weights(model.cnn, PYTORCH_CNN_CHECKPOINT, map_location)
    full_state_dict = torch.load(str(PYTORCH_CNN_VIT_CHECKPOINT), map_location=map_location)
    if isinstance(full_state_dict, dict) and "state_dict" in full_state_dict:
        full_state_dict = full_state_dict["state_dict"]
    model.load_state_dict(full_state_dict, strict=False)

    return model.to(device).eval()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Run `model` over `loader`, collect softmax class-1 probabilities and
    hard labels, and compute accuracy/precision/recall/F1/roc_auc via the
    sibling project's binary_classification_metrics.
    """
    model.eval()
    all_labels: list[int] = []
    all_scores: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_scores.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

    return binary_classification_metrics(
        y_true=np.array(all_labels),
        y_score=np.array(all_scores),
    )

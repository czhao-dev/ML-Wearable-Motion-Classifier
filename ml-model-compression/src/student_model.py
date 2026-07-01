"""Lightweight 3-block student CNN for knowledge distillation."""

from __future__ import annotations

import torch.nn as nn


class StudentCNN(nn.Module):
    """3-block CNN following the first three blocks of build_satellite_cnn's
    structure (channels 32 -> 64 -> 128, 5x5 conv + ReLU + MaxPool2d(2) +
    BatchNorm2d per block), then global average pooling and a single linear
    classifier head (no hidden FC layer, unlike the teacher's 2-layer head,
    to keep the student maximally compact).

    ~259K params (~1.0 MB FP32) by direct computation. This is well beyond
    "~12x smaller" than the teacher CNN-ViT stated in the README's
    Highlights -- the actual measured ratio (computed at runtime via
    benchmark.measure_model_size on both models) is closer to 30-90x
    depending on which teacher size figure is used as the denominator.
    Report the real measured ratio in reports/results_summary.md rather
    than repeating "12x".
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

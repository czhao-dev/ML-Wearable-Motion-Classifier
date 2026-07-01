#!/usr/bin/env python3
"""Knowledge distillation: CNN-ViT teacher -> lightweight StudentCNN."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.benchmark import measure_latency, measure_model_size, record_result, write_results_summary
from src.eval_utils import build_canonical_split, evaluate_model, load_fp32_cnn_vit
from src.paths import FIGURES_DIR, TRAINED_MODELS_DIR
from src.student_model import StudentCNN

DEVICE = "cpu"
EPOCHS = 30
ALPHA = 0.3
TEMPERATURE = 4.0
LR = 1e-3


def distillation_loss(student_logits, teacher_logits, hard_labels, alpha=ALPHA, T=TEMPERATURE):
    """L = alpha * CE(student, hard) + (1 - alpha) * T^2 * KL(softmax(student/T), softmax(teacher/T))

    NOTE: the README's written formula omits the T^2 scaling factor on the KL
    term. Standard KD practice (Hinton et al. 2015) includes T^2 because the
    gradient of a temperature-softened KL term shrinks by ~1/T^2; multiplying
    back by T^2 keeps the soft-loss gradient magnitude comparable to the hard
    CE loss across different T values. This implementation follows standard
    practice.
    """
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss


def train_student(train_loader, val_loader, teacher=None) -> tuple[dict, dict]:
    """Train a fresh StudentCNN for EPOCHS. If `teacher` is None, trains on
    hard labels only (the baseline control). Otherwise distills from the
    frozen teacher's soft labels combined with hard labels via
    distillation_loss.

    Returns (best_state_dict, history) where history has 'train_loss',
    'val_loss', 'train_acc', 'val_acc' per-epoch lists, and best_state_dict
    is the state dict with the highest val accuracy across all epochs.
    """
    student = StudentCNN(num_classes=2).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        student.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            student_logits = student(images)

            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = distillation_loss(student_logits, teacher_logits, labels)
            else:
                loss = F.cross_entropy(student_logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (student_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        val_metrics_epoch, val_loss = _evaluate_epoch(student, val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics_epoch["accuracy"])

        if val_metrics_epoch["accuracy"] > best_val_acc:
            best_val_acc = val_metrics_epoch["accuracy"]
            best_state = copy.deepcopy(student.state_dict())

        kind = "distilled" if teacher is not None else "baseline"
        print(
            f"[{kind}] epoch {epoch + 1}/{EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics_epoch['accuracy']:.4f}"
        )

    return best_state, history


@torch.no_grad()
def _evaluate_epoch(model, val_loader):
    model.eval()
    metrics = evaluate_model(model, val_loader, DEVICE)
    total_loss, total = 0.0, 0
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels).item() * images.size(0)
        total += labels.size(0)
    return metrics, total_loss / total


def plot_distillation_curves(history_baseline: dict, history_distilled: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history_baseline["train_loss"], label="Baseline train")
    axes[0].plot(history_baseline["val_loss"], label="Baseline val")
    axes[0].plot(history_distilled["train_loss"], label="Distilled train")
    axes[0].plot(history_distilled["val_loss"], label="Distilled val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_baseline["val_acc"], label="Baseline val acc")
    axes[1].plot(history_distilled["val_acc"], label="Distilled val acc")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distillation_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    train_loader, val_loader = build_canonical_split()
    teacher = load_fp32_cnn_vit(DEVICE)

    print("Training baseline student (hard labels only)...")
    baseline_state, baseline_history = train_student(train_loader, val_loader, teacher=None)

    print("Training distilled student (CNN-ViT teacher)...")
    distilled_state, distilled_history = train_student(train_loader, val_loader, teacher=teacher)

    plot_distillation_curves(baseline_history, distilled_history)

    for name, state in (
        ("StudentCNN (hard labels only)", baseline_state),
        ("StudentCNN (distilled from CNN-ViT)", distilled_state),
    ):
        student = StudentCNN(num_classes=2)
        student.load_state_dict(state)
        student.to(DEVICE).eval()

        metrics = evaluate_model(student, val_loader, DEVICE)
        size_mb = measure_model_size(student)
        bench = measure_latency(student, DEVICE)

        checkpoint_name = "student_baseline.pth" if "hard labels" in name else "student_distilled.pth"
        torch.save(state, TRAINED_MODELS_DIR / checkpoint_name)
        record_result({
            "variant": name,
            "technique": "Distillation",
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "size_mb": size_mb,
            **bench,
        })
        print(f"{name}: accuracy={metrics['accuracy']:.4f} size={size_mb:.2f}MB latency={bench['latency_ms']:.3f}ms")

    write_results_summary()


if __name__ == "__main__":
    main()

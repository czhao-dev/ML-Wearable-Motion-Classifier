#!/usr/bin/env python3
"""Evidently drift report: reference vs. current satellite-tile batches.

Raw image pixels aren't tabular, so this extracts a small set of per-image
summary statistics -- per-channel mean/std, overall brightness, contrast,
and (when a trained pytorch_cnn checkpoint is available) the served model's
predicted confidence -- and feeds those as tabular features into Evidently's
DataDriftPreset.

To produce a genuine drift signal rather than a null result, the "current"
batch is a brightness-shifted copy of a disjoint sample of the same dataset,
standing in for a real-world lighting/sensor shift; unaffected columns
(e.g. mean_g on an evenly-lit shift) are expected to show little or no
drift, which is itself part of the honest signal this report is checking.

Usage:
    python monitoring/generate_drift_report.py
"""
from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATASET_DIR = REPO_ROOT / "data" / "raw" / "images_dataSAT"
CHECKPOINT_PATH = REPO_ROOT / "models" / "trained" / "ai_capstone_pytorch_state_dict.pth"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

IMG_SIZE = 64
SEED = 42
SAMPLE_SIZE = 150
BRIGHTNESS_SHIFT_FACTOR = 1.6  # simulated drift: overexposed tiles


def _load_model():
    """Load the trained pytorch_cnn checkpoint, if present, to add a confidence feature."""
    if not CHECKPOINT_PATH.exists():
        return None
    import torch
    from serve.pytorch_models import build_satellite_cnn

    model = build_satellite_cnn()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def _load_image(path: Path, brightness_factor: float = 1.0) -> Image.Image:
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    if brightness_factor != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    return image


def _image_features(image: Image.Image, model) -> dict:
    arr = np.asarray(image, dtype=np.float32)
    features = {
        "mean_r": float(arr[:, :, 0].mean()),
        "mean_g": float(arr[:, :, 1].mean()),
        "mean_b": float(arr[:, :, 2].mean()),
        "std_r": float(arr[:, :, 0].std()),
        "std_g": float(arr[:, :, 1].std()),
        "std_b": float(arr[:, :, 2].std()),
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
    }
    if model is not None:
        import torch
        from serve.preprocessing import preprocess_for_pytorch

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        tensor = preprocess_for_pytorch(buffer.getvalue())
        with torch.no_grad():
            confidence = torch.softmax(model(tensor), dim=1).max().item()
        features["model_confidence"] = confidence
    return features


def build_dataframe(paths: list[Path], model, brightness_factor: float = 1.0) -> pd.DataFrame:
    rows = [_image_features(_load_image(p, brightness_factor), model) for p in paths]
    return pd.DataFrame(rows)


def _print_drift_summary(snapshot) -> None:
    result = snapshot.dict()
    print("\nDrift summary:")
    for metric in result["metrics"]:
        name = metric["metric_name"]
        value = metric["value"]
        if name.startswith("DriftedColumnsCount"):
            print(f"  {name}: {value}")
        elif name.startswith("ValueDrift"):
            drifted = isinstance(value, (int, float)) and value < 0.05
            flag = "DRIFTED" if drifted else "stable"
            print(f"  {name}: p={value:.4g} [{flag}]")


def main() -> None:
    random.seed(SEED)
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"{DATASET_DIR} not found -- run scripts/05_pytorch_cnn_classifier.py "
            "(or any script that downloads the dataset) first."
        )

    all_paths = sorted(DATASET_DIR.rglob("*.jpg"))
    if len(all_paths) < SAMPLE_SIZE * 2:
        raise RuntimeError(f"Expected at least {SAMPLE_SIZE * 2} images, found {len(all_paths)}")

    random.shuffle(all_paths)
    reference_paths = all_paths[:SAMPLE_SIZE]
    current_paths = all_paths[SAMPLE_SIZE : SAMPLE_SIZE * 2]

    model = _load_model()
    print(f"Model confidence feature: {'enabled' if model is not None else 'skipped (no checkpoint found)'}")

    reference_df = build_dataframe(reference_paths, model, brightness_factor=1.0)
    current_df = build_dataframe(current_paths, model, brightness_factor=BRIGHTNESS_SHIFT_FACTOR)

    from evidently import Dataset, Report
    from evidently.presets import DataDriftPreset

    reference_dataset = Dataset.from_pandas(reference_df)
    current_dataset = Dataset.from_pandas(current_df)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current_dataset, reference_data=reference_dataset)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / "drift_report.html"
    json_path = REPORTS_DIR / "drift_report.json"
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    print(f"Wrote {html_path}")
    print(f"Wrote {json_path}")
    _print_drift_summary(snapshot)


if __name__ == "__main__":
    main()

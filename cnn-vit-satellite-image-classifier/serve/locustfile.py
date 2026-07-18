"""Locust load test for the FastAPI inference server (POST /predict, GET /health).

Usage (against a running server):
    locust --headless -u 20 -r 5 -t 60s --host http://localhost:8000 \
        --csv=reports/load_test -f serve/locustfile.py

Cycles through real held-out satellite tiles from data/raw/images_dataSAT/
when the dataset has already been downloaded locally (see scripts/05);
otherwise falls back to synthetic in-memory RGB tiles (same technique as
tests/conftest.py's rgb_png_bytes fixture) so this still runs standalone.
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from locust import HttpUser, between, task
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "data" / "raw" / "images_dataSAT"


def _load_real_images(limit: int = 200) -> list[bytes]:
    if not DATASET_DIR.exists():
        return []
    paths = list(DATASET_DIR.rglob("*.jpg"))[:limit]
    return [path.read_bytes() for path in paths]


def _synthetic_image() -> bytes:
    color = tuple(random.randint(0, 255) for _ in range(3))
    image = Image.fromarray(np.full((64, 64, 3), color, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


_REAL_IMAGES = _load_real_images()


def _sample_image_bytes() -> bytes:
    if _REAL_IMAGES:
        return random.choice(_REAL_IMAGES)
    return _synthetic_image()


class ClassifierUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def predict(self):
        image_bytes = _sample_image_bytes()
        self.client.post(
            "/predict",
            params={"model": "pytorch_cnn"},
            files={"file": ("tile.jpg", image_bytes, "image/jpeg")},
            name="/predict?model=pytorch_cnn",
        )

    @task(1)
    def health(self):
        self.client.get("/health")

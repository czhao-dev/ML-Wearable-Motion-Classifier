#!/usr/bin/env python3
"""Prefect orchestration flow for the PyTorch retraining pipeline.

Chains: train the CNN baseline (scripts/05) -> train the CNN-ViT hybrid on
top of it (scripts/08) -> read both runs' held-out accuracy back from MLflow
-> register the CNN baseline in the MLflow Model Registry only if it beats
the currently-registered version.

Each training step shells out to the existing numbered script rather than
reimplementing it, mirroring how these scripts are actually run today (see
scripts/README.md). Keras models are out of scope here -- see the root
README for why this pipeline only covers the PyTorch track.
"""
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task, get_run_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data" / "raw"
TRACKING_URI = "sqlite:///" + str(REPO_ROOT / "mlruns.db")
EXPERIMENT_NAME = "cnn-vit-satellite-image-classifier"
REGISTERED_MODEL_NAME = "cnn-vit-satellite-pytorch-cnn"


def _run_script(script_name: str) -> None:
    logger = get_run_logger()
    script_path = SCRIPTS_DIR / script_name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running {script_path} (cwd={DATA_DIR})")
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(DATA_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


@task(name="train_cnn")
def train_cnn() -> None:
    """Train the PyTorch CNN baseline (scripts/05)."""
    _run_script("05_pytorch_cnn_classifier.py")


@task(name="train_hybrid")
def train_hybrid() -> None:
    """Train the CNN-ViT hybrid on top of the CNN checkpoint (scripts/08)."""
    _run_script("08_pytorch_cnn_vit_hybrid.py")


@task(name="evaluate")
def evaluate() -> dict:
    """Read back the two just-completed runs' held-out accuracy from MLflow."""
    logger = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=2,
    )
    results = {}
    for run in runs:
        model_name = run.data.params.get("model")
        accuracy = run.data.metrics.get("held_out_accuracy")
        results[model_name] = {"run_id": run.info.run_id, "accuracy": accuracy}
        logger.info(f"{model_name}: held_out_accuracy={accuracy} (run_id={run.info.run_id})")
    return results


@task(name="register_if_better")
def register_if_better(results: dict) -> None:
    """Register the new pytorch_cnn run only if it beats the current best registered version."""
    logger = get_run_logger()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    cnn_result = results.get("pytorch_cnn")
    if cnn_result is None or cnn_result["accuracy"] is None:
        logger.warning("No pytorch_cnn run with a held_out_accuracy metric found -- skipping registration.")
        return

    new_accuracy = cnn_result["accuracy"]
    run_id = cnn_result["run_id"]

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    current_best = 0.0
    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        latest_run = client.get_run(latest.run_id)
        current_best = latest_run.data.metrics.get("held_out_accuracy", 0.0)

    if new_accuracy > current_best:
        logger.info(
            f"New pytorch_cnn accuracy {new_accuracy:.4f} beats currently-registered "
            f"best {current_best:.4f} -- registering."
        )
        mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    else:
        logger.info(
            f"New pytorch_cnn accuracy {new_accuracy:.4f} does not beat currently-registered "
            f"best {current_best:.4f} -- not registering."
        )


@flow(name="cnn-vit-pytorch-retrain")
def retrain_flow() -> None:
    train_cnn()
    train_hybrid()
    results = evaluate()
    register_if_better(results)


if __name__ == "__main__":
    retrain_flow()

"""Latency/throughput benchmarking and the shared results cache."""

from __future__ import annotations

import io
import json
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.paths import FIGURES_DIR, RESULTS_CACHE_JSON, RESULTS_SUMMARY_MD

# ---------------------------------------------------------------------------
# Latency / size measurement
# ---------------------------------------------------------------------------


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    device: torch.device | str = "cpu",
    input_shape: tuple[int, int, int, int] = (1, 3, 64, 64),
    n: int = 500,
    warmup: int = 50,
) -> dict[str, float]:
    """Measure mean single-image CPU inference latency and derived throughput.

    Runs `warmup` untimed forward passes, then times `n` sequential
    single-image (batch size 1) forward passes with time.perf_counter,
    matching the README's stated methodology.
    """
    model.eval().to(device)
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        model(dummy)

    start = perf_counter()
    for _ in range(n):
        model(dummy)
    elapsed = perf_counter() - start

    latency_ms = (elapsed / n) * 1000.0
    throughput_ips = n / elapsed
    return {"latency_ms": latency_ms, "throughput_ips": throughput_ips}


def measure_model_size(model_or_state_dict: nn.Module | dict) -> float:
    """Serialize a model's state_dict to an in-memory buffer via torch.save
    and return its size in MB. Avoids writing throwaway files to disk.
    """
    state_dict = (
        model_or_state_dict.state_dict()
        if isinstance(model_or_state_dict, nn.Module)
        else model_or_state_dict
    )
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Results cache (reports/results.json) -> markdown table + figures
# ---------------------------------------------------------------------------


def _load_results() -> list[dict[str, Any]]:
    if RESULTS_CACHE_JSON.exists():
        return json.loads(RESULTS_CACHE_JSON.read_text())
    return []


def record_result(row: dict[str, Any]) -> None:
    """Upsert one benchmark row into reports/results.json, keyed by
    row["variant"].

    Expected row schema (extra script-specific keys allowed, e.g.
    "sparsity", "prune_type", "qscheme"):
      {
        "variant": str,        # unique display name
        "technique": str,      # "Pruning" | "Quantization" | "Distillation"
        "accuracy": float,
        "f1": float,
        "size_mb": float,
        "latency_ms": float,
        "throughput_ips": float,
      }
    """
    rows = _load_results()
    rows = [r for r in rows if r.get("variant") != row.get("variant")]
    rows.append(row)
    RESULTS_CACHE_JSON.write_text(json.dumps(rows, indent=2))


def write_results_summary() -> None:
    """Regenerate reports/results_summary.md (full markdown table) and
    reports/figures/{accuracy_vs_sparsity,size_vs_latency}.png from the
    current contents of results.json.

    Safe to call after every script run; only plots data that exists so far.
    distillation_curves.png is written directly by 03_distillation.py (it
    needs per-epoch loss history, not the row-level cache).
    """
    rows = _load_results()
    _write_markdown_table(rows)
    if rows:
        _plot_accuracy_vs_sparsity(rows)
        _plot_size_vs_latency(rows)


def _write_markdown_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "| Model | Compression | Accuracy | F1 | Size (MB) | Latency (ms/img) | Throughput (img/s) |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for r in sorted(rows, key=lambda r: (r.get("technique", ""), r.get("variant", ""))):
        lines.append(
            f"| {r['variant']} | {r.get('technique', '')} | {r['accuracy'] * 100:.2f}% | "
            f"{r['f1']:.4f} | {r['size_mb']:.2f} | {r['latency_ms']:.3f} | {r['throughput_ips']:.1f} |\n"
        )
    body = "".join(lines)
    content = (
        "# Benchmark Results\n\n"
        "Full run log for all compression techniques. Regenerated automatically "
        "each time a script in scripts/ is run.\n\n"
        + body
        + "\n## Notes\n\n"
        "- Latency: mean of 500 single-image CPU forward passes after 50 warmup "
        "steps (`time.perf_counter`).\n"
        "- Accuracy/F1: fixed SEED=42 held-out validation split (1,200 images), "
        "identical for every row.\n"
    )
    RESULTS_SUMMARY_MD.write_text(content)


def _plot_accuracy_vs_sparsity(rows: list[dict[str, Any]]) -> None:
    pruning_rows = [r for r in rows if r.get("technique") == "Pruning" and "sparsity" in r]
    if not pruning_rows:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for prune_type in sorted({r["prune_type"] for r in pruning_rows}):
        subset = sorted((r for r in pruning_rows if r["prune_type"] == prune_type), key=lambda r: r["sparsity"])
        ax.plot(
            [r["sparsity"] * 100 for r in subset],
            [r["accuracy"] * 100 for r in subset],
            marker="o",
            label=prune_type,
        )
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "accuracy_vs_sparsity.png", dpi=150)
    plt.close(fig)


def _plot_size_vs_latency(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        ax.scatter(r["size_mb"], r["latency_ms"])
        ax.annotate(
            r["variant"],
            (r["size_mb"], r["latency_ms"]),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Latency (ms/image)")
    ax.set_title("Size vs. Latency (all variants)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "size_vs_latency.png", dpi=150)
    plt.close(fig)

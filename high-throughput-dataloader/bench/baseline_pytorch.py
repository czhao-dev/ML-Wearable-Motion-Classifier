"""PyTorch DataLoader baseline for the throughput comparison (spec section 13).

Not algorithmically identical to the C++ ImageGenerator -- this uses a
vectorized torch.rand() call (as a real PyTorch synthetic Dataset would),
not a hand-rolled per-pixel shape-rasterization loop. It matches the same
tensor shape (3x64x64) and a comparable, wall-clock-calibrated per-sample
cost (a busy-wait, not an iteration count), so the comparison reflects the
two data-loading INFRASTRUCTURES' throughput rather than incidental
differences in how the two languages execute nested pixel loops.

Reports both "first epoch" (includes multiprocessing worker startup -- on
macOS, Python's default "spawn" start method re-imports the module in each
worker process, which is not cheap) and "steady state" (a later epoch, with
persistent_workers=True so the worker pool is reused) throughput, since the
two tell very different stories for a process-based DataLoader.

Usage: python3 baseline_pytorch.py
Requires: torch (see bench/requirements.txt)
"""

import time

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples: int, cost_us: float):
        self.num_samples = num_samples
        self.cost_us = cost_us

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        image = torch.rand(3, 64, 64)
        label = int(torch.randint(0, 10, (1,)).item())

        if self.cost_us > 0:
            deadline = time.perf_counter() + self.cost_us / 1e6
            while time.perf_counter() < deadline:
                pass

        return image, label


def bench(num_samples: int, num_workers: int, batch_size: int, cost_us: float, num_epochs: int = 3):
    """Returns (first_epoch_rate, steady_state_rate) in samples/sec."""
    dataset = SyntheticImageDataset(num_samples, cost_us)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    epoch_times = []
    for _ in range(num_epochs):
        start = time.perf_counter()
        count = 0
        for images, _labels in loader:
            count += images.shape[0]
        epoch_times.append(time.perf_counter() - start)

    first_epoch_rate = count / epoch_times[0]
    steady_state_rate = count / epoch_times[-1]
    return first_epoch_rate, steady_state_rate


def main() -> None:
    num_samples = 2000
    batch_size = 32
    # Calibrated to roughly match the C++ pipeline's "cheap" generation
    # regime (see bench/p3_cost_sweep_bench.cpp, cost_iterations=0..20000,
    # which measured ~145us/sample of fixed per-sample cost at 4 workers).
    cost_us = 150

    print(f"{'workers':>8} {'first_epoch/s':>14} {'steady_state/s':>15}")
    for num_workers in (0, 1, 2, 4, 8):
        first, steady = bench(num_samples, num_workers, batch_size, cost_us)
        print(f"{num_workers:>8} {first:>14.1f} {steady:>15.1f}")


if __name__ == "__main__":
    main()

"""Generate synthetic IMU recordings by augmenting the real WMFT data/ recordings.

Unlike generate_synthetic_data.py (which fabricates motion from hand-picked
amplitude assumptions per class), this script perturbs the *actual* labeled
recordings in data/: it splits each real trial into a rest baseline and a
motion component (using the same initial-window detection the real pipeline
uses), then creates new trials by time-warping, scaling, and adding noise to
the motion component before reassembling and re-encoding it in the original
sensor file format. This keeps each class's real motion signature (true
quaternion track, true acceleration shape) while producing many plausible
variations for training.

This is still NOT real data. It cannot invent variation the original 1-4
trials per class didn't already contain (e.g. a different subject, a
different impairment level). Treat a model trained on it as a pipeline
sanity check, not a validated classifier.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from wmft_motion.io import load_sensor_file
from wmft_motion.preprocessing import determine_initial_window
from wmft_motion.quaternion import multiply, normalize, rotate_vectors

ACCEL_SENSITIVITY_16G = 2048.0
GYRO_SENSITIVITY = 16.4
QUATERNION_SENSITIVITY = 1_073_741_824.0
SAMPLE_RATE_HZ = 200.0

HEADER = "Receiver time, Accel X,Y,Z, Gyro X,Y,Z, Quaternion W,X,Y,Z, dt(or)index"

# Wider than a first pass: narrow ranges make augmented variants near-duplicates
# of their parent trial, which understates real trial-to-trial variability and
# makes classes look more separable than they are.
GAIN_RANGE = (0.7, 1.3)
WARP_RANGE = (0.7, 1.3)
BASELINE_JITTER_RANGE = (0.95, 1.05)
ACCEL_NOISE_STD_G = 0.008
GYRO_NOISE_STD_DPS = 1.5
ORIENTATION_JITTER_DEG = 6.0


def random_small_rotation(rng: np.random.Generator) -> np.ndarray:
    """One-time small-angle quaternion offset simulating sensor mounting variation."""

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-ORIENTATION_JITTER_DEG, ORIENTATION_JITTER_DEG))
    half = angle / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def time_warp(matrix: np.ndarray, new_n: int) -> np.ndarray:
    n = matrix.shape[0]
    old_idx = np.linspace(0.0, 1.0, n)
    new_idx = np.linspace(0.0, 1.0, new_n)
    return np.column_stack([np.interp(new_idx, old_idx, matrix[:, k]) for k in range(matrix.shape[1])])


def augment_trial(path: Path, rng: np.random.Generator) -> np.ndarray:
    frame = load_sensor_file(path)
    matrix = frame.as_processing_matrix()  # accel_g(3), gyro_dps(3), quaternion_wxyz(4)
    init_window = determine_initial_window(matrix)

    accel_baseline = matrix[:init_window, 0:3].mean(axis=0)
    gyro_baseline = matrix[:init_window, 3:6].mean(axis=0)
    accel_motion = matrix[:, 0:3] - accel_baseline
    gyro_motion = matrix[:, 3:6] - gyro_baseline
    quaternion = matrix[:, 6:10]

    n = matrix.shape[0]

    # Apply a one-time small-angle rotation to the whole trial, simulating a
    # slightly different sensor mounting angle. The trajectory pipeline
    # re-derives a canonical frame from gravity + displacement, so this is
    # invisible to most trajectory features -- but per-axis std features
    # (accel_*_std, gyro_*_std) are computed in raw sensor axes and do pick
    # up the change, adding realistic mounting-angle variability there.
    offset = random_small_rotation(rng)
    offset_n = np.tile(offset, (n, 1))
    accel_baseline = rotate_vectors(offset[None, :], accel_baseline[None, :])[0]
    gyro_baseline = rotate_vectors(offset[None, :], gyro_baseline[None, :])[0]
    accel_motion = rotate_vectors(offset_n, accel_motion)
    gyro_motion = rotate_vectors(offset_n, gyro_motion)
    quaternion = normalize(multiply(offset_n, quaternion))
    warp_factor = rng.uniform(*WARP_RANGE)
    new_n = max(40, int(round(n * warp_factor)))

    accel_motion_w = time_warp(accel_motion, new_n)
    gyro_motion_w = time_warp(gyro_motion, new_n)
    quaternion_w = normalize(time_warp(quaternion, new_n))

    gain = rng.uniform(*GAIN_RANGE)
    accel_motion_aug = accel_motion_w * gain + rng.normal(0.0, ACCEL_NOISE_STD_G, size=accel_motion_w.shape)
    gyro_motion_aug = gyro_motion_w * gain + rng.normal(0.0, GYRO_NOISE_STD_DPS, size=gyro_motion_w.shape)

    baseline_jitter = accel_baseline * rng.uniform(*BASELINE_JITTER_RANGE, size=3)
    accel_g_aug = baseline_jitter[None, :] + accel_motion_aug
    gyro_dps_aug = gyro_baseline[None, :] + gyro_motion_aug

    raw_accel_x = -accel_g_aug[:, 0] * ACCEL_SENSITIVITY_16G
    raw_accel_y = -accel_g_aug[:, 1] * ACCEL_SENSITIVITY_16G
    raw_accel_z = accel_g_aug[:, 2] * ACCEL_SENSITIVITY_16G
    raw_gyro = gyro_dps_aug * GYRO_SENSITIVITY
    raw_quat = quaternion_w * QUATERNION_SENSITIVITY

    base_time_ms = rng.integers(1_400_000_000_000, 1_700_000_000_000)
    receiver_time = base_time_ms + np.arange(new_n) * 5
    index = np.arange(new_n)

    return np.column_stack(
        [receiver_time, raw_accel_x, raw_accel_y, raw_accel_z, raw_gyro, raw_quat, index]
    )


def write_trial(path: Path, rows: np.ndarray) -> None:
    fmt = ["%d"] * 12
    np.savetxt(path, rows, delimiter=",", header=HEADER, comments="", fmt=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory containing real labeled recordings")
    parser.add_argument("--out-dir", default="synthetic_data", help="Directory to write augmented recordings")
    parser.add_argument("--variants-per-trial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_files = sorted(data_dir.glob("*.txt"))
    manifest_rows = []
    for real_path in real_files:
        match = re.match(r"(\d+)_", real_path.name)
        if not match:
            continue
        class_id = int(match.group(1))
        stem = real_path.stem
        for variant in range(args.variants_per_trial):
            rng = np.random.default_rng(args.seed * 1_000_000 + class_id * 10_000 + hash(stem) % 1000 * 100 + variant)
            rows = augment_trial(real_path, rng)
            file_name = f"{stem}_aug{variant:02d}.txt"
            write_trial(out_dir / file_name, rows)
            manifest_rows.append({"path": file_name, "label": class_id})

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Augmented {len(real_files)} real trials into {len(manifest_rows)} synthetic recordings.")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()

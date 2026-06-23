"""Generate synthetic AirInterface-format IMU recordings for pipeline testing.

This produces FAKE data shaped like real WMFT recordings so the training/
prediction pipeline can be exercised end to end before real sensor captures
are available. The motion parameters per class are not derived from real
biomechanics -- they only give each class a distinct, repeatable signature
so a classifier has something learnable. Replace this dataset with real
recordings before drawing any conclusions about classifier accuracy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ACCEL_SENSITIVITY_16G = 2048.0
GYRO_SENSITIVITY = 16.4
QUATERNION_SENSITIVITY = 1_073_741_824.0
SAMPLE_RATE_HZ = 200.0

HEADER = "Receiver time, Accel X,Y,Z, Gyro X,Y,Z, Quaternion W,X,Y,Z, dt(or)index"

# amplitude: peak displacement (x, y, z) in meters during the motion phase.
# mode: "transition" moves from rest to a new resting position and holds it;
#   "round_trip" moves out and back to the starting position.
# n_cycles: oscillation count within the motion envelope (>1 for repetitive tasks).
# rot_amp: peak synthetic gyro signal (deg/s) per axis, just for feature variety.
# motion_s: duration in seconds of the motion phase.
CLASS_PARAMS = {
    1: dict(amplitude=(0.25, 0.05, 0.05), mode="transition", n_cycles=1, rot_amp=(5, 5, 5), motion_s=1.5),
    2: dict(amplitude=(0.30, 0.05, 0.10), mode="transition", n_cycles=1, rot_amp=(5, 5, 5), motion_s=1.6),
    3: dict(amplitude=(0.35, 0.02, 0.02), mode="round_trip", n_cycles=1, rot_amp=(3, 3, 3), motion_s=1.8),
    4: dict(amplitude=(0.35, 0.02, 0.02), mode="round_trip", n_cycles=1, rot_amp=(3, 3, 3), motion_s=2.2),
    5: dict(amplitude=(0.20, 0.05, 0.04), mode="transition", n_cycles=1, rot_amp=(5, 5, 5), motion_s=1.4),
    6: dict(amplitude=(0.25, 0.05, 0.12), mode="transition", n_cycles=1, rot_amp=(5, 5, 5), motion_s=1.6),
    7: dict(amplitude=(0.30, 0.05, 0.15), mode="transition", n_cycles=1, rot_amp=(8, 8, 8), motion_s=2.0),
    8: dict(amplitude=(0.25, 0.20, 0.05), mode="round_trip", n_cycles=1, rot_amp=(10, 10, 10), motion_s=2.4),
    9: dict(amplitude=(0.15, 0.05, 0.10), mode="round_trip", n_cycles=1, rot_amp=(5, 5, 5), motion_s=1.8),
    10: dict(amplitude=(0.10, 0.03, 0.05), mode="round_trip", n_cycles=1, rot_amp=(15, 15, 5), motion_s=1.6),
    11: dict(amplitude=(0.08, 0.03, 0.04), mode="round_trip", n_cycles=1, rot_amp=(15, 15, 5), motion_s=1.6),
    12: dict(amplitude=(0.06, 0.02, 0.03), mode="round_trip", n_cycles=4, rot_amp=(10, 10, 10), motion_s=3.0),
    13: dict(amplitude=(0.05, 0.02, 0.02), mode="round_trip", n_cycles=5, rot_amp=(20, 20, 20), motion_s=3.0),
    14: dict(amplitude=(0.01, 0.01, 0.01), mode="round_trip", n_cycles=1, rot_amp=(2, 2, 2), motion_s=1.5),
    15: dict(amplitude=(0.05, 0.02, 0.02), mode="round_trip", n_cycles=2, rot_amp=(40, 10, 10), motion_s=2.0),
    16: dict(amplitude=(0.15, 0.10, 0.05), mode="round_trip", n_cycles=3, rot_amp=(15, 15, 10), motion_s=3.2),
    17: dict(amplitude=(0.30, 0.10, 0.20), mode="transition", n_cycles=1, rot_amp=(8, 8, 8), motion_s=2.2),
}

GRAVITY_VEC_G = np.array([0.0, 0.0, -1.0])
ACCEL_NOISE_STD_G = 0.01
GYRO_NOISE_STD_DPS = 1.0
REST_S = 1.0


def _position_profile(rng: np.random.Generator, params: dict) -> np.ndarray:
    motion_s = params["motion_s"] * rng.uniform(0.85, 1.15)
    rest_samples = int(round(REST_S * rng.uniform(0.8, 1.3) * SAMPLE_RATE_HZ))
    motion_samples = max(2, int(round(motion_s * SAMPLE_RATE_HZ)))

    amplitude = np.array(params["amplitude"]) * rng.uniform(0.8, 1.2, size=3)
    frac = np.linspace(0.0, 1.0, motion_samples)

    if params["mode"] == "round_trip":
        envelope = np.sin(np.pi * frac) ** 2
        phase = rng.uniform(0.0, 2 * np.pi)
        wave = np.sin(2 * np.pi * params["n_cycles"] * frac + phase)
        shape = envelope * wave
    else:
        shape = 10 * frac**3 - 15 * frac**4 + 6 * frac**5

    motion_position = shape[:, None] * amplitude[None, :]

    before = np.zeros((rest_samples, 3))
    after = np.tile(motion_position[-1], (rest_samples, 1))
    return np.concatenate([before, motion_position, after], axis=0)


def _rotation_profile(rng: np.random.Generator, params: dict, n_samples: int, motion_slice: slice) -> np.ndarray:
    rot_amp = np.array(params["rot_amp"]) * rng.uniform(0.8, 1.2, size=3)
    motion_samples = motion_slice.stop - motion_slice.start
    frac = np.linspace(0.0, 1.0, motion_samples)
    envelope = np.sin(np.pi * frac) ** 2
    phase = rng.uniform(0.0, 2 * np.pi)
    wave = np.sin(2 * np.pi * max(params["n_cycles"], 1) * frac + phase)

    gyro = np.zeros((n_samples, 3))
    gyro[motion_slice] = (envelope * wave)[:, None] * rot_amp[None, :]
    return gyro


def generate_trial(class_id: int, rng: np.random.Generator) -> np.ndarray:
    params = CLASS_PARAMS[class_id]
    position = _position_profile(rng, params)
    n_samples = position.shape[0]
    dt = 1.0 / SAMPLE_RATE_HZ

    velocity = np.gradient(position, dt, axis=0)
    accel_m_s2 = np.gradient(velocity, dt, axis=0)
    accel_motion_g = accel_m_s2 / 9.8

    rest_samples_guess = int(round(REST_S * SAMPLE_RATE_HZ))
    motion_slice = slice(rest_samples_guess, n_samples - rest_samples_guess)
    gyro_dps = _rotation_profile(rng, params, n_samples, motion_slice)
    gyro_dps += rng.normal(0.0, GYRO_NOISE_STD_DPS, size=gyro_dps.shape)

    accel_g = GRAVITY_VEC_G[None, :] + accel_motion_g + rng.normal(0.0, ACCEL_NOISE_STD_G, size=(n_samples, 3))

    raw_accel_x = -accel_g[:, 0] * ACCEL_SENSITIVITY_16G
    raw_accel_y = -accel_g[:, 1] * ACCEL_SENSITIVITY_16G
    raw_accel_z = accel_g[:, 2] * ACCEL_SENSITIVITY_16G
    raw_gyro = gyro_dps * GYRO_SENSITIVITY

    raw_quat = np.tile([QUATERNION_SENSITIVITY, 0.0, 0.0, 0.0], (n_samples, 1))

    base_time_ms = rng.integers(1_400_000_000_000, 1_700_000_000_000)
    receiver_time = base_time_ms + np.arange(n_samples) * 5
    index = np.arange(n_samples)

    return np.column_stack(
        [receiver_time, raw_accel_x, raw_accel_y, raw_accel_z, raw_gyro, raw_quat, index]
    )


def write_trial(path: Path, rows: np.ndarray) -> None:
    fmt = ["%d"] + ["%d"] * 10 + ["%d"]
    np.savetxt(path, rows, delimiter=",", header=HEADER, comments="", fmt=fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="synthetic_data", help="Directory to write recordings and manifest.csv")
    parser.add_argument("--trials-per-class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=sorted(CLASS_PARAMS.keys()),
        help="WMFT class IDs to generate (default: all defined classes 1-17)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for class_id in args.classes:
        if class_id not in CLASS_PARAMS:
            raise ValueError(f"No synthetic profile defined for class {class_id}")
        for trial in range(args.trials_per_class):
            rng = np.random.default_rng(args.seed * 100000 + class_id * 1000 + trial)
            rows = generate_trial(class_id, rng)
            file_name = f"class{class_id:02d}_trial{trial:02d}.txt"
            write_trial(out_dir / file_name, rows)
            manifest_rows.append({"path": file_name, "label": class_id})

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} synthetic recordings and manifest to {manifest_path}")


if __name__ == "__main__":
    main()

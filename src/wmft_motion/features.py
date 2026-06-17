"""Feature extraction for rule-based and trainable WMFT classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .constants import SAMPLE_RATE_HZ
from .io import load_sensor_file
from .preprocessing import process_motion


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    horizontal = np.sqrt(x**2 + y**2)
    elevation = np.arctan2(z, horizontal)
    return azimuth, elevation, radius


def average_power(values: np.ndarray, origin: float | None = None, negative_only: bool = False) -> float:
    if len(values) < 2:
        return 0.0
    origin_value = float(values[0] if origin is None else origin)
    changed = values[1:] != values[:-1]
    selected = values[1:][changed]
    if negative_only:
        selected = selected[selected < 0]
    if len(selected) == 0:
        return 0.0
    return float(np.sum(np.abs(selected - origin_value) ** 2) / len(values) * 100.0)


def acceleration_power(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    changed = values[1:] != values[:-1]
    selected = values[1:][changed]
    if len(selected) == 0:
        return 0.0
    return float(np.sum(np.abs(selected) ** 2) / len(values) * 100.0)


def extract_features(path: str | Path) -> dict[str, float]:
    """Extract trajectory, spectral, and descriptive features from one recording."""

    frame = load_sensor_file(path)
    processed = process_motion(frame.as_processing_matrix())
    trajectory = processed.trajectory_m
    corrected_accel = processed.corrected_accel_g

    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    ax, ay, az = corrected_accel[:, 0], corrected_accel[:, 1], corrected_accel[:, 2]
    azimuth, elevation, radius = cartesian_to_spherical(x, y, z)
    accel_azimuth, accel_elevation, accel_radius = cartesian_to_spherical(ax, ay, az)

    displacement = trajectory[-1] - trajectory[0]
    azimuth_diff, elevation_diff, radius_diff = cartesian_to_spherical(
        np.asarray([displacement[0]]),
        np.asarray([displacement[1]]),
        np.asarray([displacement[2]]),
    )

    features: dict[str, float] = {
        "sample_count": float(frame.sample_count),
        "duration_s": float(frame.sample_count / SAMPLE_RATE_HZ),
        "x_diff": float(displacement[0]),
        "y_diff": float(displacement[1]),
        "z_diff": float(displacement[2]),
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))),
        "x_power": average_power(x),
        "y_power": average_power(y),
        "z_power": average_power(z, negative_only=True),
        "azimuth_power": average_power(azimuth),
        "elevation_power": average_power(elevation),
        "radius_power": average_power(radius),
        "accel_radius_power": acceleration_power(accel_radius),
        "azimuth_diff": float(azimuth_diff[0]),
        "elevation_diff": float(elevation_diff[0]),
        "radius_diff": float(radius_diff[0]),
        "azimuth_var": float(np.var(azimuth)),
        "elevation_var": float(np.var(elevation)),
        "radius_var": float(np.var(radius)),
        "z_mean": float(np.mean(z)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "z_range": float(np.max(z) - np.min(z)),
        "z_end_minus_min_abs": float(abs(np.min(z) - z[-1])),
        "z_peaks_min_count": float(len(find_peaks(-z)[0])),
        "z_peaks_max_count": float(len(find_peaks(z)[0])),
        "azimuth_peaks_count": float(len(find_peaks(azimuth)[0])),
        "accel_x_std": float(np.std(ax)),
        "accel_y_std": float(np.std(ay)),
        "accel_z_std": float(np.std(az)),
        "gyro_x_std": float(np.std(frame.gyro_dps[:, 0])),
        "gyro_y_std": float(np.std(frame.gyro_dps[:, 1])),
        "gyro_z_std": float(np.std(frame.gyro_dps[:, 2])),
        "zero_velocity_window_count": float(len(processed.zero_velocity_windows)),
    }

    for axis_name, series in {
        "x": x,
        "y": y,
        "z": z,
        "azimuth": azimuth,
        "elevation": elevation,
        "radius": radius,
        "accel_azimuth": accel_azimuth,
        "accel_elevation": accel_elevation,
    }.items():
        features[f"{axis_name}_mean"] = float(np.mean(series))
        features[f"{axis_name}_std"] = float(np.std(series))
        features[f"{axis_name}_iqr"] = float(np.percentile(series, 75) - np.percentile(series, 25))

    return features

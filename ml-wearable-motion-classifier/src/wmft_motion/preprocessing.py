"""Trajectory reconstruction utilities for wrist IMU recordings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GRAVITY_M_S2, SAMPLE_RATE_HZ
from .quaternion import divide, normalize, rotate_vectors


@dataclass(frozen=True)
class ProcessedMotion:
    corrected_accel_g: np.ndarray
    trajectory_m: np.ndarray
    initial_accel_mean_g: np.ndarray
    zero_velocity_windows: np.ndarray


def determine_initial_window(data: np.ndarray, window_len: int = 20, threshold: float = 0.003) -> int:
    """Find the index where the recording leaves its initial stillness period."""

    doubled = 2 * window_len
    if len(data) <= doubled:
        return 1

    start = 0
    for start in range(0, len(data) - doubled, doubled + 1):
        window = data[start : start + doubled + 1, :3]
        acc_var = np.var(np.sum(window**2, axis=1))
        if acc_var >= threshold:
            break

    matlab_index = start + 1
    if matlab_index == 1:
        return 1
    return max(1, matlab_index - doubled // 2)


def subtract_gravity(data: np.ndarray, initial_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Align quaternions to the initial pose and subtract the initial gravity vector."""

    window = max(1, initial_window)
    accel = data[:, :3]
    q = normalize(data[:, 6:10])
    accel_mean = np.mean(accel[:window], axis=0)
    q_ref = normalize(np.mean(q[:window], axis=0, keepdims=True))[0]
    q_relative = divide(q, q_ref)
    corrected = rotate_vectors(q_relative, accel) - accel_mean
    return corrected, accel_mean


def process_motion(data: np.ndarray) -> ProcessedMotion:
    initial_window = determine_initial_window(data)
    corrected_accel, accel_mean = subtract_gravity(data, initial_window)
    extended = np.column_stack([data, corrected_accel])
    trajectory, windows = estimate_trajectory(corrected_accel, extended, accel_mean)
    return ProcessedMotion(
        corrected_accel_g=corrected_accel,
        trajectory_m=trajectory,
        initial_accel_mean_g=accel_mean,
        zero_velocity_windows=windows,
    )


def estimate_trajectory(
    corrected_accel_g: np.ndarray,
    data_with_corrected_accel: np.ndarray,
    accel_mean_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / SAMPLE_RATE_HZ
    velocity = np.cumsum(corrected_accel_g * GRAVITY_M_S2, axis=0) * dt
    velocity_corrected, zero_velocity_windows, first_motion_segment = zero_velocity_update(
        data_with_corrected_accel, velocity
    )
    trajectory = np.cumsum(velocity_corrected, axis=0) * dt
    if first_motion_segment is not None:
        trajectory = construct_frame(accel_mean_g, trajectory, first_motion_segment)
    return trajectory, zero_velocity_windows


def zero_velocity_update(data: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    corrected_accel = data[:, 10:13]
    gyro = data[:, 3:6]
    raw_accel = data[:, :3]
    windows = detect_zero_velocity_windows(corrected_accel, gyro, raw_accel)

    if len(windows) < 2:
        return velocity.copy(), windows, None

    first_motion_segment = (int(windows[0, 1]), int(windows[1, 0]))
    corrected = np.zeros_like(velocity)
    j = 0
    for i in range(len(velocity)):
        if i >= windows[j, 0] and i < windows[-1, 1]:
            while j + 1 < len(windows) and i > windows[j + 1, 1]:
                j += 1
            if j + 1 >= len(windows):
                break
            if windows[j, 1] <= i <= windows[j + 1, 0]:
                left = windows[j, 1]
                right = windows[j + 1, 0]
                denom = max(1, right - left)
                drift = (velocity[right] * (i - left) + velocity[left] * (right - i)) / denom
                corrected[i] = velocity[i] - drift
            else:
                corrected[i] = 0.0
    return corrected, windows, first_motion_segment


def detect_zero_velocity_windows(
    corrected_accel: np.ndarray,
    gyro: np.ndarray,
    raw_accel: np.ndarray,
    window_len: int = 40,
    thresholds: tuple[float, float, float] = (0.02, 10.0, np.inf),
) -> np.ndarray:
    acc_energy = np.sum(corrected_accel**2, axis=1)
    gyro_energy = np.sum(np.abs(gyro), axis=1)
    raw_accel_energy = np.abs(np.sum(raw_accel**2, axis=1) - 1.0)

    zero_points: set[int] = set()
    for i in range(len(acc_energy)):
        start = max(0, i - window_len)
        end = min(len(acc_energy), i + window_len + 1)
        if (
            np.mean(acc_energy[start:end]) < thresholds[0]
            and np.mean(gyro_energy[start:end]) < thresholds[1]
            and np.mean(raw_accel_energy[start:end]) < thresholds[2]
        ):
            zero_points.update(range(start, end))

    windows = merge_zero_velocity_points(sorted(zero_points))
    if len(windows) == 0:
        return windows

    adjusted = windows.copy()
    interior_lengths = []
    for i in range(1, len(adjusted) - 1):
        if adjusted[i, 1] - adjusted[i, 0] > 2 * window_len:
            adjusted[i, 0] += window_len
            adjusted[i, 1] -= window_len
        interior_lengths.append(adjusted[i, 1] - adjusted[i, 0])

    mean_len = int(round(np.mean(interior_lengths))) if interior_lengths else window_len
    if adjusted[0, 1] - window_len > adjusted[0, 0]:
        adjusted[0, 1] -= window_len
        if adjusted[0, 1] - mean_len > adjusted[0, 0]:
            adjusted[0, 0] = adjusted[0, 1] - mean_len
    if adjusted[-1, 0] + window_len < adjusted[-1, 1]:
        adjusted[-1, 0] += window_len
        if adjusted[-1, 0] + mean_len < adjusted[-1, 1]:
            adjusted[-1, 1] = adjusted[-1, 0] + mean_len
    return adjusted


def merge_zero_velocity_points(points: list[int]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=int)

    windows = []
    start = previous = points[0]
    for point in points[1:]:
        if point - previous > 1:
            windows.append((start, previous))
            start = point
        previous = point
    windows.append((start, previous))
    return np.asarray(windows, dtype=int)


def construct_frame(accel_mean_g: np.ndarray, position: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    first, second = center
    if first < 0 or second < 0 or first >= len(position) or second >= len(position) or first == second:
        return position

    gravity_norm = np.linalg.norm(accel_mean_g)
    if gravity_norm == 0:
        return position

    z_axis = accel_mean_g / gravity_norm
    displacement = position[second] - position[first]
    displacement_norm = np.linalg.norm(displacement)
    if displacement_norm == 0:
        return position

    x_axis = displacement / displacement_norm
    y_axis = np.cross(x_axis, z_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm == 0:
        return position

    y_axis /= y_norm
    x_axis = np.cross(z_axis, y_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.linalg.solve(np.vstack([x_axis, y_axis, z_axis]), np.eye(3))
    return position @ rotation

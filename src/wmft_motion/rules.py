"""Deterministic rule-based baseline classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import MOVEMENT_LABELS
from .features import extract_features


def classify_with_rules(path: str | Path) -> tuple[int, str, dict[str, float]]:
    features = extract_features(path)
    movement_id = classify_features_with_rules(features)
    return movement_id, MOVEMENT_LABELS[movement_id], features


def classify_features_with_rules(features: dict[str, float]) -> int:
    x_power_threshold = 0.001
    y_power_threshold = 0.002
    z_power_threshold_upper = 0.3
    z_power_threshold_lower = 0.1
    azimuth_power_threshold_1lb = 65
    elevation_var_threshold = 0.3
    radius_power_threshold_lower = 1.5
    radius_power_threshold_upper = 6

    z_power = features["z_power"]
    z_peaks_min = features["z_peaks_min_count"]
    azimuth_peaks = features["azimuth_peaks_count"]
    elevation_var = features["elevation_var"]
    radius_power = features["radius_power"]
    azimuth_diff = features["azimuth_diff"]
    elevation_diff = features["elevation_diff"]
    azimuth_var = features["azimuth_var"]
    x_power = features["x_power"]
    y_power = features["y_power"]
    azimuth_power = features["azimuth_power"]

    if z_power > z_power_threshold_upper:
        if z_peaks_min < 6 or azimuth_peaks < 5:
            if features["z_end_minus_min_abs"] > 0.02:
                if elevation_var > elevation_var_threshold:
                    return 6 if radius_power > radius_power_threshold_lower else 5
                if radius_power > radius_power_threshold_upper:
                    return 17
                ratio = azimuth_diff / elevation_diff if elevation_diff != 0 else np.inf
                if ratio < 0.5:
                    return 7
                if ratio < 0.66:
                    return 2
                return 1
            if z_power > 9:
                return 9
            if 3 < z_power < 5:
                return 10
            return 11
        if 0.3 < azimuth_var < 0.7:
            return 13
        if 0.7 < azimuth_var < 1:
            return 16
        return 12

    if z_power < z_power_threshold_lower:
        if x_power < x_power_threshold and y_power < y_power_threshold:
            return 14
        if azimuth_diff > 0:
            return 8
        if azimuth_power > azimuth_power_threshold_1lb:
            return 15 if z_peaks_min >= 3 else 4
        return 3

    return 18

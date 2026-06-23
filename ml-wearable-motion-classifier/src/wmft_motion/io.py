"""Input parsing for AirInterface MPU-9150 text captures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    ACCEL_SENSITIVITY_16G,
    GYRO_SENSITIVITY,
    QUATERNION_SENSITIVITY,
)


@dataclass(frozen=True)
class SensorFrame:
    """Scaled sensor arrays from one WMFT recording."""

    path: Path
    receiver_time: np.ndarray
    accel_g: np.ndarray
    gyro_dps: np.ndarray
    quaternion: np.ndarray
    index: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.accel_g.shape[0])

    def as_processing_matrix(self) -> np.ndarray:
        """Return processing columns: accelerometer, gyroscope, quaternion."""

        return np.column_stack([self.accel_g, self.gyro_dps, self.quaternion])


def load_sensor_file(path: str | Path) -> SensorFrame:
    """Load a raw AirInterface text file.

    The expected format is a header line followed by comma-separated rows:
    receiver time, accel xyz, gyro xyz, quaternion wxyz, dt/index.
    """

    file_path = Path(path)
    raw = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 12:
        raise ValueError(f"{file_path} has {raw.shape[1]} columns; expected at least 12")

    adjusted = raw.copy()
    adjusted[:, 1] *= -1.0
    adjusted[:, 2] *= -1.0

    return SensorFrame(
        path=file_path,
        receiver_time=adjusted[:, 0],
        accel_g=adjusted[:, 1:4] / ACCEL_SENSITIVITY_16G,
        gyro_dps=adjusted[:, 4:7] / GYRO_SENSITIVITY,
        quaternion=adjusted[:, 7:11] / QUATERNION_SENSITIVITY,
        index=adjusted[:, 11],
    )

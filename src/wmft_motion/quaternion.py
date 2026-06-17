"""Small quaternion helpers using WXYZ component order."""

from __future__ import annotations

import numpy as np


def normalize(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.divide(q, norms, out=np.zeros_like(q, dtype=float), where=norms != 0)


def conjugate(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for WXYZ quaternions."""

    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a / b for unit quaternions."""

    return multiply(a, conjugate(b))


def rotate_vectors(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate 3D vectors by WXYZ quaternions."""

    zeros = np.zeros((vectors.shape[0], 1), dtype=float)
    vector_quat = np.column_stack([zeros, vectors])
    rotated = multiply(multiply(q, vector_quat), conjugate(q))
    return rotated[:, 1:4]

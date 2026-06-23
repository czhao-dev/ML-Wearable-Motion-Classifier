import numpy as np

from wmft_motion.quaternion import multiply, normalize, rotate_vectors


def test_normalize_preserves_zero_quaternion_without_nan():
    result = normalize(np.asarray([[0.0, 0.0, 0.0, 0.0]]))

    np.testing.assert_allclose(result, [[0.0, 0.0, 0.0, 0.0]])


def test_multiply_identity_quaternion():
    q = np.asarray([[0.5, 0.5, -0.5, 0.5]])
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0]])

    np.testing.assert_allclose(multiply(identity, q), q)
    np.testing.assert_allclose(multiply(q, identity), q)


def test_rotate_vectors_around_z_axis():
    angle = np.pi / 2
    q = np.asarray([[np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]])
    vector = np.asarray([[1.0, 0.0, 0.0]])

    rotated = rotate_vectors(q, vector)

    np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-12)

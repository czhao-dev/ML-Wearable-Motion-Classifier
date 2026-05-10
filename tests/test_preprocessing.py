import numpy as np

from wmft_motion.preprocessing import (
    construct_frame,
    detect_zero_velocity_windows,
    determine_initial_window,
    merge_zero_velocity_points,
)


def test_determine_initial_window_returns_one_for_short_recordings():
    data = np.zeros((10, 10))

    assert determine_initial_window(data) == 1


def test_merge_zero_velocity_points_groups_contiguous_runs():
    points = [0, 1, 2, 5, 6, 10]

    windows = merge_zero_velocity_points(points)

    np.testing.assert_array_equal(windows, np.asarray([[0, 2], [5, 6], [10, 10]]))


def test_detect_zero_velocity_windows_for_stationary_signal():
    corrected_accel = np.zeros((100, 3))
    gyro = np.zeros((100, 3))
    raw_accel = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (100, 1))

    windows = detect_zero_velocity_windows(corrected_accel, gyro, raw_accel)

    assert windows.shape == (1, 2)
    assert windows[0, 0] <= windows[0, 1]


def test_construct_frame_returns_original_when_reference_is_degenerate():
    position = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    np.testing.assert_allclose(construct_frame(np.zeros(3), position, (0, 1)), position)
    np.testing.assert_allclose(construct_frame(np.asarray([0.0, 0.0, 1.0]), position, (0, 99)), position)

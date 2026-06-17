import numpy as np
import pytest

from wmft_motion.io import load_sensor_file


def test_load_sensor_file_scales_and_flips_expected_axes(tmp_path):
    sample = tmp_path / "capture.txt"
    sample.write_text(
        "Receiver time, Accel X,Y,Z, Gyro X,Y,Z, Quaternion W,X,Y,Z, dt(or)index\n"
        "100,2048,-4096,1024,16.4,-32.8,49.2,1073741824,536870912,0,-536870912,7\n"
    )

    frame = load_sensor_file(sample)

    assert frame.sample_count == 1
    np.testing.assert_allclose(frame.accel_g[0], [-1.0, 2.0, 0.5])
    np.testing.assert_allclose(frame.gyro_dps[0], [1.0, -2.0, 3.0])
    np.testing.assert_allclose(frame.quaternion[0], [1.0, 0.5, 0.0, -0.5])
    assert frame.receiver_time[0] == 100
    assert frame.index[0] == 7


def test_load_sensor_file_rejects_too_few_columns(tmp_path):
    sample = tmp_path / "bad_capture.txt"
    sample.write_text("header\n1,2,3\n")

    with pytest.raises(ValueError, match="expected at least 12"):
        load_sensor_file(sample)

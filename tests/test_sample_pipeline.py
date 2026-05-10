from pathlib import Path

from wmft_motion import classify_with_rules, extract_features, load_sensor_file


ROOT = Path(__file__).resolve().parents[1]


def test_sample_file_loads():
    frame = load_sensor_file(ROOT / "sample_data.txt")

    assert frame.sample_count == 1592
    assert frame.accel_g.shape == (1592, 3)
    assert frame.gyro_dps.shape == (1592, 3)
    assert frame.quaternion.shape == (1592, 4)


def test_sample_features_and_rule_prediction():
    features = extract_features(ROOT / "sample_data.txt")
    movement_id, label, returned_features = classify_with_rules(ROOT / "sample_data.txt")

    assert "z_power" in features
    assert "path_length_m" in features
    assert 1 <= movement_id <= 18
    assert label
    assert returned_features["sample_count"] == features["sample_count"]

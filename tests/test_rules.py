from wmft_motion.rules import classify_features_with_rules


def base_features(**overrides):
    features = {
        "azimuth_diff": -1.0,
        "azimuth_peaks_count": 0.0,
        "azimuth_power": 0.0,
        "azimuth_var": 0.0,
        "elevation_diff": 1.0,
        "elevation_var": 0.0,
        "radius_power": 0.0,
        "x_power": 1.0,
        "y_power": 1.0,
        "z_end_minus_min_abs": 0.0,
        "z_peaks_min_count": 0.0,
        "z_power": 0.2,
    }
    features.update(overrides)
    return features


def test_rule_classifier_returns_undefined_for_middle_z_power():
    assert classify_features_with_rules(base_features()) == 18


def test_rule_classifier_detects_grip_strength_for_low_motion_power():
    assert classify_features_with_rules(base_features(z_power=0.0, x_power=0.0, y_power=0.0)) == 14


def test_rule_classifier_detects_lift_basket_for_large_radius_power():
    features = base_features(
        z_power=1.0,
        z_end_minus_min_abs=0.1,
        radius_power=7.0,
        elevation_var=0.0,
    )

    assert classify_features_with_rules(features) == 17


def test_rule_classifier_detects_repeated_azimuth_motion():
    features = base_features(
        z_power=1.0,
        z_peaks_min_count=6.0,
        azimuth_peaks_count=5.0,
        azimuth_var=0.5,
    )

    assert classify_features_with_rules(features) == 13

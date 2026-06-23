"""Small smoke test for a local development environment with dependencies installed."""

from __future__ import annotations

from pathlib import Path

from wmft_motion import classify_with_rules, extract_features, load_sensor_file


def main() -> None:
    sample = Path(__file__).resolve().parents[1] / "sample_data.txt"
    frame = load_sensor_file(sample)
    features = extract_features(sample)
    movement_id, label, _ = classify_with_rules(sample)

    print(f"samples={frame.sample_count}")
    print(f"features={len(features)}")
    print(f"rule_prediction={movement_id},{label}")


if __name__ == "__main__":
    main()

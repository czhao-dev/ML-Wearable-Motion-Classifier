# Upper Body Motion Classifier

This project classifies upper-body movements from wrist-worn IMU recordings collected during the Wolf Motor Function Test (WMFT). It provides a Python package for signal processing, feature extraction, rule-based baseline classification, and trainable machine-learning models.

The system is intended for research and prototyping in rehabilitation monitoring. It is not approved for clinical use.

## What It Does

- Reads AirInterface MPU-9150 text captures with accelerometer, gyroscope, and quaternion columns.
- Aligns the sensor frame, subtracts gravity, applies zero-velocity updates, and reconstructs wrist trajectory.
- Extracts trajectory and motion-shape features such as vertical power, azimuth rotation, peak counts, path length, variance, and acceleration statistics.
- Provides a deterministic rule-based baseline for quick classification before a trained dataset is available.
- Trains more advanced tabular ML classifiers using scikit-learn: SVM, random forest, histogram gradient boosting, or a soft-voting ensemble.

## Repository Layout

```text
.
├── LICENSE
├── README.md
├── pyproject.toml                # Python package and dependencies
├── requirements.txt              # Runtime dependency list
├── sample_data.txt               # Example raw IMU capture
├── examples/
│   └── manifest.example.csv      # Training manifest template
├── scripts/
│   └── smoke_test.py             # Quick local pipeline smoke test
├── src/wmft_motion/
│   ├── __init__.py               # Public package exports
│   ├── cli.py                    # wmft-motion command line interface
│   ├── constants.py              # WMFT labels and sensor constants
│   ├── features.py               # Feature extraction
│   ├── io.py                     # Sensor text parser
│   ├── models.py                 # ML training and prediction
│   ├── preprocessing.py          # Gravity correction, ZUPT, trajectory
│   ├── quaternion.py             # Quaternion math
│   └── rules.py                  # Rule-based baseline classifier
└── tests/
    ├── test_cli.py               # Command line interface tests
    ├── test_io.py                # Sensor parser tests
    ├── test_models.py            # ML utility regression tests
    ├── test_preprocessing.py     # Trajectory preprocessing tests
    ├── test_quaternion.py        # Quaternion math tests
    ├── test_rules.py             # Rule-based classifier tests
    └── test_sample_pipeline.py   # Sample data pipeline tests
```

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Use The Rule-Based Baseline

The baseline uses hand-tuned trajectory thresholds and is useful before you have enough labeled training data.

```bash
wmft-motion classify-rule sample_data.txt
```

## Extract Features

```bash
wmft-motion extract-features sample_data.txt --json
```

## Train A Machine-Learning Model

Create a labeled manifest CSV with two columns:

```csv
path,label
../recordings/subject01_trial01.txt,1
../recordings/subject01_trial02.txt,WMFT 8: Reach and retrieve
```

Then build features and train:

```bash
wmft-motion build-features examples/manifest.csv features.csv
wmft-motion train features.csv wmft_model.joblib --kind ensemble --evaluate
```

You can also train directly from a manifest:

```bash
wmft-motion train examples/manifest.csv wmft_model.joblib --from-manifest --kind random_forest
```

## Predict With A Trained Model

```bash
wmft-motion predict wmft_model.joblib sample_data.txt
```

## Data Notes

The raw input format is expected to match `sample_data.txt`:

```text
Receiver time, Accel X,Y,Z, Gyro X,Y,Z, Quaternion W,X,Y,Z, dt(or)index
```

The Python pipeline uses these sensor assumptions:

- Accelerometer: 2048 counts per g for 16g mode
- Gyroscope: 16.4 counts per degree/second
- Quaternion: 1073741824 fixed-point scale
- Sampling rate assumption: 200 Hz

## Modeling Direction

The rule-based baseline is intentionally simple and deterministic. The trainable model path can improve once you collect labeled examples across users, impairment levels, and repeated trials.

For small to medium datasets, start with `--kind ensemble` or `--kind random_forest`. For larger datasets with many repeated trials, the next upgrade would be a sequence model over sliding windows, such as a temporal convolutional network or transformer encoder.

## References

- Wolf Motor Function Test Manual - UAB CI Therapy Research Group
- InvenSense MPU-9150 Product Specification
- Wolfram MathWorld: Spherical Coordinates

## License

Apache License 2.0.

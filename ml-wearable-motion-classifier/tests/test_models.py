from pathlib import Path

import pandas as pd
import pytest

from wmft_motion.features import extract_features
from wmft_motion.models import evaluate_with_cross_validation, predict_files, split_features_and_labels, train_model


ROOT = Path(__file__).resolve().parents[1]


def test_split_features_requires_label_column():
    with pytest.raises(ValueError, match="label"):
        split_features_and_labels(pd.DataFrame({"feature": [1.0, 2.0]}))


def test_train_model_requires_two_classes(tmp_path):
    table = pd.DataFrame({"feature": [1.0, 2.0], "label": ["same", "same"]})

    with pytest.raises(ValueError, match="At least two classes"):
        train_model(table, tmp_path / "model.joblib", kind="random_forest")


def test_cross_validation_requires_two_samples_per_class():
    table = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "label": ["a", "a", "b"]})

    with pytest.raises(ValueError, match="two samples"):
        evaluate_with_cross_validation(table, kind="svm")


def test_cross_validation_requires_at_least_two_folds():
    table = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "label": ["a", "a", "b", "b"]})

    with pytest.raises(ValueError, match="two folds"):
        evaluate_with_cross_validation(table, kind="random_forest", folds=1)


def test_predict_files_accepts_empty_path_list(tmp_path):
    table = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "label": ["a", "a", "b", "b"]})
    model_path = tmp_path / "model.joblib"
    train_model(table, model_path, kind="random_forest")

    result = predict_files(model_path, [])

    assert list(result.columns) == ["path", "prediction", "confidence"]
    assert result.empty


def test_train_and_predict_with_extracted_feature_columns(tmp_path):
    sample_features = extract_features(ROOT / "sample_data.txt")
    table = pd.DataFrame(
        [
            {**sample_features, "path": "sample-a", "label": "a"},
            {**sample_features, "path": "sample-b", "label": "b", "x_power": sample_features["x_power"] + 1.0},
        ]
    )
    model_path = tmp_path / "model.joblib"

    train_model(table, model_path, kind="random_forest")
    result = predict_files(model_path, [ROOT / "sample_data.txt"])

    assert result.loc[0, "path"].endswith("sample_data.txt")
    assert result.loc[0, "prediction"] in {"a", "b"}
    assert 0.0 <= result.loc[0, "confidence"] <= 1.0


def test_cross_validation_success_returns_report():
    table = pd.DataFrame({"feature": [1.0, 1.1, 9.0, 9.1], "label": ["a", "a", "b", "b"]})

    report = evaluate_with_cross_validation(table, kind="random_forest", folds=2)

    assert "precision" in report
    assert "Confusion matrix" in report

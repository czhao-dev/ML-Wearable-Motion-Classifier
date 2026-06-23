"""Trainable ML models for WMFT feature tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import extract_features


def build_feature_table(manifest_path: str | Path) -> pd.DataFrame:
    """Build one feature row per labeled recording.

    The manifest must contain `path` and `label` columns. Labels may be WMFT
    integer IDs or human-readable class names.
    """

    manifest = pd.read_csv(manifest_path)
    required = {"path", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    base_dir = Path(manifest_path).resolve().parent
    rows = []
    for _, row in manifest.iterrows():
        sample_path = Path(row["path"])
        if not sample_path.is_absolute():
            sample_path = base_dir / sample_path
        features = extract_features(sample_path)
        features["path"] = str(sample_path)
        features["label"] = row["label"]
        rows.append(features)
    return pd.DataFrame(rows)


def split_features_and_labels(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "label" not in table.columns:
        raise ValueError("Feature table must contain a 'label' column")

    drop_columns = [col for col in ["label", "path"] if col in table.columns]
    features = table.drop(columns=drop_columns)
    if features.empty:
        raise ValueError("Feature table must contain at least one feature column")
    return features, table["label"]


def validate_labels(y: pd.Series, *, require_cross_validation: bool = False) -> None:
    if y.isna().any():
        raise ValueError("Labels must not contain missing values")

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two classes are required to train or evaluate a classifier")

    if require_cross_validation and int(class_counts.min()) < 2:
        raise ValueError("Cross-validation requires at least two samples in every class")


def make_classifier(kind: str = "ensemble", random_state: int = 42) -> BaseEstimator:
    """Create a modern tabular classifier for extracted motion features."""

    svm = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=10.0, gamma="scale", probability=True, class_weight="balanced")),
        ]
    )
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
            ),
        ]
    )
    gradient_boosting = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(random_state=random_state)),
        ]
    )

    if kind == "svm":
        return svm
    if kind == "random_forest":
        return forest
    if kind == "gradient_boosting":
        return gradient_boosting
    if kind != "ensemble":
        raise ValueError("kind must be one of: ensemble, svm, random_forest, gradient_boosting")

    return VotingClassifier(
        estimators=[
            ("rf", forest),
            ("hgb", gradient_boosting),
            ("svm", svm),
        ],
        voting="soft",
    )


def train_model(
    feature_table: pd.DataFrame,
    model_path: str | Path,
    kind: str = "ensemble",
    random_state: int = 42,
) -> BaseEstimator:
    x, y = split_features_and_labels(feature_table)
    validate_labels(y)
    model = make_classifier(kind=kind, random_state=random_state)
    model.fit(x, y)
    artifact = {
        "model": model,
        "feature_columns": list(x.columns),
        "label_values": sorted(map(str, pd.Series(y).unique())),
    }
    joblib.dump(artifact, model_path)
    return model


def evaluate_with_cross_validation(
    feature_table: pd.DataFrame,
    kind: str = "ensemble",
    folds: int = 5,
    random_state: int = 42,
) -> str:
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds")

    x, y = split_features_and_labels(feature_table)
    validate_labels(y, require_cross_validation=True)
    min_class_count = int(pd.Series(y).value_counts().min())
    actual_folds = min(folds, min_class_count)
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
    predictions = cross_val_predict(make_classifier(kind=kind, random_state=random_state), x, y, cv=cv)
    labels = sorted(pd.Series(y).unique(), key=str)
    report = classification_report(y, predictions, zero_division=0)
    matrix = confusion_matrix(y, predictions, labels=labels)
    return f"{report}\nConfusion matrix:\n{matrix}"


def load_model(model_path: str | Path) -> dict:
    return joblib.load(model_path)


def predict_files(model_path: str | Path, paths: Iterable[str | Path]) -> pd.DataFrame:
    artifact = load_model(model_path)
    rows = []
    for path in paths:
        features = extract_features(path)
        rows.append({"path": str(path), **features})
    if not rows:
        return pd.DataFrame(columns=["path", "prediction", "confidence"])

    table = pd.DataFrame(rows)
    x = table[artifact["feature_columns"]]
    predictions = artifact["model"].predict(x)
    result = table[["path"]].copy()
    result["prediction"] = predictions
    if hasattr(artifact["model"], "predict_proba"):
        result["confidence"] = np.max(artifact["model"].predict_proba(x), axis=1)
    return result

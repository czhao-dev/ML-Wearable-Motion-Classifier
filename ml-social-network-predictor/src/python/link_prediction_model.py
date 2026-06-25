"""Link-prediction classifiers trained on engineered features, with an
explicit comparison between the pre-existing graph heuristics
(embeddedness/dispersion/common-neighbors/community) and the learned
embedding features."""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import EMBEDDING_FEATURE_COLUMNS, HEURISTIC_FEATURE_COLUMNS

FEATURE_SETS = {
    "heuristics_only": HEURISTIC_FEATURE_COLUMNS,
    "embeddings_only": EMBEDDING_FEATURE_COLUMNS,
    "combined": HEURISTIC_FEATURE_COLUMNS + EMBEDDING_FEATURE_COLUMNS,
}


def train_classifier(train_df, feature_columns, model="logistic"):
    if model == "logistic":
        # Heuristic feature scales vary by orders of magnitude (e.g. dispersion
        # up to ~37,000 vs. jaccard_coefficient in [0, 1]), which stalls
        # LogisticRegression's solver well short of convergence without scaling.
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif model == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier()
    else:
        raise ValueError(f"Unknown model: {model}")
    clf.fit(train_df[feature_columns], train_df["label"])
    return clf


def evaluate_classifier(model, test_df, feature_columns):
    y_true = test_df["label"]
    y_proba = model.predict_proba(test_df[feature_columns])[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "y_true": y_true,
        "y_proba": y_proba,
    }


def run_comparison(train_df, test_df):
    rows = []
    curves = {}
    for feature_set, columns in FEATURE_SETS.items():
        model = train_classifier(train_df, columns)
        metrics = evaluate_classifier(model, test_df, columns)
        rows.append({
            "feature_set": feature_set,
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
        curves[feature_set] = (metrics["y_true"], metrics["y_proba"])

    import pandas as pd
    return pd.DataFrame(rows), curves


def plot_roc_curve(curves, path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for feature_set, (y_true, y_proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, label=feature_set)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Link Prediction ROC Curve")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def save_classifier(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

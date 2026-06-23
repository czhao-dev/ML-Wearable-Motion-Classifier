"""Group-aware cross-validation for feature tables containing augmented variants.

models.evaluate_with_cross_validation uses plain StratifiedKFold, which splits
by row. When a feature table contains multiple augmented variants of the same
source recording, that scatters near-duplicate siblings across train and test
folds, so the model partly memorizes the source trial instead of learning the
movement class -- accuracy comes back unrealistically high.

This script instead groups rows by source recording (stripping any "_augNN"
suffix from the file stem) and uses StratifiedGroupKFold, so every variant of
a given real trial stays on the same side of the split. The reported accuracy
is the honest one to look at when deciding if an augmented dataset is useful.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from wmft_motion.models import make_classifier, split_features_and_labels


def source_group(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"_aug\d+$", "", stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv")
    parser.add_argument("--kind", default="ensemble", choices=["ensemble", "svm", "random_forest", "gradient_boosting"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    table = pd.read_csv(args.features_csv)
    table = table.assign(group=table["path"].map(source_group))
    n_unique_groups_per_class = table.groupby("label")["group"].nunique()

    keep_labels = n_unique_groups_per_class[n_unique_groups_per_class >= 2].index
    excluded = sorted(set(table["label"]) - set(keep_labels))
    if excluded:
        print(f"Excluding classes with fewer than 2 distinct source recordings: {excluded}")
    table = table[table["label"].isin(keep_labels)]

    actual_folds = min(args.folds, int(n_unique_groups_per_class[keep_labels].min()))
    groups = table["group"]

    x, y = split_features_and_labels(table.drop(columns=["path", "group"]))
    cv = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=args.seed)
    predictions = cross_val_predict(make_classifier(kind=args.kind, random_state=args.seed), x, y, cv=cv, groups=groups)

    labels = sorted(pd.Series(y).unique(), key=str)
    print(f"Grouped {actual_folds}-fold CV over {groups.nunique()} source recordings ({len(table)} rows)")
    print(classification_report(y, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=labels))


if __name__ == "__main__":
    main()

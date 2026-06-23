"""Command line interface for the WMFT motion classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import extract_features
from .models import (
    build_feature_table,
    evaluate_with_cross_validation,
    predict_files,
    train_model,
)
from .rules import classify_with_rules


def main() -> None:
    parser = argparse.ArgumentParser(prog="wmft-motion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rule_parser = subparsers.add_parser("classify-rule", help="Classify one recording with the rule-based baseline")
    rule_parser.add_argument("file")

    features_parser = subparsers.add_parser("extract-features", help="Extract features for one recording")
    features_parser.add_argument("file")
    features_parser.add_argument("--json", action="store_true", help="Print JSON instead of CSV-like key/value output")

    build_parser = subparsers.add_parser("build-features", help="Build a feature CSV from a labeled manifest")
    build_parser.add_argument("manifest")
    build_parser.add_argument("output_csv")

    train_parser = subparsers.add_parser("train", help="Train a model from a feature CSV or labeled manifest")
    train_parser.add_argument("input")
    train_parser.add_argument("model_output")
    train_parser.add_argument("--kind", default="ensemble", choices=["ensemble", "svm", "random_forest", "gradient_boosting"])
    train_parser.add_argument("--from-manifest", action="store_true", help="Treat input as a path,label manifest")
    train_parser.add_argument("--evaluate", action="store_true", help="Run cross-validation before final training")

    predict_parser = subparsers.add_parser("predict", help="Predict WMFT labels for one or more recordings")
    predict_parser.add_argument("model")
    predict_parser.add_argument("files", nargs="+")

    args = parser.parse_args()

    if args.command == "classify-rule":
        movement_id, label, _ = classify_with_rules(args.file)
        print(f"{movement_id},{label}")
        return

    if args.command == "extract-features":
        features = extract_features(args.file)
        if args.json:
            print(json.dumps(features, indent=2, sort_keys=True))
        else:
            for key, value in sorted(features.items()):
                print(f"{key},{value}")
        return

    if args.command == "build-features":
        table = build_feature_table(args.manifest)
        table.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(table)} rows to {args.output_csv}")
        return

    if args.command == "train":
        import pandas as pd

        table = build_feature_table(args.input) if args.from_manifest else pd.read_csv(args.input)
        if args.evaluate:
            print(evaluate_with_cross_validation(table, kind=args.kind))
        train_model(table, args.model_output, kind=args.kind)
        print(f"Wrote model to {args.model_output}")
        return

    if args.command == "predict":
        result = predict_files(args.model, [Path(path) for path in args.files])
        print(result.to_csv(index=False))
        return


if __name__ == "__main__":
    main()

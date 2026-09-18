from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from rail_cdm.features import extract_feature_table
from rail_cdm.io import read_labels, validate_training_inventory

RANDOM_SEED = 42


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    min_samples_leaf=1,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and validate the Rail baseline.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--feature-cache", type=Path, default=Path("data/processed/rail_features.csv")
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = read_labels(args.labels)
    paths = validate_training_inventory(args.train_dir, labels)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)

    if args.feature_cache.exists() and not args.rebuild_features:
        print(f"Loading cached features: {args.feature_cache}")
        features = pd.read_csv(args.feature_cache)
    else:
        features = extract_feature_table(paths)
        features.to_csv(args.feature_cache, index=False)
        print(f"Saved feature cache: {args.feature_cache}")

    dataset = labels.merge(features, on="filename", how="inner", validate="one_to_one")
    if len(dataset) != len(labels):
        raise ValueError("Some labels did not match extracted features.")

    feature_columns = [column for column in features.columns if column != "filename"]
    x = dataset[feature_columns]
    y = dataset["label"]

    smallest_class = int(y.value_counts().min())
    folds = min(5, smallest_class)
    if folds < 2:
        raise ValueError("At least two files are required in every class for cross-validation.")

    model = make_model()
    cross_validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    validation_predictions = cross_val_predict(model, x, y, cv=cross_validation, n_jobs=-1)

    labels_in_order = ["Normal", "Side I", "Side II"]
    macro_f1 = float(f1_score(y, validation_predictions, average="macro"))
    report = classification_report(
        y, validation_predictions, labels=labels_in_order, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y, validation_predictions, labels=labels_in_order).tolist()
    metrics = {
        "cross_validation_folds": folds,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix_labels": labels_in_order,
        "confusion_matrix": matrix,
    }

    print(f"\nCross-validated macro F1: {macro_f1:.4f}\n")
    print(classification_report(y, validation_predictions, labels=labels_in_order, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted)")
    print(pd.DataFrame(matrix, index=labels_in_order, columns=labels_in_order))

    validation_frame = dataset[["filename", "label"]].copy()
    validation_frame["prediction"] = validation_predictions
    validation_frame.to_csv(args.artifact_dir / "cross_validation_predictions.csv", index=False)
    (args.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    model.fit(x, y)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "allowed_labels": labels_in_order,
        "feature_version": 1,
    }
    model_path = args.artifact_dir / "rail_model.joblib"
    joblib.dump(bundle, model_path)
    print(f"\nSaved final model: {model_path}")


if __name__ == "__main__":
    main()

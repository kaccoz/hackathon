from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from rail_cdm.calibration import ProbabilityAdjustedClassifier
from rail_cdm.features import FEATURE_VERSION, extract_feature_table
from rail_cdm.io import read_labels, validate_training_inventory

RANDOM_SEED = 42
SIDE_I_PROBABILITY_MULTIPLIER = 1.5
SMOTE_TARGET_PER_FAULT_CLASS = 48
SMOTE_NEIGHBORS = 2


def make_model(
    *,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    max_features: str | float = "sqrt",
    class_weight: str | dict[str, float] | None = "balanced",
    side_i_multiplier: float = SIDE_I_PROBABILITY_MULTIPLIER,
    random_state: int = RANDOM_SEED,
    **random_forest_options: Any,
) -> ProbabilityAdjustedClassifier:
    base_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight=class_weight,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    n_jobs=-1,
                    random_state=random_state,
                    **random_forest_options,
                ),
            ),
        ]
    )
    return ProbabilityAdjustedClassifier(
        estimator=base_model,
        class_multipliers=(("Side I", side_i_multiplier),),
    )


def make_final_model() -> ImbalancedPipeline:
    """Build the promoted model with fold-safe moderate SMOTE oversampling."""
    return ImbalancedPipeline(
        [
            (
                "sampler",
                SMOTE(
                    sampling_strategy={
                        "Side I": SMOTE_TARGET_PER_FAULT_CLASS,
                        "Side II": SMOTE_TARGET_PER_FAULT_CLASS,
                    },
                    k_neighbors=SMOTE_NEIGHBORS,
                    random_state=RANDOM_SEED,
                ),
            ),
            ("model", make_model(min_samples_split=4, class_weight=None)),
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

    model = make_final_model()
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
        "decision_adjustment": {"Side I": SIDE_I_PROBABILITY_MULTIPLIER},
        "training_configuration": {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_split": 4,
            "smote_target_per_fault_class": SMOTE_TARGET_PER_FAULT_CLASS,
            "smote_neighbors": SMOTE_NEIGHBORS,
        },
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
        "feature_version": FEATURE_VERSION,
    }
    model_path = args.artifact_dir / "rail_model.joblib"
    joblib.dump(bundle, model_path)
    print(f"\nSaved final model: {model_path}")


if __name__ == "__main__":
    main()

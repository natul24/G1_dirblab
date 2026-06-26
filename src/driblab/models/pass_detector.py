"""Train a simple XGBoost pass-detection model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
)

from driblab.config import (
    ARTIFACTS_DIR,
    MODEL_BASE_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    project_path,
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".matplotlib_cache"),
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = [
    # Existing engineered features
    "ball_speed_avg_xy",
    "closest_player_team_id",

    # Raw ball/context features
    "t.ball_x",
    "t.ball_y",
    "t.ball_z",
    "t.player_count",
    "t.visible_player_count",

    # New engineered features
    "closest_player_distance_to_ball",
    "ball_position_missing",
    "ball_z_missing",
    "ball_speed_missing",
    "closest_player_missing",
    "ball_direction_x_5f",
    "ball_direction_y_5f",
    "ball_distance_5f",
    "ball_acceleration_xy",
    "closest_player_distance_change",
    "closest_player_changed",
    "closest_team_changed",
]
TARGET = "is_pass"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = REPORTS_DIR
FIGURE_DIR = REPORTS_DIR / "figures"


def load_training_tables(
    input_dir: Path = MODEL_BASE_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test training tables."""
    input_dir = input_dir.expanduser().resolve()
    train_df = pd.read_parquet(input_dir / "training_table_train.parquet")
    validation_df = pd.read_parquet(
        input_dir / "training_table_validation.parquet",
    )
    test_df = pd.read_parquet(input_dir / "training_table_test.parquet")
    return train_df, validation_df, test_df


def _prepare_matrix(
    table: pd.DataFrame,
    features: list[str] = FEATURES,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series, xgb.DMatrix]:
    missing = [column for column in [*features, target] if column not in table]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")

    encoded = {}
    for col in features:
        series = table[col]
        if pd.api.types.is_numeric_dtype(series):
            encoded[col] = series.to_numpy(dtype=float, na_value=np.nan)
        else:
            codes = pd.Categorical(series).codes.astype(float)
            codes[codes == -1] = np.nan
            encoded[col] = codes
    x_values = pd.DataFrame(encoded, index=table.index).fillna(0)
    y_values = table[target].astype(int)
    dmatrix = xgb.DMatrix(
        x_values,
        label=y_values,
        feature_names=features,
    )
    return x_values, y_values, dmatrix


def train_pass_detector(
    input_dir: Path = MODEL_BASE_DATA_DIR,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
    figure_dir: Path = FIGURE_DIR,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
) -> dict[str, Any]:
    """Train, evaluate, and save the XGBoost pass detector."""
    train_df, validation_df, test_df = load_training_tables(input_dir)
    _, y_train, dtrain = _prepare_matrix(train_df)
    _, y_validation, dvalidation = _prepare_matrix(validation_df)
    _, y_test, dtest = _prepare_matrix(test_df)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )

    best_threshold, threshold_results = find_best_f1_threshold(
        model=model,
        dmatrix=dvalidation,
        y_true=y_validation,
        start=0.0,
        stop=0.5,
        step=0.05,
    )

    print("\nValidation F1 by threshold:")
    print(
        threshold_results[
            ["threshold", "accuracy", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
        ].to_string(index=False)
    )

    print("\nBest validation threshold:")
    print(f"  Threshold: {best_threshold:.2f}")
    print(
        f"  Validation F1: "
        f"{threshold_results.loc[threshold_results['threshold'] == best_threshold, 'f1'].iloc[0]:.4f}"
    )

    metrics = {
        "train": evaluate_split(model, dtrain, y_train, threshold=best_threshold),
        "validation": evaluate_split(
         model,
            dvalidation,
            y_validation,
            threshold=best_threshold,
        ),
        "test": evaluate_split(model, dtest, y_test, threshold=best_threshold),
    }
    
    predictions = {
        "train": {"y_true": y_train, "y_proba": model.predict(dtrain)},
        "validation": {
            "y_true": y_validation,
            "y_proba": model.predict(dvalidation),
        },
        "test": {"y_true": y_test, "y_proba": model.predict(dtest)},
    }
    importance = feature_importance(model)

    model_dir = model_dir.expanduser().resolve()
    report_dir = report_dir.expanduser().resolve()
    figure_dir = figure_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pass_detector.json"
    metadata_path = model_dir / "pass_detector_metadata.json"
    encoders_path = model_dir / "feature_encoders.pkl"
    report_paths = save_reports(
        report_dir,
        figure_dir,
        metrics,
        importance,
        predictions,
    )

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "threshold": best_threshold,
        "params": params,
        "best_iteration": int(model.best_iteration),
    }

    model.save_model(model_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
    )
    joblib.dump({}, encoders_path)

    print(f"\nBest round: {model.best_iteration + 1}")
    _print_metrics(metrics)
    print("\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))
    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved feature encoders: {encoders_path}")
    for report_name, report_path in report_paths.items():
        print(f"Saved {report_name}: {report_path}")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "feature_encoders_path": str(encoders_path),
        "report_paths": {
            report_name: str(report_path)
            for report_name, report_path in report_paths.items()
        },
        "metrics": metrics,
    }


def evaluate_split(
    model: xgb.Booster,
    dmatrix: xgb.DMatrix,
    y_true: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one split with a chosen probability threshold."""
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

def find_best_f1_threshold(
    model: xgb.Booster,
    dmatrix: xgb.DMatrix,
    y_true: pd.Series,
    start: float = 0.0,
    stop: float = 0.5,
    step: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """Test thresholds and return the one with the best validation F1."""
    rows = []

    thresholds = np.arange(start, stop + step, step)

    for threshold in thresholds:
        threshold = round(float(threshold), 2)
        metrics = evaluate_split(
            model=model,
            dmatrix=dmatrix,
            y_true=y_true,
            threshold=threshold,
        )

        rows.append(metrics)

    threshold_results = pd.DataFrame(rows)

    best_row = threshold_results.sort_values(
        ["f1", "accuracy"],
        ascending=False,
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    return best_threshold, threshold_results

def feature_importance(model: xgb.Booster) -> pd.DataFrame:
    """Return feature importance sorted by split count."""
    importance = model.get_score(importance_type="weight")
    rows = [
        {"feature": feature, "importance": importance.get(feature, 0)}
        for feature in FEATURES
    ]
    return pd.DataFrame(rows).sort_values(
        "importance",
        ascending=False,
    )


def save_reports(
    report_dir: Path,
    figure_dir: Path,
    metrics: dict[str, dict[str, Any]],
    importance: pd.DataFrame,
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Save report JSON and stakeholder-facing model plots."""
    evaluation_path = report_dir / "model_evaluation_results.json"
    importance_path = figure_dir / "feature_importance.png"
    roc_path = figure_dir / "roc_curve.png"
    confusion_path = figure_dir / "confusion_matrices.png"

    evaluation_path.write_text(
        json.dumps(_evaluation_report(metrics), indent=2),
    )
    _plot_feature_importance(importance, importance_path)
    _plot_roc_curves(predictions, roc_path)
    _plot_confusion_matrices(metrics, confusion_path)

    return {
        "model_evaluation_results": evaluation_path,
        "feature_importance": importance_path,
        "roc_curve": roc_path,
        "confusion_matrices": confusion_path,
    }


def _evaluation_report(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "target": TARGET,
        "threshold": metrics["validation"]["threshold"],
        "features": FEATURES,
        "splits": {
            split_name: {
                "accuracy": split_metrics["accuracy"],
                "f1": split_metrics["f1"],
                "roc_auc": split_metrics["roc_auc"],
                "confusion_matrix": [
                    [split_metrics["tn"], split_metrics["fp"]],
                    [split_metrics["fn"], split_metrics["tp"]],
                ],
                "confusion_matrix_labels": {
                    "rows": ["actual_no_pass", "actual_pass"],
                    "columns": ["predicted_no_pass", "predicted_pass"],
                },
            }
            for split_name, split_metrics in metrics.items()
        },
    }


def _plot_feature_importance(
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2f6f9f")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Split count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_roc_curves(
    predictions: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for split_name, split_predictions in predictions.items():
        y_true = split_predictions["y_true"]
        y_proba = split_predictions["y_proba"]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            y_proba,
        )
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{split_name} AUC={auc:.3f}",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrices(
    metrics: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (split_name, split_metrics) in zip(axes, metrics.items()):
        matrix = [
            [split_metrics["tn"], split_metrics["fp"]],
            [split_metrics["fn"], split_metrics["tp"]],
        ]
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(split_name.capitalize())
        ax.set_xticks([0, 1], labels=["No pass", "Pass"])
        ax.set_yticks([0, 1], labels=["No pass", "Pass"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        max_value = max(max(row) for row in matrix)
        for row_idx, row in enumerate(matrix):
            for col_idx, value in enumerate(row):
                color = "white" if value > max_value / 2 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                )
    fig.suptitle("Confusion Matrices")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _print_metrics(metrics: dict[str, dict[str, Any]]) -> None:
    for split_name, split_metrics in metrics.items():
        print(f"\n{split_name.upper()}")
        print(f"  Threshold: {split_metrics['threshold']:.2f}")
        print(f"  Accuracy:  {split_metrics['accuracy']:.4f}")
        print(f"  F1:        {split_metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {split_metrics['roc_auc']:.4f}")
        print("  Confusion Matrix:")
        print(f"    TN={split_metrics['tn']}, FP={split_metrics['fp']}")
        print(f"    FN={split_metrics['fn']}, TP={split_metrics['tp']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost pass detector.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=MODEL_BASE_DATA_DIR,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR,
    )
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_pass_detector(
        input_dir=project_path(args.input_dir),
        model_dir=project_path(args.model_dir),
        report_dir=project_path(args.report_dir),
        figure_dir=project_path(args.figure_dir),
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )


if __name__ == "__main__":
    main()

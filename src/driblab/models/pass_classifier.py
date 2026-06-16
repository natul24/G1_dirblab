"""Binary logistic regression pass classifier.

This module builds the first supervised model in the project: a frame-level
binary classifier that predicts whether `event_label` represents a pass. It
creates the `is_pass` target from configured event labels, selects tracking
features from the Step 2 master join table, trains a scikit-learn pipeline,
evaluates train/validation/test splits, writes metrics, and saves the trained
`.joblib` model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from driblab.config import CONFIG_PATH
from driblab.config import MODEL_BASE_DATA_DIR
from driblab.config import PASS_CLASSIFIER_DATA_DIR
from driblab.config import PASS_CLASSIFIER_MODEL_DIR
from driblab.evaluation.classification import binary_classification_metrics
from driblab.features.match_splits import add_data_split_column
from driblab.features.match_splits import load_match_splits
from driblab.validation import require_columns
from driblab.validation import validate_binary_target


@dataclass
class PassModelConfig:
    """Configuration for training and saving the pass classifier."""

    input_table: Path = MODEL_BASE_DATA_DIR / "master_join_table.parquet"
    metrics_dir: Path = PASS_CLASSIFIER_DATA_DIR
    model_dir: Path = PASS_CLASSIFIER_MODEL_DIR
    match_splits: Path = CONFIG_PATH
    threshold: float = 0.50
    c_value: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    random_state: int = 42
    class_weight: str | None = "balanced"
    positive_labels: tuple[str, ...] = ("PASS",)
    feature_columns: tuple[str, ...] = ()


PASS_FEATURE_LINEAGE: dict[str, dict[str, str]] = {
    "period_id": {
        "feature_kind": "original master-join column",
        "created_or_calculated": "No",
        "description": "Match period kept from the tracking/event join.",
    },
    "match_clock_min": {
        "feature_kind": "original master-join column",
        "created_or_calculated": "No",
        "description": "Minute component kept from the tracking clock.",
    },
    "match_clock_sec": {
        "feature_kind": "original master-join column",
        "created_or_calculated": "No",
        "description": "Second component kept from the tracking clock.",
    },
    "cam_present": {
        "feature_kind": "original master-join column",
        "created_or_calculated": "No",
        "description": "Tracking camera/live-play availability flag.",
    },
    "ball_z_m_raw": {
        "feature_kind": "original master-join column",
        "created_or_calculated": "No",
        "description": "Raw tracking ball height kept in meters.",
    },
    "ball_x_raw": {
        "feature_kind": "transformed master-join column",
        "created_or_calculated": "Transformed",
        "description": "Raw tracking ball x converted from meters to 0-100.",
    },
    "ball_y_raw": {
        "feature_kind": "transformed master-join column",
        "created_or_calculated": "Transformed",
        "description": "Raw tracking ball y converted from meters to 0-100.",
    },
    "tracking_match_clock_seconds": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Continuous 10 Hz time from clock and frame order.",
    },
    "ball_x": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Normalized ball x after short-gap interpolation.",
    },
    "ball_y": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Normalized ball y after short-gap interpolation.",
    },
    "ball_z_m": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Ball height after short-gap interpolation.",
    },
    "ball_present_raw": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Flag that raw ball x/y/z were present.",
    },
    "ball_interpolated": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Flag that Step 2 filled a short ball gap.",
    },
    "dt_sec": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Seconds elapsed since the previous live frame.",
    },
    "ball_speed_xy_mps": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Horizontal ball speed calculated in meters/second.",
    },
    "ball_speed_mps": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "3D ball speed calculated in meters/second.",
    },
    "ball_acceleration_mps2": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Frame-to-frame ball acceleration.",
    },
    "nearest_player_visible": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": (
            "Whether the nearest player was directly visible in tracking."
        ),
    },
    "nearest_player_distance_to_ball_m": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Distance from the nearest player to the ball.",
    },
    "has_possession": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Raw possession flag from distance and ball speed.",
    },
    "player_count": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Number of player rows available in the frame.",
    },
    "visible_player_count": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Number of directly visible player rows.",
    },
    "min_distance_to_ball_m": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Minimum player-to-ball distance in the frame.",
    },
    "mean_distance_to_ball_m": {
        "feature_kind": "calculated master-join feature",
        "created_or_calculated": "Yes",
        "description": "Average player-to-ball distance in the frame.",
    },
}


def describe_pass_features(feature_columns: list[str]) -> pd.DataFrame:
    """Return a lineage table for the configured pass-model features."""
    rows = []
    for column in feature_columns:
        metadata = PASS_FEATURE_LINEAGE.get(
            column,
            {
                "feature_kind": "unclassified master-join column",
                "created_or_calculated": "Unknown",
                "description": "Feature is configured but not in lineage map.",
            },
        )
        rows.append(
            {
                "feature": column,
                "source_table": "master_join_table.parquet",
                **metadata,
            }
        )
    return pd.DataFrame(rows)


def is_pass_event(value: Any, positive_labels: set[str]) -> bool:
    """Return True when an event label contains a pass event type."""
    if pd.isna(value):
        return False
    labels = {
        part.strip().upper()
        for part in str(value).split("|")
        if part.strip()
    }
    return bool(labels & positive_labels)


def available_tracking_features(
    table: pd.DataFrame,
    configured_features: tuple[str, ...],
) -> list[str]:
    """Return tracking-derived model features present in the table."""
    return [
        column
        for column in configured_features
        if column in table.columns
    ]


def prepare_pass_model_frame(
    master_join_table: pd.DataFrame,
    splits: dict[str, Any],
    positive_labels: tuple[str, ...],
    configured_features: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    """Create modelling frame with split, target, and tracking features."""
    model_frame = add_data_split_column(master_join_table, splits)
    positive_label_set = {label.upper() for label in positive_labels}
    model_frame["is_pass"] = model_frame["event_label"].map(
        lambda value: is_pass_event(value, positive_label_set)
    )
    model_frame["is_pass"] = model_frame["is_pass"].astype(int)

    feature_columns = available_tracking_features(
        model_frame,
        configured_features,
    )
    if not feature_columns:
        raise ValueError("No configured pass-model feature columns were found")
    require_columns(
        model_frame,
        feature_columns,
        context="pass model frame",
    )
    for column in feature_columns:
        if model_frame[column].dtype == "bool":
            model_frame[column] = model_frame[column].astype(int)

    return model_frame, feature_columns


def build_pass_pipeline(config: PassModelConfig) -> Pipeline:
    """Build the logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=config.c_value,
                    class_weight=config.class_weight,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                    solver=config.solver,
                ),
            ),
        ]
    )


def predict_passes(
    model_frame: pd.DataFrame,
    feature_columns: list[str],
    pipeline: Pipeline,
    threshold: float,
) -> pd.DataFrame:
    """Predict pass probabilities and binary labels for every row."""
    predictions = model_frame[
        [
            "match_id",
            "data_split",
            "frame_id",
            "period_id",
            "tracking_match_clock_seconds",
            "event_label",
            "is_pass",
        ]
    ].copy()
    probabilities = pipeline.predict_proba(model_frame[feature_columns])[:, 1]
    predictions["pass_probability"] = probabilities
    predictions["pass_prediction"] = (
        predictions["pass_probability"].ge(threshold).astype(int)
    )
    return predictions


def evaluate_pass_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate binary pass predictions by split."""
    rows = []
    for split_name in ["train", "validation", "test"]:
        split_rows = predictions[predictions["data_split"].eq(split_name)]
        if split_rows.empty:
            continue
        rows.append(
            binary_classification_metrics(
                y_true=split_rows["is_pass"],
                y_pred=split_rows["pass_prediction"],
                y_score=split_rows["pass_probability"],
                split_name=split_name,
            )
        )
    return pd.DataFrame(rows)


def run_pass_model(config: PassModelConfig) -> dict[str, Any]:
    """Train and evaluate the binary pass logistic regression model."""
    input_table = config.input_table.expanduser().resolve()
    metrics_dir = config.metrics_dir.expanduser().resolve()
    model_dir = config.model_dir.expanduser().resolve()
    splits_path = config.match_splits.expanduser().resolve()

    master_join_table = pd.read_parquet(input_table)
    splits = load_match_splits(splits_path)
    model_frame, feature_columns = prepare_pass_model_frame(
        master_join_table,
        splits,
        config.positive_labels,
        config.feature_columns,
    )

    train_rows = model_frame[model_frame["data_split"].eq("train")]
    if train_rows.empty:
        raise ValueError(
            "No train rows found. Check config.yaml match_splits."
        )
    validate_binary_target(train_rows["is_pass"], context="train is_pass")

    pipeline = build_pass_pipeline(config)
    pipeline.fit(train_rows[feature_columns], train_rows["is_pass"])

    predictions = predict_passes(
        model_frame=model_frame,
        feature_columns=feature_columns,
        pipeline=pipeline,
        threshold=config.threshold,
    )
    metrics = evaluate_pass_predictions(predictions)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "pass_logistic_regression.joblib"
    metrics_path = metrics_dir / "pass_model_metrics.parquet"

    joblib.dump(pipeline, model_path)
    metrics.to_parquet(metrics_path, index=False)

    return {
        "model": pipeline,
        "model_frame": model_frame,
        "predictions": predictions,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "outputs": {
            "model": str(model_path),
            "metrics": str(metrics_path),
        },
    }

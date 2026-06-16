"""Step 4 deterministic rule-based event detector.

This module contains the baseline detector that converts Step 3 smoothed
possession changes plus Step 2 ball movement features into broad event
predictions. It maps provider labels to evaluation classes, creates lagged rule
features, predicts pass/interception/tackle/shot/out/corner/no-event classes,
and evaluates the predictions on a configured held-out split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driblab.config import POSSESSION_SEQUENCE_DATA_DIR, RULE_BASED_DATA_DIR
from driblab.evaluation.classification import (
    confusion_matrix_long,
    per_class_precision_recall_f1,
)


@dataclass
class Step4Config:
    """Configuration for Step 4 rule thresholds and evaluation output."""

    input_table: Path = (
        POSSESSION_SEQUENCE_DATA_DIR / "possession_sequence_table.parquet"
    )
    output_dir: Path = RULE_BASED_DATA_DIR
    evaluation_split: str = "test"
    shot_min_speed_mps: float = 14.0
    shot_min_attacking_x: float = 70.0
    shot_min_dx_attacking: float = 0.25
    interception_min_ball_speed_mps: float = 4.0
    boundary_margin: float = 0.5
    corner_y_margin: float = 12.0
    rule_classes: tuple[str, ...] = (
        "no event",
        "pass",
        "interception",
        "tackle",
        "shot",
        "out",
        "corner",
        "other_event",
    )
    label_groups: dict[str, tuple[str, ...]] | None = None


def _split_labels(value: Any) -> set[str]:
    if pd.isna(value):
        return {"no event"}
    labels = {
        part.strip().upper()
        for part in str(value).split("|")
        if part.strip()
    }
    return labels or {"no event"}


def canonical_event_class(
    value: Any,
    label_groups: dict[str, set[str]],
) -> str:
    """Map provider event labels into the Step 4 rule classes."""
    labels = _split_labels(value)
    if labels == {"NO EVENT"} or labels == {"no event"}:
        return "no event"
    if "NO EVENT" in labels or "no event" in labels:
        labels.discard("NO EVENT")
        labels.discard("no event")
    if not labels:
        return "no event"
    if labels & label_groups.get("shot", set()):
        return "shot"
    if labels & label_groups.get("corner", set()):
        return "corner"
    if labels & label_groups.get("out", set()):
        return "out"
    if labels & label_groups.get("tackle", set()):
        return "tackle"
    if labels & label_groups.get("interception", set()):
        return "interception"
    if labels & label_groups.get("pass", set()):
        return "pass"
    return "other_event"


def add_rule_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add lagged ball and possession features used by the rules."""
    output = table.sort_values(
        ["match_id", "period_id", "tracking_match_clock_seconds", "frame_id"]
    ).copy()
    group_cols = ["match_id", "period_id"]
    output["rule_prev_smoothed_possession_team_id"] = output.groupby(
        group_cols
    )["smoothed_possession_team_id"].shift()
    output["rule_prev_smoothed_possession_player_id"] = output.groupby(
        group_cols
    )["smoothed_possession_player_id"].shift()
    output["rule_ball_dx_attacking"] = output.groupby(group_cols)[
        "ball_x_attacking"
    ].diff()
    output["rule_ball_dy_attacking"] = output.groupby(group_cols)[
        "ball_y_attacking"
    ].diff()
    return output


def predict_rule_based_events(
    table: pd.DataFrame,
    config: Step4Config,
) -> pd.DataFrame:
    """Predict event classes with possession and ball-movement rules."""
    output = add_rule_features(table)
    label_groups = {
        name: {label.upper() for label in labels}
        for name, labels in (config.label_groups or {}).items()
    }
    output["true_event_class"] = output["event_label"].map(
        lambda value: canonical_event_class(value, label_groups)
    )

    ball_speed = pd.to_numeric(output["ball_speed_mps"], errors="coerce")
    ball_x = pd.to_numeric(output["ball_x"], errors="coerce")
    ball_y = pd.to_numeric(output["ball_y"], errors="coerce")
    ball_x_attacking = pd.to_numeric(
        output["ball_x_attacking"],
        errors="coerce",
    )
    ball_dx_attacking = pd.to_numeric(
        output["rule_ball_dx_attacking"],
        errors="coerce",
    )

    team_changed = output["possession_team_change"].fillna(False)
    player_changed = output["possession_player_change"].fillna(False)
    same_team_player_changed = player_changed & ~team_changed
    opponent_gained_ball = team_changed

    near_touchline = ball_y.le(config.boundary_margin) | ball_y.ge(
        100.0 - config.boundary_margin
    )
    near_goal_line = ball_x.le(config.boundary_margin) | ball_x.ge(
        100.0 - config.boundary_margin
    )
    near_corner = near_goal_line & (
        ball_y.le(config.corner_y_margin)
        | ball_y.ge(100.0 - config.corner_y_margin)
    )
    shot_rule = (
        ball_speed.ge(config.shot_min_speed_mps)
        & ball_x_attacking.ge(config.shot_min_attacking_x)
        & ball_dx_attacking.ge(config.shot_min_dx_attacking)
    )

    predictions = pd.Series("no event", index=output.index, dtype="object")
    predictions.loc[same_team_player_changed] = "pass"
    fast_opponent_gain = opponent_gained_ball & ball_speed.ge(
        config.interception_min_ball_speed_mps
    )
    slow_opponent_gain = opponent_gained_ball & ball_speed.lt(
        config.interception_min_ball_speed_mps
    )
    predictions.loc[fast_opponent_gain] = "interception"
    predictions.loc[slow_opponent_gain] = "tackle"
    predictions.loc[near_touchline | near_goal_line] = "out"
    predictions.loc[near_corner] = "corner"
    predictions.loc[shot_rule] = "shot"
    allowed_prediction_classes = set(config.rule_classes)
    predictions = predictions.where(
        predictions.isin(allowed_prediction_classes),
        "other_event",
    )

    output["rule_event_class"] = predictions
    output["rule_reason"] = np.select(
        [
            shot_rule,
            near_corner,
            near_touchline | near_goal_line,
            same_team_player_changed,
            opponent_gained_ball
            & ball_speed.ge(config.interception_min_ball_speed_mps),
            opponent_gained_ball
            & ball_speed.lt(config.interception_min_ball_speed_mps),
        ],
        [
            "fast_ball_toward_goal",
            "ball_near_corner_boundary",
            "ball_near_pitch_boundary",
            "same_team_player_change",
            "opponent_gain_fast_ball",
            "opponent_gain_slow_ball",
        ],
        default="no_rule_fired",
    )
    return output


def evaluate_rule_predictions(
    predictions: pd.DataFrame,
    evaluation_split: str,
    rule_classes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate on the requested held-out split."""
    if "data_split" in predictions.columns:
        evaluation_rows = predictions[
            predictions["data_split"].eq(evaluation_split)
        ].copy()
    else:
        evaluation_rows = predictions.copy()
        evaluation_rows["data_split"] = "all"

    metrics = per_class_precision_recall_f1(
        evaluation_rows["true_event_class"],
        evaluation_rows["rule_event_class"],
        labels=list(rule_classes),
    )
    confusion = confusion_matrix_long(
        evaluation_rows["true_event_class"],
        evaluation_rows["rule_event_class"],
    )
    summary = pd.DataFrame(
        [
            {
                "evaluation_split": evaluation_split,
                "rows": len(evaluation_rows),
                "matches": evaluation_rows["match_id"].astype(str).nunique(),
                "macro_f1": metrics["f1"].mean(),
                "weighted_f1": (
                    (metrics["f1"] * metrics["support"]).sum()
                    / metrics["support"].sum()
                    if metrics["support"].sum()
                    else 0.0
                ),
            }
        ]
    )
    return metrics, confusion, summary


def run_step4(config: Step4Config) -> dict[str, Any]:
    """Run the rule-based detector and held-out evaluation."""
    input_table = config.input_table.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(input_table)
    predictions = predict_rule_based_events(table, config)
    metrics, confusion, summary = evaluate_rule_predictions(
        predictions,
        evaluation_split=config.evaluation_split,
        rule_classes=config.rule_classes,
    )

    predictions_path = output_dir / "rule_based_predictions.parquet"
    metrics_path = output_dir / "rule_based_metrics_by_class.csv"
    confusion_path = output_dir / "rule_based_confusion_matrix.csv"
    summary_path = output_dir / "rule_based_summary.csv"
    metadata_path = output_dir / "rule_based_metadata.json"

    predictions.to_parquet(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "config": {
                    **asdict(config),
                    "input_table": str(input_table),
                    "output_dir": str(output_dir),
                },
                "rows": int(len(predictions)),
                "outputs": {
                    "predictions": str(predictions_path),
                    "metrics": str(metrics_path),
                    "confusion": str(confusion_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        )
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "confusion": confusion,
        "summary": summary,
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "confusion": str(confusion_path),
            "summary": str(summary_path),
            "metadata": str(metadata_path),
        },
    }

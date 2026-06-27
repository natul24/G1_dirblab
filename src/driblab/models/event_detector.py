"""Train a multi-class XGBoost event detector.

Detects PASS, BALL TOUCH, AERIAL, TACKLE, BALL RECOVERY, FOUL, and TAKEON
from tracking-data features. Uses the same training tables as the binary
pass detector but with a multi-class target derived from p.event_label.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

from driblab.config import (
    ARTIFACTS_DIR,
    MODEL_BASE_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    project_path,
)

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVENT_CLASSES = [
    "no event",
    "PASS",
    "BALL TOUCH",
    "AERIAL",
    "TACKLE",
    "BALL RECOVERY",
    "FOUL",
    "TAKEON",
]
CLASS_TO_ID = {name: i for i, name in enumerate(EVENT_CLASSES)}
NUM_CLASSES = len(EVENT_CLASSES)
EVAL_CLASSES = EVENT_CLASSES[1:]

FEATURES = [
    "ball_speed_raw",
    "ball_speed_avg_xy",
    "ball_speed_avg_15f",
    "closest_player_team_id",
    "t.ball_x",
    "t.ball_y",
    "t.ball_z",
    "t.player_count",
    "t.visible_player_count",
    "closest_player_distance_to_ball",
    "second_closest_player_distance",
    "second_closest_same_team",
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
    "ball_speed_ratio",
    "ball_z_change_5f",
    "distance_gap",
]

TARGET = "event_label_id"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = REPORTS_DIR
FIGURE_DIR = REPORTS_DIR / "figures"


def _load_training_tables(
    input_dir: Path = MODEL_BASE_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir).expanduser().resolve()
    train = pd.read_parquet(input_dir / "training_table_train.parquet")
    val = pd.read_parquet(input_dir / "training_table_validation.parquet")
    test = pd.read_parquet(input_dir / "training_table_test.parquet")
    return train, val, test


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET] = df["p.event_label"].map(CLASS_TO_ID).fillna(0).astype(int)
    return df


def _prepare_matrix(
    table: pd.DataFrame,
    class_weights: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.Series, xgb.DMatrix]:
    encoded = {}
    for col in FEATURES:
        series = table[col]
        if pd.api.types.is_numeric_dtype(series):
            encoded[col] = series.to_numpy(dtype=float, na_value=np.nan)
        else:
            codes = pd.Categorical(series).codes.astype(float)
            codes[codes == -1] = np.nan
            encoded[col] = codes
    x = pd.DataFrame(encoded, index=table.index).fillna(0)
    y = table[TARGET].astype(int)
    dm = xgb.DMatrix(x, label=y, feature_names=FEATURES)
    if class_weights is not None:
        weights = y.map(class_weights).to_numpy(dtype=float)
        dm.set_weight(weights)
    return x, y, dm


# ------------------------------------------------------------------
# NMS helpers (generalised from pass_detector for any event class)
# ------------------------------------------------------------------

def _run_class_nms_eval(
    table: pd.DataFrame,
    y_true: pd.Series,
    class_proba: np.ndarray,
    class_id: int,
    threshold: float,
    suppress_window_sec: float,
    tolerance_sec: float = 1.0,
) -> dict[str, Any]:
    pred_df = table[
        ["t.match_id", "t.period", "t.frame", "t.Videotimestamp"]
    ].copy().rename(columns={
        "t.match_id": "match_id",
        "t.period": "period",
        "t.frame": "frame",
        "t.Videotimestamp": "time_sec",
    })
    pred_df["is_true"] = (y_true.to_numpy() == class_id).astype(int)
    pred_df["score"] = class_proba

    # --- temporal NMS ---
    pred_df["pred_nms"] = False
    candidates = pred_df.loc[
        pred_df["score"] > threshold
    ].dropna(subset=["time_sec", "score"])

    kept_indices: list[int] = []
    for _, group in candidates.groupby(["match_id", "period"], sort=False):
        group = group.sort_values("score", ascending=False)
        kept_times: list[float] = []
        for idx, row in group.iterrows():
            t = row["time_sec"]
            if not any(abs(t - kt) <= suppress_window_sec for kt in kept_times):
                kept_indices.append(idx)
                kept_times.append(t)
    pred_df.loc[kept_indices, "pred_nms"] = True

    # --- cluster true events ---
    true_events: list[dict] = []
    positive = pred_df[pred_df["is_true"] == 1].dropna(subset=["time_sec"])
    for group_key, group in positive.groupby(["match_id", "period"], sort=False):
        group = group.sort_values("time_sec")
        td = group["time_sec"].diff().fillna(999)
        cluster_id = (td > 1.0).cumsum()
        for _, cluster in group.groupby(cluster_id):
            true_events.append({
                "match_id": group_key[0],
                "period": group_key[1],
                "true_event_time": float(cluster["time_sec"].median()),
            })
    true_df = pd.DataFrame(true_events)

    # --- extract detections ---
    det = pred_df[pred_df["pred_nms"]].dropna(subset=["time_sec"])
    det_df = det[["match_id", "period", "time_sec", "score"]].rename(
        columns={"time_sec": "pred_event_time", "score": "pred_score"},
    ).reset_index(drop=True)

    # --- greedy matching ---
    if true_df.empty:
        return _metrics_dict(0, len(det_df), 0, len(det_df), 0)

    true_df = true_df.reset_index(drop=True)
    true_df["matched"] = False
    det_df["matched"] = False

    for di, d in det_df.iterrows():
        cands = true_df[
            (true_df["match_id"] == d["match_id"])
            & (true_df["period"] == d["period"])
            & (~true_df["matched"])
        ].copy()
        if cands.empty:
            continue
        cands["err"] = (cands["true_event_time"] - d["pred_event_time"]).abs()
        cands = cands[cands["err"] <= tolerance_sec]
        if cands.empty:
            continue
        true_df.loc[cands["err"].idxmin(), "matched"] = True
        det_df.loc[di, "matched"] = True

    tp = int(det_df["matched"].sum())
    fp = int((~det_df["matched"]).sum())
    fn = int((~true_df["matched"]).sum())
    return _metrics_dict(len(true_df), len(det_df), tp, fp, fn)


def _metrics_dict(
    true_events: int, detections: int, tp: int, fp: int, fn: int,
) -> dict[str, Any]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "true_events": true_events,
        "detections": detections,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def _plot_feature_importance(imp_df: pd.DataFrame, path: Path) -> None:
    plot_df = imp_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2f6f9f")
    ax.set_title("Multi-class Event Detector — Feature Importance")
    ax.set_xlabel("Split count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(EVENT_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(EVENT_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Event Detector — Confusion Matrix (Test)")
    max_val = cm.max()
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm[i, j]
            color = "white" if val > max_val / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    color=color, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train_event_detector(
    input_dir: Path = MODEL_BASE_DATA_DIR,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
    figure_dir: Path = FIGURE_DIR,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    """Train, evaluate, and save the multi-class event detector."""

    train_df, val_df, test_df = _load_training_tables(input_dir)
    train_df = _encode_target(train_df)
    val_df = _encode_target(val_df)
    test_df = _encode_target(test_df)

    # Compute class weights: sqrt(max_count / class_count)
    class_counts = train_df[TARGET].value_counts()
    max_count = class_counts.max()
    class_weights = {
        cls_id: float(np.sqrt(max_count / class_counts.get(cls_id, 1)))
        for cls_id in range(NUM_CLASSES)
    }

    _, y_train, dtrain = _prepare_matrix(train_df, class_weights=class_weights)
    _, y_val, dval = _prepare_matrix(val_df)
    _, y_test, dtest = _prepare_matrix(test_df)

    print("Training class distribution & weights:")
    for cls_name in EVENT_CLASSES:
        cls_id = CLASS_TO_ID[cls_name]
        n = int((y_train == cls_id).sum())
        w = class_weights[cls_id]
        print(f"  {cls_id} {cls_name:15s}: {n:>7,}  ({n / len(y_train):.1%})"
              f"  weight={w:.2f}")

    params = {
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "eval_metric": "mlogloss",
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )
    print(f"\nBest round: {model.best_iteration + 1}")

    # ---- frame-level predictions ----
    proba_val = model.predict(dval)
    proba_test = model.predict(dtest)
    pred_val = proba_val.argmax(axis=1)
    pred_test = proba_test.argmax(axis=1)

    print("\n=== Validation — Frame-level Classification Report ===")
    print(classification_report(
        y_val, pred_val,
        labels=list(range(NUM_CLASSES)),
        target_names=EVENT_CLASSES,
        zero_division=0,
    ))
    print("=== Test — Frame-level Classification Report ===")
    print(classification_report(
        y_test, pred_test,
        labels=list(range(NUM_CLASSES)),
        target_names=EVENT_CLASSES,
        zero_division=0,
    ))

    # ---- per-class NMS evaluation ----
    print("=" * 60)
    print("Event-level NMS evaluation (tuned on validation, tested)")
    print("=" * 60)

    nms_results: dict[str, dict[str, Any]] = {}

    for cls_name in EVAL_CLASSES:
        cls_id = CLASS_TO_ID[cls_name]
        best_f1 = -1.0
        best_thresh = 0.10
        best_window = 1.5

        for thresh in np.arange(0.10, 0.61, 0.05):
            for window in [1.5, 1.75, 2.0, 2.5]:
                r = _run_class_nms_eval(
                    table=val_df,
                    y_true=y_val,
                    class_proba=proba_val[:, cls_id],
                    class_id=cls_id,
                    threshold=round(float(thresh), 2),
                    suppress_window_sec=window,
                )
                if r["f1"] > best_f1:
                    best_f1 = r["f1"]
                    best_thresh = round(float(thresh), 2)
                    best_window = window

        test_r = _run_class_nms_eval(
            table=test_df,
            y_true=y_test,
            class_proba=proba_test[:, cls_id],
            class_id=cls_id,
            threshold=best_thresh,
            suppress_window_sec=best_window,
        )
        test_r["threshold"] = best_thresh
        test_r["suppress_window_sec"] = best_window
        nms_results[cls_name] = test_r

        print(f"\n{cls_name}  (threshold={best_thresh:.2f}, "
              f"window={best_window:.2f}s)")
        print(f"  True events: {test_r['true_events']:,}  |  "
              f"Detections: {test_r['detections']:,}")
        print(f"  TP={test_r['tp']:,}  FP={test_r['fp']:,}  "
              f"FN={test_r['fn']:,}")
        print(f"  Precision={test_r['precision']:.4f}  "
              f"Recall={test_r['recall']:.4f}  "
              f"F1={test_r['f1']:.4f}")

    # ---- summary table ----
    print("\n" + "=" * 60)
    print(f"{'Event':17s} {'P':>7s} {'R':>7s} {'F1':>7s} "
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s} {'True':>5s}")
    print("-" * 60)
    for cls_name in EVAL_CLASSES:
        r = nms_results[cls_name]
        print(f"{cls_name:17s} {r['precision']:7.4f} {r['recall']:7.4f} "
              f"{r['f1']:7.4f} {r['tp']:5d} {r['fp']:5d} "
              f"{r['fn']:5d} {r['true_events']:5d}")
    print("=" * 60)

    # ---- save model & reports ----
    model_dir = Path(model_dir).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    figure_dir = Path(figure_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "event_detector.json"
    metadata_path = model_dir / "event_detector_metadata.json"

    model.save_model(model_path)

    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "event_classes": EVENT_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "params": params,
        "best_iteration": int(model.best_iteration),
        "nms_results": nms_results,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    importance = model.get_score(importance_type="weight")
    imp_df = pd.DataFrame([
        {"feature": f, "importance": importance.get(f, 0)} for f in FEATURES
    ]).sort_values("importance", ascending=False)

    print("\nTop 10 Features:")
    print(imp_df.head(10).to_string(index=False))

    _plot_feature_importance(imp_df, figure_dir / "event_feature_importance.png")
    _plot_confusion_matrix(y_test.to_numpy(), pred_test, figure_dir / "event_confusion_matrix.png")

    report = {
        "event_classes": EVENT_CLASSES,
        "classification_report_test": classification_report(
            y_test, pred_test,
            labels=list(range(NUM_CLASSES)),
            target_names=EVENT_CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "nms_results": nms_results,
    }
    report_path = report_dir / "event_detector_results.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nSaved model:    {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved report:   {report_path}")
    print(f"Saved figures:  {figure_dir / 'event_feature_importance.png'}")
    print(f"                {figure_dir / 'event_confusion_matrix.png'}")

    return {"model_path": str(model_path), "nms_results": nms_results}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the multi-class XGBoost event detector.",
    )
    parser.add_argument("--input-dir", type=Path, default=MODEL_BASE_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_event_detector(
        input_dir=project_path(args.input_dir),
        model_dir=project_path(args.model_dir),
        report_dir=project_path(args.report_dir),
        figure_dir=project_path(args.figure_dir),
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )


if __name__ == "__main__":
    main()

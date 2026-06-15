"""Classification metric utilities for project evaluation.

This module contains reusable metric functions for both multiclass event
detection and binary pass prediction. It provides per-class precision, recall,
F1, support, long-format confusion matrices, and common binary metrics such as
accuracy and ROC-AUC.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def per_class_precision_recall_f1(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-class precision, recall, and F1."""
    truth = y_true.fillna("no event").astype(str)
    pred = y_pred.fillna("no event").astype(str)
    if labels is None:
        labels = sorted(set(truth.unique()) | set(pred.unique()))

    rows = []
    for label in labels:
        true_positive = int(((truth == label) & (pred == label)).sum())
        false_positive = int(((truth != label) & (pred == label)).sum())
        false_negative = int(((truth == label) & (pred != label)).sum())
        support = int((truth == label).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "event_class": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )

    return pd.DataFrame(rows)


def confusion_matrix_long(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """Return confusion matrix in long format for easy CSV inspection."""
    matrix = pd.crosstab(
        y_true.fillna("no event").astype(str),
        y_pred.fillna("no event").astype(str),
        rownames=["true_event_class"],
        colnames=["predicted_event_class"],
        dropna=False,
    )
    return (
        matrix.reset_index()
        .melt(
            id_vars="true_event_class",
            var_name="predicted_event_class",
            value_name="rows",
        )
        .sort_values(["true_event_class", "predicted_event_class"])
        .reset_index(drop=True)
    )


def binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
    split_name: str,
) -> dict[str, float | int | str]:
    """Compute common binary classification metrics for one split."""
    truth = y_true.astype(int)
    pred = y_pred.astype(int)
    score = y_score.astype(float)

    if truth.nunique() == 2:
        roc_auc = float(roc_auc_score(truth, score))
    else:
        roc_auc = float("nan")

    return {
        "split": split_name,
        "rows": int(len(truth)),
        "positive_rows": int(truth.sum()),
        "positive_rate": float(truth.mean()),
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(
            precision_score(truth, pred, zero_division=0)
        ),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "roc_auc": roc_auc,
    }

"""Fail-fast validation helpers shared across pipeline stages.

This module contains small reusable checks that stop the pipeline early when
required columns, match split definitions, or binary targets are invalid. The
helpers are intentionally lightweight so ETL, feature-building, and model code
can validate assumptions before producing misleading outputs.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def require_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    context: str,
) -> None:
    """Fail if required columns are missing from a dataframe."""
    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{context} is missing required columns: {missing}"
        )


def validate_match_splits(splits: dict[str, object]) -> None:
    """Fail if split definitions overlap or omit required split names."""
    required_names = {"train", "validation", "test"}
    missing_names = required_names - set(splits)
    if missing_names:
        raise ValueError(f"Match splits missing keys: {sorted(missing_names)}")

    seen: dict[str, str] = {}
    duplicates = []
    for split_name in sorted(required_names):
        match_ids = splits.get(split_name, [])
        if not isinstance(match_ids, list):
            raise TypeError(f"{split_name} split must be a list")
        for match_id in match_ids:
            match_id_text = str(match_id)
            if match_id_text in seen:
                duplicates.append(
                    (match_id_text, seen[match_id_text], split_name)
                )
            seen[match_id_text] = split_name

    if duplicates:
        raise ValueError(
            "Match IDs assigned to multiple splits: "
            f"{duplicates}"
        )


def validate_binary_target(
    target: pd.Series,
    context: str,
) -> None:
    """Fail if a binary target does not contain both classes."""
    unique_values = set(target.dropna().astype(int).unique())
    if not unique_values <= {0, 1}:
        raise ValueError(f"{context} must contain only 0/1 values")
    if unique_values != {0, 1}:
        raise ValueError(f"{context} must contain both 0 and 1")

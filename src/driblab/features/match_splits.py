"""Match-level train, validation, and test split helpers.

This module loads split definitions from `config.yaml` or JSON, validates that
matches are assigned to exactly one split, maps `match_id` values to split
names, and appends a `data_split` column to frame-level tables. Splits happen
by full match, not by row, because adjacent 10 Hz tracking frames from the same
match are highly correlated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from driblab.validation import validate_match_splits


SPLIT_NAMES = ("train", "validation", "test")


def load_match_splits(path: Path) -> dict[str, Any]:
    """Load the explicit match split config."""
    if path.suffix in {".yaml", ".yml"}:
        config = yaml.safe_load(path.read_text())
        splits = config["match_splits"]
    else:
        splits = json.loads(path.read_text())
    validate_match_splits(splits)
    return splits


def split_lookup(splits: dict[str, Any]) -> dict[str, str]:
    """Return match_id -> split name from a split config."""
    lookup: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for match_id in splits.get(split_name, []):
            lookup[str(match_id)] = split_name
    return lookup


def add_data_split_column(
    table: pd.DataFrame,
    splits: dict[str, Any],
    match_col: str = "match_id",
) -> pd.DataFrame:
    """Attach a data_split column using match-level holdout assignments."""
    output = table.copy()
    lookup = split_lookup(splits)
    output["data_split"] = (
        output[match_col].astype(str).map(lookup).fillna("unassigned")
    )
    return output


def summarize_splits(table: pd.DataFrame) -> pd.DataFrame:
    """Count rows and matches per split for auditability."""
    summary = (
        table.groupby("data_split", dropna=False)
        .agg(
            rows=("match_id", "size"),
            matches=("match_id", lambda values: values.astype(str).nunique()),
        )
        .reset_index()
        .sort_values("data_split")
    )
    return summary

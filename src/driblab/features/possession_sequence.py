"""Step 3 smoothed possession sequence construction.

This module reads the Step 2 master join table, assigns raw possession from the
nearest-player fields, smooths short no-possession gaps and unstable one-frame
flips, tracks possession starts and changes, adds match-level split labels, and
writes `possession_sequence_table.parquet`. The resulting possession sequence
is the skeleton used by Step 4 rules and later learned event models.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driblab.config import (
    CONFIG_PATH,
    MODEL_BASE_DATA_DIR,
    POSSESSION_SEQUENCE_DATA_DIR,
)
from driblab.features.match_splits import (
    add_data_split_column,
    load_match_splits,
)


@dataclass
class Step3Config:
    """Configuration for Step 3 possession sequence generation."""

    input_table: Path = MODEL_BASE_DATA_DIR / "master_join_table.parquet"
    output_dir: Path = POSSESSION_SEQUENCE_DATA_DIR
    match_splits: Path = CONFIG_PATH
    match_id: str | None = None
    max_gap_frames: int = 3
    min_stable_frames: int = 3


def _clean_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in {"", "0", "<NA>", "None", "nan", "no event"}:
        return None
    if text.endswith(".0"):
        return text[:-2]
    return text


def _make_possession_key(
    team_id: Any,
    player_id: Any,
    has_possession: Any,
) -> str | None:
    if not bool(has_possession):
        return None
    clean_team_id = _clean_id(team_id)
    clean_player_id = _clean_id(player_id)
    if clean_team_id is None or clean_player_id is None:
        return None
    return f"{clean_team_id}|{clean_player_id}"


def _key_part(key: Any, index: int) -> str | None:
    if pd.isna(key) or key is None:
        return None
    parts = str(key).split("|", 1)
    if len(parts) <= index:
        return None
    return parts[index]


def _smooth_keys(
    keys: pd.Series,
    max_gap_frames: int,
    min_stable_frames: int,
) -> pd.Series:
    """Fill short no-possession gaps and remove one-off possession flips."""
    smoothed = keys.astype("object").where(keys.notna(), None).copy()

    previous_key = smoothed.ffill(limit=max_gap_frames)
    next_key = smoothed.bfill(limit=max_gap_frames)
    gap_mask = (
        smoothed.isna()
        & previous_key.notna()
        & previous_key.eq(next_key)
    )
    smoothed.loc[gap_mask] = previous_key.loc[gap_mask]

    if min_stable_frames <= 1 or len(smoothed) == 0:
        return smoothed

    values = smoothed.to_numpy(dtype=object).copy()
    segments: list[tuple[int, int, object]] = []
    start = 0
    for idx in range(1, len(values) + 1):
        if idx == len(values) or values[idx] != values[start]:
            segments.append((start, idx, values[start]))
            start = idx

    for segment_idx, (start, end, value) in enumerate(segments):
        segment_len = end - start
        if value is None or segment_len >= min_stable_frames:
            continue
        if segment_idx == 0 or segment_idx == len(segments) - 1:
            continue
        previous_value = segments[segment_idx - 1][2]
        next_value = segments[segment_idx + 1][2]
        if previous_value is not None and previous_value == next_value:
            values[start:end] = previous_value

    return pd.Series(values, index=keys.index, dtype="object")


def _name_lookup(table: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    rows = table[
        [
            "raw_possession_key",
            "possessing_team_name",
            "possessing_player_name",
        ]
    ].dropna(subset=["raw_possession_key"])
    team_names = (
        rows.dropna(subset=["possessing_team_name"])
        .drop_duplicates("raw_possession_key")
        .set_index("raw_possession_key")["possessing_team_name"]
        .to_dict()
    )
    player_names = (
        rows.dropna(subset=["possessing_player_name"])
        .drop_duplicates("raw_possession_key")
        .set_index("raw_possession_key")["possessing_player_name"]
        .to_dict()
    )
    return team_names, player_names


def _add_possession_sequence_for_group(
    group: pd.DataFrame,
    max_gap_frames: int,
    min_stable_frames: int,
) -> pd.DataFrame:
    output = group.copy()
    output["smoothed_possession_key"] = _smooth_keys(
        output["raw_possession_key"],
        max_gap_frames=max_gap_frames,
        min_stable_frames=min_stable_frames,
    )

    key = output["smoothed_possession_key"]
    output["smoothed_possession_team_id"] = key.map(
        lambda value: _key_part(value, 0)
    )
    output["smoothed_possession_player_id"] = key.map(
        lambda value: _key_part(value, 1)
    )
    output["smoothed_has_possession"] = key.notna()

    previous_active_key = key.ffill().shift()
    output["previous_smoothed_possession_key"] = previous_active_key
    output["previous_smoothed_possession_team_id"] = previous_active_key.map(
        lambda value: _key_part(value, 0)
    )
    output["previous_smoothed_possession_player_id"] = previous_active_key.map(
        lambda value: _key_part(value, 1)
    )
    output["smoothed_possession_start"] = (
        key.notna() & previous_active_key.isna()
    )
    output["smoothed_possession_change"] = (
        key.notna() & previous_active_key.notna() & key.ne(previous_active_key)
    )
    output["possession_team_change"] = (
        output["smoothed_possession_change"]
        & output["smoothed_possession_team_id"].ne(
            output["previous_smoothed_possession_team_id"]
        )
    )
    output["possession_player_change"] = (
        output["smoothed_possession_change"]
        & output["smoothed_possession_player_id"].ne(
            output["previous_smoothed_possession_player_id"]
        )
    )

    segment_key = key.fillna("__NO_POSSESSION__")
    segment_number = segment_key.ne(segment_key.shift()).cumsum()
    output["possession_sequence_number"] = np.where(
        key.notna(),
        segment_number,
        np.nan,
    )
    output["possession_sequence_id"] = np.where(
        key.notna(),
        output["match_id"].astype(str)
        + "_P"
        + output["period_id"].astype(str)
        + "_"
        + segment_number.astype(str),
        pd.NA,
    )
    output["possession_sequence_frame_number"] = np.where(
        key.notna(),
        output.groupby(segment_number).cumcount() + 1,
        np.nan,
    )
    sequence_sizes = output.groupby(segment_number)["frame_id"].transform(
        "size"
    )
    output["possession_sequence_duration_sec"] = np.where(
        key.notna(),
        sequence_sizes * output["dt_sec"].median(skipna=True),
        np.nan,
    )
    return output


def build_possession_sequence(
    master_join_table: pd.DataFrame,
    splits: dict[str, Any],
    max_gap_frames: int,
    min_stable_frames: int,
) -> pd.DataFrame:
    """Add smoothed possession assignment and change columns."""
    table = master_join_table.copy()
    table = add_data_split_column(table, splits)
    table = table.sort_values(
        ["match_id", "period_id", "tracking_match_clock_seconds", "frame_id"]
    ).reset_index(drop=True)

    table["raw_possession_key"] = [
        _make_possession_key(team_id, player_id, has_possession)
        for team_id, player_id, has_possession in zip(
            table["possessing_team_id"],
            table["possessing_player_id"],
            table["has_possession"],
        )
    ]

    sequence_parts = []
    for _, group in table.groupby(["match_id", "period_id"], sort=False):
        sequence_parts.append(
            _add_possession_sequence_for_group(
                group,
                max_gap_frames=max_gap_frames,
                min_stable_frames=min_stable_frames,
            )
        )

    possession_sequence = pd.concat(sequence_parts, ignore_index=True)
    team_names, player_names = _name_lookup(possession_sequence)
    possession_sequence["smoothed_possession_team_name"] = (
        possession_sequence["smoothed_possession_key"].map(team_names)
    )
    possession_sequence["smoothed_possession_player_name"] = (
        possession_sequence["smoothed_possession_key"].map(player_names)
    )
    possession_sequence["possession_change_type"] = np.select(
        [
            possession_sequence["possession_team_change"],
            possession_sequence["possession_player_change"],
            possession_sequence["smoothed_possession_start"],
        ],
        ["team_change", "player_change", "possession_start"],
        default="no_change",
    )
    return possession_sequence


def summarize_possession_sequence(
    possession_sequence: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row per match."""
    return (
        possession_sequence.groupby("match_id", dropna=False)
        .agg(
            rows=("frame_id", "size"),
            raw_possession_frames=(
                "raw_possession_key",
                lambda values: values.notna().sum(),
            ),
            smoothed_possession_frames=(
                "smoothed_possession_key",
                lambda values: values.notna().sum(),
            ),
            possession_changes=("smoothed_possession_change", "sum"),
            team_changes=("possession_team_change", "sum"),
            player_changes=("possession_player_change", "sum"),
            split=("data_split", "first"),
        )
        .reset_index()
    )


def run_step3(config: Step3Config) -> dict[str, Any]:
    """Run Step 3 from the Step 2 master join table."""
    input_table = config.input_table.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    splits_path = config.match_splits.expanduser().resolve()

    table = pd.read_parquet(input_table)
    if config.match_id is not None:
        table = table[
            table["match_id"].astype(str).eq(str(config.match_id))
        ].copy()
        if table.empty:
            raise ValueError(f"No rows found for match_id={config.match_id}")

    splits = load_match_splits(splits_path)
    possession_sequence = build_possession_sequence(
        table,
        splits=splits,
        max_gap_frames=config.max_gap_frames,
        min_stable_frames=config.min_stable_frames,
    )
    summary = summarize_possession_sequence(possession_sequence)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{config.match_id}" if config.match_id else ""
    table_path = output_dir / f"possession_sequence_table{suffix}.parquet"
    summary_path = output_dir / f"possession_sequence_summary{suffix}.csv"
    metadata_path = output_dir / f"possession_sequence_metadata{suffix}.json"

    possession_sequence.to_parquet(table_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "config": {
                    **asdict(config),
                    "input_table": str(input_table),
                    "output_dir": str(output_dir),
                    "match_splits": str(splits_path),
                },
                "rows": int(len(possession_sequence)),
                "matches": int(
                    possession_sequence["match_id"].astype(str).nunique()
                ),
                "outputs": {
                    "table": str(table_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        )
    )

    return {
        "table": possession_sequence,
        "summary": summary,
        "outputs": {
            "table": str(table_path),
            "summary": str(summary_path),
            "metadata": str(metadata_path),
        },
    }

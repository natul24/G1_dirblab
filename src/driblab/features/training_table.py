"""Build training table for pass classification.

Reads pre_training_table.parquet, computes rolling 2D ball speed over
±5 frames, identifies the closest visible player to the ball for every
row, assigns match-level data splits, and derives the pass binary target.
Writes one parquet per split.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from driblab.config import CONFIG_PATH, MODEL_BASE_DATA_DIR
from driblab.features import match_splits


SPLIT_OUTPUT_PATHS = {
    split: MODEL_BASE_DATA_DIR / f"training_table_{split}.parquet"
    for split in ["train", "validation", "test"]
}

_PLAYER_X_PATTERN = re.compile(r"t\.player_(\d+)_x$")


def _player_slots(schema_names: set[str]) -> list[str]:
    """Return sorted slot identifiers for player columns present in the schema."""
    return sorted(
        _PLAYER_X_PATTERN.match(col).group(1)
        for col in schema_names
        if _PLAYER_X_PATTERN.match(col)
    )


def _closest_player(
    rows: pd.DataFrame, slots: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized closest visible player to ball for every row.

    Returns (closest_player_id, closest_player_team_id) as object arrays
    aligned to rows. Values are NaN where ball position is missing or no
    visible player with known coordinates exists.
    """
    n = len(rows)
    nan_col: np.ndarray = np.full(n, np.nan, dtype=object)

    if n == 0 or not slots:
        return nan_col, nan_col.copy()

    ball_x = pd.to_numeric(rows["t.ball_x"], errors="coerce").to_numpy(dtype=float)
    ball_y = pd.to_numeric(rows["t.ball_y"], errors="coerce").to_numpy(dtype=float)
    ball_missing = np.isnan(ball_x) | np.isnan(ball_y)

    n_slots = len(slots)
    dists = np.full((n, n_slots), np.inf)
    pid_arr = np.full((n, n_slots), np.nan, dtype=object)
    team_arr = np.full((n, n_slots), np.nan, dtype=object)

    for j, slot in enumerate(slots):
        x_col = f"t.player_{slot}_x"
        y_col = f"t.player_{slot}_y"
        vis_col = f"t.player_{slot}_visible"
        id_col = f"t.player_{slot}_id"
        team_col = f"t.player_{slot}_team_id"

        if x_col not in rows.columns:
            continue

        px = pd.to_numeric(rows[x_col], errors="coerce").to_numpy(dtype=float)
        py = pd.to_numeric(rows[y_col], errors="coerce").to_numpy(dtype=float)

        vis_raw = rows[vis_col].astype(str).str.lower()
        visible = (vis_raw == "true").to_numpy(dtype=bool)

        valid = visible & ~np.isnan(px) & ~np.isnan(py) & ~ball_missing
        dists[valid, j] = np.sqrt(
            (px[valid] - ball_x[valid]) ** 2 + (py[valid] - ball_y[valid]) ** 2
        )

        if id_col in rows.columns:
            pid_arr[:, j] = rows[id_col].to_numpy(dtype=object)
        if team_col in rows.columns:
            team_arr[:, j] = rows[team_col].to_numpy(dtype=object)

    min_idx = np.argmin(dists, axis=1)
    all_inf = np.all(dists == np.inf, axis=1)
    row_idx = np.arange(n)

    closest_ids = np.where(all_inf, np.nan, pid_arr[row_idx, min_idx])
    closest_teams = np.where(all_inf, np.nan, team_arr[row_idx, min_idx])

    return closest_ids, closest_teams


def _add_ball_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Add ball_speed_avg_xy: rolling mean of 2D frame-to-frame speed over ±5 frames.

    Computed per (match_id, period) group so steps never cross boundaries.
    The rolling window is 11 frames wide (center=True), giving each row an
    average speed drawn from 5 frames before and 5 frames after.
    """
    parts = []
    for _, group in df.groupby(["t.match_id", "t.period"], sort=False):
        bx = pd.to_numeric(group["t.ball_x"], errors="coerce")
        by = pd.to_numeric(group["t.ball_y"], errors="coerce")
        step = np.sqrt(bx.diff() ** 2 + by.diff() ** 2)
        parts.append(step.rolling(window=11, center=True, min_periods=1).mean())

    df = df.copy()
    df["ball_speed_avg_xy"] = pd.concat(parts).reindex(df.index)
    return df


def build_training_table(pre_training_path: Path) -> pd.DataFrame:
    schema_names = set(pq.read_schema(pre_training_path).names)
    slots = _player_slots(schema_names)

    df = pd.read_parquet(pre_training_path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns\n")

    splits = match_splits.load_match_splits(CONFIG_PATH)
    df = match_splits.add_data_split_column(df, splits, match_col="t.match_id")
    df = df.sort_values(
        ["t.match_id", "t.period", "t.frame"],
        kind="mergesort",
    ).reset_index(drop=True)

    df = _add_ball_speed(df)

    closest_ids, closest_teams = _closest_player(df, slots)
    df["closest_player_id"] = closest_ids
    df["closest_player_team_id"] = closest_teams

    df["is_pass"] = (df["p.event_label"] == "PASS").astype(int)

    t_cols = [c for c in df.columns if c.startswith("t.")]
    added_cols = [
        "p.event_label",
        "data_split",
        "is_pass",
        "ball_speed_avg_xy",
        "closest_player_id",
        "closest_player_team_id",
    ]
    df = df[t_cols + added_cols]

    # Sample: keep the first frame of every non-overlapping 5-frame window.
    # Within each (match_id, period) group (already sorted by t.frame), rows at
    # positions 0, 5, 10, … are selected — one row per 0.5-second window.
    row_in_group = df.groupby(
        ["t.match_id", "t.period"], sort=False
    ).cumcount()
    df = df.loc[row_in_group % 5 == 0].reset_index(drop=True)

    print(f"Built {len(df):,} rows × {df.shape[1]} columns  (1 per 5-frame window)\n")
    return df


def main() -> None:
    pre_training_path = MODEL_BASE_DATA_DIR / "pre_training_table.parquet"
    training_table = build_training_table(pre_training_path)

    MODEL_BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, path in SPLIT_OUTPUT_PATHS.items():
        split_df = training_table[training_table["data_split"] == split_name]
        split_df.to_parquet(path, index=False)
        print(f"Saved {split_name:12s}: {len(split_df):,} rows -> {path.name}")


if __name__ == "__main__":
    main()

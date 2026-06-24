"""Build training windows for pass classification.

Reads pre_training_table.parquet, groups frames into 5-frame non-overlapping
windows, computes 2D ball speed, selects the primary event per window,
and computes the closest visible player to the ball at that event frame.
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


WINDOW_SIZE = 5

OUTPUT_COLUMNS = [
    "t.match_id",
    "t.period",
    "window_time",
    "primary_event_frame",
    "data_split",
    "p.event_label",
    "is_pass",
    "ball_speed_avg_xy",
    "closest_player_id",
    "closest_player_team_id",
]

SPLIT_OUTPUT_PATHS = {
    split: MODEL_BASE_DATA_DIR / f"training_table_{split}.parquet"
    for split in ["train", "validation", "test"]
}

_BASE_INPUT_COLUMNS = [
    "t.match_id",
    "t.period",
    "t.frame",
    "t.ball_x",
    "t.ball_y",
    "p.event_label",
    "p.dist_to_actual_event",
]

_PLAYER_X_PATTERN = re.compile(r"t\.player_(\d+)_x$")


def _player_columns(parquet_path: Path) -> tuple[list[str], list[str]]:
    """Return (columns_to_load, slot_list) for player data in the parquet."""
    schema_names = set(pq.read_schema(parquet_path).names)
    slots = sorted(
        _PLAYER_X_PATTERN.match(col).group(1)
        for col in schema_names
        if _PLAYER_X_PATTERN.match(col)
    )
    cols = []
    for slot in slots:
        for suffix in ("_x", "_y", "_visible", "_id", "_team_id"):
            col = f"t.player_{slot}{suffix}"
            if col in schema_names:
                cols.append(col)
    return cols, slots


def _closest_player(
    rows: pd.DataFrame, slots: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized closest visible player for each row in `rows`.

    Returns (closest_player_id, closest_player_team_id) as object arrays
    aligned to rows. Values are NaN where ball position is missing or no
    visible player with known coordinates exists.
    """
    n = len(rows)
    nan_col: np.ndarray = np.full(n, np.nan, dtype=object)

    if n == 0 or not slots:
        return nan_col, nan_col.copy()

    ball_x = pd.to_numeric(
        rows["t.ball_x"],
        errors="coerce",
    ).to_numpy(dtype=float)
    ball_y = pd.to_numeric(
        rows["t.ball_y"],
        errors="coerce",
    ).to_numpy(dtype=float)
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


def build_training_table(pre_training_path: Path) -> pd.DataFrame:
    player_load_cols, player_slots = _player_columns(pre_training_path)
    input_columns = _BASE_INPUT_COLUMNS + player_load_cols

    df = pd.read_parquet(pre_training_path, columns=input_columns)
    print(f"Loaded {len(df):,} rows\n")

    splits = match_splits.load_match_splits(CONFIG_PATH)
    df = match_splits.add_data_split_column(df, splits, match_col="t.match_id")
    df = df.sort_values(
        ["t.match_id", "t.period", "t.frame"],
        kind="mergesort",
    ).reset_index(drop=True)

    group_cols = ["t.match_id", "t.period"]
    grouped = df.groupby(group_cols, sort=False)
    row_in_group = grouped.cumcount()
    group_size = grouped["t.frame"].transform("size")
    complete_row_count = (group_size // WINDOW_SIZE) * WINDOW_SIZE
    df = df.loc[row_in_group < complete_row_count].copy()
    row_in_group = row_in_group.loc[df.index]

    df["_window_idx"] = row_in_group // WINDOW_SIZE
    df["_frame_in_window"] = row_in_group % WINDOW_SIZE
    window_cols = ["t.match_id", "t.period", "_window_idx"]

    dx = pd.to_numeric(df["t.ball_x"], errors="coerce").diff()
    dy = pd.to_numeric(df["t.ball_y"], errors="coerce").diff()
    same_window_step = df["_frame_in_window"] > 0
    df["_ball_step_xy"] = np.where(
        same_window_step,
        np.sqrt(dx**2 + dy**2),
        np.nan,
    )

    base_windows = (
        df.loc[df["_frame_in_window"] == 0]
        .assign(
            window_time=lambda t: (t["_window_idx"] + 1) * 0.5,
            **{"p.event_label": "no event"},
        )[
            [
                "t.match_id",
                "t.period",
                "_window_idx",
                "window_time",
                "data_split",
                "p.event_label",
            ]
        ]
    )

    speed_by_window = (
        df.groupby(window_cols, sort=False)["_ball_step_xy"]
        .mean()
        .rename("ball_speed_avg_xy")
        .reset_index()
    )

    event_rows = df.loc[df["p.event_label"] != "no event"].copy()
    if event_rows.empty:
        event_by_window = pd.DataFrame(
            columns=[
                *window_cols,
                "p.event_label",
                "primary_event_frame",
                "closest_player_id",
                "closest_player_team_id",
            ]
        )
    else:
        event_rows["_event_distance"] = pd.to_numeric(
            event_rows["p.dist_to_actual_event"], errors="coerce"
        ).fillna(np.inf)
        event_indices = (
            event_rows.groupby(window_cols, sort=False)["_event_distance"]
            .idxmin()
            .to_numpy()
        )
        primary_rows = event_rows.loc[event_indices]
        closest_ids, closest_teams = _closest_player(
            primary_rows,
            player_slots,
        )

        event_by_window = (
            primary_rows[[*window_cols, "p.event_label", "t.frame"]]
            .rename(columns={"t.frame": "primary_event_frame"})
            .assign(
                closest_player_id=closest_ids,
                closest_player_team_id=closest_teams,
            )
        )

    training_table = (
        base_windows.drop(columns=["p.event_label"])
        .merge(event_by_window, on=window_cols, how="left")
        .merge(speed_by_window, on=window_cols, how="left")
    )
    training_table["p.event_label"] = training_table[
        "p.event_label"
    ].fillna("no event")
    training_table["is_pass"] = (
        training_table["p.event_label"] == "PASS"
    ).astype(int)
    training_table = training_table[OUTPUT_COLUMNS]
    print(f"Created {len(training_table):,} windows\n")
    return training_table


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

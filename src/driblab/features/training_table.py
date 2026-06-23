"""Build 7-column training windows for pass classification.

Reads pre_training_table.parquet, groups frames into 5-frame non-overlapping
windows, computes 2D ball speed, selects the primary event per window, and
writes training_table_simple.parquet with exactly 7 output columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driblab.config import CONFIG_PATH, MODEL_BASE_DATA_DIR
from driblab.features import match_splits


WINDOW_SIZE = 5

OUTPUT_COLUMNS = [
    "t.match_id",
    "t.period",
    "window_time",
    "data_split",
    "p.event_label",
    "is_pass",
    "ball_speed_avg_xy",
]

SPLIT_OUTPUT_PATHS = {
    split: MODEL_BASE_DATA_DIR / f"training_table_{split}.parquet"
    for split in ["train", "validation", "test"]
}
INPUT_COLUMNS = [
    "t.match_id",
    "t.period",
    "t.frame",
    "t.ball_x",
    "t.ball_y",
    "p.event_label",
    "p.dist_to_actual_event",
]


def build_training_table(pre_training_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pre_training_path, columns=INPUT_COLUMNS)
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
            window_time=lambda table: (
                (table["_window_idx"] + 1) * 0.5
            ),
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
            columns=[*window_cols, "p.event_label"],
        )
    else:
        event_rows["_event_distance"] = pd.to_numeric(
            event_rows["p.dist_to_actual_event"],
            errors="coerce",
        ).fillna(np.inf)
        event_indices = (
            event_rows.groupby(window_cols, sort=False)["_event_distance"]
            .idxmin()
            .to_numpy()
        )
        event_by_window = event_rows.loc[
            event_indices,
            [*window_cols, "p.event_label"],
        ]

    training_table = (
        base_windows.drop(columns=["p.event_label"])
        .merge(event_by_window, on=window_cols, how="left")
        .merge(speed_by_window, on=window_cols, how="left")
    )
    training_table["p.event_label"] = training_table["p.event_label"].fillna(
        "no event",
    )
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

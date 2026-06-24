"""Build pre-training table from the master join.

Reads master_join_table.parquet. For each tracking frame, assigns
p.event_label by finding the nearest real event anchor within ±1 second
using merge_asof per (match_id, period) group.

Output: all t.* tracking columns + p.event_label.
Intermediate columns p.actual_event_frame and p.dist_to_actual_event
are used during labeling but are not written to the parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from driblab.config import MODEL_BASE_DATA_DIR


WINDOW_SEC = 1.0

INPUT_PATH = MODEL_BASE_DATA_DIR / "master_join_table.parquet"
OUTPUT_PATH = MODEL_BASE_DATA_DIR / "pre_training_table.parquet"


def build_pre_training_table(master_join_path: Path) -> pd.DataFrame:
    master_join = pd.read_parquet(master_join_path)
    master_join["t.Videotimestamp"] = pd.to_numeric(
        master_join["t.Videotimestamp"], errors="coerce"
    )
    print(f"Loaded {len(master_join):,} rows\n")

    # Event anchor table: only real events, "no event" rows never become anchors
    events_df = master_join.loc[
        master_join["e.event.event_type_name"] != "no event",
        ["t.match_id", "t.period", "t.Videotimestamp", "e.event.event_type_name"],
    ].rename(columns={
        "t.Videotimestamp"         : "p.actual_event_frame",
        "e.event.event_type_name"  : "p.nearest_event_label",
    })

    labeled_chunks = []

    for (match_id, period), period_tracking in master_join.groupby(
        ["t.match_id", "t.period"], sort=False
    ):
        period_events = (
            events_df[
                (events_df["t.match_id"] == match_id)
                & (events_df["t.period"] == period)
            ]
            .sort_values("p.actual_event_frame")
        )

        chunk = period_tracking.sort_values("t.Videotimestamp").copy()

        if period_events.empty:
            chunk["p.event_label"] = "no event"
            labeled_chunks.append(chunk)
            continue

        merged = pd.merge_asof(
            chunk,
            period_events[["p.actual_event_frame", "p.nearest_event_label"]],
            left_on="t.Videotimestamp",
            right_on="p.actual_event_frame",
            direction="nearest",
        )

        dist = (merged["t.Videotimestamp"] - merged["p.actual_event_frame"]).abs()
        within = dist <= WINDOW_SEC

        merged["p.event_label"] = np.where(
            within, merged["p.nearest_event_label"], "no event"
        )
        merged = merged.drop(columns=["p.nearest_event_label", "p.actual_event_frame"])
        labeled_chunks.append(merged)

    pre_training = pd.concat(labeled_chunks, ignore_index=True)

    # Drop all e.* columns and any leftover p.* intermediates
    drop_cols = [c for c in pre_training.columns if c.startswith("e.")]
    pre_training = pre_training.drop(columns=drop_cols)

    t_cols = [c for c in pre_training.columns if c.startswith("t.")]
    return pre_training[t_cols + ["p.event_label"]]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}\n"
            "Run: python main.py master-join"
        )

    pre_training = build_pre_training_table(INPUT_PATH)

    MODEL_BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pre_training.to_parquet(OUTPUT_PATH, index=False)

    t_cols = sum(c.startswith("t.") for c in pre_training.columns)
    print(f"Saved {len(pre_training):,} rows → {OUTPUT_PATH}")
    print(f"Columns: {pre_training.shape[1]}  (t.*: {t_cols}, p.*: 1)")
    print()
    print("Label distribution:")
    for label, count in pre_training["p.event_label"].value_counts().items():
        print(f"  {label}: {count:,}  ({count / len(pre_training) * 100:.2f}%)")


if __name__ == "__main__":
    main()

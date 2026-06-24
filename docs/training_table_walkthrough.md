# Training Table Walkthrough

This guide documents what `src/driblab/features/training_table.py` currently
builds.

The current training table is a compact pass-classification table built from
`pre_training_table.parquet`. It groups tracking frames into 0.5-second
windows, selects one event label per window, computes one ball-speed feature,
adds the closest visible player at the selected event frame, and writes one
parquet file per split.

## 1. Inputs and Outputs

The module reads:

```text
data/processed/model_base/pre_training_table.parquet
config.yaml
```

`pre_training_table.parquet` is created by
`notebooks/pre_training_table.ipynb`. It contains all `t.*` tracking columns
plus these pre-training event-label columns:

```text
p.actual_event_frame
p.event_label
p.dist_to_actual_event
```

The training-table module writes:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
```

It does not write summary CSVs, scaler artifacts, or model artifacts.

## 2. High-Level Workflow

The actual order in `training_table.py` is:

1. Read only the columns needed from `pre_training_table.parquet`.
2. Detect available player-slot columns from the parquet schema.
3. Load match-level train, validation, and test split definitions.
4. Add `data_split` before feature engineering.
5. Sort rows by `t.match_id`, `t.period`, and `t.frame`.
6. Build consecutive non-overlapping 5-frame windows inside each match-period.
7. Compute average 2D ball movement per window.
8. Select the primary labelled event per window from `p.event_label`.
9. Find the closest visible player to the ball at the selected event frame.
10. Create `is_pass`.
11. Split the table by `data_split`.
12. Write one parquet file per split.

No standardization is applied in this module. All output values are in raw
units or identifiers.

## 3. Split Rule

Splits are assigned with:

```text
driblab.features.match_splits.add_data_split_column
```

The split values are:

```text
train
validation
test
```

The split is match-level, so all frames from the same match stay in the same
split. This prevents row-level leakage between train, validation, and test.

## 4. Window Construction

Rows are grouped by:

```text
t.match_id
t.period
```

Inside each group, rows are sorted by:

```text
t.frame
```

The module creates consecutive non-overlapping 5-frame windows:

```text
frames 1-5   -> window_time = 0.5
frames 6-10  -> window_time = 1.0
frames 11-15 -> window_time = 1.5
```

Trailing partial groups with fewer than 5 frames are discarded.

Windows with missing ball coordinates are kept. If no valid ball movement step
exists inside a window, `ball_speed_avg_xy` is `NaN`.

## 5. Ball-Speed Feature

The ball-speed feature is:

```text
ball_speed_avg_xy
```

For each 5-frame window, the module computes up to four frame-to-frame 2D
movements:

```text
sqrt((x[i+1] - x[i])^2 + (y[i+1] - y[i])^2)
```

Only valid steps where both endpoints have non-missing `t.ball_x` and
`t.ball_y` contribute to the mean.

The unit is meters per frame. At 10 Hz, multiply by 10 to express it as meters
per second.

## 6. Event Selection

Each frame already has:

```text
p.event_label
p.dist_to_actual_event
```

The module ignores frames where:

```text
p.event_label == "no event"
```

If at least one labelled event exists inside a 5-frame window, the primary
event is the row with the smallest numeric `p.dist_to_actual_event`. Missing
distances are treated as infinity for sorting.

The selected row provides:

```text
p.event_label
primary_event_frame
```

If the window has no labelled event:

```text
p.event_label = "no event"
primary_event_frame = missing
```

## 7. Closest Player At Event Frame

For windows with a selected event row, the module finds the closest visible
player to the ball on that same selected frame.

The module detects player slots from columns like:

```text
t.player_01_x
t.player_01_y
t.player_01_visible
t.player_01_id
t.player_01_team_id
```

A player is eligible when:

```text
t.player_XX_visible == true
t.player_XX_x is present
t.player_XX_y is present
```

Distance uses raw x/y tracking coordinates:

```text
sqrt((player_x - ball_x)^2 + (player_y - ball_y)^2)
```

The output columns are:

```text
closest_player_id
closest_player_team_id
```

They are missing when the window has no selected event, when ball position is
missing on the selected event frame, or when no visible player with valid x/y
coordinates is available.

## 8. Target

The binary target is:

```text
is_pass = 1 when p.event_label == "PASS"
is_pass = 0 otherwise
```

## 9. Output Columns

The schema is fixed by `OUTPUT_COLUMNS` in `training_table.py`.

The output table has exactly 10 columns:

| # | Column | Description |
| ---: | --- | --- |
| 1 | `t.match_id` | Source match ID. |
| 2 | `t.period` | Source tracking period. |
| 3 | `window_time` | End time of the 5-frame window in seconds within the period. |
| 4 | `primary_event_frame` | Source `t.frame` for the selected labelled event row. Missing for no-event windows. |
| 5 | `data_split` | Match-level split: `train`, `validation`, or `test`. |
| 6 | `p.event_label` | Selected event label for the window, or `"no event"`. |
| 7 | `is_pass` | Binary target derived from `p.event_label == "PASS"`. |
| 8 | `ball_speed_avg_xy` | Mean valid 2D frame-to-frame ball movement in meters per frame. |
| 9 | `closest_player_id` | ID of the closest visible player to the ball at `primary_event_frame`. |
| 10 | `closest_player_team_id` | Team ID for `closest_player_id`. |

## 10. Rebuild Command

Before running, make sure `pre_training_table.parquet` exists. If it does not,
run `notebooks/pre_training_table.ipynb` first.

From the project root:

```bash
conda activate driblabvenv
python -m driblab.features.training_table
```

Expected console output:

```text
Loaded 1,986,630 rows
Created 397,297 windows
Saved train      : 279,437 rows -> training_table_train.parquet
Saved validation : 59,372 rows -> training_table_validation.parquet
Saved test       : 58,488 rows -> training_table_test.parquet
```

## 11. Most Recent Output Summary

The most recent generated split tables contained:

| Split | Rows | Pass rows | Pass rate |
| --- | ---: | ---: | ---: |
| train | 279,437 | 88,363 | 31.62% |
| validation | 59,372 | 20,073 | 33.81% |
| test | 58,488 | 21,108 | 36.09% |

Event-label counts by split:

| Split | no event | PASS | BALL TOUCH | TACKLE | BALL RECOVERY | AERIAL | TAKEON | FOUL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 164,677 | 88,363 | 11,295 | 3,993 | 2,951 | 3,063 | 2,714 | 2,381 |
| validation | 33,618 | 20,073 | 2,777 | 682 | 554 | 444 | 732 | 492 |
| test | 31,829 | 21,108 | 2,334 | 959 | 616 | 544 | 599 | 499 |

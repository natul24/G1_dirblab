# Training Table Walkthrough

This guide documents what `src/driblab/features/training_table.py` builds.

## 1. Inputs and Outputs

The module reads:

```text
data/processed/model_base/pre_training_table.parquet
config.yaml
```

`pre_training_table.parquet` is built by running `notebooks/pre_training_table.ipynb`.
It extends the master join table with `p.*` event label columns assigned to every tracking frame.

The module writes one file per split:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
```

No summary CSVs or scaler artifacts are written by this module.

## 2. High-Level Workflow

1. Load the pre-training table (selected columns only).
2. Load match-level split definitions from `config.yaml`.
3. Add `data_split` column before any feature engineering.
4. Sort all rows by `(t.match_id, t.period, t.frame)`.
5. Build non-overlapping 5-frame windows across every (match, period) group.
6. Compute 2D ball speed for each window.
7. Select the primary event per window from `p.event_label`.
8. Derive the binary target `is_pass`.
9. Write the output table.

No standardization is applied. The output values are in raw units.

## 3. Split Rule

The split is assigned before any feature engineering using:

```text
driblab.features.match_splits.add_data_split_column
```

The `data_split` column values are:

```text
train
validation
test
```

Every frame from the same match lands in the same split, preventing data leakage between splits.
All windows are built together in one pass; the `data_split` column identifies each window's split.

## 4. Window Construction

Rows are sorted by `(t.match_id, t.period, t.frame)` and grouped by `(t.match_id, t.period)`.

Within each group, consecutive non-overlapping 5-frame windows are assigned:

```text
frames 1–5   → window_time = 0.5
frames 6–10  → window_time = 1.0
frames 11–15 → window_time = 1.5
…
```

`window_time` is `(window_index + 1) * 0.5` and measures elapsed seconds within the
(match, period). It resets at the start of every new period.

Trailing partial groups with fewer than 5 frames are discarded.
Windows where all ball coordinates are missing are kept — `ball_speed_avg_xy` is `NaN`
for those windows.

## 5. Ball Speed Feature

One feature is computed per window:

```text
ball_speed_avg_xy
```

This is the mean of the four frame-to-frame 2D Euclidean distances inside the window:

```text
sqrt((x[i+1] - x[i])^2 + (y[i+1] - y[i])^2)
```

Only steps where both endpoints have non-missing `t.ball_x` and `t.ball_y` contribute to
the mean. If no valid step exists, `ball_speed_avg_xy` is `NaN`.

The unit is metres per frame. At 10 Hz, multiply by 10 to get m/s.

## 6. Event Selection

Each frame already has a `p.event_label` from the pre-training step.
Frames where `p.event_label == "no event"` are excluded from event selection.

Within each window, among all frames with a real event label, the primary event is the
frame with the smallest `p.dist_to_actual_event` (time distance in seconds to the event
anchor). The label from that frame becomes the window's `p.event_label`.
Missing distances are treated as infinity for sorting.

If the window has no labeled frames, `p.event_label = "no event"`.

## 7. Target

```text
is_pass = 1  when p.event_label == "PASS"
is_pass = 0  otherwise
```

## 8. Output Columns

The output table has exactly 7 columns:

| # | Column | Description |
|---:|---|---|
| 1 | `t.match_id` | Source match ID. |
| 2 | `t.period` | Source tracking period. |
| 3 | `window_time` | End time of the 5-frame window in seconds within the period. |
| 4 | `data_split` | Match-level split: `train`, `validation`, or `test`. |
| 5 | `p.event_label` | Selected event type for the window, or `"no event"`. |
| 6 | `is_pass` | Binary target: `1` = PASS, `0` = anything else. |
| 7 | `ball_speed_avg_xy` | Mean 2D frame-to-frame ball speed in metres per frame. `NaN` when ball tracking is unavailable. |

## 9. Rebuild Command

Before running, ensure `pre_training_table.parquet` exists by running
`notebooks/pre_training_table.ipynb` first.

```bash
conda activate driblabvenv
python -m driblab.features.training_table
```

This writes:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
```

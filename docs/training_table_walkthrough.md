# Training Table Walkthrough

This guide documents what `src/driblab/features/training_table.py` currently
builds.

The current training table is a compact pass-classification table built from
`pre_training_table.parquet`. It keeps the tracking columns, assigns
match-level train/validation/test splits, computes one rolling ball-speed
feature, identifies the closest visible player to the ball, creates the
`is_pass` binary target, and writes one parquet file per split.

## 1. Inputs and Outputs

The module reads:

```text
data/processed/model_base/pre_training_table.parquet
config.yaml
```

`pre_training_table.parquet` is created by:

```bash
python main.py pre-training
```

It contains all `t.*` tracking columns plus one pre-training label column:

```text
p.event_label
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

1. Read `pre_training_table.parquet`.
2. Detect available player-slot columns from the parquet schema.
3. Load match-level train, validation, and test split definitions.
4. Add `data_split`.
5. Sort rows by `t.match_id`, `t.period`, and `t.frame`.
6. Compute rolling 2D ball speed per match-period.
7. Find the closest visible player to the ball for each row.
8. Create `is_pass` from `p.event_label`.
9. Keep all `t.*` columns plus the six added model columns.
10. Sample the first row of every non-overlapping 5-frame interval.
11. Split the table by `data_split`.
12. Write one parquet file per split.

No standardization is applied in this module. All output values are in raw
tracking units or identifiers.

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

## 4. Sampling Rule

Rows are grouped by:

```text
t.match_id
t.period
```

Inside each group, rows are sorted by:

```text
t.frame
```

The module keeps rows at positions:

```text
0, 5, 10, 15, ...
```

This produces one sampled row per 5-frame interval, approximately one row every
0.5 seconds at 10 Hz. The sampled row keeps the original `t.*` values and the
pre-training `p.event_label` from that frame.

## 5. Ball-Speed Feature

The ball-speed feature is:

```text
ball_speed_avg_xy
```

For each `(t.match_id, t.period)` group, the module computes frame-to-frame 2D
ball movement:

```text
sqrt((x[i] - x[i-1])^2 + (y[i] - y[i-1])^2)
```

It then applies a centered rolling mean over 11 frames, equivalent to up to 5
frames before and 5 frames after the current row:

```text
step.rolling(window=11, center=True, min_periods=1).mean()
```

The unit is meters per frame. At 10 Hz, multiply by 10 to express it as meters
per second.

## 6. Event Labels

Each row already has:

```text
p.event_label
```

That label comes from the pre-training stage. The training-table stage does not
reselect a primary event inside each 5-frame interval and does not use event
coordinates as model features.

## 7. Closest Player

The module finds the closest visible player to the ball for each row before the
5-frame sampling step.

It detects player slots from columns like:

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

They are missing when ball position is missing or when no visible player with
valid x/y coordinates is available.

## 8. Target

The binary target is:

```text
is_pass = 1 when p.event_label == "PASS"
is_pass = 0 otherwise
```

## 9. Output Columns

The current generated training tables have 127 columns:

- 121 `t.*` tracking columns from `pre_training_table.parquet`
- 6 added training/model columns

The six added columns are:

| Column | Description |
| --- | --- |
| `p.event_label` | Event label assigned in the pre-training stage. |
| `data_split` | Match-level split: `train`, `validation`, or `test`. |
| `is_pass` | Binary target derived from `p.event_label == "PASS"`. |
| `ball_speed_avg_xy` | Centered rolling mean of 2D frame-to-frame ball movement in meters per frame. |
| `closest_player_id` | ID of the closest visible player to the ball at the sampled row. |
| `closest_player_team_id` | Team ID for `closest_player_id`. |

## 10. Rebuild Command

Before running, make sure `pre_training_table.parquet` exists. If it does not,
build the upstream stages first.

From the project root:

```bash
conda activate driblabvenv
python main.py training-table
```

Or run the full current pipeline:

```bash
python main.py all
```

Expected console output includes:

```text
Loaded 1,986,630 rows, 122 columns
Built 397,351 rows x 127 columns  (1 per 5-frame window)
Saved train       : 279,474 rows -> training_table_train.parquet
Saved validation  : 59,382 rows -> training_table_validation.parquet
Saved test        : 58,495 rows -> training_table_test.parquet
```

## 11. Most Recent Output Summary

The most recent generated split tables contained:

| Split | Rows | Pass rows | Pass rate |
| --- | ---: | ---: | ---: |
| train | 279,474 | 78,561 | 28.11% |
| validation | 59,382 | 17,849 | 30.06% |
| test | 58,495 | 18,820 | 32.17% |

Event-label counts by split:

| Split | no event | PASS | BALL TOUCH | TACKLE | BALL RECOVERY | AERIAL | TAKEON | FOUL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 177,085 | 78,561 | 10,261 | 3,603 | 2,727 | 2,792 | 2,426 | 2,019 |
| validation | 36,399 | 17,849 | 2,525 | 627 | 507 | 404 | 654 | 417 |
| test | 34,675 | 18,820 | 2,095 | 877 | 570 | 498 | 534 | 426 |

# Training Table Walkthrough

This guide explains the pass-detection training table built from the Step 2
master join output.

The training-table module creates one 0.5-second window table per data split:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
```

It also creates one summary CSV per split:

```text
data/processed/model_base/training_table_summary_train.csv
data/processed/model_base/training_table_summary_validation.csv
data/processed/model_base/training_table_summary_test.csv
```

## 1. Why This Table Exists

The master join table is frame-level data. Each row is one tracking frame at
about 10 Hz.

The pass model needs a more useful modelling grain: short windows of play. The
training table converts the frame-level master join into non-overlapping
5-frame windows, which represent about 0.5 seconds.

Each output row answers this question:

```text
During this 0.5-second window, does the primary attached event look like a pass?
```

The target column is:

```text
is_pass = 1 when primary_event == "PASS"
is_pass = 0 otherwise
```

## 2. The Critical Split Rule

The split is applied before feature engineering.

The workflow is:

1. Load `data/processed/model_base/master_join_table.parquet`.
2. Load match split definitions from `config.yaml`.
3. Add `data_split` using
   `driblab.features.match_splits.add_data_split_column`.
4. Filter to one split at a time.
5. Build windows and features inside that split only.
6. Write separate train, validation, and test files.

This prevents leakage between train, validation, and test because no windowing
or feature computation is allowed to mix matches from different splits.

The split definitions are match-level holdouts:

```text
train
validation
test
```

## 3. Window Construction

Within each `t.match_id` and `t.period`, rows are sorted by:

```text
t.match_id
t.period
t.frame
```

Then the module creates consecutive non-overlapping 5-frame windows:

```text
frames 1-5   -> window_time = 0.5
frames 6-10  -> window_time = 1.0
frames 11-15 -> window_time = 1.5
```

Any trailing partial group with fewer than 5 frames is ignored. This keeps the
output contract simple: every training row represents the same number of input
frames.

## 4. Confirmed Skip Rule

A window is skipped only when all ball position values are missing across the
whole 5-frame window:

```text
t.ball_x
t.ball_y
t.ball_z
```

If some ball coordinates are missing but at least one ball coordinate exists in
the window, the row is kept. The missing values remain missing in the output.

This is intentional because XGBoost can handle missing numeric values, and
partial tracking gaps should not automatically remove a useful training
example.

The most recent build skipped:

| Split | Skipped all-ball-missing windows |
| --- | ---: |
| train | 117,932 |
| validation | 22,653 |
| test | 23,403 |

## 5. Ball Features

For each window, the module computes average raw tracking ball coordinates:

```text
ball_x_avg
ball_y_avg
ball_z_avg
```

These stay in the raw tracking coordinate system. No normalization, flipping,
or event-coordinate conversion is applied at this stage.

The module also computes:

```text
ball_speed_avg
```

This is the mean Euclidean frame-to-frame distance across the window:

```text
sqrt((x[i+1] - x[i])^2 + (y[i+1] - y[i])^2 + (z[i+1] - z[i])^2)
```

Then those inter-frame distances are averaged.

## 6. Closest-Player Features

For the first and fifth frame of each window, the module finds the visible
player closest to the ball.

Only player slots with these conditions are considered:

```text
t.player_XX_visible == True
t.player_XX_team_id is present
t.player_XX_x and t.player_XX_y are present
```

The distance uses raw tracking x/y coordinates:

```text
sqrt((player_x - ball_x)^2 + (player_y - ball_y)^2)
```

The output columns are:

```text
closest_player_dist_start
closest_player_team_start
closest_player_dist_end
closest_player_team_end
```

If no visible player can be found, the distance is missing and the team is:

```text
unknown
```

## 7. Player Change Signal

The pass-signal feature is:

```text
player_changed_same_team
```

It is `1` when:

- the closest player at the start is different from the closest player at the
  end
- both closest players are on the same team
- both closest players are known

Otherwise it is `0`.

This feature is not the target. It is a behavioural signal: the ball may have
moved from one teammate to another during the 0.5-second window.

## 8. Event Selection

A window can contain zero, one, or several attached Step 2 events.

The module looks at:

```text
e.event.event_type_name
nearest_timestamp_distance_sec
```

Rows where `e.event.event_type_name == "no event"` are ignored for primary
event selection.

If the window has at least one event, the primary event is the event with the
smallest `nearest_timestamp_distance_sec`.

If the window has no event:

```text
primary_event = "no event"
secondary_events = ""
is_pass = 0
```

All non-primary events in the same window are stored as a comma-separated
string:

```text
secondary_events
```

## 9. Output Columns

The training table columns are:

| Column | Meaning |
| --- | --- |
| `t.match_id` | Source match id. |
| `t.period` | Source tracking period. |
| `window_time` | End time of the 5-frame window in seconds within the period. |
| `data_split` | Match-level split: `train`, `validation`, or `test`. |
| `primary_event` | Event selected for the window, or `"no event"`. |
| `is_pass` | Binary target, `1` when `primary_event == "PASS"`. |
| `secondary_events` | Other event types in the window, comma-separated. |
| `ball_x_avg` | Mean raw tracking ball x coordinate in the window. |
| `ball_y_avg` | Mean raw tracking ball y coordinate in the window. |
| `ball_z_avg` | Mean raw tracking ball z coordinate in the window. |
| `ball_speed_avg` | Mean raw tracking ball movement per frame. |
| `closest_player_dist_start` | Closest visible player distance in frame 1. |
| `closest_player_team_start` | Closest visible player team in frame 1. |
| `closest_player_dist_end` | Closest visible player distance in frame 5. |
| `closest_player_team_end` | Closest visible player team in frame 5. |
| `player_changed_same_team` | `1` when closest player changes to a teammate. |

## 10. How to Rebuild

From the project root, use the project conda environment:

```bash
conda activate driblabvenv
PYTHONPATH=src python -m driblab.features.training_table
```

If the package is installed in editable mode inside the environment, the
`PYTHONPATH=src` prefix is optional:

```bash
python -m driblab.features.training_table
```

The build reads:

```text
data/processed/model_base/master_join_table.parquet
config.yaml
```

and writes the split outputs into:

```text
data/processed/model_base/
```

## 11. Most Recent Output Summary

The most recent generated files contained:

| Split | Windows | Pass windows | No-event windows | Pass percentage | Matches | Periods |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 161,505 | 15,554 | 140,892 | 9.63% | 23 | 2 |
| validation | 36,719 | 3,567 | 32,044 | 9.71% | 5 | 2 |
| test | 35,085 | 3,725 | 30,251 | 10.62% | 5 | 2 |

These summaries are useful quick checks before training a model. The pass rate
is similar across splits, which is a good sign for the match-level holdout.

## 12. Important Notes

- The table keeps raw tracking coordinates.
- Event coordinates are not used in this feature table yet.
- No feature computation crosses split boundaries.
- No possession model is used.
- No attacking-direction normalization is applied.
- Missing numeric values are preserved when the window is otherwise valid.
- The target is based only on the selected primary event in the 0.5-second
  window.
- `player_changed_same_team` is a feature, not a label.

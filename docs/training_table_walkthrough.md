# Training Table Walkthrough

This guide documents exactly what `src/driblab/features/training_table.py`
currently builds.

The module reads the Step 2 master join table, assigns match-level train,
validation, and test splits, creates non-overlapping 0.5-second windows, and
writes one training table plus one summary CSV per split. By default, the
training tables are normalized after they are built.

Output tables:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
```

Output summaries:

```text
data/processed/model_base/training_table_summary_train.csv
data/processed/model_base/training_table_summary_validation.csv
data/processed/model_base/training_table_summary_test.csv
```

Saved scaler:

```text
artifacts/models/feature_scaler.pkl
```

## 1. Target

Each row is one 5-frame window. The target is:

```text
is_pass = 1 when primary_event == "PASS"
is_pass = 0 otherwise
```

`primary_event` is selected from the events attached to the frames inside the
window. If the window has no event, `primary_event` is `"no event"` and
`is_pass` is `0`.

## 2. Split Rule

The split is assigned before feature engineering.

The workflow is:

1. Load `data/processed/model_base/master_join_table.parquet`.
2. Load match split definitions from `config.yaml`.
3. Add `data_split` with
   `driblab.features.match_splits.add_data_split_column`.
4. Build windows separately inside each split.
5. Write one table and one summary CSV per split.
6. Fit `StandardScaler` on the train table only.
7. Apply that scaler to train, validation, and test.
8. Overwrite the same three parquet files with normalized values.
9. Save the fitted scaler to `artifacts/models/feature_scaler.pkl`.

No window crosses split boundaries, and validation/test statistics are not used
to fit the scaler.

## 3. Window Construction

Rows are grouped by:

```text
t.match_id
t.period
```

Inside each match-period group, rows are sorted by:

```text
t.match_id
t.period
t.frame
```

The module then creates consecutive non-overlapping 5-frame windows:

```text
frames 1-5   -> window_time = 0.5
frames 6-10  -> window_time = 1.0
frames 11-15 -> window_time = 1.5
```

Trailing partial groups with fewer than 5 frames are ignored.

A window is skipped only when every ball coordinate value is missing across all
five frames:

```text
t.ball_x
t.ball_y
t.ball_z
```

If at least one ball coordinate exists somewhere in the window, the row is
kept. Missing continuous feature values are filled with `0` during
normalization.

## 4. Event Selection

A window can contain zero, one, or several attached events.

The module ignores rows where:

```text
e.event.event_type_name == "no event"
```

If at least one event remains, the primary event is the event with the smallest
numeric:

```text
nearest_timestamp_distance_sec
```

Missing event distances are treated as infinity for sorting. If there are
multiple events and the primary one is chosen, the other event type names are
stored in:

```text
secondary_events
```

as a comma-separated string. If there are no events in the window:

```text
primary_event = "no event"
secondary_events = ""
```

## 5. Event Coordinates and Direction

The raw event coordinates in the master join are provider coordinates on a
`0-100` attacking-direction scale. Before normalization, the training table
converts only the primary event coordinates into absolute pitch meters.

Output columns:

```text
e.x_meters_absolute
e.y_meters_absolute
is_attacking_direction
```

Conversion before scaling:

```text
x_meters = e.x * pitch_length_m / 100
y_meters = e.y * pitch_width_m / 100
```

The pitch dimensions come from `config.yaml` and default to:

```text
pitch_length_m = 105.0
pitch_width_m = 68.0
```

For period 1:

```text
e.x_meters_absolute = x_meters
is_attacking_direction = 1
```

For period 2 and later:

```text
e.x_meters_absolute = pitch_length_m - x_meters
is_attacking_direction = 0
```

`e.y_meters_absolute` is never flipped. If the window has no primary event, or
the primary event has missing/non-numeric `e.x` or `e.y`, both converted event
coordinate columns are filled with `0` for scaling and then normalized.

## 6. Ball Features

The ball columns start as raw tracking coordinates in meters:

```text
ball_x_avg
ball_y_avg
ball_z_avg
```

Each is the mean of the matching tracking coordinate over the five frames,
ignoring missing values.

The movement columns are:

```text
ball_speed_avg
ball_speed_change
ball_direction_x
ball_direction_y
```

`ball_speed_avg` starts as the mean frame-to-frame 3D Euclidean ball movement
inside the window:

```text
sqrt((x[i+1] - x[i])^2 + (y[i+1] - y[i])^2 + (z[i+1] - z[i])^2)
```

Frame-to-frame movements with missing coordinates are ignored by the final
average.

`ball_speed_change` starts as:

```text
last valid frame-to-frame movement - first valid frame-to-frame movement
```

It is filled with `0` for scaling when fewer than two valid frame-to-frame
movements exist.

`ball_direction_x` and `ball_direction_y` start as the raw tracking-coordinate
differences from frame 1 to frame 5. They are filled with `0` for scaling when
the first and fifth frames do not both have complete `t.ball_x`, `t.ball_y`,
and `t.ball_z` values.

## 7. Closest-Player Features

For frame 1 and frame 5 of each window, the module finds the visible player
closest to the ball.

A player slot is eligible when:

```text
t.player_XX_visible is true
t.player_XX_team_id is present
t.player_XX_x is present
t.player_XX_y is present
```

Distance uses raw tracking x/y meters before normalization:

```text
sqrt((player_x - ball_x)^2 + (player_y - ball_y)^2)
```

Output columns:

```text
closest_player_dist_start
closest_player_team_start
closest_player_dist_end
closest_player_team_end
closest_player_dist_change
```

`closest_player_dist_start` and `closest_player_team_start` come from frame 1.
`closest_player_dist_end` and `closest_player_team_end` come from frame 5.

`closest_player_dist_change` starts as:

```text
closest_player_dist_end - closest_player_dist_start
```

If no eligible player is found, the distance is filled with `0` for scaling and
the team is `"unknown"`.

## 8. Player Density Features

The player-density columns count unique visible player IDs across all five
frames in the window before normalization:

```text
n_players_near_ball
n_unique_players_in_frame
```

`n_players_near_ball` counts unique visible player IDs that are within 5 meters
of the ball in at least one frame. It does not require a team ID.

`n_unique_players_in_frame` counts unique visible player IDs tracked anywhere
in the window. It does not require a team ID.

## 9. Team Change Feature

The team-change column is:

```text
team_changed
```

When the primary event has a usable `e.possession_id`, `team_changed` is `1` if
any other event in the same window has a different usable possession ID.
Otherwise it is `0`.

If there is no primary event, or the primary event has no usable possession ID,
the module falls back to the closest-player teams:

```text
team_changed = 1 when closest_player_team_start != closest_player_team_end
```

The fallback returns `0` if either closest-player team is `"unknown"`.

`team_changed` is not scaled.

## 10. Normalization

The module normalizes these continuous columns:

```text
ball_x_avg
ball_y_avg
ball_z_avg
ball_speed_avg
ball_speed_change
ball_direction_x
ball_direction_y
closest_player_dist_start
closest_player_dist_end
closest_player_dist_change
n_players_near_ball
n_unique_players_in_frame
e.x_meters_absolute
e.y_meters_absolute
```

Normalization uses `sklearn.preprocessing.StandardScaler`.

The scaler is fit only on the train table:

```text
scaler.fit(train_df[continuous_features].fillna(0))
```

The fitted train-split scaler is then applied to train, validation, and test.
The same parquet files are overwritten with normalized values.

These columns are not normalized:

```text
t.match_id
t.period
window_time
data_split
is_attacking_direction
primary_event
is_pass
secondary_events
closest_player_team_start
closest_player_team_end
team_changed
```

## 11. Output Columns

The output schema is fixed by `OUTPUT_COLUMNS` in `training_table.py`:

| # | Column | Meaning |
| ---: | --- | --- |
| 1 | `t.match_id` | Source match ID as a string. |
| 2 | `t.period` | Source tracking period. |
| 3 | `window_time` | End time of the 5-frame window in seconds within the period. |
| 4 | `data_split` | Match-level split: `train`, `validation`, or `test`. |
| 5 | `is_attacking_direction` | `1` for period 1, `0` for period 2 and later. |
| 6 | `primary_event` | Selected event type for the window, or `"no event"`. |
| 7 | `is_pass` | Binary target derived from `primary_event == "PASS"`. |
| 8 | `secondary_events` | Non-primary event type names in the window, comma-separated. |
| 9 | `ball_x_avg` | Normalized ball x average. |
| 10 | `ball_y_avg` | Normalized ball y average. |
| 11 | `ball_z_avg` | Normalized ball z average. |
| 12 | `ball_speed_avg` | Normalized mean valid 3D frame-to-frame ball movement. |
| 13 | `ball_speed_change` | Normalized last-minus-first valid ball movement. |
| 14 | `ball_direction_x` | Normalized ball x difference from frame 1 to frame 5. |
| 15 | `ball_direction_y` | Normalized ball y difference from frame 1 to frame 5. |
| 16 | `e.x_meters_absolute` | Normalized primary event x in absolute pitch meters. |
| 17 | `e.y_meters_absolute` | Normalized primary event y in pitch meters. |
| 18 | `closest_player_dist_start` | Normalized closest eligible player distance in frame 1. |
| 19 | `closest_player_team_start` | Closest eligible player team in frame 1. |
| 20 | `closest_player_dist_end` | Normalized closest eligible player distance in frame 5. |
| 21 | `closest_player_team_end` | Closest eligible player team in frame 5. |
| 22 | `closest_player_dist_change` | Normalized end-minus-start closest-player distance. |
| 23 | `n_players_near_ball` | Normalized count of unique visible player IDs near the ball. |
| 24 | `n_unique_players_in_frame` | Normalized count of unique visible player IDs in the window. |
| 25 | `team_changed` | Possession-ID change within the window, with closest-team fallback. |

## 12. Rebuild Command

From the project root:

```bash
conda activate driblabvenv
PYTHONPATH=src python -m driblab.features.training_table
```

To rebuild without normalization:

```bash
PYTHONPATH=src python -m driblab.features.training_table --no-normalize
```

The default command reads:

```text
data/processed/model_base/master_join_table.parquet
config.yaml
```

and writes:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
artifacts/models/feature_scaler.pkl
```

## 13. Most Recent Output Summary

The most recent generated files contained:

| Split | Windows | Pass windows | No-event windows | Pass percentage | Matches | Periods |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 161,505 | 15,554 | 140,892 | 9.63% | 23 | 2 |
| validation | 36,719 | 3,567 | 32,044 | 9.71% | 5 | 2 |
| test | 35,085 | 3,725 | 30,251 | 10.62% | 5 | 2 |

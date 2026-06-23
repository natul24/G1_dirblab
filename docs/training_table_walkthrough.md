# Training Table Walkthrough

This guide documents what `src/driblab/features/training_table.py` builds.

Read it in this order: the table is first built in interpretable raw units,
then the continuous model features are standardized at the end. The default
command saves standardized parquet tables because that is what the model uses,
but the feature definitions below describe the values before standardization.

## 1. Inputs And Outputs

The module reads:

```text
data/processed/model_base/master_join_table.parquet
config.yaml
```

It writes one training table and one summary CSV per split:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
data/processed/model_base/training_table_summary_train.csv
data/processed/model_base/training_table_summary_validation.csv
data/processed/model_base/training_table_summary_test.csv
```

It also saves the train-fitted scaler:

```text
artifacts/models/feature_scaler.pkl
```

Important: by default, the parquet tables above are overwritten with
standardized continuous features after the raw table is built. To inspect raw
pre-standardized features, run the module with `--no-normalize` and preferably
write to a separate output directory.

## 2. High-Level Workflow

The actual order in `training_table.py` is:

1. Load the master join table.
2. Load match-level train, validation, and test split definitions.
3. Add `data_split` before feature engineering.
4. Build raw 5-frame training windows separately for each split.
5. Calculate target, event, ball, player, density, and team-change features in
   interpretable units.
6. Write raw split tables to parquet.
7. Write split summary CSVs.
8. If normalization is enabled, fit `StandardScaler` on the raw train table.
9. Apply that scaler to train, validation, and test.
10. Overwrite the same parquet files with standardized values.
11. Save the fitted scaler.

No window crosses split boundaries. Validation and test statistics are never
used to fit the scaler.

## 3. Split Rule

The split is assigned before any feature engineering with:

```text
driblab.features.match_splits.add_data_split_column
```

The split column is:

```text
data_split
```

Its values are:

```text
train
validation
test
```

Each split is built independently.

## 4. Window Construction

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

The module creates consecutive non-overlapping 5-frame windows:

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
kept.

## 5. Target

Each row is one 5-frame window. The target is:

```text
is_pass = 1 when primary_event == "PASS"
is_pass = 0 otherwise
```

`primary_event` is selected from events attached to frames inside the window.
If the window has no event, then:

```text
primary_event = "no event"
is_pass = 0
```

## 6. Event Selection

A window can contain zero, one, or several attached events.

The module ignores rows where:

```text
e.event.event_type_name == "no event"
```

If at least one real event remains, the primary event is the event with the
smallest numeric:

```text
nearest_timestamp_distance_sec
```

Missing event distances are treated as infinity for sorting.

If there are other real events in the same window, their event type names are
stored in:

```text
secondary_events
```

as a comma-separated string. If there are no real events:

```text
primary_event = "no event"
secondary_events = ""
```

## 7. Attacking Direction

The training table keeps one directional context column:

```text
is_attacking_direction
```

It is derived only from the tracking period:

```text
is_attacking_direction = 1 for period 1
is_attacking_direction = 0 for period 2 and later
```

The event coordinate columns from the master join are intentionally excluded
from the training table because they are event-derived locations and would leak
information into the pass detector.

## 8. Raw Ball Features

The ball average columns are raw tracking-coordinate means in meters before
standardization:

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

`ball_speed_avg` is the mean frame-to-frame 3D Euclidean ball movement inside
the window:

```text
sqrt((x[i+1] - x[i])^2 + (y[i+1] - y[i])^2 + (z[i+1] - z[i])^2)
```

Frame-to-frame movements with missing coordinates are ignored by the final
average.

`ball_speed_change` is:

```text
last valid frame-to-frame movement - first valid frame-to-frame movement
```

It is missing before standardization when fewer than two valid frame-to-frame
movements exist.

`ball_direction_x` and `ball_direction_y` are tracking-coordinate differences
from frame 1 to frame 5:

```text
ball_direction_x = frame_5_ball_x - frame_1_ball_x
ball_direction_y = frame_5_ball_y - frame_1_ball_y
```

They are missing before standardization when the first and fifth frames do not
both have complete `t.ball_x`, `t.ball_y`, and `t.ball_z` values.

## 9. Raw Closest-Player Features

For frame 1 and frame 5 of each window, the module finds the visible player
closest to the ball.

A player slot is eligible when:

```text
t.player_XX_visible is true
t.player_XX_team_id is present
t.player_XX_x is present
t.player_XX_y is present
```

Distance uses raw tracking x/y meters before standardization:

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

`closest_player_dist_change` is:

```text
closest_player_dist_end - closest_player_dist_start
```

If no eligible player is found, the distance is missing before
standardization and the team is:

```text
unknown
```

## 10. Raw Player Density Features

The player-density columns count unique visible player IDs across all five
frames in the window before standardization:

```text
n_players_near_ball
n_unique_players_in_frame
```

`n_players_near_ball` counts unique visible player IDs that are within 5 meters
of the ball in at least one frame. It does not require a team ID.

`n_unique_players_in_frame` counts unique visible player IDs tracked anywhere
in the window. It does not require a team ID.

## 11. Team Change Feature

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

The fallback returns `0` if either closest-player team is:

```text
unknown
```

`team_changed` is not standardized.

## 12. Raw Output Columns Before Standardization

The raw training table schema is fixed by `OUTPUT_COLUMNS` in
`training_table.py`.

| # | Column | Raw meaning before standardization |
| ---: | --- | --- |
| 1 | `t.match_id` | Source match ID as a string. |
| 2 | `t.period` | Source tracking period. |
| 3 | `window_time` | End time of the 5-frame window in seconds within the period. |
| 4 | `data_split` | Match-level split: `train`, `validation`, or `test`. |
| 5 | `is_attacking_direction` | `1` for period 1, `0` for period 2 and later. |
| 6 | `primary_event` | Selected event type for the window, or `"no event"`. |
| 7 | `is_pass` | Binary target derived from `primary_event == "PASS"`. |
| 8 | `secondary_events` | Non-primary event type names in the window, comma-separated. |
| 9 | `ball_x_avg` | Raw mean ball x coordinate in tracking meters. |
| 10 | `ball_y_avg` | Raw mean ball y coordinate in tracking meters. |
| 11 | `ball_z_avg` | Raw mean ball z coordinate in tracking meters. |
| 12 | `ball_speed_avg` | Raw mean valid 3D frame-to-frame ball movement. |
| 13 | `ball_speed_change` | Raw last-minus-first valid ball movement. |
| 14 | `ball_direction_x` | Raw ball x difference from frame 1 to frame 5. |
| 15 | `ball_direction_y` | Raw ball y difference from frame 1 to frame 5. |
| 16 | `closest_player_dist_start` | Raw closest eligible player distance in frame 1. |
| 17 | `closest_player_team_start` | Closest eligible player team in frame 1. |
| 18 | `closest_player_dist_end` | Raw closest eligible player distance in frame 5. |
| 19 | `closest_player_team_end` | Closest eligible player team in frame 5. |
| 20 | `closest_player_dist_change` | Raw end-minus-start closest-player distance. |
| 21 | `n_players_near_ball` | Raw count of unique visible player IDs near the ball. |
| 22 | `n_unique_players_in_frame` | Raw count of unique visible player IDs in the window. |
| 23 | `team_changed` | Possession-ID change within the window, with closest-team fallback. |

## 13. Standardization As The Final Step

After raw tables and summaries are written, the default command standardizes
the continuous columns for model training.

The standardized columns are:

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
```

Standardization uses:

```text
sklearn.preprocessing.StandardScaler
```

The scaler is fit only on the train table:

```text
scaler.fit(train_df[continuous_features].fillna(0))
```

Before the scaler transform, missing continuous values are filled with `0`.
Then the fitted train-split scaler is applied to train, validation, and test.
The same parquet files are overwritten with standardized values.

These columns are not standardized:

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

The fitted scaler is saved to:

```text
artifacts/models/feature_scaler.pkl
```

## 14. What The Default Saved Parquet Tables Contain

The default command:

```bash
python -m driblab.features.training_table
```

builds raw tables first, then overwrites the three parquet tables with
standardized continuous features. Therefore, the default saved parquet files
contain standardized values for the continuous columns listed above.

To inspect raw, interpretable, pre-standardized features, run:

```bash
python -m driblab.features.training_table \
  --no-normalize \
  --output-dir data/processed/model_base_raw
```

That preserves raw feature units in a separate output directory and avoids
overwriting the model-ready standardized tables.

## 15. Rebuild Commands

Default model-ready build:

```bash
conda activate driblabvenv
python -m driblab.features.training_table
```

Raw pre-standardized build for inspection:

```bash
python -m driblab.features.training_table \
  --no-normalize \
  --output-dir data/processed/model_base_raw
```

The default command writes:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
data/processed/model_base/training_table_summary_train.csv
data/processed/model_base/training_table_summary_validation.csv
data/processed/model_base/training_table_summary_test.csv
artifacts/models/feature_scaler.pkl
```

## 16. Most Recent Output Summary

The most recent generated files contained:

| Split | Windows | Pass windows | No-event windows | Pass percentage | Matches | Periods |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 161,505 | 15,554 | 140,892 | 9.63% | 23 | 2 |
| validation | 36,719 | 3,567 | 32,044 | 9.71% | 5 | 2 |
| test | 35,085 | 3,725 | 30,251 | 10.62% | 5 | 2 |

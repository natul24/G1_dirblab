# XGBoost Pass Detection Model Guide

This guide documents the current XGBoost pass detector implemented in
`src/driblab/models/pass_detector.py`.

## 1. What The Model Does

Objective: predict whether a 0.5-second tracking window contains a pass event.

Input: one row from the standardized training table, where each row summarizes
five consecutive tracking frames.

Output: binary probability from XGBoost. The production script converts that
probability to a class with a fixed `0.5` threshold:

```text
1 = pass
0 = no pass
```

The target is:

```text
is_pass = 1 when primary_event == "PASS"
```

## 2. Current Pipeline

```text
Raw event + tracking files
    |
    v
Master join table
    |
    v
Training table: 5-frame windows and engineered features
    |
    v
StandardScaler fit on train continuous features only
    |
    v
XGBoost binary classifier
    |
    v
Model artifacts and evaluation reports
```

The model-ready training tables live in:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
```

## 3. Training Table Grain And Splits

One row equals one non-overlapping 5-frame window, approximately 0.5 seconds at
10 Hz.

Current split summary:

| Split | Windows | Pass windows | No-event windows | Pass rate | Matches | Periods |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 161,505 | 15,554 | 140,892 | 9.63% | 23 | 2 |
| validation | 36,719 | 3,567 | 32,044 | 9.71% | 5 | 2 |
| test | 35,085 | 3,725 | 30,251 | 10.62% | 5 | 2 |

Splits are assigned by match before feature engineering.

## 4. Model Features

The current model uses 14 input features.

### Ball Features

| Feature | Meaning before standardization |
| --- | --- |
| `ball_x_avg` | Mean ball x position across the window. |
| `ball_y_avg` | Mean ball y position across the window. |
| `ball_z_avg` | Mean ball z position across the window. |
| `ball_speed_avg` | Mean valid 3D frame-to-frame ball movement. |
| `ball_speed_change` | Last valid ball movement minus first valid ball movement. |
| `ball_direction_x` | Ball x change from frame 1 to frame 5. |
| `ball_direction_y` | Ball y change from frame 1 to frame 5. |

### Closest-Player Features

| Feature | Meaning before standardization |
| --- | --- |
| `closest_player_dist_start` | Distance from the ball to the nearest eligible visible player in frame 1. |
| `closest_player_dist_end` | Distance from the ball to the nearest eligible visible player in frame 5. |
| `closest_player_dist_change` | End closest-player distance minus start closest-player distance. |

The training table also stores `closest_player_team_start` and
`closest_player_team_end`, but these are not model input features.

### Player-Density Features

| Feature | Meaning before standardization |
| --- | --- |
| `n_players_near_ball` | Count of unique visible players within 5 meters of the ball at least once in the window. |
| `n_unique_players_in_frame` | Count of unique visible players tracked anywhere in the window. |

### Context Features

| Feature | Meaning |
| --- | --- |
| `is_attacking_direction` | `1` for period 1, `0` for period 2 and later. |
| `team_changed` | Possession/team-change signal inside the window. |

Important caveat: `team_changed` can use `e.possession_id` from the master join
when event possession is available. If the goal is a strictly tracking-only
model, this feature should be reviewed or replaced with a tracking-only
closest-player-team signal.

## 5. Target And Metadata Columns

These columns are kept in the training table but are not model input features:

| Column | Meaning |
| --- | --- |
| `is_pass` | Target variable. |
| `primary_event` | Selected event type in the window, or `"no event"`. |
| `secondary_events` | Other event types in the same window, comma-separated. |
| `data_split` | `train`, `validation`, or `test`. |
| `t.match_id` | Source match ID. |
| `t.period` | Tracking period. |
| `window_time` | Window end time in seconds within the period. |
| `closest_player_team_start` | Team of closest eligible player in frame 1. |
| `closest_player_team_end` | Team of closest eligible player in frame 5. |

## 6. Event Coordinate Leakage Fix

The earlier training table included event-location features:

```text
e.x_meters_absolute
e.y_meters_absolute
```

Those columns came from the labelled event data and created leakage risk,
because the model could learn from where an already-labelled event occurred
rather than from tracking behavior.

Current status:

```text
Event coordinate columns in training tables: none
Event coordinate columns in model features: none
```

The master join can still keep raw event columns such as `e.x` and `e.y`; they
are not passed into the training table or the XGBoost model.

## 7. Standardization

The training-table module fits `StandardScaler` on train only, then applies the
same scaler to train, validation, and test.

Standardized continuous features:

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

Not standardized:

```text
is_attacking_direction
team_changed
primary_event
is_pass
secondary_events
data_split
t.match_id
t.period
window_time
closest_player_team_start
closest_player_team_end
```

The fitted scaler is saved to:

```text
artifacts/models/feature_scaler.pkl
```

## 8. Model Configuration

The script uses:

```python
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 42,
}

num_boost_round = 200
early_stopping_rounds = 20
threshold = 0.5
```

Current saved metadata reports:

```text
best_iteration = 95
```

The script reports `Best round` as `best_iteration + 1`, so the console output
shows `96`.

## 9. Current Model Performance

These metrics come from the current `reports/model_evaluation_results.json`
after removing event coordinate leakage.

| Split | Accuracy | F1 | ROC-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.9037 | 0.0000 | 0.7152 | 145,951 | 0 | 15,554 | 0 |
| validation | 0.9029 | 0.0000 | 0.6715 | 33,152 | 0 | 3,567 | 0 |
| test | 0.8938 | 0.0000 | 0.6517 | 31,360 | 0 | 3,725 | 0 |

Interpretation:

- ROC-AUC is above random, so the model has some ranking signal.
- At the fixed `0.5` threshold, the model currently predicts every row as
  `no pass`.
- Accuracy looks high because no-pass windows dominate the data.
- F1 is `0.0` because there are no predicted positives.

This means the current model is not yet useful as a pass detector at the
default threshold. The next modelling step should be threshold tuning and/or
class-imbalance handling.

## 10. Feature Importance

Current top features by XGBoost split count:

| Feature | Split count |
| --- | ---: |
| `ball_x_avg` | 359 |
| `ball_y_avg` | 343 |
| `closest_player_dist_end` | 334 |
| `ball_speed_avg` | 310 |
| `ball_z_avg` | 289 |
| `closest_player_dist_change` | 261 |
| `ball_direction_x` | 252 |
| `ball_speed_change` | 245 |
| `ball_direction_y` | 229 |
| `closest_player_dist_start` | 213 |

Feature importance counts how often each feature is used to split XGBoost
trees. It does not prove causal importance, but it helps identify which inputs
the current model is relying on most often.

## 11. Generated Files

The training script saves:

```text
artifacts/models/pass_detector.json
artifacts/models/pass_detector_metadata.json
artifacts/models/feature_encoders.pkl
reports/model_evaluation_results.json
reports/figures/feature_importance.png
reports/figures/roc_curve.png
reports/figures/confusion_matrices.png
```

`feature_encoders.pkl` is currently an empty dictionary placeholder because the
model does not encode categorical features.

## 12. How To Run

Rebuild model-ready training tables:

```bash
python -m driblab.features.training_table
```

Train the pass detector and regenerate model reports:

```bash
python -m driblab.models.pass_detector
```

Explore the same workflow interactively:

```text
notebooks/xgboost_pass_detector.ipynb
```

## 13. Related Documentation

- Training table details: `docs/training_table_walkthrough.md`
- Master join details: `docs/master_join_walkthrough.md`
- Data dictionary: `docs/data_dictionary.md`
- Feature engineering module: `src/driblab/features/training_table.py`
- Model module: `src/driblab/models/pass_detector.py`
- Model notebook: `notebooks/xgboost_pass_detector.ipynb`

## 14. Practical Next Steps

The leakage fix changed the model behavior substantially. Recommended next
steps:

1. Tune the decision threshold using validation precision, recall, and F1.
2. Add class-imbalance handling, such as `scale_pos_weight`.
3. Compare against a baseline model that always predicts no pass.
4. Audit `team_changed` if the target deployment must be strictly tracking-only.
5. Re-run the notebook and reports after every feature or threshold change.

# XGBoost Pass Detection Model Guide

This guide documents the current XGBoost pass detector implemented in
`src/driblab/models/pass_detector.py`.

## 1. What The Model Does

Objective: predict whether a sampled tracking row from a 5-frame interval
represents a pass event.

Input: one row from the training table. Each row keeps the source `t.*`
tracking columns from the sampled frame plus the engineered model columns.

Output: binary probability from XGBoost. The production script converts that
probability to a class with a fixed `0.5` threshold:

```text
1 = pass
0 = no pass
```

The target is:

```text
is_pass = 1 when p.event_label == "PASS"
```

## 2. Current Pipeline

```text
Raw event + tracking files
    |
    v
Master join table
    |
    v
Pre-training table: event labels assigned per frame
    |
    v
Training table: 5-frame sampling, rolling ball speed, closest player
    |
    v
XGBoost binary classifier
    |
    v
Model artifacts and evaluation reports
```

The model-ready training tables are:

```text
data/processed/model_base/training_table_train.parquet
data/processed/model_base/training_table_validation.parquet
data/processed/model_base/training_table_test.parquet
```

## 3. Training Table Grain And Splits

One row equals one sampled row per non-overlapping 5-frame interval,
approximately 0.5 seconds at 10 Hz.

The `data_split` column identifies the split for each row. Splits are assigned
at the match level before feature engineering to prevent leakage.

## 4. Training Table Columns

The training table has 127 columns: all 121 `t.*` tracking columns from the
pre-training table, plus 6 added columns:

| Column | Description |
|---|---|
| `p.event_label` | Event label assigned by the pre-training stage. |
| `data_split` | `train`, `validation`, or `test` - assigned at the match level. |
| `is_pass` | Binary target: `1` = PASS, `0` = other. |
| `ball_speed_avg_xy` | Rolling mean of 2D frame-to-frame ball speed over +/-5 frames (m / frame). |
| `closest_player_id` | ID of the visible player closest to the ball at this row's frame. |
| `closest_player_team_id` | Team ID of the closest visible player at this row's frame. |

## 5. Model Features

The model currently uses two features:

```python
FEATURES = [
    "ball_speed_avg_xy",
    "closest_player_team_id",
]
```

`ball_speed_avg_xy` is numeric. `closest_player_team_id` is a team identifier
that is converted to categorical integer codes before training. Missing feature
values are filled with `0`.

## 6. Model Configuration

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

## 7. Generated Files

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

`feature_encoders.pkl` is currently an empty dictionary placeholder.

## 8. How To Run

Run every stage through `main.py` from the project root:

```bash
conda activate driblabvenv
python main.py master-join
python main.py pre-training
python main.py training-table
python main.py pass-detector
```

Or run the current full pipeline in one command:

```bash
python main.py all
```

Explore the same workflow interactively:

```text
notebooks/xgboost_pass_detector.ipynb
```

## 9. Most Recent Saved Metrics

The current `reports/model_evaluation_results.json` contains:

| Split | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| train | 0.8210 | 0.6824 | 0.8050 |
| validation | 0.8055 | 0.6733 | 0.7912 |
| test | 0.7752 | 0.6192 | 0.7349 |

Confusion matrices use rows as actual labels and columns as predicted labels:

| Split | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| train | 175,678 | 15,396 | 34,620 | 53,743 |
| validation | 35,921 | 3,378 | 8,172 | 11,901 |
| test | 34,650 | 2,730 | 10,419 | 10,689 |

## 10. Related Documentation

- Training table details: `docs/training_table_walkthrough.md`
- Master join details: `docs/master_join_walkthrough.md`
- Data dictionary: `docs/data_dictionary.md`
- Feature engineering module: `src/driblab/features/training_table.py`
- Model module: `src/driblab/models/pass_detector.py`
- Model notebook: `notebooks/xgboost_pass_detector.ipynb`

## 11. Practical Next Steps

1. Tune the decision threshold using validation precision, recall, and F1.
2. Add class-imbalance handling with `scale_pos_weight`.
3. Expand the feature set with additional columns from the training table.
4. Compare against a baseline model that always predicts no pass.
5. Re-run the notebook and reports after every feature or threshold change.

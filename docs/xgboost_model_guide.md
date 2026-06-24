# XGBoost Pass Detection Model Guide

This guide documents the current XGBoost pass detector implemented in
`src/driblab/models/pass_detector.py`.

## 1. What The Model Does

Objective: predict whether a 0.5-second tracking window contains a pass event.

Input: one row from the training table, where each row summarizes five consecutive
tracking frames.

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
Pre-training table: event labels assigned per frame (notebook)
    |
    v
Training table: 5-frame windows with ball speed feature
    |
    v
XGBoost binary classifier
    |
    v
Model artifacts and evaluation reports
```

The model-ready training table is:

```text
data/processed/model_base/training_table_simple.parquet
```

## 3. Training Table Grain And Splits

One row equals one non-overlapping 5-frame window, approximately 0.5 seconds at 10 Hz.

The `data_split` column identifies the split for each window. Splits are assigned at
the match level before feature engineering to prevent leakage.

## 4. Training Table Columns

The training table has 7 columns:

| Column | Description |
|---|---|
| `t.match_id` | Source match ID. |
| `t.period` | Tracking period. |
| `window_time` | Window end time in seconds within the period. |
| `data_split` | `train`, `validation`, or `test`. |
| `p.event_label` | Primary event type in the window, or `"no event"`. |
| `is_pass` | Binary target: `1` = PASS, `0` = other. |
| `ball_speed_avg_xy` | Mean 2D frame-to-frame ball speed in metres per frame. |

## 5. Model Configuration

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

## 6. Generated Files

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

## 7. How To Run

Run `notebooks/pre_training_table.ipynb` first to build the pre-training table, then:

```bash
conda activate driblabvenv
python -m driblab.features.training_table
python -m driblab.models.pass_detector
```

Explore the same workflow interactively:

```text
notebooks/xgboost_pass_detector.ipynb
```

## 8. Related Documentation

- Training table details: `docs/training_table_walkthrough.md`
- Master join details: `docs/master_join_walkthrough.md`
- Data dictionary: `docs/data_dictionary.md`
- Feature engineering module: `src/driblab/features/training_table.py`
- Model module: `src/driblab/models/pass_detector.py`
- Model notebook: `notebooks/xgboost_pass_detector.ipynb`

## 9. Practical Next Steps

1. Tune the decision threshold using validation precision, recall, and F1.
2. Add class-imbalance handling, such as `scale_pos_weight`.
3. Compare against a baseline model that always predicts no pass.
4. Re-run the notebook and reports after every feature or threshold change.

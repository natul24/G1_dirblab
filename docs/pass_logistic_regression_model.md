# Binary Pass Model: Logistic Regression

This document explains the first supervised model in the project. The model is
a simple binary classifier that predicts whether a tracking frame corresponds
to a pass event.

## Goal

The model answers this question:

```text
Given the tracking-derived state of a frame, is this frame a pass?
```

It is intentionally simple so the project has a clear first machine-learning
baseline before moving to multiclass event detection.

## Input

```text
data/processed/model_base/master_join_table.parquet
```

The input is the Step 2 master join table. It has one row per tracking frame.
Event data is joined onto the tracking frame only to create the target label.

## Target

The binary target is:

```text
is_pass
```

`is_pass = 1` when `event_label` contains one of the configured pass labels in
`config.yaml` under `pass_model.positive_labels`. Otherwise, `is_pass = 0`.

The current positive labels are:

```text
PASS
```

Rows with `event_label = "no event"` become negative examples.

## Features

Feature columns are configured in:

```text
config.yaml -> pass_model.feature_columns
```

The current model uses tracking-derived and frame-state features from the master
join table, including:

- match-clock and period fields
- normalized ball position
- ball height
- ball interpolation flags
- ball speed and acceleration
- nearest-player distance to the ball
- raw possession flag
- player-count and visibility aggregates
- player-to-ball distance aggregates

The model does not use detailed event columns such as `event_type_name`,
`event_team_id`, `event_player_id`, `event_x`, or `event_y` as input features.
Those columns describe the answer after the event has been labelled, so they are
used only to build `is_pass`, not to train the model.

The complete feature lineage is documented in
[`docs/data_dictionary.md`](data_dictionary.md) under "Pass Model Feature
Columns".

## Split Logic

Train, validation, and test splits are assigned by complete `match_id` in:

```text
config.yaml -> match_splits
```

The model is fit on `train` matches. Metrics are reported for train,
validation, and test. Splitting by full match avoids leakage from adjacent 10 Hz
frames in the same match.

## Model

The estimator is a scikit-learn pipeline:

1. `SimpleImputer`
2. `StandardScaler`
3. `LogisticRegression`

Current hyperparameters are configured in:

```text
config.yaml -> pass_model
```

The current classifier uses balanced class weights because pass frames are much
rarer than non-pass frames.

## Outputs

```text
artifacts/models/pass_classifier/pass_logistic_regression.joblib
data/processed/pass_classifier/pass_model_metrics.parquet
```

The `.joblib` file is the trained model artifact. It is ignored by Git because
it can be regenerated from the tracked code, config, and local data.

The metrics table is a processed project output. It includes:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- positive row count
- positive rate

## How To Run

From the project root, after activating the environment:

```bash
python main.py pass_model
```

Use a different classification threshold:

```bash
python main.py pass_model --threshold 0.4
```

Rerunning the command overwrites the same model artifact and metrics file.

## Notebook

The walkthrough notebook is:

```text
notebooks/pass_logistic_regression_model.ipynb
```

It shows the train/test split, target creation, model fitting, evaluation
metrics, and evaluation visuals.

The notebook visuals include:

- metrics by split
- ROC curve by split
- test confusion matrix
- test pass probability distribution
- top logistic regression coefficients

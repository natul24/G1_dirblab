# Step 4: Rule-Based Detection

Step 4 is the first event detector. It is a deterministic baseline, not a
trained machine-learning model. The goal is to turn the Step 3 possession
sequence plus ball movement into simple event predictions that can be evaluated
before building a learned classifier.

This step is useful because it gives the project a clear floor: if later ML
models cannot beat these rules, the ML model is not adding value yet.

## What This Step Does

For every reliable live-play tracking frame, Step 4:

1. Reads the Step 3 possession sequence table.
2. Adds a few rule-specific lag and ball-movement features.
3. Converts provider event labels into broad evaluation classes.
4. Applies simple football rules to predict an event class.
5. Evaluates the predicted classes against provider-derived classes on a
   held-out split.

The output still has one row per tracking frame. Rows with no rule match are
predicted as `no event`.

## Input Table

```text
data/processed/possession_sequence/possession_sequence_table.parquet
```

This table already includes:

- all Step 2 master join columns
- Step 3 smoothed possession columns
- `data_split`, assigned by full `match_id`

Step 4 expects the input table to contain the columns below.

| Column | Source step | How Step 4 uses it |
| --- | --- | --- |
| `match_id` | Step 2 | Keeps rows grouped by match and supports match-level evaluation splits. |
| `period_id` | Step 2 | Keeps lag/diff features inside one period. |
| `frame_id` | Step 2 | Keeps deterministic frame ordering when timestamps tie. |
| `tracking_match_clock_seconds` | Step 2 | Sorts frames in match-clock order before calculating lag/diff features. |
| `data_split` | Step 3 / `config.yaml` | Filters evaluation rows to `train`, `validation`, or `test`. |
| `event_label` | Step 2 event join | Provider event label used to create `true_event_class`. |
| `ball_x` | Step 2 | Detects pitch boundary and goal-line conditions on the normalized `0-100` pitch. |
| `ball_y` | Step 2 | Detects touchline and corner conditions on the normalized `0-100` pitch. |
| `ball_x_attacking` | Step 2 | Detects ball movement toward the attacking goal. |
| `ball_speed_mps` | Step 2 | Separates fast opponent gains from slow opponent gains and helps identify shots. |
| `smoothed_possession_team_id` | Step 3 | Builds previous-frame possession features. |
| `smoothed_possession_player_id` | Step 3 | Builds previous-frame possession features. |
| `possession_team_change` | Step 3 | Detects opponent possession gains. |
| `possession_player_change` | Step 3 | Detects player changes, including teammate changes. |

## Features Created Inside Step 4

These are temporary rule features added to the Step 3 table before prediction.
They are saved in `rule_based_predictions.parquet`.

| Column | How it is calculated | Why it is used |
| --- | --- | --- |
| `rule_prev_smoothed_possession_team_id` | Previous frame's `smoothed_possession_team_id` within the same `match_id` and `period_id`. | Makes possession transitions auditable. |
| `rule_prev_smoothed_possession_player_id` | Previous frame's `smoothed_possession_player_id` within the same `match_id` and `period_id`. | Makes player-level possession transitions auditable. |
| `rule_ball_dx_attacking` | Frame-to-frame difference of `ball_x_attacking` within the same match and period. | Positive values mean the ball moved toward the reference team's attacking goal. |
| `rule_ball_dy_attacking` | Frame-to-frame difference of `ball_y_attacking` within the same match and period. | Captures sideways ball movement in attacking orientation. |

## Target Class Mapping

Step 4 evaluates against broad event classes, not every raw provider event
name. The broad class is stored as:

```text
true_event_class
```

It is created from `event_label` using `step4.label_groups` in `config.yaml`.
For example, labels such as `PASS`, `CROSS`, and `KEY PASS` map to `pass`;
labels such as `GOAL`, `SAVED SHOT`, and `MISSED SHOT` map to `shot`.

If a provider event does not belong to any configured group, it becomes:

```text
other_event
```

If the tracking frame has no matched event, it becomes:

```text
no event
```

## Prediction Classes

The detector predicts these classes:

```text
no event, pass, interception, tackle, shot, out, corner, other_event
```

`other_event` is included in evaluation because the provider can label events
outside the rule groups. The current rules do not actively predict
`other_event`; they predict a rule class or leave the row as `no event`.

## Rule Logic

Every row starts with:

```text
rule_event_class = no event
rule_reason = no_rule_fired
```

Then the detector applies these rules.

| Predicted class | Rule condition | Main columns used | Config threshold |
| --- | --- | --- | --- |
| `pass` | Smoothed possession changes to a different player on the same team. | `possession_player_change`, `possession_team_change` | None |
| `interception` | Smoothed possession changes to the opponent and the ball is moving at or above the interception speed threshold. | `possession_team_change`, `ball_speed_mps` | `interception_min_ball_speed_mps` |
| `tackle` | Smoothed possession changes to the opponent and the ball is moving below the interception speed threshold. | `possession_team_change`, `ball_speed_mps` | `interception_min_ball_speed_mps` |
| `out` | Ball is near either touchline or goal line on the normalized pitch. | `ball_x`, `ball_y` | `boundary_margin` |
| `corner` | Ball is near the goal line and near either corner area. | `ball_x`, `ball_y` | `boundary_margin`, `corner_y_margin` |
| `shot` | Ball is fast, already in an advanced attacking x location, and moving toward the attacking goal. | `ball_speed_mps`, `ball_x_attacking`, `rule_ball_dx_attacking` | `shot_min_speed_mps`, `shot_min_attacking_x`, `shot_min_dx_attacking` |

The current implementation writes predictions in this order:

1. teammate player change -> `pass`
2. fast opponent gain -> `interception`
3. slow opponent gain -> `tackle`
4. boundary -> `out`
5. corner -> `corner`
6. fast attacking ball -> `shot`

Because later assignments can overwrite earlier ones, `shot` has the highest
priority in the current implementation, then `corner`, then `out`, then
possession-change rules.

## Current Thresholds

Current thresholds are configured in `config.yaml` under `step4`.

| Parameter | Current value | Meaning |
| --- | ---: | --- |
| `shot_min_speed_mps` | `14.0` | Minimum ball speed for the shot rule. |
| `shot_min_attacking_x` | `70.0` | Minimum attacking x location for the shot rule on the `0-100` pitch. |
| `shot_min_dx_attacking` | `0.25` | Minimum frame-to-frame attacking x movement for the shot rule. |
| `interception_min_ball_speed_mps` | `4.0` | Opponent gains at or above this ball speed are labeled `interception`; below it, `tackle`. |
| `boundary_margin` | `0.5` | Distance from pitch boundary on the `0-100` scale for `out`. |
| `corner_y_margin` | `12.0` | Y-zone near either corner for `corner`. |

## Train, Validation, and Test Split

The split is defined in `config.yaml` under `match_splits`. Splits are assigned
by complete `match_id`, not by individual frame. This matters because adjacent
10 Hz tracking frames from the same match are highly related; splitting by row
would leak match context between train and test.

Current split sizes:

| Split | Matches | Purpose in Step 4 |
| --- | ---: | --- |
| `train` | 23 | Available for future learned models. Step 4 does not fit parameters on it. |
| `validation` | 5 | Use this split to adjust rule thresholds and rule logic. |
| `test` | 5 | Final held-out evaluation after thresholds are frozen. |

Current split membership:

| Split | Match IDs |
| --- | --- |
| `train` | `678949`, `679026`, `679053`, `679072`, `679075`, `679088`, `679104`, `679128`, `682607`, `682717`, `682815`, `683132`, `683190`, `683253`, `683261`, `683309`, `683425`, `684014`, `684119`, `684139`, `684141`, `684147`, `689340` |
| `validation` | `689526`, `689552`, `689556`, `713886`, `713893` |
| `test` | `713910`, `713946`, `713998`, `714000`, `745399` |

Step 4 is not trained, so the `train` split is not used to fit model weights.
The important discipline is:

- tune rules using `validation`
- report final performance using `test`
- do not keep changing thresholds after looking at the test results

Default evaluation uses:

```text
evaluation_split: test
```

Run validation evaluation while tuning:

```bash
python main.py step4 --evaluation-split validation
```

Run final held-out evaluation:

```bash
python main.py step4 --evaluation-split test
```

## Evaluation Metrics

Evaluation compares:

```text
true_event_class vs rule_event_class
```

It calculates per-class:

- precision
- recall
- F1
- support
- true positives
- false positives
- false negatives

It also writes a long-format confusion matrix and a one-row summary containing
macro F1 and weighted F1.

## Output Columns Added

The prediction table keeps the full input table and adds these Step 4 columns.

| Column | Meaning |
| --- | --- |
| `rule_prev_smoothed_possession_team_id` | Previous frame's smoothed possession team in the same match and period. |
| `rule_prev_smoothed_possession_player_id` | Previous frame's smoothed possession player in the same match and period. |
| `rule_ball_dx_attacking` | Frame-to-frame attacking x movement of the ball. |
| `rule_ball_dy_attacking` | Frame-to-frame attacking y movement of the ball. |
| `true_event_class` | Provider-derived broad class used as the evaluation target. |
| `rule_event_class` | Rule-based predicted class. |
| `rule_reason` | Short explanation of which rule fired. |

## Output Files

```text
data/processed/rule_based_detection/rule_based_predictions.parquet
data/processed/rule_based_detection/rule_based_metrics_by_class.csv
data/processed/rule_based_detection/rule_based_confusion_matrix.csv
data/processed/rule_based_detection/rule_based_summary.csv
data/processed/rule_based_detection/rule_based_metadata.json
```

| File | Purpose |
| --- | --- |
| `rule_based_predictions.parquet` | Full frame-level table with input columns, rule features, true class, predicted class, and rule reason. |
| `rule_based_metrics_by_class.csv` | Precision, recall, F1, and support by event class for the selected evaluation split. |
| `rule_based_confusion_matrix.csv` | Count of each true/predicted class combination. |
| `rule_based_summary.csv` | Overall evaluation split, rows, matches, macro F1, and weighted F1. |
| `rule_based_metadata.json` | Run configuration and output paths for auditability. |

Run the default Step 4 pipeline with:

```bash
python main.py step4
```

## Notebook Visuals

The walkthrough notebook is:

```text
notebooks/step4_rule_based_detection.ipynb
```

It includes three evaluation visuals:

- per-class precision, recall, and F1
- normalized confusion matrix
- true versus predicted class volume

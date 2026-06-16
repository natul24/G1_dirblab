# Driblab Pipeline Data Dictionary

This dictionary explains the columns used by the current project pipeline.
It is written for someone who has not seen the code or notebooks before.

The main modeling unit is a **tracking frame**. Tracking is sampled at about
10 Hz, so most output tables have one row per live tracking frame. Events are
joined onto those tracking rows by match clock.

## Current Schema Version

This dictionary reflects the current Step 2 through pass-model schema. Step 3
columns are named by what they represent:

| Concept | Current column-name pattern |
| --- | --- |
| Smoothed possession assignment | `smoothed_possession_*` |
| Previous smoothed possession assignment | `previous_smoothed_possession_*` |
| Possession changes | `smoothed_possession_change`, `possession_team_change`, `possession_player_change`, `possession_change_type` |
| Possession sequences | `possession_sequence_*` |
| Step 4 lagged possession features | `rule_prev_smoothed_possession_*` |

Step 3 appends these exact columns to the master join table:

| Column |
| --- |
| `data_split` |
| `raw_possession_key` |
| `smoothed_possession_key` |
| `smoothed_possession_team_id` |
| `smoothed_possession_player_id` |
| `smoothed_has_possession` |
| `previous_smoothed_possession_key` |
| `previous_smoothed_possession_team_id` |
| `previous_smoothed_possession_player_id` |
| `smoothed_possession_start` |
| `smoothed_possession_change` |
| `possession_team_change` |
| `possession_player_change` |
| `possession_sequence_number` |
| `possession_sequence_id` |
| `possession_sequence_frame_number` |
| `possession_sequence_duration_sec` |
| `smoothed_possession_team_name` |
| `smoothed_possession_player_name` |
| `possession_change_type` |

Current output widths:

| Table | Columns |
| --- | ---: |
| Master join table | 81 |
| Smoothed possession sequence table | 101 |
| Rule-based predictions table | 108 |

## Main Tables

| Table | Path | Grain | Purpose |
| --- | --- | --- | --- |
| Master join table | `data/processed/model_base/master_join_table.parquet` | One row per tracking frame | Step 2 output. This is the base table that joins tracking, selected event labels, ball features, player-distance features, and attacking-direction coordinates. |
| Smoothed possession sequence table | `data/processed/possession_sequence/possession_sequence_table.parquet` | One row per tracking frame | Step 3 output. It keeps all master join columns and adds smoothed possession assignment, possession-change, and possession-sequence columns. |
| Rule-based predictions table | `data/processed/rule_based_detection/rule_based_predictions.parquet` | One row per tracking frame | Step 4 output. It keeps all smoothed possession sequence columns and adds rule features, true event class, predicted event class, and rule reason. |

## Coordinate Rules

| Concept | Meaning |
| --- | --- |
| Event coordinates | Provider event x/y fields are already normalized from `0` to `100`, with the eventing team always attacking toward `x = 100`. They are not rescaled in Step 2. |
| Tracking coordinates | Tracking x/y starts as physical pitch coordinates. Step 2 normalizes saved tracking x/y fields to `0` to `100`. |
| Ball z coordinate | Ball height stays in meters. It is renamed to `ball_z_m` or `ball_z_m_raw`; it is not normalized to `0` to `100`. |
| Attacking tracking coordinates | `*_attacking` columns flip tracking x when needed so that the relevant reference team attacks toward `x = 100`. Y is not flipped. |
| Reference team for attacking orientation | Step 2 uses `event_team_id` when available, then `possessing_team_id`, then `nearest_team_id`. This decision is stored in `tracking_reference_source`. |

The modelling convention is:

```text
x = 0      defensive goal / own-goal side
x = 100    attacking goal
y = 0      one touchline
y = 100    opposite touchline
```

Tracking x/y values start in meters and are normalized before being saved in the
master join table:

```text
tracking_x_0_100 = clip(tracking_x_m / 105 * 100, 0, 100)
tracking_y_0_100 = clip(tracking_y_m / 68 * 100, 0, 100)
```

For attacking-perspective tracking columns, Step 2 keeps `ball_x_attacking =
ball_x` when the reference team attacks in the physical `+x` direction and uses
`ball_x_attacking = 100 - ball_x` when the reference team attacks in the
physical `-x` direction.

## Label Rules

| Concept | Meaning |
| --- | --- |
| `event_label` | Pipe-separated provider event names matched to that tracking frame. If no event matched, it is `"no event"`. |
| Multiple events in one frame | More than one provider event can match the same 10 Hz tracking frame. In that case event names, IDs, team IDs, and player IDs are joined with `|`. |
| First event detail columns | When several events match one frame, the detailed `event_*` columns such as `event_x`, `event_team_name`, and `event_player_name` come from the first event in that frame after sorting by event time and event ID. |
| `true_event_class` | Step 4 broad class created from `event_label` using label groups in `config.yaml`. |
| `rule_event_class` | Step 4 broad class predicted by possession and ball-movement rules. |
| `is_pass` | Binary pass target for the logistic regression model. `1` means `event_label` contains one of the pass labels configured in `config.yaml`; otherwise `0`. |

## Master Join Table Columns

These columns appear in `master_join_table.parquet`. They are also inherited by
the Step 3 smoothed possession sequence table and Step 4 rule-based prediction table.

### Player Tracking Rule

Step 2 uses player positions to identify the nearest player to the ball and to
create frame-level player availability and distance-to-ball aggregates. Player
speed aggregates are not stored in the master join table.

### Ball Speed Timing Rule

Ball speed and acceleration are also frame-to-frame calculations. For ball
features, a consecutive frame means the next tracking row in the same
`period_id` after sorting by `video_timestamp`; it is not simply `frame_id + 1`.
Step 2 keeps all tracking rows, interpolates short ball gaps, and then
calculates `dt_sec` from the previous tracking row in the same period.

The timing gap is considered valid only when `0 < dt_sec <= max_speed_dt_sec`.
The default `max_speed_dt_sec` is `0.50` seconds. Ball speed and acceleration
are set to missing for larger gaps so missing tracking stretches do not create
artificial speed spikes.

`ball_acceleration_mps2` is calculated as change in 3D ball speed divided by
`dt_sec`: `(current ball_speed_mps - previous ball_speed_mps) / dt_sec`. It is
therefore a speed-change acceleration, not a full x/y/z acceleration vector.

| Column | Type | Source | Description |
| --- | --- | --- | --- |
| `match_id` | string | Raw file name / match metadata | Match identifier. This is the key used to keep train, validation, and test splits separate by full match. |
| `frame_id` | integer | Tracking | Tracking frame number from the tracking JSONL file. |
| `period_id` | integer | Tracking and events | Match period, usually `1` for first half and `2` for second half. Used with match clock for event-to-frame matching. |
| `match_clock_min` | integer | Tracking | Minute component of the tracking match clock. This clock counts match time and does not reset at half-time in the current data. |
| `match_clock_sec` | integer | Tracking | Second component of the tracking match clock. |
| `video_timestamp` | float | Tracking | Video timestamp in seconds from the tracking feed. Used for frame-to-frame time differences and speed calculations. |
| `cam_present` | boolean | Tracking | Whether the tracking frame has camera/live-play data available. Step 2 keeps all tracking rows and uses this field to mark whether speed/possession calculations are reliable. |
| `ball_x_raw` | float | Tracking, normalized | Ball x position after converting the original tracking x coordinate to the `0-100` pitch scale. Despite the `_raw` suffix, this saved column is normalized. |
| `ball_y_raw` | float | Tracking, normalized | Ball y position after converting the original tracking y coordinate to the `0-100` pitch scale. Despite the `_raw` suffix, this saved column is normalized. |
| `ball_z_m_raw` | float | Tracking | Raw ball height in meters before interpolation. This is not normalized. |
| `tracking_match_clock_seconds` | float | Tracking-derived | Continuous tracking time in seconds. Built from `match_clock_min * 60 + match_clock_sec + frame_index_within_same_second / FPS`. This creates 10 Hz timing inside each whole-second clock value. |
| `ball_x` | float | Tracking, normalized/interpolated | Ball x position on the `0-100` scale after short-gap interpolation within each period. |
| `ball_y` | float | Tracking, normalized/interpolated | Ball y position on the `0-100` scale after short-gap interpolation within each period. |
| `ball_z_m` | float | Tracking/interpolated | Ball height in meters after short-gap interpolation. Not normalized. |
| `ball_present_raw` | boolean | Tracking-derived | True when raw ball x, y, and z were all present before interpolation. |
| `ball_interpolated` | boolean | Tracking-derived | True when raw ball data was missing but Step 2 filled the ball x/y/z values using short-gap linear interpolation. |
| `dt_sec` | float | Tracking-derived | Time difference in seconds from the previous tracking row in the same period, based on `video_timestamp`. |
| `ball_speed_xy_mps` | float | Tracking-derived | Ball speed in meters per second using only horizontal x/y movement: `sqrt(dx^2 + dy^2) / dt_sec`. Calculated before saved x/y normalization. |
| `ball_speed_mps` | float | Tracking-derived | Ball speed in meters per second using x, y, and z movement: `sqrt(dx^2 + dy^2 + dz^2) / dt_sec`. |
| `ball_acceleration_mps2` | float | Tracking-derived | Frame-to-frame change in `ball_speed_mps` divided by `dt_sec`: `(current speed - previous speed) / dt_sec`. |
| `nearest_team_id` | string | Tracking-derived from players + ball | Team ID of the player nearest to the ball in that frame. |
| `nearest_team_name` | string | Tracking metadata + nearest player | Team name of the player nearest to the ball. |
| `nearest_player_id` | string | Tracking-derived from players + ball | Player ID of the player nearest to the ball in that frame. |
| `nearest_player_name` | string | Tracking metadata + nearest player | Player name of the player nearest to the ball. |
| `nearest_player_visible` | boolean/object | Tracking-derived from nearest player | Visibility flag for the nearest player. True means the player's position was directly visible in the tracking frame. False means the provider still gave a coordinate, but the player was not directly visible, so the coordinate may be estimated/imputed by the provider. This is not an AI imputation created by this project. |
| `nearest_player_distance_to_ball_m` | float | Tracking-derived | Distance in meters between the nearest player and the ball. Built from player x/y and ball x/y in physical tracking coordinates. |
| `has_possession` | boolean | Tracking-derived | Raw Step 2 possession flag. True when the nearest player is within the configured possession distance and ball speed is below the configured possession speed threshold. |
| `possessing_team_id` | string | Tracking-derived | Team ID assigned raw possession when `has_possession` is true; missing otherwise. |
| `possessing_team_name` | string | Tracking-derived | Team name assigned raw possession when `has_possession` is true; missing otherwise. |
| `possessing_player_id` | string | Tracking-derived | Player ID assigned raw possession when `has_possession` is true; missing otherwise. |
| `possessing_player_name` | string | Tracking-derived | Player name assigned raw possession when `has_possession` is true; missing otherwise. |
| `player_count` | integer | Tracking-derived aggregate | Number of player rows available for the frame. Usually players from both teams. |
| `visible_player_count` | integer | Tracking-derived aggregate | Number of players in the frame with `player_visible = True`. |
| `min_distance_to_ball_m` | float | Tracking-derived aggregate | Minimum player-to-ball distance in meters across all players in the frame. Equivalent to the nearest-player distance when valid. |
| `mean_distance_to_ball_m` | float | Tracking-derived aggregate | Average player-to-ball distance in meters across all players in the frame. |
| `event_count_at_frame` | integer | Event-to-frame aggregation | Number of provider events matched to this tracking frame. `0` means no provider event matched. |
| `event_type_names_at_frame` | string | Event-to-frame aggregation | Pipe-separated provider event names matched to this frame, such as `PASS|BALL TOUCH`. Filled as `"no event"` when no event matched. |
| `event_type_ids_at_frame` | string | Event-to-frame aggregation | Pipe-separated provider event type IDs matched to this frame. Empty string when no event matched. |
| `event_ids_at_frame` | string | Event-to-frame aggregation | Pipe-separated provider event IDs matched to this frame. Empty string when no event matched. |
| `event_team_ids_at_frame` | string | Event-to-frame aggregation | Pipe-separated team IDs from events matched to this frame. Empty string when no event matched. |
| `event_player_ids_at_frame` | string | Event-to-frame aggregation | Pipe-separated player IDs from events matched to this frame. Empty string when no event matched. |
| `first_event_match_clock_seconds` | float | Event-to-frame aggregation | Earliest event match-clock time among the events matched to this frame. Missing when no event matched. |
| `event_label` | string | Event-to-frame aggregation | Modeling target label. Equal to `event_type_names_at_frame`; `"no event"` when no event matched. |
| `event_match_id` | float | Events | Match ID from the first provider event matched to the frame. Missing for no-event rows. |
| `event_period_id` | float | Events | Period ID from the first provider event matched to the frame. Missing for no-event rows. |
| `event_min` | float | Events | Minute value from the first provider event matched to the frame. Missing for no-event rows. |
| `event_sec` | float | Events | Second value from the first provider event matched to the frame. Missing for no-event rows. |
| `event_x` | float | Events | Event x coordinate from the provider, already on the `0-100` attacking-direction scale. Missing for no-event rows. |
| `event_y` | float | Events | Event y coordinate from the provider, already on the `0-100` scale. Missing for no-event rows. |
| `event_outcome` | boolean/object | Events | Provider outcome field for the first matched event. Meaning depends on event type; for some event types it is always true, while for others it can represent success/failure. |
| `event_qualifiers` | string | Events | Serialized JSON-like event qualifiers from the first matched event. Qualifiers can include extra details such as pass end location. |
| `event_possession_id` | float | Events | Provider possession sequence ID from the first matched event. |
| `event_xa` | float | Events | Provider expected-assist value for the first matched event, when available. |
| `event_xg` | float | Events | Provider expected-goal value for the first matched event, when available. |
| `event_xt` | float | Events | Provider expected-threat value for the first matched event, when available. |
| `event_x_start` | float | Events | Provider start x coordinate for the first matched event, already on the `0-100` attacking-direction scale. |
| `event_y_start` | float | Events | Provider start y coordinate for the first matched event, already on the `0-100` scale. |
| `event_x_end` | float | Events | Provider end x coordinate for the first matched event, already on the `0-100` attacking-direction scale. |
| `event_y_end` | float | Events | Provider end y coordinate for the first matched event, already on the `0-100` scale. |
| `event_milisec` | float | Events | Provider millisecond value for the first matched event. Used with `event_min` and `event_sec` to calculate event match-clock seconds. The source field is spelled `milisec`. |
| `event_id` | float | Events | Provider event ID from the first event matched to the frame. |
| `event_type_id` | float | Events | Provider event type ID from the first event matched to the frame. |
| `event_type_name` | string | Events | Provider event name from the first event matched to the frame. Filled as `"no event"` for no-event rows. |
| `event_team_id` | float | Events | Provider team ID from the first event matched to the frame. |
| `event_team_name` | string | Events | Provider team name from the first event matched to the frame. |
| `event_player_id` | float | Events | Provider player ID from the first event matched to the frame. |
| `event_player_name` | string | Events | Provider player name from the first event matched to the frame. |
| `event_match_clock_seconds` | float | Event-derived | Event time in seconds, calculated as `min * 60 + sec + milisec / 1000`. |
| `event_matched_tracking_match_clock_seconds` | float | Event-to-frame join | Tracking-frame time selected for the first matched event using nearest match-clock join within the configured tolerance. |
| `event_match_clock_join_error_sec` | float | Event-to-frame join | Difference between event time and matched tracking time: `event_match_clock_seconds - event_matched_tracking_match_clock_seconds`. |
| `event_attack_direction` | string | Event/tracking alignment | Inferred tracking attack direction for the event team in that period: `+x` or `-x`. Missing for no-event rows or neutral team rows. |
| `event_attack_direction_sign` | float | Event/tracking alignment | Numeric form of `event_attack_direction`: `1` for `+x`, `-1` for `-x`. |
| `is_event_frame` | boolean | Event-to-frame aggregation | True when `event_count_at_frame > 0`. False for tracking rows labeled `"no event"`. |
| `tracking_reference_team_id` | string | Tracking/event alignment | Team ID used to orient tracking coordinates for `*_attacking` columns. Chosen from event team, possession team, then nearest-player team. |
| `tracking_reference_source` | string | Tracking/event alignment | Source used for `tracking_reference_team_id`: `event_team`, `possession_team`, or `nearest_team`. |
| `tracking_attack_direction_sign` | float | Tracking/event alignment | Direction sign for the reference team in that period: `1` means the team attacks toward increasing x, `-1` means it attacks toward decreasing x. |
| `tracking_attack_direction` | string | Tracking/event alignment | Text version of `tracking_attack_direction_sign`: `+x` or `-x`. |
| `ball_x_raw_attacking` | float | Tracking-derived alignment | `ball_x_raw` flipped into the reference team's attacking orientation. If `tracking_attack_direction_sign = -1`, this is `100 - ball_x_raw`; otherwise it equals `ball_x_raw`. |
| `ball_x_attacking` | float | Tracking-derived alignment | Interpolated `ball_x` flipped into the reference team's attacking orientation. If `tracking_attack_direction_sign = -1`, this is `100 - ball_x`; otherwise it equals `ball_x`. |
| `ball_y_raw_attacking` | float | Tracking-derived alignment | `ball_y_raw` in the attacking-coordinate view. Y is not flipped, so this equals `ball_y_raw` when a reference team exists. |
| `ball_y_attacking` | float | Tracking-derived alignment | Interpolated `ball_y` in the attacking-coordinate view. Y is not flipped, so this equals `ball_y` when a reference team exists. |

## Step 3 Smoothed Possession Sequence Columns

These columns are added by `possession_sequence_table.parquet`. The table keeps
all master join columns and appends the fields below.

| Column | Type | Source | Description |
| --- | --- | --- | --- |
| `data_split` | string | `config.yaml` match split | Split assignment for the row: `train`, `validation`, or `test`. Assigned by complete `match_id`, not by individual frame. |
| `raw_possession_key` | string | Step 2 possession columns | Combined key built as `possessing_team_id|possessing_player_id` when `has_possession` is true. Missing when Step 2 does not assign possession. |
| `smoothed_possession_key` | string | Step 3 smoothing | Smoothed possession key after filling short no-possession gaps and removing unstable one-frame flips. Same format as `team_id|player_id`. |
| `smoothed_possession_team_id` | string | Step 3 smoothing | Team ID extracted from `smoothed_possession_key`. |
| `smoothed_possession_player_id` | string | Step 3 smoothing | Player ID extracted from `smoothed_possession_key`. |
| `smoothed_has_possession` | boolean | Step 3 smoothing | True when `smoothed_possession_key` is present after smoothing. |
| `previous_smoothed_possession_key` | string | Step 3 sequence logic | Previous active smoothed possession key, forward-filled over no-possession gaps and shifted by one frame. |
| `previous_smoothed_possession_team_id` | string | Step 3 sequence logic | Team ID extracted from `previous_smoothed_possession_key`. |
| `previous_smoothed_possession_player_id` | string | Step 3 sequence logic | Player ID extracted from `previous_smoothed_possession_key`. |
| `smoothed_possession_start` | boolean | Step 3 sequence logic | True when the current frame has smoothed possession and there was no previous active smoothed possession key. |
| `smoothed_possession_change` | boolean | Step 3 sequence logic | True when current smoothed possession exists, previous smoothed possession exists, and the current key is different from the previous key. |
| `possession_team_change` | boolean | Step 3 sequence logic | True when `smoothed_possession_change` is true and the team ID changed. This is part of the event skeleton for opponent gains, interceptions, and tackles. |
| `possession_player_change` | boolean | Step 3 sequence logic | True when `smoothed_possession_change` is true and the player ID changed. This includes teammate changes and opponent changes. |
| `possession_sequence_number` | float | Step 3 sequence logic | Numeric segment ID within each match/period. It increases whenever the smoothed possession key changes. Missing for no-possession frames. |
| `possession_sequence_id` | string | Step 3 sequence logic | Human-readable possession segment ID built from `match_id`, `period_id`, and `possession_sequence_number`. |
| `possession_sequence_frame_number` | float | Step 3 sequence logic | Frame number within the current smoothed possession sequence, starting at 1. Missing for no-possession frames. |
| `possession_sequence_duration_sec` | float | Step 3 sequence logic | Estimated duration of the current smoothed possession sequence. Built as sequence frame count times median frame `dt_sec` for that group. |
| `smoothed_possession_team_name` | string | Step 3 lookup | Team name mapped from `smoothed_possession_key` using names observed in raw Step 2 possession rows. |
| `smoothed_possession_player_name` | string | Step 3 lookup | Player name mapped from `smoothed_possession_key` using names observed in raw Step 2 possession rows. |
| `possession_change_type` | string | Step 3 sequence logic | Broad change label: `team_change`, `player_change`, `possession_start`, or `no_change`. Team changes take priority over player changes. |

## Step 4 Rule-Based Prediction Columns

These columns are added by `rule_based_predictions.parquet`. The table keeps all
master join and Step 3 columns, then appends the rule fields below.

| Column | Type | Source | Description |
| --- | --- | --- | --- |
| `rule_prev_smoothed_possession_team_id` | string | Step 4 lag feature | Previous frame's `smoothed_possession_team_id` within the same match and period. Used by rules to detect possession transitions. |
| `rule_prev_smoothed_possession_player_id` | string | Step 4 lag feature | Previous frame's `smoothed_possession_player_id` within the same match and period. |
| `rule_ball_dx_attacking` | float | Step 4 ball movement | Frame-to-frame change in `ball_x_attacking` within the same match and period. Positive values mean the ball moved toward the reference team's attacking goal. |
| `rule_ball_dy_attacking` | float | Step 4 ball movement | Frame-to-frame change in `ball_y_attacking` within the same match and period. |
| `true_event_class` | string | Event label mapping | Broad evaluation class created from `event_label` using the label groups in `config.yaml`. Possible classes include `no event`, `pass`, `interception`, `tackle`, `shot`, `out`, `corner`, and `other_event`. |
| `rule_event_class` | string | Step 4 rules | Event class predicted by rules. Examples: same-team player change becomes `pass`; opponent gain with fast ball becomes `interception`; opponent gain with slow ball becomes `tackle`; fast ball toward goal becomes `shot`; boundary locations become `out` or `corner`. |
| `rule_reason` | string | Step 4 rules | Human-readable reason for the prediction. Values include `same_team_player_change`, `opponent_gain_fast_ball`, `opponent_gain_slow_ball`, `fast_ball_toward_goal`, `ball_near_pitch_boundary`, `ball_near_corner_boundary`, and `no_rule_fired`. |

## Pass Model In-Memory Prediction Columns

These columns are created in memory by `predict_passes` for notebook plots and
metric calculation. They are not saved as a separate file by default, because
the simple pass model only persists the trained model and the metrics table.

| Column | Type | Source | Description |
| --- | --- | --- | --- |
| `match_id` | string | Master join | Match identifier for the frame. |
| `data_split` | string | `config.yaml` match split | Train, validation, or test split assigned by complete match ID. |
| `frame_id` | integer | Master join | Tracking frame number. |
| `period_id` | integer | Master join | Match period. |
| `tracking_match_clock_seconds` | float | Master join | Continuous tracking time in seconds used to locate the frame. |
| `event_label` | string | Master join | Provider event label joined to the frame, or `"no event"`. |
| `is_pass` | integer | Pass model target | Binary target. `1` if `event_label` contains a configured pass label; `0` otherwise. |
| `pass_probability` | float | Logistic regression output | Predicted probability that the frame is a pass frame. Produced by `predict_proba`. |
| `pass_prediction` | integer | Logistic regression output | Binary prediction after applying the configured threshold to `pass_probability`. `1` means predicted pass; `0` means predicted not-pass. |

## Pass Model Feature Columns

The logistic regression pass model is trained with the feature list in
`config.yaml`. The feature list is returned in memory when the pass model runs;
it is not saved as a separate CSV file. Every input feature below is read from
`master_join_table.parquet`, but the table distinguishes fields that are kept
from the original join from fields calculated during Step 2.

| Feature | Feature kind | Created or calculated? | Description |
| --- | --- | --- | --- |
| `period_id` | Original master-join column | No | Match period kept from the tracking/event join. |
| `match_clock_min` | Original master-join column | No | Minute component kept from the tracking clock. |
| `match_clock_sec` | Original master-join column | No | Second component kept from the tracking clock. |
| `cam_present` | Original master-join column | No | Tracking camera/live-play availability flag. |
| `ball_z_m_raw` | Original master-join column | No | Raw tracking ball height kept in meters. |
| `ball_x_raw` | Transformed master-join column | Transformed | Raw tracking ball x converted from meters to `0-100`. |
| `ball_y_raw` | Transformed master-join column | Transformed | Raw tracking ball y converted from meters to `0-100`. |
| `tracking_match_clock_seconds` | Calculated master-join feature | Yes | Continuous 10 Hz time from clock and frame order. |
| `ball_x` | Calculated master-join feature | Yes | Normalized ball x after short-gap interpolation. |
| `ball_y` | Calculated master-join feature | Yes | Normalized ball y after short-gap interpolation. |
| `ball_z_m` | Calculated master-join feature | Yes | Ball height after short-gap interpolation. |
| `ball_present_raw` | Calculated master-join feature | Yes | Flag that raw ball x/y/z were present. |
| `ball_interpolated` | Calculated master-join feature | Yes | Flag that Step 2 filled a short ball gap. |
| `dt_sec` | Calculated master-join feature | Yes | Seconds elapsed since the previous tracking frame in the same period. |
| `ball_speed_xy_mps` | Calculated master-join feature | Yes | Horizontal ball speed calculated in meters/second. |
| `ball_speed_mps` | Calculated master-join feature | Yes | 3D ball speed calculated in meters/second. |
| `ball_acceleration_mps2` | Calculated master-join feature | Yes | Frame-to-frame ball acceleration. |
| `nearest_player_visible` | Calculated master-join feature | Yes | Whether the player nearest to the ball was directly visible in tracking. False can indicate a provider-estimated/non-visible coordinate. |
| `nearest_player_distance_to_ball_m` | Calculated master-join feature | Yes | Distance from the nearest player to the ball. |
| `has_possession` | Calculated master-join feature | Yes | Raw possession flag from distance and ball speed. |
| `player_count` | Calculated master-join feature | Yes | Number of player rows available in the frame. |
| `visible_player_count` | Calculated master-join feature | Yes | Number of directly visible player rows. |
| `min_distance_to_ball_m` | Calculated master-join feature | Yes | Minimum player-to-ball distance in the frame. |
| `mean_distance_to_ball_m` | Calculated master-join feature | Yes | Average player-to-ball distance in the frame. |

### Pass Model Created Columns

These columns are created for modeling or evaluation. They are not input
features used to train the logistic regression model.

| Column | Source | Description |
| --- | --- | --- |
| `data_split` | `config.yaml` match split | Train, validation, or test assignment by complete match ID. |
| `is_pass` | Created from `event_label` | Binary target. `1` means the matched event label contains a configured pass label. |
| `pass_probability` | Logistic regression output | Predicted probability that the frame is a pass. |
| `pass_prediction` | Logistic regression output | Binary prediction after applying the configured threshold. |

## Summary and Evaluation Tables

### Master Join Summary

Path: `data/processed/model_base/master_join_summary.csv`

| Column | Description |
| --- | --- |
| `match_id` | Match identifier. |
| `master_join_table_rows` | Number of tracking rows in the master join table for the match. |
| `master_join_event_rows` | Number of master join rows where at least one event matched. |
| `aligned_events` | Number of provider event rows processed for event-to-frame alignment. |
| `matched_events` | Number of provider events that found a tracking frame within the configured join tolerance. |
| `median_abs_match_clock_join_error_sec` | Median absolute event-to-frame time difference in seconds. |
| `p95_abs_match_clock_join_error_sec` | 95th percentile absolute event-to-frame time difference in seconds. |
| `live_frames` | Number of tracking frames with `cam_present = True`. |
| `frames_with_possession` | Number of frames where Step 2 assigned raw possession. |

### Smoothed Possession Sequence Summary

Path: `data/processed/possession_sequence/possession_sequence_summary.csv`

| Column | Description |
| --- | --- |
| `match_id` | Match identifier. |
| `rows` | Number of tracking-frame rows for the match. |
| `raw_possession_frames` | Number of rows with a non-missing `raw_possession_key`. |
| `smoothed_possession_frames` | Number of rows with a non-missing `smoothed_possession_key` after smoothing. |
| `possession_changes` | Number of rows where `smoothed_possession_change` is true. |
| `team_changes` | Number of rows where `possession_team_change` is true. |
| `player_changes` | Number of rows where `possession_player_change` is true. |
| `split` | Train, validation, or test split for the match. |

### Rule-Based Metrics by Class

Path: `data/processed/rule_based_detection/rule_based_metrics_by_class.csv`

| Column | Description |
| --- | --- |
| `event_class` | Event class being evaluated. |
| `precision` | `true_positive / (true_positive + false_positive)` for that class. Measures how often predictions of the class were correct. |
| `recall` | `true_positive / (true_positive + false_negative)` for that class. Measures how often true examples of the class were found. |
| `f1` | Harmonic mean of precision and recall for that class. |
| `support` | Number of true rows for that class in the evaluation split. |
| `true_positive` | Count of rows where true class and predicted class both equal `event_class`. |
| `false_positive` | Count of rows predicted as `event_class` where the true class was different. |
| `false_negative` | Count of rows truly belonging to `event_class` but predicted as another class. |

### Rule-Based Confusion Matrix

Path: `data/processed/rule_based_detection/rule_based_confusion_matrix.csv`

| Column | Description |
| --- | --- |
| `true_event_class` | Provider-derived broad class. |
| `predicted_event_class` | Rule-based predicted broad class. |
| `rows` | Number of evaluation rows with that true/predicted class combination. |

### Rule-Based Summary

Path: `data/processed/rule_based_detection/rule_based_summary.csv`

| Column | Description |
| --- | --- |
| `evaluation_split` | Split used for evaluation, normally `test`. |
| `rows` | Number of rows evaluated. |
| `matches` | Number of matches represented in the evaluation split. |
| `macro_f1` | Unweighted average F1 across configured rule classes. |
| `weighted_f1` | Average F1 weighted by class support. |

### Pass Model Metrics

Path: `data/processed/pass_classifier/pass_model_metrics.parquet`

| Column | Description |
| --- | --- |
| `split` | Split being evaluated: train, validation, or test. |
| `rows` | Number of frame rows in the split. |
| `positive_rows` | Number of rows where `is_pass = 1`. |
| `positive_rate` | `positive_rows / rows`. |
| `accuracy` | Share of rows where `pass_prediction` equals `is_pass`. |
| `precision` | Among rows predicted as pass, the share that were true passes. |
| `recall` | Among true pass rows, the share predicted as pass. |
| `f1` | Harmonic mean of precision and recall. |
| `roc_auc` | Area under the ROC curve using `pass_probability`. Measures ranking quality independent of the classification threshold. |

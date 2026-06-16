# Driblab Pipeline Data Dictionary

This dictionary explains the columns used by the current project pipeline.
It is written for someone who has not seen the code or notebooks before.

The main modeling unit is a **tracking frame**. Tracking is sampled at about
10 Hz, so current output tables have one row per tracking frame. Events are
joined onto those tracking rows by match clock.

## Current Schema Version

This dictionary reflects the current Step 2 master-join schema.

Current output widths:

| Table | Columns |
| --- | ---: |
| Master join table | 81 |

## Main Tables

| Table | Path | Grain | Purpose |
| --- | --- | --- | --- |
| Master join table | `data/processed/model_base/master_join_table.parquet` | One row per tracking frame | Step 2 output. This is the base table that joins tracking, selected event labels, ball features, player-distance features, and attacking-direction coordinates. |

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

## Master Join Table Columns

These columns appear in `master_join_table.parquet`.

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

## Summary Tables

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

# Step 2 Foundation: Tracking-First Training Table

Step 2 turns the raw Driblab tracking and event files into one table that can
be used as the base for modelling.

The central idea is:

```text
tracking frames are the main table
events are joined onto the closest tracking frame by match-clock time
```

This means the final dataset has one row per reliable tracking frame. If an
event happened at that frame time, the event columns are filled. If no event
happened at that frame time, the row is labelled `no event`.

## Inputs

Raw files live in `data/raw/`.

For each match, Step 2 expects:

```text
<match_id>_tracking_data.jsonl
<match_id>_events.json
```

The tracking file is JSONL:

- first line: match metadata, teams, players, FPS
- following lines: one tracking frame per row

The event file is flattened into a pandas table. Across the current data, each
match has the same 24 raw event columns.

## Match-Clock Sync and Drift Correction

The event-to-tracking sync is the drift-correction step. The pipeline uses
match clock only; it does not use `Videotimestamp` for the event join.

For events, the pipeline creates:

```text
event_match_clock_seconds = min * 60 + sec + milisec / 1000
```

For tracking, the raw match clock is only minute and second, but tracking is 10
Hz. Multiple frames share the same integer second, so the pipeline creates:

```text
tracking_match_clock_seconds =
    match_clock_min * 60
    + match_clock_sec
    + frame_position_inside_that_second / FPS
```

Example at 10 FPS:

```text
match_clock = 0:02
4th frame inside second 2
tracking_match_clock_seconds = 2 + 4 / 10 = 2.4
```

Then, for each period separately, events are matched to the nearest tracking
frame with:

```text
pd.merge_asof(
    left_on="event_match_clock_seconds",
    right_on="tracking_match_clock_seconds",
    direction="nearest",
    tolerance=0.5,
)
```

The QA field is:

```text
event_match_clock_join_error_sec =
    event_match_clock_seconds - event_matched_tracking_match_clock_seconds
```

This nearest-frame match corrects the practical timestamp drift between event
milliseconds and 10 Hz tracking frames by snapping each event to the closest
tracking frame on the same match-clock timeline. A small absolute join error
means little residual drift after syncing. In the current all-match output, the
average match median absolute join error is about `0.025` seconds.

## Final Master Join Table Shape

Main output:

```text
data/processed/model_base/master_join_table.parquet
```

Current generated shape:

```text
matches: 33
rows: 1,986,630
columns: 81
event-prefixed columns: 36
non-event tracking/feature columns: 45
tracking rows with no event: 1,944,246
tracking rows with at least one selected event: 42,384
```

The table is tracking-first:

- one row per tracking frame
- tracking/frame columns stay on every row
- event columns are filled only when one of the selected event types matched
  that frame
- event columns use the `event_` prefix
- rows without matched events have `event_label = "no event"` and
  `event_type_name = "no event"`

Step 2 currently keeps only these provider event types as modelling labels:

```text
PASS, BALL TOUCH, AERIAL, TACKLE, BALL RECOVERY, FOUL, TAKEON
```

All other provider events are ignored by the Step 2 event join and therefore
remain `no event` rows in the master join table.

## Coordinate System

The ETL confirmed that event coordinates are already in a normalized `0-100`
system where teams attack toward `x=100`.

Step 2 now uses normalized `0-100` field coordinates as the shared modelling
coordinate system for x/y positions:

- tracking ball x/y coordinates are converted from meters and clipped to `0-100`
- tracking ball x/y coordinates are also expressed from the relevant team's
  attacking perspective in columns such as `ball_x_attacking`
- original event coordinates are preserved as `event_x`, `event_y`,
  `event_x_start`, `event_y_start`, `event_x_end`, and `event_y_end`

The normalized field convention is:

```text
x = 0      defensive goal / own-goal side
x = 100    attacking goal
y = 0      one touchline
y = 100    opposite touchline
```

Event coordinates already follow this attacking-direction convention. Tracking
coordinates do not: they start as physical meter coordinates on a `105 x 68`
pitch. Step 2 converts saved tracking x/y fields with:

```text
tracking_x_0_100 = clip(tracking_x_m / 105 * 100, 0, 100)
tracking_y_0_100 = clip(tracking_y_m / 68 * 100, 0, 100)
```

Then Step 2 creates attacking-perspective tracking columns so the reference team
always attacks toward high x:

```text
if reference team attacks +x:
    ball_x_attacking = ball_x

if reference team attacks -x:
    ball_x_attacking = 100 - ball_x
```

So the ETL finding is reflected in Step 2 through the alignment logic: event
coordinates arrive as normalized `0-100`, then Step 2 converts tracking x/y to
the same scale and flips tracking x when the reference team is attacking in the
negative physical x direction. The final master join table does not keep x/y field
coordinates in meters. Ball height is vertical, not a field x/y coordinate, so
it is kept explicitly as `ball_z_m_raw` and `ball_z_m`.

The reference team for attacking-perspective tracking columns is chosen in this
order: event team on event rows, possession team when possession is detected,
then nearest team to the ball. The chosen reference is stored in
`tracking_reference_team_id` and `tracking_reference_source`.

The attacking-perspective tracking columns answer this question:

```text
Where is the tracking ball position from the reference team's attacking view?
```

For x-coordinates:

```text
if tracking_attack_direction = "+x":
    ball_x_attacking = ball_x

if tracking_attack_direction = "-x":
    ball_x_attacking = 100 - ball_x
```

For y-coordinates:

```text
ball_y_attacking = ball_y
```

Example: `ball_x_attacking = 20` means the ball is about 20% up the pitch from
the reference team's own goal toward the goal they are attacking. On a 105m
pitch, that is about `21m` from the reference team's own goal line. It does not
mean the ball came from the event file; it is still tracking-derived.

Tracking can contain occasional off-pitch or noisy x/y values. After converting
meters to the field scale, Step 2 clips x/y coordinates to `[0, 100]` so the
master join table only contains valid field-coordinate values.

### Coordinate Column Treatment

The table below separates columns that remain as-is from columns that are
normalized or added for alignment. The left side is the data kept in its
original coordinate meaning; the right side is the data converted or flipped
for the model-ready table.

| Source | Remained the same | Normalized / transformed |
| --- | --- | --- |
| Events | `event_x`, `event_y`, `event_x_start`, `event_y_start`, `event_x_end`, `event_y_end` keep the provider's original `0-100` attacking-direction coordinates | Not normalized or flipped |
| Tracking ball x/y | Original meter x/y values are used internally for interpolation, speed, acceleration, distance, possession, and attack-direction inference | Final table columns `ball_x_raw`, `ball_y_raw`, `ball_x`, and `ball_y` are converted from tracking meters to `0-100` and clipped to `[0, 100]`; `ball_x_raw_attacking`, `ball_y_raw_attacking`, `ball_x_attacking`, and `ball_y_attacking` express those tracking coordinates from the reference team's attacking perspective |
| Tracking ball z | `ball_z_m_raw` and `ball_z_m` stay in meters because height is not a field x/y coordinate | Not normalized |
| Tracking player x/y | Original meter player positions are used internally to identify nearest-player and distance-to-ball features | Player-level x/y rows are not saved in the public Step 2 outputs; only frame-level player count, visibility count, and distance-to-ball aggregates are kept in the master join table |
| Physical features | `ball_speed_xy_mps`, `ball_speed_mps`, `ball_acceleration_mps2`, and distance-to-ball fields stay in metric units | Not normalized |

If more than one event matches the same tracking frame, Step 2 keeps aggregate
frame labels:

- `event_count_at_frame`
- `event_type_names_at_frame`
- `event_type_ids_at_frame`
- `event_ids_at_frame`
- `event_team_ids_at_frame`
- `event_player_ids_at_frame`

For the full event detail columns, the pipeline keeps the first event in that
frame after sorting by event match-clock time and event id.

## Column Groups

Tracking/frame columns include:

- `match_id`
- `frame_id`
- `period_id`
- `match_clock_min`
- `match_clock_sec`
- `video_timestamp`
- `cam_present`
- `ball_x_raw`, `ball_y_raw`
- `ball_z_m_raw`
- `tracking_match_clock_seconds`

Clean ball features include:

- `ball_x`, `ball_y`
- `ball_x_raw_attacking`, `ball_y_raw_attacking`
- `ball_x_attacking`, `ball_y_attacking`
- `tracking_reference_team_id`, `tracking_reference_source`
- `tracking_attack_direction`, `tracking_attack_direction_sign`
- `ball_z_m`
- `ball_present_raw`
- `ball_interpolated`
- `dt_sec`
- `ball_speed_xy_mps`
- `ball_speed_mps`
- `ball_acceleration_mps2`

Possession and player aggregate features include:

- nearest player/team fields
- possession player/team fields
- player counts
- visible player count
- min/mean distance to ball

Event label columns include:

- `event_label`
- `event_count_at_frame`
- `event_type_names_at_frame`
- `first_event_match_clock_seconds`
- `is_event_frame`

Original event columns are preserved with the `event_` prefix, for example:

- `event_match_id`
- `event_period_id`
- `event_min`, `event_sec`, `event_milisec`
- `event_x`, `event_y`
- `event_x_start`, `event_y_start`, `event_x_end`, `event_y_end`
- `event_id`
- `event_type_id`
- `event_type_name`
- `event_team_id`, `event_team_name`
- `event_player_id`, `event_player_name`
- `event_qualifiers`
- `event_xg`, `event_xa`, `event_xt`

Step 2 also adds event-side join/coordinate fields:

- `event_match_clock_seconds`
- `event_matched_tracking_match_clock_seconds`
- `event_match_clock_join_error_sec`
- `event_attack_direction`
- `event_attack_direction_sign`

The raw nested tracking JSON player lists are not stored in the final Parquet
table for now. The table keeps flattened frame-level tracking fields and
engineered player aggregates instead.

### Ball Speed and Acceleration Features

Ball speed and acceleration are calculated from tracking-derived ball positions
before the saved x/y columns are normalized to the `0-100` field scale. The
calculations use metric tracking coordinates, so the output units remain metric.

A consecutive frame for these calculations means the next tracking row in the
same `period_id` after sorting by `video_timestamp`. It is not assumed to be
`frame_id + 1`.

The elapsed time field is:

```text
dt_sec = current video_timestamp - previous live-frame video_timestamp
```

Step 2 only treats the frame-to-frame jump as valid when:

```text
0 < dt_sec <= max_speed_dt_sec
```

The current default for `max_speed_dt_sec` is `0.50` seconds. Larger gaps are
left as missing for speed and acceleration so that long camera or tracking gaps
do not create artificial movement spikes.

For valid frame gaps, horizontal and 3D ball speed are:

```text
ball_speed_xy_mps = sqrt(dx^2 + dy^2) / dt_sec
ball_speed_mps    = sqrt(dx^2 + dy^2 + dz^2) / dt_sec
```

Ball acceleration is then the frame-to-frame change in 3D ball speed:

```text
ball_acceleration_mps2 =
    (current ball_speed_mps - previous ball_speed_mps) / dt_sec
```

This is acceleration based on change in speed, not a full x/y/z acceleration
vector.

## Cleaning and Feature Logic

Step 2 also does the foundation cleaning needed before modelling:

1. Syncs event timestamps to tracking frames by nearest match-clock frame; this
   is the timestamp drift-correction step.
2. Keeps all tracking frames and uses `cam_present` as a reliability field.
3. Interpolates short ball gaps inside live play.
4. Computes ball speed and acceleration.
5. Builds nearest-player, player-count, visibility-count, and distance-to-ball
   features.
6. Estimates possession from the nearest player when the player is close enough
   and the ball is not moving too fast.
7. Infers attacking direction by comparing provider-normalized event
   coordinates to normalized tracking ball coordinates.
8. Converts tracking ball x/y into attacking-perspective `0-100` columns using
   the event, possession, or nearest-player reference team.

## How To Run

From the project root, run the default all-match build configured in
`config.yaml`:

```bash
python main.py step2
```

Run a specific match:

```bash
python main.py step2 --match-id 678949
```

Run all matches explicitly:

```bash
python main.py step2 --all-matches
```

Rerunning Step 2 overwrites the same master join table and summary files. It
does not create duplicate dated or timestamped outputs.

## Outputs

Default all-match output:

```text
data/processed/model_base/master_join_table.parquet
data/processed/model_base/master_join_summary.csv
```

Specific-match output with `--match-id`:

```text
data/processed/model_base/master_join_table_<match_id>.parquet
data/processed/model_base/master_join_summary_<match_id>.csv
```

# Master Join Walkthrough

This guide explains Step 2 for someone who has not looked at the code.

Step 2 builds one table:

```text
data/processed/model_base/master_join_table.parquet
```

That table combines tracking data and selected event data. Tracking is the base
of the table, so every output row represents one original tracking frame.

## 1. What Goes In

Step 2 reads two raw files per match from `data/raw/`:

```text
<match_id>_tracking_data.jsonl
<match_id>_events.json
```

The tracking file is a JSONL file:

- line 1 is match metadata, including teams, players, and FPS
- every later line is one tracking frame

The event file is a JSON file:

- every row is one provider event
- nested fields are flattened with `pandas.json_normalize`
- examples of flattened event columns are `event.event_type_name`,
  `team.team_name`, and `player.player_name`

## 2. The Table Starts From Tracking

The master join table is tracking-first.

That means:

- no tracking rows are dropped
- the final row count should equal the number of raw tracking frames
- most rows have no event, because tracking is sampled about 10 times per second
  and events happen less often

Every original tracking frame column is kept with a `t.` prefix. Step 2 also
adds `t.match_id` from the source filename so every row in the all-match table
can be traced back to its match, including rows with `"no event"`.

Examples:

```text
t.match_id
t.period
t.frame
t.Videotimestamp
t.match_clock
t.cam
```

The raw nested tracking fields `ball` and `data` are not kept as packed columns
in the master join. Step 2 unpacks the important position fields from those
nested values into normal `t.` columns:

```text
t.ball_x
t.ball_y
t.ball_z
t.player_count
t.visible_player_count
t.player_01_team_id
t.player_01_id
t.player_01_x
t.player_01_y
t.player_01_visible
...
```

The repeated `t.player_XX_*` columns are player slots within the tracking
frame. Each slot stores the tracked player's team id, player id, x/y pitch
position, and visibility flag. The number of slots is set from the maximum
number of player rows observed in the selected matches, so the all-match table
has a stable schema.

## 3. Which Events Are Kept

Step 2 does not join every provider event. It keeps only the event types that
cumulatively cover about 91% of all events in the current raw data.

The selected event types are:

```text
PASS
BALL TOUCH
AERIAL
TACKLE
BALL RECOVERY
FOUL
TAKEON
```

In the current raw files, these sum to about `91.355%` of provider events.

Events after `TAKEON` in the frequency table, such as lower-frequency shots,
cards, substitutions, or clearances, are not attached to tracking frames in this
Step 2 table. The tracking frames still remain; their event columns are filled
with `"no event"` unless one of the selected event types is attached.

Every original flattened event column is kept with an `e.` prefix.

Examples:

```text
e.match_id
e.period_id
e.min
e.sec
e.milisec
e.x
e.y
e.event.event_type_name
e.team.team_name
e.player.player_name
```

## 4. How Event Time Is Compared to Tracking Time

Events and tracking use match clock, but they store it differently.

For events, Step 2 creates an internal time in seconds:

```text
event_match_clock_seconds = min * 60 + sec + milisec / 1000
```

For tracking, each frame has `match_clock`, usually like:

```text
[minute, second]
```

Because tracking is sampled around 10 FPS, several frames share the same
integer second. Step 2 creates an internal tracking time by adding the frame's
position inside that second:

```text
tracking_match_clock_seconds =
    match_clock_min * 60
    + match_clock_sec
    + frame_position_inside_that_second / FPS
```

Example at 10 FPS:

```text
match_clock = [0, 2]
4th frame inside second 2
tracking_match_clock_seconds = 2 + 4 / 10 = 2.4
```

These internal time fields are used only for matching. They are not saved as
normal output columns.

## 5. How Events Are Attached to Frames

Step 2 matches events and tracking separately within each period.

For each selected event:

1. Look only at tracking frames from the same period.
2. Find the tracking frame with the nearest internal match-clock timestamp.
3. Attach the event to that tracking frame.
4. Save the absolute time distance in:

```text
nearest_timestamp_distance_sec
```

There is no tolerance window. Step 2 does not use `0.5` seconds or `1.0`
seconds as a cutoff. The nearest-frame rule is the rule, and the distance column
is only for quality checking.

If two selected events choose the same tracking frame, Step 2 keeps only one:

1. the event with the smallest `nearest_timestamp_distance_sec`
2. if still tied, the event that appeared earlier in the raw event table

## 6. What Happens When There Is No Selected Event

Most tracking rows do not get one of the selected events.

For those rows:

- all `t.*` tracking columns are still present, including `t.match_id`
- all `e.*` event columns are filled with `"no event"`
- `nearest_timestamp_distance_sec` is missing

This is why the table can be used to distinguish event frames from ordinary
tracking frames without losing the full tracking timeline.

## 7. What Step 2 Does Not Do

Step 2 is intentionally a raw master join. It does not create modelling
features.

It does not:

- normalize tracking coordinates
- flip coordinates into attacking direction
- convert event coordinates to meters
- calculate ball speed
- calculate ball acceleration
- calculate possession-change features
- calculate player-speed aggregates
- run a pass classifier

Event coordinates stay exactly as provider event columns. Tracking coordinates
stay in their raw tracking coordinate system. Ball and player x/y values are
unpacked from raw `ball` and `data`, but they are not normalized or flipped.
The training-table step intentionally excludes event coordinate columns to
avoid leakage into the pass detector, and it adds an explicit period-based
attacking-direction flag.

## 8. Output Files

Default all-match output:

```text
data/processed/model_base/master_join_table.parquet
data/processed/model_base/master_join_summary.csv
```

Specific-match output:

```text
data/processed/model_base/master_join_table_<match_id>.parquet
data/processed/model_base/master_join_summary_<match_id>.csv
```

The summary file has one row per match and includes:

- `tracking_rows`: raw tracking frame count
- `raw_events`: all raw provider events
- `selected_events`: events kept by the 91% cumulative event filter
- `event_type_names`: selected event types used by Step 2
- `matched_events`: selected events kept after nearest-frame matching and
  same-frame deduplication
- `master_join_rows`: final output row count
- `master_join_event_rows`: rows with an attached selected event
- median and p95 nearest timestamp distance

## 9. Sanity Checks

After Step 2 runs, these checks should be true:

```text
master_join_rows == tracking_rows
```

The only non-prefixed output column should be:

```text
nearest_timestamp_distance_sec
```

All source columns should be clearly marked:

```text
t.* = tracking source columns
e.* = event source columns
```

`t.match_id` should be present so every row can be grouped by match even when
`e.match_id` is `"no event"`.

The only attached event names should be:

```text
PASS, BALL TOUCH, AERIAL, TACKLE, BALL RECOVERY, FOUL, TAKEON
```

Rows without one of those events should show:

```text
e.event.event_type_name = "no event"
```

## 10. How To Run

Run the default all-match build configured in `config.yaml`:

```bash
python main.py master-join
```

`main.py master-join` builds the all-match table from every raw match that has
both an events file and a tracking file. Rerunning it overwrites the same master
join table and summary files. It does not create duplicate dated or timestamped
outputs.

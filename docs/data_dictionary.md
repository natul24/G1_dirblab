# Driblab Pipeline Data Dictionary

This dictionary describes the current Step 2 master join output.

The main unit is a tracking frame. Tracking is sampled at about 10 Hz, so the
master join table has one row per original tracking row. Selected events are
attached to the nearest tracking frame in the same period.

For the full plain-language build walkthrough, see
[`docs/master_join_walkthrough.md`](master_join_walkthrough.md).

## Main Tables

| Table | Path | Grain | Purpose |
| --- | --- | --- | --- |
| Master join table | `data/processed/model_base/master_join_table.parquet` | One row per original tracking frame | Raw Step 2 table joining tracking rows to nearest selected event rows. |
| Master join summary | `data/processed/model_base/master_join_summary.csv` | One row per match | QA counts and nearest timestamp distance summary. |

## Column Rules

| Column group | Meaning |
| --- | --- |
| `t.*` | Tracking-source columns. Original scalar JSONL frame fields are preserved, `t.match_id` is added from the source filename, and nested `ball`/`data` values are unpacked into normal `t.` columns. |
| `e.*` | Original flattened event columns from `pandas.json_normalize`. Empty event fields on tracking frames with no attached selected event are filled with `"no event"`. |
| `nearest_timestamp_distance_sec` | Absolute time distance between the event timestamp and the selected nearest tracking frame timestamp. Missing for tracking rows with no attached event. |

To read one row:

- `t.*` tells you what the tracking feed reported at that frame.
- `e.*` tells you which selected event, if any, was attached to that frame.
- `"no event"` in `e.*` means that tracking frame did not receive one of the
  selected Step 2 events.
- `nearest_timestamp_distance_sec` tells you how far the attached event time was
  from the selected tracking frame time.

Step 2 does not add modelling features. It does not normalize coordinates,
calculate speed, calculate acceleration, calculate possession, or calculate
player speed aggregates. It does unpack raw ball and player position fields from
the nested tracking columns.

## Selected Event Types

Step 2 keeps only the event types that cumulatively cover about 91% of all
provider events in the current raw data:

```text
PASS, BALL TOUCH, AERIAL, TACKLE, BALL RECOVERY, FOUL, TAKEON
```

Provider events after `TAKEON` in the frequency table are not joined into the
master table. Tracking rows remain present, and rows without one of the selected
events have `e.*` columns filled with `"no event"`.

## Coordinate Rules

| Source | Meaning |
| --- | --- |
| Event coordinates | Event x/y fields are provider columns. They are already on the provider `0-100` attacking-direction scale, where the eventing team attacks toward high x. Step 2 keeps them as-is in columns such as `e.x`, `e.y`, `e.x_start`, and `e.x_end`. |
| Tracking coordinates | Tracking coordinate values are unpacked from raw `ball` and `data` fields into `t.ball_*` and `t.player_XX_*` columns. Step 2 does not normalize or flip them. |

The two systems are intentionally not reconciled in the master join. Future
feature notebooks can create normalized or attacking-direction coordinates from
the raw columns when those features are needed.

## Match-Clock Join

Internal event time:

```text
event_match_clock_seconds = min * 60 + sec + milisec / 1000
```

Internal tracking time:

```text
tracking_match_clock_seconds =
    match_clock_min * 60
    + match_clock_sec
    + frame_position_inside_that_second / FPS
```

Selected events are matched to tracking frames separately within each period
using the nearest timestamp. There is no tolerance window. If multiple selected
events choose the same tracking frame, only the event with the smallest
`nearest_timestamp_distance_sec` is kept.

## Tracking Columns

The current tracking files usually include these raw frame fields:

| Column | Description |
| --- | --- |
| `t.match_id` | Match identifier from the source tracking filename. Added by Step 2 so every tracking row can be traced to a match, including `"no event"` rows. |
| `t.period` | Tracking period from the raw frame row. |
| `t.frame` | Tracking frame number from the raw frame row. |
| `t.Videotimestamp` | Provider video timestamp from the raw frame row. |
| `t.match_clock` | Raw tracking match clock list, usually `[minute, second]`. |
| `t.ball_x`, `t.ball_y`, `t.ball_z` | Ball coordinates unpacked from the raw `ball` list. These stay in the raw tracking coordinate system. |
| `t.player_count` | Number of player rows unpacked from raw `data` for that frame. |
| `t.visible_player_count` | Number of unpacked player rows where the raw visibility flag is `True`. |
| `t.player_XX_team_id` | Team id for player slot `XX`, unpacked from raw `data`. |
| `t.player_XX_id` | Player id for player slot `XX`, unpacked from raw `data`. |
| `t.player_XX_x`, `t.player_XX_y` | Player x/y coordinates for player slot `XX`, unpacked from raw `data` and kept in the raw tracking coordinate system. |
| `t.player_XX_visible` | Raw visibility flag for player slot `XX`. |

Additional `t.*` columns are preserved if they exist in a match file.

## Event Columns

The current event files usually include these flattened fields:

| Column | Description |
| --- | --- |
| `e.match_id` | Provider match identifier. |
| `e.period_id` | Event period. |
| `e.min` | Event minute. |
| `e.sec` | Event second. |
| `e.milisec` | Event millisecond field. The provider field is spelled `milisec`. |
| `e.x`, `e.y` | Provider event location. |
| `e.x_start`, `e.y_start` | Provider event start location when available. |
| `e.x_end`, `e.y_end` | Provider event end location when available. |
| `e.outcome` | Provider outcome field. |
| `e.qualifiers` | Provider qualifiers, serialized if nested. |
| `e.possession_id` | Provider possession identifier. |
| `e.xa`, `e.xg`, `e.xt` | Provider value fields when available. |
| `e.event.id` | Provider event id. |
| `e.event.event_type_id` | Provider event type id. |
| `e.event.event_type_name` | Provider event type name. |
| `e.team.team_id`, `e.team.team_name` | Event team fields. |
| `e.player.player_id`, `e.player.player_name` | Event player fields when available. |

Additional `e.*` columns are preserved if they exist in a match file.

## Summary Columns

| Column | Description |
| --- | --- |
| `match_id` | Match identifier. |
| `tracking_rows` | Number of raw tracking frame rows. |
| `raw_events` | Number of raw event rows loaded. |
| `selected_events` | Number of raw event rows kept after the 91% cumulative event-type filter. |
| `event_type_names` | Pipe-separated selected event type names used by Step 2. |
| `matched_events` | Number of events assigned to a tracking frame before same-frame deduplication effects are reflected in the final event row count. |
| `master_join_rows` | Number of output rows. This should equal `tracking_rows`. |
| `master_join_event_rows` | Number of output rows where an event was attached. |
| `median_abs_nearest_timestamp_distance_sec` | Median absolute nearest-frame time distance for attached events. |
| `p95_abs_nearest_timestamp_distance_sec` | 95th percentile nearest-frame time distance for attached events. |

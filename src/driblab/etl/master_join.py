"""Step 2 raw tracking-event master join.

This module builds a tracking-first table with one row per tracking frame. It
keeps original per-frame tracking fields and original flattened event fields,
adds prefixes to identify each source, and attaches at most one event to each
tracking frame by nearest match-clock timestamp.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driblab.config import DEFAULT_MATCH_ID
from driblab.etl.pipeline import load_event_file, load_tracking_file


DEFAULT_STEP2_EVENT_TYPES = (
    "PASS",
    "BALL TOUCH",
    "AERIAL",
    "TACKLE",
    "BALL RECOVERY",
    "FOUL",
    "TAKEON",
)


@dataclass
class Step2Config:
    """Configuration for building Step 2 outputs for one match."""

    data_dir: Path
    output_dir: Path
    match_id: str = DEFAULT_MATCH_ID
    tracking_columns: tuple[str, ...] | None = None
    event_columns: tuple[str, ...] | None = None
    max_player_slots: int | None = None
    event_type_names: tuple[str, ...] = DEFAULT_STEP2_EVENT_TYPES
    save_match_outputs: bool = False


@dataclass
class Step2BatchConfig:
    """Configuration for building Step 2 outputs for one or many matches."""

    data_dir: Path
    output_dir: Path
    model_base_dir: Path
    match_id: str = DEFAULT_MATCH_ID
    all_matches: bool = False
    max_player_slots: int | None = None
    event_type_names: tuple[str, ...] = DEFAULT_STEP2_EVENT_TYPES
    save_match_outputs: bool = False


def available_matches(data_dir: Path) -> list[str]:
    """Return match IDs with both events and tracking files."""
    event_ids = {
        path.name.split("_", 1)[0]
        for path in data_dir.glob("*_events.json")
    }
    tracking_ids = {
        path.name.split("_", 1)[0]
        for path in data_dir.glob("*_tracking_data.jsonl")
    }
    return sorted(event_ids & tracking_ids)


def _append_parquet_chunk(
    path: Path,
    chunk: pd.DataFrame,
    writer: object | None,
    schema: object | None,
) -> tuple[object, object]:
    """Append one dataframe chunk to a single Parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Parquet output requires pyarrow. Install it with "
            "`conda env update -f environment.yml --prune`."
        ) from exc

    if schema is None:
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
        schema = table.schema
    else:
        table = pa.Table.from_pandas(
            chunk,
            schema=schema,
            preserve_index=False,
        )

    writer.write_table(table)
    return writer, schema


def _ordered_keys(records: list[dict[str, Any]]) -> list[str]:
    """Return keys in first-seen order across raw dictionaries."""
    keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _serialize_cell(value: Any) -> Any:
    """Keep scalars as-is and serialize nested original JSON values."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _safe_period(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def _tracking_match_seconds(
    tracking_frames: list[dict[str, Any]],
    fps: float,
) -> pd.Series:
    """Create internal 10 Hz match-clock seconds for tracking frames."""
    rows = []
    for row_id, frame in enumerate(tracking_frames):
        clock = frame.get("match_clock") or [np.nan, np.nan]
        minute = clock[0] if len(clock) > 0 else np.nan
        second = clock[1] if len(clock) > 1 else np.nan
        whole_second = (
            pd.to_numeric(pd.Series([minute]), errors="coerce").iloc[0] * 60
            + pd.to_numeric(pd.Series([second]), errors="coerce").iloc[0]
        )
        rows.append(
            {
                "_tracking_row_id": row_id,
                "_period_id": _safe_period(frame.get("period")),
                "_whole_second": whole_second,
            }
        )

    timing = pd.DataFrame(rows)
    frame_index_inside_second = timing.groupby(
        ["_period_id", "_whole_second"],
        dropna=False,
    ).cumcount()
    return timing["_whole_second"].astype(float) + (
        frame_index_inside_second.astype(float) / fps
    )


def _tracking_player_rows(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Return player tracking rows from one raw frame in stable order."""
    players: list[dict[str, Any]] = []
    for team_id, team_players in sorted((frame.get("data") or {}).items()):
        for player in team_players or []:
            players.append(
                {
                    "team_id": str(team_id),
                    "player_id": player.get("id"),
                    "player_x": player.get("x"),
                    "player_y": player.get("y"),
                    "player_visible": player.get("vis"),
                }
            )
    return players


def _max_player_slots(tracking_frames: list[dict[str, Any]]) -> int:
    """Return the maximum number of player rows in any frame."""
    if not tracking_frames:
        return 0
    return max(len(_tracking_player_rows(frame)) for frame in tracking_frames)


def build_tracking_frame_table(
    tracking_header: dict[str, Any],
    tracking_frames: list[dict[str, Any]],
    tracking_columns: tuple[str, ...] | None = None,
    match_id: str | None = None,
    max_player_slots: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build prefixed original tracking rows plus internal match columns."""
    frame_columns = list(tracking_columns or _ordered_keys(tracking_frames))
    fps = float(tracking_header.get("FPS", 10))
    match_seconds = _tracking_match_seconds(tracking_frames, fps=fps)
    player_slots = (
        int(max_player_slots)
        if max_player_slots is not None
        else _max_player_slots(tracking_frames)
    )

    rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for row_id, frame in enumerate(tracking_frames):
        row = {"_tracking_row_id": row_id}
        if match_id is not None:
            row["t.match_id"] = str(match_id)
        for column in frame_columns:
            if column in {"ball", "data"}:
                continue
            row[f"t.{column}"] = _serialize_cell(frame.get(column, pd.NA))

        ball = frame.get("ball") or [pd.NA, pd.NA, pd.NA]
        row["t.ball_x"] = ball[0] if len(ball) > 0 else pd.NA
        row["t.ball_y"] = ball[1] if len(ball) > 1 else pd.NA
        row["t.ball_z"] = ball[2] if len(ball) > 2 else pd.NA

        player_rows = _tracking_player_rows(frame)
        row["t.player_count"] = len(player_rows)
        row["t.visible_player_count"] = sum(
            1 for player in player_rows if player["player_visible"] is True
        )
        for slot_idx in range(player_slots):
            prefix = f"t.player_{slot_idx + 1:02d}"
            player = player_rows[slot_idx] if slot_idx < len(player_rows) else {}
            row[f"{prefix}_team_id"] = player.get("team_id", pd.NA)
            row[f"{prefix}_id"] = player.get("player_id", pd.NA)
            row[f"{prefix}_x"] = player.get("player_x", pd.NA)
            row[f"{prefix}_y"] = player.get("player_y", pd.NA)
            row[f"{prefix}_visible"] = player.get("player_visible", pd.NA)

        rows.append(row)
        match_rows.append(
            {
                "_tracking_row_id": row_id,
                "_period_id": _safe_period(frame.get("period")),
                "_tracking_match_clock_seconds": match_seconds.iloc[row_id],
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(match_rows)


def _event_seconds(df_events: pd.DataFrame) -> pd.Series:
    millis = (
        pd.to_numeric(df_events.get("milisec", 0), errors="coerce")
        .fillna(0)
        .astype(float)
    )
    return (
        pd.to_numeric(df_events["min"], errors="coerce").astype(float) * 60
        + pd.to_numeric(df_events["sec"], errors="coerce").astype(float)
        + millis / 1000.0
    )


def filter_step2_events(
    df_events: pd.DataFrame,
    event_type_names: tuple[str, ...] = DEFAULT_STEP2_EVENT_TYPES,
) -> pd.DataFrame:
    """Keep only the selected high-coverage event types for Step 2."""
    if not event_type_names or "event.event_type_name" not in df_events.columns:
        return df_events.copy()
    return df_events[
        df_events["event.event_type_name"].isin(event_type_names)
    ].copy()


def build_event_table(
    df_events: pd.DataFrame,
    event_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build prefixed original event rows plus internal match columns."""
    columns = list(event_columns or df_events.columns)
    detail_rows: list[dict[str, Any]] = []
    for row_id, (_, event) in enumerate(df_events.iterrows()):
        row = {"_event_row_id": row_id}
        for column in columns:
            row[f"e.{column}"] = _serialize_cell(event.get(column, pd.NA))
        detail_rows.append(row)

    details = pd.DataFrame(detail_rows)
    if df_events.empty:
        match_columns = [
            "_event_row_id",
            "_period_id",
            "_event_match_clock_seconds",
        ]
        return details, pd.DataFrame(columns=match_columns)

    match_data = pd.DataFrame(
        {
            "_event_row_id": range(len(df_events)),
            "_period_id": pd.to_numeric(
                df_events["period_id"],
                errors="coerce",
            ).astype(float),
            "_event_match_clock_seconds": _event_seconds(df_events),
        }
    )
    return details, match_data


def match_events_to_frames(
    event_match_data: pd.DataFrame,
    tracking_match_data: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each event to its nearest tracking frame within the same period."""
    if event_match_data.empty or tracking_match_data.empty:
        return pd.DataFrame(
            columns=[
                "_tracking_row_id",
                "_event_row_id",
                "nearest_timestamp_distance_sec",
            ]
        )

    matches: list[pd.DataFrame] = []
    tracking_sorted = tracking_match_data.sort_values(
        ["_period_id", "_tracking_match_clock_seconds", "_tracking_row_id"]
    )
    events_sorted = event_match_data.sort_values(
        ["_period_id", "_event_match_clock_seconds", "_event_row_id"]
    )

    for period_id, period_events in events_sorted.groupby(
        "_period_id",
        dropna=False,
    ):
        period_tracking = tracking_sorted[
            tracking_sorted["_period_id"].eq(period_id)
        ]
        if period_tracking.empty:
            continue
        matched = pd.merge_asof(
            period_events.sort_values("_event_match_clock_seconds"),
            period_tracking.sort_values("_tracking_match_clock_seconds"),
            left_on="_event_match_clock_seconds",
            right_on="_tracking_match_clock_seconds",
            direction="nearest",
        )
        matched = matched[matched["_tracking_row_id"].notna()].copy()
        if matched.empty:
            continue
        matched["nearest_timestamp_distance_sec"] = (
            matched["_event_match_clock_seconds"]
            - matched["_tracking_match_clock_seconds"]
        ).abs()
        matches.append(
            matched[
                [
                    "_tracking_row_id",
                    "_event_row_id",
                    "nearest_timestamp_distance_sec",
                ]
            ]
        )

    if not matches:
        return pd.DataFrame(
            columns=[
                "_tracking_row_id",
                "_event_row_id",
                "nearest_timestamp_distance_sec",
            ]
        )

    candidates = pd.concat(matches, ignore_index=True)
    candidates["_tracking_row_id"] = candidates["_tracking_row_id"].astype(int)
    candidates["_event_row_id"] = candidates["_event_row_id"].astype(int)
    return (
        candidates.sort_values(
            [
                "_tracking_row_id",
                "nearest_timestamp_distance_sec",
                "_event_row_id",
            ]
        )
        .drop_duplicates("_tracking_row_id", keep="first")
        .reset_index(drop=True)
    )


def build_master_join_table(
    tracking_table: pd.DataFrame,
    event_table: pd.DataFrame,
    event_frame_matches: pd.DataFrame,
    event_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Create one output row per tracking frame with one nearest event at most."""
    event_prefixed_columns = [f"e.{column}" for column in event_columns]
    output = tracking_table.merge(
        event_frame_matches,
        on="_tracking_row_id",
        how="left",
    ).merge(
        event_table,
        on="_event_row_id",
        how="left",
    )

    unmatched = output["_event_row_id"].isna()
    for column in event_prefixed_columns:
        if column not in output.columns:
            output[column] = pd.NA
        output[column] = output[column].astype("string")
        output.loc[unmatched, column] = "no event"

    output = output.drop(columns=["_tracking_row_id", "_event_row_id"])
    ordered_columns = [
        column
        for column in output.columns
        if column.startswith("t.")
    ]
    ordered_columns.extend(event_prefixed_columns)
    ordered_columns.append("nearest_timestamp_distance_sec")
    return output[ordered_columns]


def _columns_for_matches(
    data_dir: Path,
    match_ids: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """Collect stable original tracking/event columns across selected matches."""
    tracking_columns: list[str] = []
    event_columns: list[str] = []
    seen_tracking: set[str] = set()
    seen_events: set[str] = set()
    max_player_slots = 0

    for match_id in match_ids:
        tracking_path = data_dir / f"{match_id}_tracking_data.jsonl"
        _, tracking_frames = load_tracking_file(tracking_path)
        max_player_slots = max(max_player_slots, _max_player_slots(tracking_frames))
        for column in _ordered_keys(tracking_frames):
            if column not in seen_tracking:
                tracking_columns.append(column)
                seen_tracking.add(column)

        events_path = data_dir / f"{match_id}_events.json"
        df_events = pd.json_normalize(load_event_file(events_path))
        for column in df_events.columns:
            if column not in seen_events:
                event_columns.append(column)
                seen_events.add(column)

    return tuple(tracking_columns), tuple(event_columns), max_player_slots


def run_step2(config: Step2Config) -> dict[str, Any]:
    """Run the raw tracking-event join for one match."""
    data_dir = config.data_dir.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()

    events_path = data_dir / f"{config.match_id}_events.json"
    tracking_path = data_dir / f"{config.match_id}_tracking_data.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    if not tracking_path.exists():
        raise FileNotFoundError(f"Missing tracking file: {tracking_path}")

    df_events_raw = pd.json_normalize(load_event_file(events_path))
    df_events = filter_step2_events(
        df_events_raw,
        event_type_names=tuple(config.event_type_names),
    )
    tracking_header, tracking_frames = load_tracking_file(tracking_path)
    tracking_columns = config.tracking_columns or tuple(
        _ordered_keys(tracking_frames)
    )
    event_columns = config.event_columns or tuple(df_events_raw.columns)

    tracking_table, tracking_match_data = build_tracking_frame_table(
        tracking_header,
        tracking_frames,
        tracking_columns=tracking_columns,
        match_id=str(config.match_id),
        max_player_slots=config.max_player_slots,
    )
    event_table, event_match_data = build_event_table(
        df_events,
        event_columns=event_columns,
    )
    event_frame_matches = match_events_to_frames(
        event_match_data,
        tracking_match_data,
    )
    master_join_table = build_master_join_table(
        tracking_table,
        event_table,
        event_frame_matches,
        event_columns=event_columns,
    )

    event_rows = int(master_join_table["e.event.event_type_name"].ne(
        "no event"
    ).sum()) if "e.event.event_type_name" in master_join_table.columns else 0
    summary = {
        "config": {
            **asdict(config),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
        },
        "match_id": str(config.match_id),
        "tracking_rows": int(len(tracking_frames)),
        "raw_events": int(len(df_events_raw)),
        "selected_events": int(len(df_events)),
        "event_type_names": "|".join(config.event_type_names),
        "matched_events": int(len(event_frame_matches)),
        "master_join_rows": int(len(master_join_table)),
        "master_join_event_rows": event_rows,
        "median_abs_nearest_timestamp_distance_sec": (
            float(event_frame_matches["nearest_timestamp_distance_sec"].median())
            if len(event_frame_matches)
            else np.nan
        ),
        "p95_abs_nearest_timestamp_distance_sec": (
            float(
                event_frame_matches[
                    "nearest_timestamp_distance_sec"
                ].quantile(0.95)
            )
            if len(event_frame_matches)
            else np.nan
        ),
    }

    outputs: dict[str, str] = {}
    if config.save_match_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / f"master_join_table_{config.match_id}.parquet"
        summary_path = output_dir / f"master_join_summary_{config.match_id}.csv"
        master_join_table.to_parquet(table_path, index=False)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        outputs = {"table": str(table_path), "summary": str(summary_path)}

    return {
        "table": master_join_table,
        "summary": summary,
        "outputs": outputs,
    }


def run_step2_batch(config: Step2BatchConfig) -> dict[str, Any]:
    """Run Step 2 for one match or all matched event/tracking pairs."""
    data_dir = config.data_dir.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    model_base_dir = config.model_base_dir.expanduser().resolve()

    match_ids = (
        available_matches(data_dir)
        if config.all_matches
        else [str(config.match_id)]
    )
    tracking_columns, event_columns, max_player_slots = _columns_for_matches(
        data_dir,
        match_ids,
    )
    if config.max_player_slots is not None:
        max_player_slots = int(config.max_player_slots)

    if config.all_matches:
        table_name = "master_join_table.parquet"
        summary_name = "master_join_summary.csv"
    else:
        table_name = f"master_join_table_{config.match_id}.parquet"
        summary_name = f"master_join_summary_{config.match_id}.csv"

    master_join_table_path = model_base_dir / table_name
    combined_summary_path = model_base_dir / summary_name
    total_rows = 0
    summary_rows: list[dict[str, Any]] = []
    per_match_results = []
    parquet_writer = None
    parquet_schema = None

    model_base_dir.mkdir(parents=True, exist_ok=True)
    if master_join_table_path.exists():
        master_join_table_path.unlink()
    if combined_summary_path.exists():
        combined_summary_path.unlink()

    try:
        for match_id in match_ids:
            step_config = Step2Config(
                data_dir=data_dir,
                output_dir=output_dir,
                match_id=match_id,
                tracking_columns=tracking_columns,
                event_columns=event_columns,
                max_player_slots=max_player_slots,
                event_type_names=tuple(config.event_type_names),
                save_match_outputs=config.save_match_outputs,
            )
            result = run_step2(step_config)
            table = result["table"]
            summary = result["summary"]
            total_rows += len(table)

            parquet_writer, parquet_schema = _append_parquet_chunk(
                master_join_table_path,
                table,
                parquet_writer,
                parquet_schema,
            )
            summary_rows.append(
                {
                    "match_id": match_id,
                    "tracking_rows": summary["tracking_rows"],
                    "raw_events": summary["raw_events"],
                    "selected_events": summary["selected_events"],
                    "event_type_names": summary["event_type_names"],
                    "matched_events": summary["matched_events"],
                    "master_join_rows": summary["master_join_rows"],
                    "master_join_event_rows": summary[
                        "master_join_event_rows"
                    ],
                    "median_abs_nearest_timestamp_distance_sec": summary[
                        "median_abs_nearest_timestamp_distance_sec"
                    ],
                    "p95_abs_nearest_timestamp_distance_sec": summary[
                        "p95_abs_nearest_timestamp_distance_sec"
                    ],
                }
            )
            per_match_results.append(result)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    combined_summary = pd.DataFrame(summary_rows)
    combined_summary.to_csv(combined_summary_path, index=False)
    return {
        "match_ids": match_ids,
        "rows": total_rows,
        "summary": combined_summary,
        "per_match_results": per_match_results,
        "outputs": {
            "table": str(master_join_table_path),
            "summary": str(combined_summary_path),
        },
    }

"""Step 2 master-join pipeline.

This module contains the code that turns raw Driblab event and tracking files
into the frame-level master join table used for modeling. It loads matched
event/tracking files, normalizes tracking coordinates to the 0-100 pitch,
matches selected events to tracking frames by match clock, keeps every tracking
frame, interpolates short ball gaps, creates ball/player/possession features,
and writes `master_join_table.parquet` plus a compact summary.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driblab.config import DEFAULT_MATCH_ID
from driblab.etl.pipeline import (
    load_event_file,
    load_tracking_file,
    normalize_tracking_x_series,
    normalize_tracking_y_series,
)


COORD_X_COLS = ["x", "x_start", "x_end"]
COORD_Y_COLS = ["y", "y_start", "y_end"]
SHOT_TYPES = ["GOAL", "SAVED SHOT", "MISSED SHOT", "SHOT ON POST"]
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
    max_ball_gap_frames: int = 10
    possession_distance_m: float = 2.0
    possession_max_ball_speed_mps: float = 12.0
    max_sync_tolerance_sec: float = 0.5
    direction_score_tolerance_sec: float = 0.30
    max_speed_dt_sec: float = 0.50
    event_type_names: tuple[str, ...] = DEFAULT_STEP2_EVENT_TYPES
    save_player_features: bool = False
    save_match_outputs: bool = False


@dataclass
class Step2BatchConfig:
    """Configuration for building Step 2 outputs for one or many matches."""

    data_dir: Path
    output_dir: Path
    model_base_dir: Path
    match_id: str = DEFAULT_MATCH_ID
    all_matches: bool = False
    max_ball_gap_frames: int = 10
    possession_distance_m: float = 2.0
    possession_max_ball_speed_mps: float = 12.0
    max_sync_tolerance_sec: float = 0.5
    direction_score_tolerance_sec: float = 0.30
    max_speed_dt_sec: float = 0.50
    event_type_names: tuple[str, ...] = DEFAULT_STEP2_EVENT_TYPES
    save_player_features: bool = False
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


def _team_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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


def _player_lookup(header: dict[str, Any]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for team_id, players in header.get("players_data", {}).items():
        for player_id, player in players.items():
            lookup[(str(team_id), str(player_id))] = player.get("name", "")
    return lookup


def _team_lookup(header: dict[str, Any]) -> dict[str, str]:
    teams = header.get("teams_data", {})
    return {
        str(side_data["id"]): side_data.get("name", side)
        for side, side_data in teams.items()
    }


def build_tracking_frame_table(
    tracking_header: dict[str, Any],
    tracking_frames: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one row per tracking frame plus video-clock diagnostics."""
    rows: list[dict[str, Any]] = []
    fps = float(tracking_header.get("FPS", 10))

    for frame in tracking_frames:
        ball = frame.get("ball") or [None, None, None]
        clock = frame.get("match_clock") or [np.nan, np.nan]
        video_timestamp = frame.get("Videotimestamp")
        rows.append(
            {
                "frame_id": int(frame["frame"]),
                "period_id": int(frame["period"]),
                "match_clock_min": clock[0],
                "match_clock_sec": clock[1],
                "video_timestamp": float(video_timestamp)
                if video_timestamp is not None
                else np.nan,
                "cam_present": frame.get("cam") is not None,
                "ball_x_raw": ball[0],
                "ball_y_raw": ball[1],
                "ball_z_raw": ball[2],
            }
        )

    frame_df = (
        pd.DataFrame(rows)
        .sort_values(["period_id", "frame_id"])
        .reset_index(drop=True)
    )
    tracking_clock_whole_second = (
        pd.to_numeric(
            frame_df["match_clock_min"],
            errors="coerce",
        ).astype(float)
        * 60
        + pd.to_numeric(
            frame_df["match_clock_sec"],
            errors="coerce",
        ).astype(float)
    )
    clock_frame_index = frame_df.groupby(
        [frame_df["period_id"], tracking_clock_whole_second]
    ).cumcount()
    frame_df["tracking_match_clock_seconds"] = (
        tracking_clock_whole_second + clock_frame_index / fps
    )

    diagnostic_rows: list[dict[str, Any]] = []
    for period_id, period_frames in frame_df.groupby("period_id"):
        valid = period_frames.dropna(
            subset=["tracking_match_clock_seconds", "video_timestamp"]
        )
        if len(valid) >= 2:
            slope, intercept = np.polyfit(
                valid["tracking_match_clock_seconds"],
                valid["video_timestamp"],
                deg=1,
            )
        else:
            slope, intercept = 1.0, 0.0

        predicted = (
            valid["tracking_match_clock_seconds"] * slope + intercept
            if len(valid)
            else pd.Series(dtype=float)
        )
        residual = (
            valid["video_timestamp"] - predicted
            if len(valid)
            else pd.Series(dtype=float)
        )
        diagnostic_rows.append(
            {
                "period_id": int(period_id),
                "slope": float(slope),
                "intercept": float(intercept),
                "fps": fps,
                "frames": int(len(period_frames)),
                "median_abs_residual_sec": float(residual.abs().median())
                if len(residual)
                else np.nan,
            }
        )

    video_clock_diagnostics = pd.DataFrame(diagnostic_rows)
    return frame_df, video_clock_diagnostics


def add_event_match_clock_seconds(
    df_events: pd.DataFrame,
) -> pd.DataFrame:
    """Add event time in match-clock seconds for joining to tracking."""
    events = df_events.copy()
    events["event_match_clock_seconds"] = _event_seconds(events)
    return events


def filter_step2_events(
    df_events: pd.DataFrame,
    event_type_names: tuple[str, ...],
) -> pd.DataFrame:
    """Keep only event types used by the Step 2 modelling target."""
    if not event_type_names:
        return df_events.copy()
    allowed = {event_type.upper() for event_type in event_type_names}
    event_names = df_events["event.event_type_name"].astype(str).str.upper()
    return df_events[event_names.isin(allowed)].copy()


def match_events_to_frames(
    df_events: pd.DataFrame,
    frame_df: pd.DataFrame,
    tolerance_sec: float,
) -> pd.DataFrame:
    """Attach each event to the closest tracking frame in-period."""
    event_parts: list[pd.DataFrame] = []
    frames = frame_df[
        [
            "period_id",
            "frame_id",
            "match_clock_min",
            "match_clock_sec",
            "tracking_match_clock_seconds",
            "cam_present",
            "ball_x_raw",
            "ball_y_raw",
            "ball_z_raw",
        ]
    ].sort_values(["period_id", "tracking_match_clock_seconds"])

    for period_id, period_events in df_events.groupby(
        "period_id",
        dropna=False,
    ):
        period_frames = frames[frames["period_id"] == period_id]
        if period_frames.empty:
            aligned = period_events.copy()
            aligned["frame_id"] = np.nan
            aligned["matched_tracking_match_clock_seconds"] = np.nan
            aligned["match_clock_join_error_sec"] = np.nan
            aligned["matched_cam_present"] = False
            event_parts.append(aligned)
            continue

        aligned = pd.merge_asof(
            period_events.sort_values("event_match_clock_seconds"),
            period_frames.drop(columns=["period_id"]).sort_values(
                "tracking_match_clock_seconds"
            ),
            left_on="event_match_clock_seconds",
            right_on="tracking_match_clock_seconds",
            direction="nearest",
            tolerance=tolerance_sec,
        )
        aligned = aligned.rename(
            columns={
                "tracking_match_clock_seconds": (
                    "matched_tracking_match_clock_seconds"
                ),
                "match_clock_min": "matched_match_clock_min",
                "match_clock_sec": "matched_match_clock_sec",
                "cam_present": "matched_cam_present",
                "ball_x_raw": "matched_ball_x_raw",
                "ball_y_raw": "matched_ball_y_raw",
                "ball_z_raw": "matched_ball_z_raw",
            }
        )
        aligned["match_clock_join_error_sec"] = (
            aligned["event_match_clock_seconds"]
            - aligned["matched_tracking_match_clock_seconds"]
        )
        event_parts.append(aligned)

    return (
        pd.concat(event_parts, ignore_index=True)
        .sort_values(["period_id", "event_match_clock_seconds", "event.id"])
        .reset_index(drop=True)
    )


def _candidate_direction_map(
    header: dict[str, Any],
    home_first_half_direction: int,
) -> dict[tuple[str, int], int]:
    teams = header.get("teams_data", {})
    home_id = str(teams["home"]["id"])
    away_id = str(teams["away"]["id"])

    return {
        (home_id, 1): home_first_half_direction,
        (away_id, 1): -home_first_half_direction,
        (home_id, 2): -home_first_half_direction,
        (away_id, 2): home_first_half_direction,
    }


def infer_attack_directions(
    df_events_aligned: pd.DataFrame,
    tracking_header: dict[str, Any],
    tolerance_sec: float,
) -> tuple[dict[tuple[str, int], int], pd.DataFrame]:
    """Infer whether each team attacks +x or -x in each period.

    The event provider normalizes x so the attacking goal is always x=100.
    Tracking x starts in meters, so we normalize it to the same 0-100 scale
    before scoring the two legal football orientations: home +x in period 1,
    or home -x in period 1. Teams then swap at half-time.
    """
    scored_events = df_events_aligned.copy()
    scored_events["team_id_str"] = scored_events["team.team_id"].map(_team_id)
    scored_events["matched_ball_x_norm"] = normalize_tracking_x_series(
        scored_events["matched_ball_x_raw"]
    )
    scored_events = scored_events[
        (scored_events["team_id_str"] != "0")
        & scored_events["x"].notna()
        & scored_events["matched_ball_x_norm"].notna()
        & scored_events["match_clock_join_error_sec"].abs().le(tolerance_sec)
    ]

    score_rows: list[dict[str, Any]] = []
    for home_direction in (1, -1):
        directions = _candidate_direction_map(tracking_header, home_direction)
        distances: list[float] = []

        for _, row in scored_events.iterrows():
            direction = directions.get(
                (row["team_id_str"], int(row["period_id"])))
            if direction is None:
                continue
            event_x = float(row["x"])
            tracking_x_attacking = (
                float(row["matched_ball_x_norm"])
                if direction == 1
                else 100.0 - float(row["matched_ball_x_norm"])
            )
            distances.append(abs(event_x - tracking_x_attacking))

        score_rows.append(
            {
                "candidate": "home_attacks_plus_x_p1"
                if home_direction == 1
                else "home_attacks_minus_x_p1",
                "home_first_half_direction": home_direction,
                "samples": len(distances),
                "median_x_error_norm": float(np.median(distances))
                if distances
                else np.inf,
                "mean_x_error_norm": float(np.mean(distances))
                if distances
                else np.inf,
            }
        )

    score_df = pd.DataFrame(score_rows)
    if score_df["samples"].max() == 0:
        best_home_direction = 1
    else:
        best_home_direction = int(
            score_df.sort_values(["median_x_error_norm", "mean_x_error_norm"])
            .iloc[0]["home_first_half_direction"]
        )

    directions = _candidate_direction_map(tracking_header, best_home_direction)
    direction_rows = []
    team_names = _team_lookup(tracking_header)
    for (team_id, period_id), direction in directions.items():
        direction_rows.append(
            {
                "team_id": team_id,
                "team_name": team_names.get(team_id, ""),
                "period_id": period_id,
                "attack_direction": "+x" if direction == 1 else "-x",
                "attack_direction_sign": direction,
            }
        )

    direction_df = pd.DataFrame(direction_rows).sort_values([
        "period_id", "team_id"])
    direction_df["chosen_orientation"] = (
        "home_attacks_plus_x_p1"
        if best_home_direction == 1
        else "home_attacks_minus_x_p1"
    )

    score_df["chosen"] = score_df["home_first_half_direction"].eq(
        best_home_direction)
    return directions, direction_df.merge(
        score_df[score_df["chosen"]], how="cross")


def add_event_attack_directions(
    df_events_aligned: pd.DataFrame,
    directions: dict[tuple[str, int], int],
) -> pd.DataFrame:
    """Add inferred attacking direction metadata to matched event rows."""
    events = df_events_aligned.copy()
    events["team_id_str"] = events["team.team_id"].map(_team_id)
    events["attack_direction_sign"] = events.apply(lambda row: directions.get(
        (row["team_id_str"], int(row["period_id"])), np.nan), axis=1, )
    events["attack_direction"] = events["attack_direction_sign"].map(
        {1: "+x", -1: "-x"}
    )
    return events


def build_live_frame_features(
    frame_df: pd.DataFrame,
    max_ball_gap_frames: int,
    max_speed_dt_sec: float,
) -> pd.DataFrame:
    frame_features = (
        frame_df.sort_values(["period_id", "video_timestamp", "frame_id"])
        .reset_index(drop=True)
    )

    for axis in ("x", "y", "z"):
        raw_col = f"ball_{axis}_raw"
        clean_col = f"ball_{axis}"
        frame_features[clean_col] = frame_features.groupby("period_id")[
            raw_col
        ].transform(
            lambda series: series.interpolate(
                method="linear",
                limit=max_ball_gap_frames,
                limit_area="inside",
            )
        )

    frame_features["ball_present_raw"] = frame_features[
        ["ball_x_raw", "ball_y_raw", "ball_z_raw"]
    ].notna().all(axis=1)
    frame_features["ball_interpolated"] = (
        ~frame_features["ball_present_raw"]
        & frame_features[["ball_x", "ball_y", "ball_z"]].notna().all(axis=1)
    )

    frame_features["dt_sec"] = frame_features.groupby(
        "period_id")["video_timestamp"].diff()
    for axis in ("x", "y", "z"):
        frame_features[f"ball_d{axis}"] = frame_features.groupby("period_id")[
            f"ball_{axis}"
        ].diff()

    previous_cam_present = frame_features.groupby("period_id")[
        "cam_present"
    ].shift(1).fillna(False)
    valid_dt = (
        frame_features["dt_sec"].gt(0)
        & frame_features["dt_sec"].le(max_speed_dt_sec)
        & frame_features["cam_present"]
        & previous_cam_present
    )
    frame_features["ball_speed_xy_mps"] = np.where(
        valid_dt,
        np.sqrt(frame_features["ball_dx"] ** 2 + frame_features["ball_dy"] ** 2)
        / frame_features["dt_sec"],
        np.nan,
    )
    frame_features["ball_speed_mps"] = np.where(
        valid_dt,
        np.sqrt(
            frame_features["ball_dx"] ** 2
            + frame_features["ball_dy"] ** 2
            + frame_features["ball_dz"] ** 2
        )
        / frame_features["dt_sec"],
        np.nan,
    )
    frame_features["ball_acceleration_mps2"] = (
        frame_features.groupby("period_id")["ball_speed_mps"].diff()
        / frame_features["dt_sec"]
    )
    frame_features.loc[~valid_dt, "ball_acceleration_mps2"] = np.nan

    return frame_features.drop(columns=["ball_dx", "ball_dy", "ball_dz"])


def build_player_features(
    tracking_header: dict[str, Any],
    tracking_frames: list[dict[str, Any]],
    live_frame_features: pd.DataFrame,
    max_speed_dt_sec: float,
) -> pd.DataFrame:
    live_ids = set(live_frame_features["frame_id"])
    team_names = _team_lookup(tracking_header)
    player_names = _player_lookup(tracking_header)

    rows: list[dict[str, Any]] = []
    for frame in tracking_frames:
        frame_id = int(frame["frame"])
        if frame_id not in live_ids:
            continue
        video_timestamp = frame.get("Videotimestamp")
        video_timestamp = (
            float(video_timestamp) if video_timestamp is not None else np.nan
        )
        period_id = int(frame["period"])

        for team_id, players in (frame.get("data") or {}).items():
            team_id_str = str(team_id)
            for player in players:
                player_id = str(player.get("id"))
                rows.append({"frame_id": frame_id,
                             "period_id": period_id,
                             "video_timestamp": video_timestamp,
                             "team_id": team_id_str,
                             "team_name": team_names.get(team_id_str,
                                                         ""),
                             "player_id": player_id,
                             "player_name": player_names.get((team_id_str,
                                                              player_id),
                                                             ""),
                             "player_x": player.get("x"),
                             "player_y": player.get("y"),
                             "player_visible": bool(player.get("vis")),
                             })

    players = pd.DataFrame(rows).sort_values(
        ["team_id", "player_id", "period_id", "video_timestamp"]
    )
    players["dt_sec"] = players.groupby(["team_id", "player_id", "period_id"])[
        "video_timestamp"
    ].diff()
    players["dx"] = players.groupby(["team_id", "player_id", "period_id"])[
        "player_x"
    ].diff()
    players["dy"] = players.groupby(["team_id", "player_id", "period_id"])[
        "player_y"
    ].diff()
    valid_dt = players["dt_sec"].gt(0) & players["dt_sec"].le(max_speed_dt_sec)
    players["player_speed_mps"] = np.where(
        valid_dt,
        np.sqrt(players["dx"] ** 2 + players["dy"] ** 2) / players["dt_sec"],
        np.nan,
    )
    players["player_speed_reliable"] = (
        valid_dt
        & players["player_visible"]
        & players.groupby(["team_id", "player_id", "period_id"])[
            "player_visible"
        ].shift(1).fillna(False)
    )

    ball_cols = live_frame_features[
        ["frame_id", "ball_x", "ball_y", "ball_z", "ball_speed_mps"]
    ]
    players = players.merge(ball_cols, on="frame_id", how="left")
    players["distance_to_ball_m"] = np.sqrt(
        (players["player_x"] - players["ball_x"]) ** 2
        + (players["player_y"] - players["ball_y"]) ** 2
    )

    return players.drop(columns=["dx", "dy"])


def add_possession_features(
    live_frame_features: pd.DataFrame,
    player_features: pd.DataFrame,
    possession_distance_m: float,
    possession_max_ball_speed_mps: float,
) -> pd.DataFrame:
    valid_distances = player_features[
        player_features["distance_to_ball_m"].notna()
    ]
    if valid_distances.empty:
        possession = pd.DataFrame(columns=["frame_id"])
    else:
        nearest_idx = valid_distances.groupby("frame_id")[
            "distance_to_ball_m"
        ].idxmin()
        possession = valid_distances.loc[
            nearest_idx,
            [
                "frame_id",
                "team_id",
                "team_name",
                "player_id",
                "player_name",
                "player_visible",
                "distance_to_ball_m",
            ],
        ].rename(
            columns={
                "team_id": "nearest_team_id",
                "team_name": "nearest_team_name",
                "player_id": "nearest_player_id",
                "player_name": "nearest_player_name",
                "player_visible": "nearest_player_visible",
                "distance_to_ball_m": "nearest_player_distance_to_ball_m",
            }
        )

    frames = live_frame_features.merge(possession, on="frame_id", how="left")
    frames["has_possession"] = (
        frames["nearest_player_distance_to_ball_m"].le(possession_distance_m)
        & frames["ball_speed_mps"].le(possession_max_ball_speed_mps)
    )
    frames["possessing_team_id"] = np.where(
        frames["has_possession"],
        frames["nearest_team_id"],
        pd.NA,
    )
    frames["possessing_team_name"] = np.where(
        frames["has_possession"],
        frames["nearest_team_name"],
        pd.NA,
    )
    frames["possessing_player_id"] = np.where(
        frames["has_possession"],
        frames["nearest_player_id"],
        pd.NA,
    )
    frames["possessing_player_name"] = np.where(
        frames["has_possession"],
        frames["nearest_player_name"],
        pd.NA,
    )

    return frames


def _join_values(values: pd.Series) -> str:
    clean_values = values.dropna().astype(str)
    return "|".join(clean_values)


def _event_model_column_name(column: str) -> str:
    """Create clear event-side column names for the frame-level model table."""
    if column.startswith("event."):
        clean = column.split(".", 1)[1].replace(".", "_")
        if clean.startswith("event_"):
            clean = clean.removeprefix("event_")
    elif column.startswith("event_"):
        return column
    elif column.startswith("team."):
        clean = column.split(".", 1)[1].replace(".", "_")
        if not clean.startswith("team_"):
            clean = f"team_{clean}"
    elif column.startswith("player."):
        clean = column.split(".", 1)[1].replace(".", "_")
        if not clean.startswith("player_"):
            clean = f"player_{clean}"
    else:
        clean = column.replace(".", "_")
    return f"event_{clean}"


def _serialize_model_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def build_player_frame_aggregates(
        player_features: pd.DataFrame) -> pd.DataFrame:
    """Collapse player-level rows into non-speed frame-level aggregates."""
    return (
        player_features.groupby("frame_id")
        .agg(
            player_count=("player_id", "size"),
            visible_player_count=("player_visible", "sum"),
            min_distance_to_ball_m=("distance_to_ball_m", "min"),
            mean_distance_to_ball_m=("distance_to_ball_m", "mean"),
        )
        .reset_index()
    )


def build_event_frame_labels(
    df_events_aligned: pd.DataFrame,
    raw_event_columns: list[str],
) -> pd.DataFrame:
    """Collapse aligned event rows into frame-level labels for modelling."""
    events = df_events_aligned[df_events_aligned["frame_id"].notna()].copy()
    output_columns = [
        "frame_id",
        "event_count_at_frame",
        "event_label",
        "event_type_names_at_frame",
        "event_type_ids_at_frame",
        "event_ids_at_frame",
        "event_team_ids_at_frame",
        "event_player_ids_at_frame",
        "first_event_match_clock_seconds",
    ]
    if events.empty:
        return pd.DataFrame(columns=output_columns)

    events["frame_id"] = events["frame_id"].astype(int)
    events = events.sort_values(
        ["frame_id", "event_match_clock_seconds", "event.id"],
        na_position="last",
    )

    grouped = (
        events.groupby("frame_id")
        .agg(
            event_count_at_frame=("event.event_type_name", "size"),
            event_type_names_at_frame=("event.event_type_name", _join_values),
            event_type_ids_at_frame=("event.event_type_id", _join_values),
            event_ids_at_frame=("event.id", _join_values),
            event_team_ids_at_frame=("team.team_id", _join_values),
            event_player_ids_at_frame=("player.player_id", _join_values),
            first_event_match_clock_seconds=(
                "event_match_clock_seconds",
                "min",
            ),
        )
        .reset_index()
    )
    grouped["event_label"] = grouped["event_type_names_at_frame"]

    event_detail_cols = [
        col for col in raw_event_columns if col in events.columns
    ]
    event_detail_cols.extend(
        [
            "event_match_clock_seconds",
            "matched_tracking_match_clock_seconds",
            "match_clock_join_error_sec",
            "attack_direction",
            "attack_direction_sign",
        ]
    )
    event_detail_cols = list(dict.fromkeys(event_detail_cols))

    detail = events.drop_duplicates("frame_id", keep="first")[
        ["frame_id", *event_detail_cols]
    ].copy()
    for col in event_detail_cols:
        detail[col] = detail[col].map(_serialize_model_cell)
    detail = detail.rename(
        columns={
            col: _event_model_column_name(col) for col in event_detail_cols})

    return grouped.merge(detail, on="frame_id", how="left")


def fill_no_event_labels(model_base: pd.DataFrame) -> pd.DataFrame:
    """Mark live tracking frames that did not match to any event."""
    no_event_cols = [
        "event_label",
        "event_type_names_at_frame",
        "event_type_name",
    ]
    for col in no_event_cols:
        if col in model_base.columns:
            model_base[col] = model_base[col].fillna("no event")

    for col in [
        "event_type_ids_at_frame",
        "event_ids_at_frame",
        "event_team_ids_at_frame",
        "event_player_ids_at_frame",
    ]:
        if col in model_base.columns:
            model_base[col] = model_base[col].fillna("")

    return model_base


def normalize_master_join_coordinates(
        master_join_table: pd.DataFrame) -> pd.DataFrame:
    """Normalize saved x/y tracking coordinates to the 0-100 field scale."""
    normalized = master_join_table.copy()

    for col in ["ball_x_raw", "ball_x"]:
        if col in normalized.columns:
            normalized[col] = normalize_tracking_x_series(normalized[col])
    for col in ["ball_y_raw", "ball_y"]:
        if col in normalized.columns:
            normalized[col] = normalize_tracking_y_series(normalized[col])

    normalized = normalized.rename(
        columns={
            "ball_z_raw": "ball_z_m_raw",
            "ball_z": "ball_z_m",
        }
    )
    return normalized


def normalize_player_feature_coordinates(
    player_features: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize saved player-level x/y coordinates to 0-100."""
    normalized = player_features.copy()

    for col in ["player_x", "ball_x"]:
        if col in normalized.columns:
            normalized[col] = normalize_tracking_x_series(normalized[col])
    for col in ["player_y", "ball_y"]:
        if col in normalized.columns:
            normalized[col] = normalize_tracking_y_series(normalized[col])

    if "ball_z" in normalized.columns:
        normalized = normalized.rename(columns={"ball_z": "ball_z_m"})
    return normalized


def _valid_model_team_id(value: Any) -> Any:
    team_id = _team_id(value)
    if team_id in {"", "0", "<NA>", "nan", "None", "no event"}:
        return pd.NA
    return team_id


def add_tracking_attacking_coordinates(
    master_join_table: pd.DataFrame,
    directions: dict[tuple[str, int], int],
) -> pd.DataFrame:
    """Add tracking-derived x/y columns in the attacking team's orientation."""
    table = master_join_table.copy()

    event_team_id = (
        table["event_team_id"].map(_valid_model_team_id)
        if "event_team_id" in table.columns
        else pd.Series(pd.NA, index=table.index)
    )
    possession_team_id = (
        table["possessing_team_id"].map(_valid_model_team_id)
        if "possessing_team_id" in table.columns
        else pd.Series(pd.NA, index=table.index)
    )
    nearest_team_id = (
        table["nearest_team_id"].map(_valid_model_team_id)
        if "nearest_team_id" in table.columns
        else pd.Series(pd.NA, index=table.index)
    )

    reference_team_id = event_team_id.combine_first(
        possession_team_id
    ).combine_first(nearest_team_id)
    reference_source = pd.Series(pd.NA, index=table.index, dtype="object")
    reference_source.loc[event_team_id.notna()] = "event_team"
    reference_source.loc[event_team_id.isna() & possession_team_id.notna()] = (
        "possession_team"
    )
    nearest_source_mask = (
        event_team_id.isna()
        & possession_team_id.isna()
        & nearest_team_id.notna()
    )
    reference_source.loc[nearest_source_mask] = "nearest_team"

    attack_direction_sign = []
    for team_id, period_id in zip(reference_team_id, table["period_id"]):
        if pd.isna(team_id) or pd.isna(period_id):
            attack_direction_sign.append(np.nan)
            continue
        attack_direction_sign.append(
            directions.get((str(team_id), int(period_id)), np.nan)
        )

    sign = pd.Series(attack_direction_sign, index=table.index, dtype="float")
    table["tracking_reference_team_id"] = reference_team_id
    table["tracking_reference_source"] = reference_source
    table["tracking_attack_direction_sign"] = sign
    table["tracking_attack_direction"] = sign.map({1.0: "+x", -1.0: "-x"})

    for col in ["ball_x_raw", "ball_x"]:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce")
        attacking_values = values.where(sign.notna())
        attacking_values = attacking_values.mask(sign.eq(-1), 100.0 - values)
        table[f"{col}_attacking"] = attacking_values

    for col in ["ball_y_raw", "ball_y"]:
        if col not in table.columns:
            continue
        values = pd.to_numeric(table[col], errors="coerce")
        table[f"{col}_attacking"] = values.where(sign.notna())

    return table


def build_model_base_frame_table(
    match_id: str,
    frame_features: pd.DataFrame,
    player_features: pd.DataFrame,
    df_events_aligned: pd.DataFrame,
    raw_event_columns: list[str],
    directions: dict[tuple[str, int], int],
) -> pd.DataFrame:
    """Build the single-match frame-level table used for model training."""
    frames = frame_features.copy()
    frames.insert(0, "match_id", str(match_id))

    player_aggregates = build_player_frame_aggregates(player_features)
    event_labels = build_event_frame_labels(
        df_events_aligned,
        raw_event_columns,
    )

    model_base = frames.merge(player_aggregates, on="frame_id", how="left")
    model_base = model_base.merge(event_labels, on="frame_id", how="left")
    model_base["event_count_at_frame"] = (
        model_base["event_count_at_frame"].fillna(0).astype(int)
    )
    model_base["is_event_frame"] = model_base["event_count_at_frame"].gt(0)
    model_base = fill_no_event_labels(model_base)
    model_base = normalize_master_join_coordinates(model_base)
    model_base = add_tracking_attacking_coordinates(model_base, directions)

    sort_cols = ["match_id", "period_id", "video_timestamp", "frame_id"]
    return model_base.sort_values(sort_cols).reset_index(drop=True)


def build_summary(
    config: Step2Config,
    df_events_aligned: pd.DataFrame,
    live_frame_features: pd.DataFrame,
    player_features: pd.DataFrame,
    frame_features: pd.DataFrame,
    model_base_frame_table: pd.DataFrame,
    direction_df: pd.DataFrame,
    video_clock_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    join_errors = (
        df_events_aligned["match_clock_join_error_sec"]
        .abs()
        .dropna()
    )
    return {
        "config": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
        },
        "rows": {
            "aligned_events": int(len(df_events_aligned)),
            "live_frames": int(len(live_frame_features)),
            "player_features": int(len(player_features)),
            "frames_with_possession": int(
                frame_features["has_possession"].sum()
            ),
            "model_base_frame_rows": int(len(model_base_frame_table)),
            "model_base_event_rows": int(
                model_base_frame_table["is_event_frame"].sum()
            ),
        },
        "match_clock_join": {
            "matched_events": int(df_events_aligned["frame_id"].notna().sum()),
            "median_abs_error_sec": float(join_errors.median())
            if len(join_errors)
            else math.nan,
            "p95_abs_error_sec": float(join_errors.quantile(0.95))
            if len(join_errors)
            else math.nan,
        },
        "ball": {
            "raw_present_live_frames": int(
                live_frame_features["ball_present_raw"].sum()
            ),
            "interpolated_live_frames": int(
                live_frame_features["ball_interpolated"].sum()
            ),
            "max_gap_frames": int(config.max_ball_gap_frames),
        },
        "possession": {
            "distance_threshold_m": config.possession_distance_m,
            "max_ball_speed_mps": config.possession_max_ball_speed_mps,
            "share_live_frames_with_possession": round(
                float(frame_features["has_possession"].mean()), 4
            ),
        },
        "attack_directions": direction_df.to_dict(orient="records"),
        "video_clock_diagnostics": video_clock_diagnostics.to_dict(
            orient="records"
        ),
    }


def save_outputs(
    config: Step2Config,
    df_events_aligned: pd.DataFrame,
    live_frame_features: pd.DataFrame,
    player_features: pd.DataFrame,
    frame_features: pd.DataFrame,
    model_base_frame_table: pd.DataFrame,
    direction_df: pd.DataFrame,
    video_clock_diagnostics: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    output_dir = config.output_dir / str(config.match_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "event_frame_alignment": output_dir / "event_frame_alignment.csv",
        "live_frame_features": output_dir / "live_frame_features.csv",
        "frame_possession_features": (
            output_dir / "frame_possession_features.csv"
        ),
        "master_join_table": output_dir / "master_join_table.parquet",
        "attack_directions": output_dir / "attack_directions.csv",
        "video_clock_diagnostics": (
            output_dir / "video_clock_diagnostics.csv"
        ),
        "summary": output_dir / "summary.json",
    }

    live_frame_features_output = normalize_master_join_coordinates(
        live_frame_features
    )
    frame_features_output = normalize_master_join_coordinates(frame_features)

    df_events_aligned.to_csv(outputs["event_frame_alignment"], index=False)
    live_frame_features_output.to_csv(
        outputs["live_frame_features"],
        index=False,
    )
    frame_features_output.to_csv(
        outputs["frame_possession_features"],
        index=False,
    )
    model_base_frame_table.to_parquet(
        outputs["master_join_table"], index=False)
    direction_df.to_csv(outputs["attack_directions"], index=False)
    video_clock_diagnostics.to_csv(
        outputs["video_clock_diagnostics"], index=False)

    if config.save_player_features:
        outputs["player_features"] = output_dir / "player_features.csv"
        player_features_output = normalize_player_feature_coordinates(
            player_features
        )
        player_features_output.to_csv(outputs["player_features"], index=False)

    outputs["summary"].write_text(json.dumps(summary, indent=2))
    return {name: str(path) for name, path in outputs.items()}


def run_step2(config: Step2Config) -> dict[str, Any]:
    config.data_dir = config.data_dir.expanduser().resolve()
    config.output_dir = config.output_dir.expanduser().resolve()

    events_path = config.data_dir / f"{config.match_id}_events.json"
    tracking_path = config.data_dir / f"{config.match_id}_tracking_data.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events file: {events_path}")
    if not tracking_path.exists():
        raise FileNotFoundError(f"Missing tracking file: {tracking_path}")

    df_events_raw = pd.json_normalize(load_event_file(events_path))
    df_events = filter_step2_events(
        df_events_raw,
        tuple(config.event_type_names),
    )
    tracking_header, tracking_frames = load_tracking_file(tracking_path)

    frame_df, video_clock_diagnostics = build_tracking_frame_table(
        tracking_header,
        tracking_frames,
    )
    df_events_timed = add_event_match_clock_seconds(df_events)
    df_events_matched = match_events_to_frames(
        df_events_timed,
        frame_df,
        tolerance_sec=config.max_sync_tolerance_sec,
    )
    directions, direction_df = infer_attack_directions(
        df_events_matched,
        tracking_header,
        tolerance_sec=config.direction_score_tolerance_sec,
    )
    df_events_aligned = add_event_attack_directions(
        df_events_matched, directions)

    live_frame_features = build_live_frame_features(
        frame_df,
        max_ball_gap_frames=config.max_ball_gap_frames,
        max_speed_dt_sec=config.max_speed_dt_sec,
    )
    player_features = build_player_features(
        tracking_header,
        tracking_frames,
        live_frame_features,
        max_speed_dt_sec=config.max_speed_dt_sec,
    )
    frame_features = add_possession_features(
        live_frame_features,
        player_features,
        possession_distance_m=config.possession_distance_m,
        possession_max_ball_speed_mps=config.possession_max_ball_speed_mps,
    )
    model_base_frame_table = build_model_base_frame_table(
        str(config.match_id),
        frame_features,
        player_features,
        df_events_aligned,
        raw_event_columns=list(df_events.columns),
        directions=directions,
    )

    summary = build_summary(
        config,
        df_events_aligned,
        live_frame_features,
        player_features,
        frame_features,
        model_base_frame_table,
        direction_df,
        video_clock_diagnostics,
    )
    outputs: dict[str, str] = {}
    if config.save_match_outputs:
        outputs = save_outputs(
            config,
            df_events_aligned,
            live_frame_features,
            player_features,
            frame_features,
            model_base_frame_table,
            direction_df,
            video_clock_diagnostics,
            summary,
        )

    return {
        "events_aligned": df_events_aligned,
        "live_frame_features": live_frame_features,
        "player_features": player_features,
        "frame_features": frame_features,
        "model_base_frame_table": model_base_frame_table,
        "attack_directions": direction_df,
        "video_clock_diagnostics": video_clock_diagnostics,
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
    if config.all_matches:
        table_name = "master_join_table.parquet"
        summary_name = "master_join_summary.csv"
    else:
        table_name = f"master_join_table_{config.match_id}.parquet"
        summary_name = f"master_join_summary_{config.match_id}.csv"

    master_join_table_path = model_base_dir / table_name
    combined_summary_path = model_base_dir / summary_name
    total_model_rows = 0
    summary_rows = []
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
                max_ball_gap_frames=config.max_ball_gap_frames,
                possession_distance_m=config.possession_distance_m,
                possession_max_ball_speed_mps=(
                    config.possession_max_ball_speed_mps
                ),
                max_sync_tolerance_sec=config.max_sync_tolerance_sec,
                direction_score_tolerance_sec=(
                    config.direction_score_tolerance_sec
                ),
                max_speed_dt_sec=config.max_speed_dt_sec,
                event_type_names=tuple(config.event_type_names),
                save_player_features=config.save_player_features,
                save_match_outputs=config.save_match_outputs,
            )
            result = run_step2(step_config)
            summary = result["summary"]
            model_base = result["model_base_frame_table"]
            total_model_rows += len(model_base)

            parquet_writer, parquet_schema = _append_parquet_chunk(
                master_join_table_path,
                model_base,
                parquet_writer,
                parquet_schema,
            )
            summary_rows.append(
                {
                    "match_id": match_id,
                    "master_join_table_rows": summary["rows"][
                        "model_base_frame_rows"
                    ],
                    "master_join_event_rows": summary["rows"][
                        "model_base_event_rows"
                    ],
                    "aligned_events": summary["rows"]["aligned_events"],
                    "matched_events": summary["match_clock_join"][
                        "matched_events"
                    ],
                    "median_abs_match_clock_join_error_sec": summary[
                        "match_clock_join"
                    ]["median_abs_error_sec"],
                    "p95_abs_match_clock_join_error_sec": summary[
                        "match_clock_join"
                    ]["p95_abs_error_sec"],
                    "live_frames": summary["rows"]["live_frames"],
                    "frames_with_possession": summary["rows"][
                        "frames_with_possession"
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
        "rows": total_model_rows,
        "summary": combined_summary,
        "per_match_results": per_match_results,
        "outputs": {
            "table": str(master_join_table_path),
            "summary": str(combined_summary_path),
        },
    }

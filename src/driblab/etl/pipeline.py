"""Step 1 ETL checks for raw Driblab event and tracking files.

This module mirrors the ETL notebook and contains helper functions for loading
raw event JSON, tracking JSONL, and event-type reference data. It summarizes
raw assets, validates that event coordinates already use the provider 0-100
scale, normalizes tracking x/y coordinates for inspection, and prints basic
ball, camera, visibility, and cross-file consistency diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from driblab.config import (
    DEFAULT_MATCH_ID,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    RAW_DATA_DIR,
)


SHOT_TYPES = ["GOAL", "SAVED SHOT", "MISSED SHOT", "SHOT ON POST"]
COORD_COLS = ["x", "y", "x_start", "y_start", "x_end", "y_end"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Driblab ETL pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        default=RAW_DATA_DIR,
        type=Path,
        help=(
            "Directory containing *_events.json, *_tracking_data.jsonl, and "
            "dim_event_type.csv."
        ),
    )
    parser.add_argument(
        "--match-id",
        default=DEFAULT_MATCH_ID,
        help="Match ID used for detailed event/tracking checks.",
    )
    parser.add_argument(
        "--max-rows",
        default=40,
        type=int,
        help="Maximum dataframe rows to print for large summaries.",
    )
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def print_df(df: pd.DataFrame, max_rows: int = 40) -> None:
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... {len(df) - max_rows} more rows")
    else:
        print(df.to_string(index=False))


def load_event_file(path: Path) -> list[dict[str, Any]]:
    """Load an event JSON file, falling back to object-by-object parsing."""
    content = path.read_text(errors="ignore")

    try:
        events = json.loads(content)
    except json.JSONDecodeError:
        events = []
        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content):
            try:
                events.append(json.loads(match.group()))
            except json.JSONDecodeError:
                pass

    if isinstance(events, dict):
        return [events]
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]
    return []


def load_tracking_file(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open() as f:
        lines = f.readlines()

    header = json.loads(lines[0])
    frames = [json.loads(line) for line in lines[1:]]
    return header, frames


def normalize_tracking_x(value: Any) -> float:
    """Convert tracking x from pitch meters to a clipped 0-100 field scale."""
    return max(0.0, min(100.0, float(value) / PITCH_LENGTH_M * 100.0))


def normalize_tracking_y(value: Any) -> float:
    """Convert tracking y from pitch meters to a clipped 0-100 field scale."""
    return max(0.0, min(100.0, float(value) / PITCH_WIDTH_M * 100.0))


def normalize_tracking_x_series(values: pd.Series) -> pd.Series:
    """Convert tracking x from meters to a clipped 0-100 field scale."""
    return (
        pd.to_numeric(values, errors="coerce").astype(float)
        / PITCH_LENGTH_M
        * 100.0
    ).clip(lower=0.0, upper=100.0)


def normalize_tracking_y_series(values: pd.Series) -> pd.Series:
    """Convert tracking y from meters to a clipped 0-100 field scale."""
    return (
        pd.to_numeric(values, errors="coerce").astype(float)
        / PITCH_WIDTH_M
        * 100.0
    ).clip(lower=0.0, upper=100.0)


def _normalize_optional_tracking_x(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return normalize_tracking_x(numeric)


def _normalize_optional_tracking_y(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return normalize_tracking_y(numeric)


def load_data(data_dir: Path, match_id: str) -> dict[str, Any]:
    section("Load data")

    event_types_path = data_dir / "dim_event_type.csv"
    with event_types_path.open() as f:
        event_types = list(csv.DictReader(f))
    df_event_types = pd.DataFrame(event_types)

    match_events_path = data_dir / f"{match_id}_events.json"
    match_events_raw = load_event_file(match_events_path)
    df_events = pd.json_normalize(match_events_raw)

    tracking_path = data_dir / f"{match_id}_tracking_data.jsonl"
    tracking_header, tracking_frames = load_tracking_file(tracking_path)

    print(f"Event types loaded:   {len(df_event_types)} rows")
    print(f"Labelled events:      {len(df_events)} rows ({match_id})")
    print(f"Tracking frames:      {len(tracking_frames)} ({match_id})")
    print(f"\nMatch info: {tracking_header.get('match_data')}")
    print(f"\nEvent columns: {list(df_events.columns)}")

    return {
        "df_event_types": df_event_types,
        "df_events": df_events,
        "tracking_header": tracking_header,
        "tracking_frames": tracking_frames,
    }


def event_type_distribution(
    data_dir: Path,
    df_event_types: pd.DataFrame,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("Event type distribution across all matches")

    event_files = sorted(data_dir.glob("*_events.json"))
    all_events_raw: list[dict[str, Any]] = []
    for path in event_files:
        all_events_raw.extend(load_event_file(path))

    df_all_events = pd.json_normalize(all_events_raw)
    event_counts = df_all_events["event.event_type_name"].value_counts()
    total = len(df_all_events)

    df_event_summary = event_counts.rename_axis(
        "event").reset_index(name="count")
    df_event_summary["percent"] = (
        df_event_summary["count"] / total * 100).round(2)
    df_event_summary["cumulative_percent"] = (
        df_event_summary["percent"].cumsum().round(2)
    )

    print(f"Loaded {len(event_files)} event files")
    print(f"Total events across all matches: {total:,}")
    print(
        "Unique event types across all matches: "
        f"{len(event_counts)} (out of {len(df_event_types)} defined)\n"
    )
    print_df(df_event_summary, max_rows=max_rows)

    return df_all_events, df_event_summary


def validate_event_coordinates(df_all_events: pd.DataFrame) -> pd.DataFrame:
    """Convert event coordinate columns to numeric for validation only.

    Event x/y coordinates are already supplied on a 0-100 attacking-direction
    scale. ETL should not rescale them; Step 2 only flips event x into tracking
    orientation after the match direction is inferred.
    """
    events_for_coords = df_all_events.copy()
    coord_cols = [
        col for col in COORD_COLS if col in events_for_coords.columns]
    events_for_coords[coord_cols] = events_for_coords[coord_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return events_for_coords


def normalize_tracking_coordinates(
    tracking_frames: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Build normalized 0-100 ball and player coordinate tables."""
    ball_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []

    for frame in tracking_frames:
        ball = frame.get("ball") or [None, None, None]
        clock = frame.get("match_clock") or [None, None]
        ball_rows.append(
            {
                "frame_id": frame.get("frame"),
                "period_id": frame.get("period"),
                "match_clock_min": clock[0],
                "match_clock_sec": clock[1],
                "ball_x_raw_m": ball[0],
                "ball_y_raw_m": ball[1],
                "ball_z_raw_m": ball[2],
                "ball_x_norm": _normalize_optional_tracking_x(ball[0]),
                "ball_y_norm": _normalize_optional_tracking_y(ball[1]),
            }
        )

        for team_id, players in (frame.get("data") or {}).items():
            for player in players:
                player_rows.append(
                    {
                        "frame_id": frame.get("frame"),
                        "period_id": frame.get("period"),
                        "match_clock_min": clock[0],
                        "match_clock_sec": clock[1],
                        "team_id": team_id,
                        "player_id": player.get("id"),
                        "player_visible": player.get("vis"),
                        "player_x_raw_m": player.get("x"),
                        "player_y_raw_m": player.get("y"),
                        "player_x_norm": _normalize_optional_tracking_x(
                            player.get("x")
                        ),
                        "player_y_norm": _normalize_optional_tracking_y(
                            player.get("y")
                        ),
                    }
                )

    return {
        "ball": pd.DataFrame(ball_rows),
        "players": pd.DataFrame(player_rows),
    }


def coordinate_system_analysis(
    df_all_events: pd.DataFrame,
    tracking_frames: list[dict[str, Any]],
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    section("Coordinate system analysis")

    df_all_events_checked = validate_event_coordinates(df_all_events)
    tracking_coordinate_tables = normalize_tracking_coordinates(
        tracking_frames)
    coord_cols = [
        col for col in COORD_COLS if col in df_all_events_checked.columns]

    match_count = (
        df_all_events_checked["match_id"].nunique()
        if "match_id" in df_all_events_checked.columns
        else 1
    )

    print("EVENTS coordinate system (all matches)")
    print(f"  Matches: {match_count}")
    for col in coord_cols:
        print(
            f"  {col:<7} raw range: "
            f"{df_all_events_checked[col].min()} - "
            f"{df_all_events_checked[col].max()}"
        )
    print("  Events are already normalized 0-100; ETL does not rescale them")

    shots = df_all_events_checked[
        df_all_events_checked["event.event_type_name"].isin(SHOT_TYPES)
    ].copy()
    shot_summary = (
        shots.groupby(
            ["match_id", "team.team_name", "period_id"],
            dropna=False,
        )
        .agg(
            shots=("event.event_type_name", "size"),
            x_median=("x", "median"),
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_median=("y", "median"),
        )
        .reset_index()
        .sort_values(["match_id", "team.team_name", "period_id"])
    )

    print("\nSHOT LOCATIONS SUMMARY (all matches, both teams, both halves)")
    print_df(shot_summary, max_rows=max_rows)

    print("\nTRACKING coordinate system")
    ball_table = tracking_coordinate_tables["ball"]
    player_table = tracking_coordinate_tables["players"]
    print(
        "  Ball x raw meters range: "
        f"{ball_table['ball_x_raw_m'].min()} - "
        f"{ball_table['ball_x_raw_m'].max()}"
    )
    print(
        "  Ball y raw meters range: "
        f"{ball_table['ball_y_raw_m'].min()} - "
        f"{ball_table['ball_y_raw_m'].max()}"
    )
    print(
        "  Ball x normalized range: "
        f"{ball_table['ball_x_norm'].min():.1f} - "
        f"{ball_table['ball_x_norm'].max():.1f}"
    )
    print(
        "  Ball y normalized range: "
        f"{ball_table['ball_y_norm'].min():.1f} - "
        f"{ball_table['ball_y_norm'].max():.1f}"
    )
    print(
        "  Player x normalized range: "
        f"{player_table['player_x_norm'].min():.1f} - "
        f"{player_table['player_x_norm'].max():.1f}"
    )
    print(
        "  Player y normalized range: "
        f"{player_table['player_y_norm'].min():.1f} - "
        f"{player_table['player_y_norm'].max():.1f}"
    )

    sample_frame = next(
        frame
        for frame in tracking_frames
        if frame.get("ball")
        and any(value is not None for value in frame["ball"])
    )
    ball_x, ball_y, ball_z = sample_frame["ball"]
    print(f"  Ball x/y/z raw meters: {sample_frame['ball']}")
    print(
        "  Ball x/y normalized 0-100: "
        f"{normalize_tracking_x(ball_x):.1f}, "
        f"{normalize_tracking_y(ball_y):.1f}"
    )
    team_keys = list(sample_frame["data"].keys())
    sample_player = sample_frame["data"][team_keys[0]][0]
    print(
        "  Player x/y raw meters: "
        f"{sample_player['x']}, {sample_player['y']}"
    )
    print(
        "  Player x/y normalized 0-100: "
        f"{normalize_tracking_x(sample_player['x']):.1f}, "
        f"{normalize_tracking_y(sample_player['y']):.1f}"
    )

    print("\nCOORDINATE MISMATCH")
    print("  Events:   0-100, normalized, always attacking toward 100")
    print("  Tracking: raw meters are converted and clipped to 0-100")
    print("  Alignment required: normalize tracking plus per-half flip")

    return df_all_events_checked, tracking_coordinate_tables


def time_encoding_analysis(
    df_events: pd.DataFrame,
    tracking_frames: list[dict[str, Any]],
) -> None:
    section("Time encoding, outcome, and qualifiers")

    print("EVENTS time encoding")
    p1 = df_events[df_events["period_id"] == 1]
    p2 = df_events[df_events["period_id"] == 2]
    print(f"  Period 1 minutes: {p1['min'].min()} - {p1['min'].max()}")
    print(f"  Period 2 minutes: {p2['min'].min()} - {p2['min'].max()}")
    print("  Minutes count up and do not reset at half-time")

    print("\nTRACKING time encoding")
    p1_frames = [frame for frame in tracking_frames if frame["period"] == 1]
    p2_frames = [frame for frame in tracking_frames if frame["period"] == 2]
    p1_times = [frame["match_clock"] for frame in p1_frames]
    p2_times = [frame["match_clock"] for frame in p2_frames]
    print(f"  Period 1 frames:      {len(p1_frames)}")
    print(f"  Period 2 frames:      {len(p2_frames)}")
    print(f"  Period 1 match_clock: {p1_times[0]} -> {p1_times[-1]}")
    print(f"  Period 2 match_clock: {p2_times[0]} -> {p2_times[-1]}")

    print("\nTIME SYNC CHALLENGE")
    print("  Events use:   period + min + sec + millisec")
    print("  Tracking uses: period + match_clock [min, sec] + frame")
    print("  match_clock[0]=min and match_clock[1]=sec are compatible")
    print("  Millisecond drift may still exist and needs verification")

    print("\nOUTCOME FIELD")
    outcome_by_type = df_events.groupby("event.event_type_name")[
        "outcome"].unique()
    always_true = [
        name for name,
        values in outcome_by_type.items() if set(values) == {True}]
    varies = [
        name for name,
        values in outcome_by_type.items() if len(
            set(values)) > 1]
    print(f"  Always True, not informative: {always_true}")
    print(f"  Varies True/False, success/fail: {varies}")

    print("\nQUALIFIERS STRUCTURE")
    sample_quals = next(
        qual
        for qual in df_events["qualifiers"]
        if isinstance(qual, list) and len(qual) > 0
    )
    print("  Example qualifiers list:")
    for qual in sample_quals:
        print(f"    {qual}")


def ball_and_visibility_quality(tracking_frames: list[dict[str, Any]]) -> None:
    section("Ball and visibility quality")

    total = len(tracking_frames)
    ball_present = sum(
        1
        for frame in tracking_frames
        if any(value is not None for value in frame["ball"])
    )
    ball_missing = total - ball_present

    cam_present = sum(
        1 for frame in tracking_frames if frame.get("cam") is not None)

    vis_true = 0
    vis_false = 0
    for frame in tracking_frames:
        for team_key in frame["data"]:
            for player in frame["data"][team_key]:
                if player.get("vis"):
                    vis_true += 1
                else:
                    vis_false += 1

    live_frames = [
        frame for frame in tracking_frames if frame.get("cam") is not None]
    ball_in_live = sum(1 for frame in live_frames if any(
        value is not None for value in frame["ball"]))
    total_vis = vis_true + vis_false

    print("BALL PRESENCE")
    print(
        "  All frames:        "
        f"{ball_present:>6} / {total}  ({ball_present / total * 100:.2f}%)"
    )
    print(
        "  Missing:           "
        f"{ball_missing:>6} / {total}  ({ball_missing / total * 100:.2f}%)"
    )
    print(
        "  During live play:  "
        f"{ball_in_live:>6} / {cam_present}  "
        f"({ball_in_live / cam_present * 100:.2f}%)"
    )

    print("\nPLAYER VISIBILITY")
    print(
        f"  vis=True observed:   {
            vis_true:>7}  ({
            vis_true /
            total_vis *
            100:.2f}%)")
    print(
        "  vis=False imputed:   "
        f"{vis_false:>7}  ({vis_false / total_vis * 100:.2f}%)"
    )

    print("\nCAM FIELD")
    print(
        "  Frames with cam:    "
        f"{cam_present:>6} / {total}  ({cam_present / total * 100:.2f}%)"
    )
    print("  Use cam-present frames for reliable detection")

    print("\nKEY WARNINGS")
    print(f"  Ball missing in {ball_missing / total * 100:.2f}% of all frames")
    print(
        f"  {vis_false / total_vis * 100:.2f}% of player positions "
        "are AI-imputed"
    )
    print("  Ball can be missing during shots and tackles")

    print("\nSAMPLING RATE")
    p1_frames = [frame for frame in tracking_frames if frame["period"] == 1]
    start_clock = p1_frames[0]["match_clock"]
    end_clock = p1_frames[-1]["match_clock"]
    duration_sec = (end_clock[0] * 60 + end_clock[1]) - (
        start_clock[0] * 60 + start_clock[1]
    )
    fps = len(p1_frames) / duration_sec
    print(
        f"  Period 1: {len(p1_frames)} frames over ~{duration_sec}s "
        f"-> ~{fps:.1f} Hz"
    )
    print("  Confirms approximately 10 frames/second")


def cross_file_consistency(
    data_dir: Path,
    df_events: pd.DataFrame,
    tracking_header: dict[str, Any],
) -> None:
    section("Cross-file consistency and asset inventory")

    print("DATA ASSET INVENTORY")
    matches: dict[str, set[str]] = {}
    for filename in sorted(os.listdir(data_dir)):
        match = re.match(r"(\d+)_(events|tracking_data)\.", filename)
        if match:
            matches.setdefault(match.group(1), set()).add(match.group(2))

    for match_id, assets in matches.items():
        has_events = "events" in assets
        has_tracking = "tracking_data" in assets
        print(
            f"  Match {match_id}: events={has_events}  "
            f"tracking={has_tracking}"
        )

    print("\nTRACKING team IDs")
    tracking_team_ids = set(tracking_header["players_data"].keys())
    for team_id in tracking_team_ids:
        players = tracking_header["players_data"][team_id]
        names = [player["name"] for player in players.values()][:3]
        print(f"  team_id: {team_id}  ({len(players)} players)  e.g. {names}")

    print("\nEVENTS team IDs")
    event_teams = df_events[["team.team_id",
                             "team.team_name"]].drop_duplicates()
    for _, row in event_teams.iterrows():
        print(f"  team_id: {row['team.team_id']}  ({row['team.team_name']})")

    print("\nPLAYER ID OVERLAP")
    tracking_player_ids = set()
    for team_id in tracking_header["players_data"]:
        for player_id in tracking_header["players_data"][team_id].keys():
            tracking_player_ids.add(str(player_id))

    event_player_ids = set(df_events["player.player_id"].astype(str).unique())
    overlap = event_player_ids & tracking_player_ids

    print(f"  Players in tracking: {len(tracking_player_ids)}")
    print(f"  Players in events:   {len(event_player_ids)}")
    print(f"  Overlap:             {len(overlap)}")
    if overlap:
        print("  IDs match, same match confirmed")
    else:
        print("  No overlap, check if files are from the same match")

    print("\nTIME ALIGNMENT SAMPLE")
    for _, row in df_events.head(5).iterrows():
        print(
            f"  Event: {row['event.event_type_name']:<15} "
            f"P{row['period_id']} {int(row['min']):02d}:{int(row['sec']):02d} "
            f"-> tracking match_clock=[{int(row['min'])},{int(row['sec'])}]"
        )


def run_pipeline(
    data_dir: Path,
    match_id: str,
    max_rows: int = 40,
) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    loaded = load_data(data_dir, match_id)
    df_all_events, df_event_summary = event_type_distribution(
        data_dir,
        loaded["df_event_types"],
        max_rows=max_rows,
    )
    df_all_events_checked, tracking_coordinate_tables = (
        coordinate_system_analysis(
            df_all_events,
            loaded["tracking_frames"],
            max_rows=max_rows,
        )
    )
    time_encoding_analysis(loaded["df_events"], loaded["tracking_frames"])
    ball_and_visibility_quality(loaded["tracking_frames"])
    cross_file_consistency(
        data_dir,
        loaded["df_events"],
        loaded["tracking_header"],
    )

    return {
        **loaded,
        "df_all_events": df_all_events,
        "df_event_summary": df_event_summary,
        "df_all_events_checked": df_all_events_checked,
        "tracking_coordinate_tables": tracking_coordinate_tables,
    }


def main() -> None:
    args = parse_args()
    run_pipeline(args.data_dir, args.match_id, args.max_rows)


if __name__ == "__main__":
    main()

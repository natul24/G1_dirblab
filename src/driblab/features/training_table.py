"""Build 0.5-second training windows for pass classification.

The module consumes the Step 2 master join table, assigns match-level splits
before any feature engineering, and writes one training table plus one summary
CSV per split.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driblab.config import CONFIG_PATH, MODEL_BASE_DATA_DIR, project_path
from driblab.features import match_splits


LOGGER = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "t.match_id",
    "t.period",
    "window_time",
    "data_split",
    "primary_event",
    "is_pass",
    "secondary_events",
    "ball_x_avg",
    "ball_y_avg",
    "ball_z_avg",
    "ball_speed_avg",
    "ball_speed_change",
    "closest_player_dist_start",
    "closest_player_team_start",
    "closest_player_dist_end",
    "closest_player_team_end",
    "closest_player_dist_change",
    "player_changed_same_team",
    "team_changed",
    "n_players_near_ball",
    "ball_displacement_x",
    "ball_displacement_y",
]

BALL_COLUMNS = ["t.ball_x", "t.ball_y", "t.ball_z"]
EVENT_COLUMN = "e.event.event_type_name"
EVENT_DISTANCE_COLUMN = "nearest_timestamp_distance_sec"
MATCH_COLUMN = "t.match_id"
PERIOD_COLUMN = "t.period"
FRAME_COLUMN = "t.frame"
SPLIT_COLUMN = "data_split"
WINDOW_SIZE = 5
WINDOW_SECONDS = 0.5
UNKNOWN_TEAM = "unknown"
NEAR_BALL_RADIUS = 5.0

NUMERIC_FEATURES = [
    "ball_x_avg",
    "ball_y_avg",
    "ball_z_avg",
    "ball_speed_avg",
    "ball_speed_change",
    "closest_player_dist_start",
    "closest_player_dist_end",
    "closest_player_dist_change",
    "n_players_near_ball",
    "ball_displacement_x",
    "ball_displacement_y",
]

FEATURE_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "ball_x_avg": (-5, 110),
    "ball_y_avg": (-5, 73),
    "ball_z_avg": (0, 10),
    "ball_speed_avg": (0, 50),
    "ball_speed_change": (-50, 50),
    "closest_player_dist_start": (0, 130),
    "closest_player_dist_end": (0, 130),
    "closest_player_dist_change": (-130, 130),
    "n_players_near_ball": (0, 22),
    "ball_displacement_x": (-110, 110),
    "ball_displacement_y": (-110, 110),
}


def load_training_inputs(
    master_join_path: Path,
    config_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the master join table, config, and match-level split labels."""
    config = _load_yaml_config(config_path)
    splits = match_splits.load_match_splits(config_path)
    master_join = pd.read_parquet(master_join_path)
    _validate_required_columns(master_join)
    master_join = match_splits.add_data_split_column(
        master_join,
        splits,
        match_col=MATCH_COLUMN,
    )
    return master_join, config


def compute_window_features(
    window_frames: pd.DataFrame,
    window_time: float,
) -> dict[str, Any]:
    """Compute pass-classification features for one 5-frame window."""
    first_frame = window_frames.iloc[0]
    last_frame = window_frames.iloc[-1]
    start_player = _closest_visible_player(first_frame)
    end_player = _closest_visible_player(last_frame)
    primary_event, secondary_events = _window_events(window_frames)

    ball_values = window_frames[BALL_COLUMNS].apply(
        pd.to_numeric, errors="coerce",
    )
    deltas = ball_values.diff().iloc[1:]
    frame_speeds = np.sqrt((deltas**2).sum(axis=1, skipna=False))
    ball_speed_change = (
        float(frame_speeds.iloc[-1] - frame_speeds.iloc[0])
        if len(frame_speeds) >= 2
        else 0.0
    )

    team_changed = int(
        start_player["team_id"] != UNKNOWN_TEAM
        and end_player["team_id"] != UNKNOWN_TEAM
        and start_player["team_id"] != end_player["team_id"]
    )

    return {
        "t.match_id": str(first_frame[MATCH_COLUMN]),
        "t.period": first_frame[PERIOD_COLUMN],
        "window_time": float(window_time),
        "data_split": first_frame[SPLIT_COLUMN],
        "primary_event": primary_event,
        "is_pass": int(primary_event == "PASS"),
        "secondary_events": secondary_events,
        "ball_x_avg": pd.to_numeric(
            window_frames["t.ball_x"], errors="coerce",
        ).mean(),
        "ball_y_avg": pd.to_numeric(
            window_frames["t.ball_y"], errors="coerce",
        ).mean(),
        "ball_z_avg": pd.to_numeric(
            window_frames["t.ball_z"], errors="coerce",
        ).mean(),
        "ball_speed_avg": _ball_speed_avg(window_frames),
        "ball_speed_change": ball_speed_change,
        "closest_player_dist_start": start_player["distance"],
        "closest_player_team_start": start_player["team_id"],
        "closest_player_dist_end": end_player["distance"],
        "closest_player_team_end": end_player["team_id"],
        "closest_player_dist_change": (
            end_player["distance"] - start_player["distance"]
        ),
        "player_changed_same_team": _player_changed_same_team(
            start_player, end_player,
        ),
        "team_changed": team_changed,
        "n_players_near_ball": _avg_players_near_ball(window_frames),
        "ball_displacement_x": (
            _to_float(last_frame.get("t.ball_x"))
            - _to_float(first_frame.get("t.ball_x"))
        ),
        "ball_displacement_y": (
            _to_float(last_frame.get("t.ball_y"))
            - _to_float(first_frame.get("t.ball_y"))
        ),
    }


def build_training_table_for_split(
    master_join: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Build the training table for one already assigned data split."""
    split_mask = master_join[SPLIT_COLUMN] == split_name
    split_rows = master_join.loc[split_mask].copy()
    if split_rows.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    records: list[dict[str, Any]] = []
    skipped_all_nan = 0
    group_columns = [MATCH_COLUMN, PERIOD_COLUMN]
    sort_columns = group_columns + [FRAME_COLUMN]
    split_rows = split_rows.sort_values(sort_columns, kind="mergesort")

    for _, period_frames in split_rows.groupby(group_columns, sort=False):
        period_frames = period_frames.reset_index(drop=True)
        complete_rows = (len(period_frames) // WINDOW_SIZE) * WINDOW_SIZE
        if complete_rows == 0:
            continue

        window_count = complete_rows // WINDOW_SIZE
        complete_period = period_frames.iloc[:complete_rows].reset_index(
            drop=True,
        )
        window_ball = _window_ball_values(complete_period, window_count)
        all_nan_windows = np.isnan(window_ball).all(axis=(1, 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ball_means = np.nanmean(window_ball, axis=1)
            ball_speeds = _window_ball_speeds(window_ball)
            frame_deltas = np.diff(window_ball, axis=1)
            frame_spds = np.sqrt(np.sum(frame_deltas**2, axis=2))
            ball_speed_changes = frame_spds[:, -1] - frame_spds[:, 0]
            ball_displacements_x = window_ball[:, -1, 0] - window_ball[:, 0, 0]
            ball_displacements_y = window_ball[:, -1, 1] - window_ball[:, 0, 1]

        near_ball_per_frame = _count_players_near_ball(complete_period)
        near_ball_per_window = near_ball_per_frame.reshape(
            window_count, WINDOW_SIZE,
        )
        n_players_near_ball_avg = np.mean(near_ball_per_window, axis=1)

        closest_players = _closest_visible_players(complete_period)
        event_types = (
            complete_period[EVENT_COLUMN].fillna("no event").to_numpy()
        )
        event_distances = pd.to_numeric(
            complete_period[EVENT_DISTANCE_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)
        match_id = str(complete_period[MATCH_COLUMN].iloc[0])
        period = complete_period[PERIOD_COLUMN].iloc[0]
        data_split = complete_period[SPLIT_COLUMN].iloc[0]

        for window_idx in range(window_count):
            if all_nan_windows[window_idx]:
                skipped_all_nan += 1
                continue
            start_idx = window_idx * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            end_frame_idx = end_idx - 1
            primary_event, secondary_events = _window_events_from_arrays(
                event_types[start_idx:end_idx],
                event_distances[start_idx:end_idx],
            )
            start_player = _player_from_arrays(closest_players, start_idx)
            end_player = _player_from_arrays(closest_players, end_frame_idx)
            window_number = window_idx + 1
            window_time = window_number * WINDOW_SECONDS
            records.append(
                {
                    "t.match_id": match_id,
                    "t.period": period,
                    "window_time": float(window_time),
                    "data_split": data_split,
                    "primary_event": primary_event,
                    "is_pass": int(primary_event == "PASS"),
                    "secondary_events": secondary_events,
                    "ball_x_avg": ball_means[window_idx, 0],
                    "ball_y_avg": ball_means[window_idx, 1],
                    "ball_z_avg": ball_means[window_idx, 2],
                    "ball_speed_avg": ball_speeds[window_idx],
                    "ball_speed_change": ball_speed_changes[window_idx],
                    "closest_player_dist_start": start_player["distance"],
                    "closest_player_team_start": start_player["team_id"],
                    "closest_player_dist_end": end_player["distance"],
                    "closest_player_team_end": end_player["team_id"],
                    "closest_player_dist_change": (
                        end_player["distance"] - start_player["distance"]
                    ),
                    "player_changed_same_team": (
                        _player_changed_same_team(start_player, end_player)
                    ),
                    "team_changed": int(
                        start_player["team_id"] != UNKNOWN_TEAM
                        and end_player["team_id"] != UNKNOWN_TEAM
                        and start_player["team_id"] != end_player["team_id"]
                    ),
                    "n_players_near_ball": n_players_near_ball_avg[
                        window_idx
                    ],
                    "ball_displacement_x": ball_displacements_x[window_idx],
                    "ball_displacement_y": ball_displacements_y[window_idx],
                }
            )

    if skipped_all_nan:
        LOGGER.warning(
            "Skipped %s %s windows with all ball position values missing.",
            skipped_all_nan,
            split_name,
        )

    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)


def build_all_training_tables(
    master_join_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build and save train, validation, and test training tables."""
    master_join_path = master_join_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    master_join, _ = load_training_inputs(master_join_path, config_path)
    raw_tables: dict[str, pd.DataFrame] = {}

    for split_name in match_splits.SPLIT_NAMES:
        raw_tables[split_name] = build_training_table_for_split(
            master_join,
            split_name,
        )

    outputs = _normalize_tables(raw_tables, output_dir)

    for split_name, training_table in outputs.items():
        table_path = output_dir / f"training_table_{split_name}.parquet"
        summary_path = output_dir / f"training_table_summary_{split_name}.csv"
        training_table.to_parquet(table_path, index=False)
        _summarize_training_table(training_table, split_name).to_csv(
            summary_path,
            index=False,
        )
        LOGGER.info(
            "Wrote %s windows for %s to %s",
            len(training_table),
            split_name,
            table_path,
        )

    return outputs


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def _validate_required_columns(table: pd.DataFrame) -> None:
    required_columns = [
        MATCH_COLUMN,
        PERIOD_COLUMN,
        FRAME_COLUMN,
        *BALL_COLUMNS,
        EVENT_COLUMN,
        EVENT_DISTANCE_COLUMN,
    ]
    missing = [column for column in required_columns if column not in table]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _all_ball_positions_nan(window_frames: pd.DataFrame) -> bool:
    ball_values = window_frames[BALL_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return bool(ball_values.isna().all().all())


def _ball_speed_avg(window_frames: pd.DataFrame) -> float:
    ball_values = window_frames[BALL_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    deltas = ball_values.diff().iloc[1:]
    distances = np.sqrt((deltas**2).sum(axis=1, skipna=False))
    return float(distances.mean(skipna=True))


def _window_ball_values(
    period_frames: pd.DataFrame,
    window_count: int,
) -> np.ndarray:
    ball_values = period_frames[BALL_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    return ball_values.reshape(window_count, WINDOW_SIZE, len(BALL_COLUMNS))


def _window_ball_speeds(window_ball: np.ndarray) -> np.ndarray:
    deltas = np.diff(window_ball, axis=1)
    distances = np.sqrt(np.sum(deltas**2, axis=2))
    return np.nanmean(distances, axis=1)


def _closest_visible_player(frame: pd.Series) -> dict[str, Any]:
    ball_x = _to_float(frame.get("t.ball_x"))
    ball_y = _to_float(frame.get("t.ball_y"))
    if pd.isna(ball_x) or pd.isna(ball_y):
        return _unknown_player()

    closest = _unknown_player()
    closest_distance = np.inf
    for prefix in _player_prefixes(frame.index):
        team_id = frame.get(f"{prefix}_team_id")
        if pd.isna(team_id) or not _is_visible(frame.get(f"{prefix}_visible")):
            continue

        player_x = _to_float(frame.get(f"{prefix}_x"))
        player_y = _to_float(frame.get(f"{prefix}_y"))
        if pd.isna(player_x) or pd.isna(player_y):
            continue

        distance = float(np.hypot(player_x - ball_x, player_y - ball_y))
        if distance < closest_distance:
            closest_distance = distance
            player_id = frame.get(f"{prefix}_id")
            closest = {
                "distance": distance,
                "team_id": str(team_id),
                "player_id": (
                    str(player_id)
                    if not pd.isna(player_id)
                    else prefix.rsplit(".", maxsplit=1)[-1]
                ),
            }

    return closest


def _avg_players_near_ball(window_frames: pd.DataFrame) -> float:
    """Average number of visible players within NEAR_BALL_RADIUS of ball."""
    counts = []
    for _, frame in window_frames.iterrows():
        ball_x = _to_float(frame.get("t.ball_x"))
        ball_y = _to_float(frame.get("t.ball_y"))
        if pd.isna(ball_x) or pd.isna(ball_y):
            counts.append(0)
            continue
        count = 0
        for prefix in _player_prefixes(frame.index):
            if not _is_visible(frame.get(f"{prefix}_visible")):
                continue
            if pd.isna(frame.get(f"{prefix}_team_id")):
                continue
            px = _to_float(frame.get(f"{prefix}_x"))
            py = _to_float(frame.get(f"{prefix}_y"))
            if pd.isna(px) or pd.isna(py):
                continue
            if np.hypot(px - ball_x, py - ball_y) < NEAR_BALL_RADIUS:
                count += 1
        counts.append(count)
    return float(np.mean(counts))


def _unknown_player() -> dict[str, Any]:
    return {"distance": np.nan, "team_id": UNKNOWN_TEAM, "player_id": None}


def _closest_visible_players(
    period_frames: pd.DataFrame,
) -> dict[str, np.ndarray]:
    prefixes = _player_prefixes(period_frames.columns)
    row_count = len(period_frames)
    if not prefixes:
        return _unknown_player_arrays(row_count)

    team_columns = [f"{prefix}_team_id" for prefix in prefixes]
    id_columns = [f"{prefix}_id" for prefix in prefixes]
    x_columns = [f"{prefix}_x" for prefix in prefixes]
    y_columns = [f"{prefix}_y" for prefix in prefixes]
    visible_columns = [f"{prefix}_visible" for prefix in prefixes]

    player_x = period_frames[x_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    player_y = period_frames[y_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    ball_x = pd.to_numeric(
        period_frames["t.ball_x"],
        errors="coerce",
    ).to_numpy(dtype=float)[:, np.newaxis]
    ball_y = pd.to_numeric(
        period_frames["t.ball_y"],
        errors="coerce",
    ).to_numpy(dtype=float)[:, np.newaxis]

    visible = period_frames[visible_columns].apply(
        lambda column: column.map(_is_visible),
    ).to_numpy(dtype=bool)
    team_values = period_frames[team_columns].astype("object").to_numpy()
    player_ids = period_frames[id_columns].astype("object").to_numpy()
    valid_team = ~pd.isna(team_values)
    distances = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
    distances[~visible | ~valid_team | np.isnan(distances)] = np.inf

    min_idx = np.argmin(distances, axis=1)
    min_dist = distances[np.arange(row_count), min_idx]
    no_player = np.isinf(min_dist)
    teams = _chosen_values(team_values, min_idx).astype("object")
    ids = _chosen_player_ids(player_ids, prefixes, min_idx)

    min_dist = min_dist.astype(float)
    min_dist[no_player] = np.nan
    teams[no_player] = UNKNOWN_TEAM
    ids[no_player] = None

    return {"distance": min_dist, "team_id": teams, "player_id": ids}


def _unknown_player_arrays(row_count: int) -> dict[str, np.ndarray]:
    return {
        "distance": np.full(row_count, np.nan),
        "team_id": np.full(row_count, UNKNOWN_TEAM, dtype=object),
        "player_id": np.full(row_count, None, dtype=object),
    }


def _count_players_near_ball(
    period_frames: pd.DataFrame,
) -> np.ndarray:
    """Count visible players within NEAR_BALL_RADIUS of ball per frame."""
    prefixes = _player_prefixes(period_frames.columns)
    if not prefixes:
        return np.zeros(len(period_frames))

    x_columns = [f"{prefix}_x" for prefix in prefixes]
    y_columns = [f"{prefix}_y" for prefix in prefixes]
    visible_columns = [f"{prefix}_visible" for prefix in prefixes]
    team_columns = [f"{prefix}_team_id" for prefix in prefixes]

    player_x = period_frames[x_columns].apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(dtype=float)
    player_y = period_frames[y_columns].apply(
        pd.to_numeric, errors="coerce",
    ).to_numpy(dtype=float)
    ball_x = pd.to_numeric(
        period_frames["t.ball_x"], errors="coerce",
    ).to_numpy(dtype=float)[:, np.newaxis]
    ball_y = pd.to_numeric(
        period_frames["t.ball_y"], errors="coerce",
    ).to_numpy(dtype=float)[:, np.newaxis]

    visible = period_frames[visible_columns].apply(
        lambda column: column.map(_is_visible),
    ).to_numpy(dtype=bool)
    team_values = period_frames[team_columns].astype("object").to_numpy()
    valid_team = ~pd.isna(team_values)

    distances = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
    valid = visible & valid_team & ~np.isnan(distances)
    near_ball = (distances < NEAR_BALL_RADIUS) & valid
    return near_ball.sum(axis=1).astype(float)


def _chosen_values(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values)), indices]


def _chosen_player_ids(
    player_ids: np.ndarray,
    prefixes: list[str],
    indices: np.ndarray,
) -> np.ndarray:
    chosen_ids = _chosen_values(player_ids, indices).astype("object")
    missing_ids = pd.isna(chosen_ids)
    fallback_ids = np.array(
        [prefix.rsplit(".", maxsplit=1)[-1] for prefix in prefixes],
        dtype=object,
    )
    chosen_ids[missing_ids] = fallback_ids[indices[missing_ids]]
    chosen_ids[~missing_ids] = chosen_ids[~missing_ids].astype(str)
    return chosen_ids


def _player_from_arrays(
    player_arrays: dict[str, np.ndarray],
    index: int,
) -> dict[str, Any]:
    return {
        "distance": player_arrays["distance"][index],
        "team_id": player_arrays["team_id"][index],
        "player_id": player_arrays["player_id"][index],
    }


def _is_visible(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _to_float(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _player_prefixes(columns: pd.Index) -> list[str]:
    prefixes = []
    for column in columns:
        if column.startswith("t.player_") and column.endswith("_team_id"):
            prefixes.append(column.removesuffix("_team_id"))
    return sorted(prefixes)


def _player_changed_same_team(
    start_player: dict[str, Any],
    end_player: dict[str, Any],
) -> int:
    if not start_player["player_id"] or not end_player["player_id"]:
        return 0
    if start_player["team_id"] == UNKNOWN_TEAM:
        return 0
    if end_player["team_id"] == UNKNOWN_TEAM:
        return 0
    same_team = start_player["team_id"] == end_player["team_id"]
    changed_player = start_player["player_id"] != end_player["player_id"]
    return int(same_team and changed_player)


def _window_events(window_frames: pd.DataFrame) -> tuple[str, str]:
    event_rows = window_frames.loc[
        window_frames[EVENT_COLUMN].fillna("no event") != "no event",
        [EVENT_COLUMN, EVENT_DISTANCE_COLUMN],
    ].copy()
    if event_rows.empty:
        return "no event", ""

    event_rows["_distance_for_sort"] = pd.to_numeric(
        event_rows[EVENT_DISTANCE_COLUMN],
        errors="coerce",
    ).fillna(np.inf)
    primary_index = event_rows["_distance_for_sort"].idxmin()
    primary_event = str(event_rows.loc[primary_index, EVENT_COLUMN])
    secondary_events = (
        event_rows.drop(index=primary_index)[EVENT_COLUMN].astype(str)
    )
    return primary_event, ",".join(secondary_events.tolist())


def _window_events_from_arrays(
    event_types: np.ndarray,
    event_distances: np.ndarray,
) -> tuple[str, str]:
    event_mask = event_types != "no event"
    if not event_mask.any():
        return "no event", ""

    event_indices = np.flatnonzero(event_mask)
    distances = event_distances[event_indices]
    distances = np.where(np.isnan(distances), np.inf, distances)
    primary_position = int(np.argmin(distances))
    primary_index = event_indices[primary_position]
    primary_event = str(event_types[primary_index])
    secondary_indices = np.delete(event_indices, primary_position)
    secondary_events = [str(event_types[index]) for index in secondary_indices]
    return primary_event, ",".join(secondary_events)


def _clip_features(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Clip numeric features to pitch-realistic bounds."""
    table = table.copy()
    for feature in features:
        if feature in table.columns and feature in FEATURE_CLIP_BOUNDS:
            lo, hi = FEATURE_CLIP_BOUNDS[feature]
            table[feature] = table[feature].clip(lower=lo, upper=hi)
    return table


def _normalize_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Clip outliers, fit MinMaxScaler on train split, transform all splits."""
    import joblib
    from sklearn.preprocessing import MinMaxScaler

    train_table = tables.get("train")
    if train_table is None or train_table.empty:
        return tables

    present_features = [f for f in NUMERIC_FEATURES if f in train_table.columns]
    if not present_features:
        return tables

    clipped: dict[str, pd.DataFrame] = {}
    for split_name, table in tables.items():
        clipped[split_name] = _clip_features(table, present_features)

    scaler = MinMaxScaler()
    scaler.fit(clipped["train"][present_features].fillna(0.0))

    normalized: dict[str, pd.DataFrame] = {}
    for split_name, table in clipped.items():
        table = table.copy()
        table[present_features] = scaler.transform(
            table[present_features].fillna(0.0),
        )
        normalized[split_name] = table

    scaler_path = output_dir / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    LOGGER.info("Saved fitted scaler to %s", scaler_path)

    return normalized


def _summarize_training_table(
    training_table: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    total_windows = len(training_table)
    pass_windows = int(training_table["is_pass"].sum()) if total_windows else 0
    no_event_windows = (
        int((training_table["primary_event"] == "no event").sum())
        if total_windows
        else 0
    )
    pass_percentage = (
        pass_windows / total_windows * 100
        if total_windows
        else 0.0
    )
    unique_matches = (
        int(training_table[MATCH_COLUMN].astype(str).nunique())
        if total_windows
        else 0
    )
    unique_periods = (
        int(training_table[PERIOD_COLUMN].nunique())
        if total_windows
        else 0
    )

    return pd.DataFrame(
        [
            {
                "split_name": split_name,
                "total_windows": total_windows,
                "pass_windows": pass_windows,
                "no_event_windows": no_event_windows,
                "pass_percentage": pass_percentage,
                "unique_matches": unique_matches,
                "unique_periods": unique_periods,
            }
        ]
    )


def _default_master_join_path() -> Path:
    return MODEL_BASE_DATA_DIR / "master_join_table.parquet"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pass-detection training tables from master join data."
        ),
    )
    parser.add_argument(
        "--master-join-path",
        type=Path,
        default=_default_master_join_path(),
    )
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_BASE_DATA_DIR,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = _parse_args()
    build_all_training_tables(
        project_path(args.master_join_path),
        project_path(args.config_path),
        project_path(args.output_dir),
    )


if __name__ == "__main__":
    main()

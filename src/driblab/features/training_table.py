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

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from driblab.config import (
    CONFIG_PATH,
    MODEL_BASE_DATA_DIR,
    TRAINED_MODELS_DIR,
    project_path,
)
from driblab.features import match_splits


LOGGER = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "t.match_id",
    "t.period",
    "window_time",
    "data_split",
    "is_attacking_direction",
    "primary_event",
    "is_pass",
    "secondary_events",
    "ball_x_avg",
    "ball_y_avg",
    "ball_z_avg",
    "ball_speed_avg",
    "ball_speed_change",
    "ball_direction_x",
    "ball_direction_y",
    "e.x_meters_absolute",
    "e.y_meters_absolute",
    "closest_player_dist_start",
    "closest_player_team_start",
    "closest_player_dist_end",
    "closest_player_team_end",
    "closest_player_dist_change",
    "n_players_near_ball",
    "n_unique_players_in_frame",
    "team_changed",
]

CONTINUOUS_FEATURES = [
    "ball_x_avg",
    "ball_y_avg",
    "ball_z_avg",
    "ball_speed_avg",
    "ball_speed_change",
    "ball_direction_x",
    "ball_direction_y",
    "closest_player_dist_start",
    "closest_player_dist_end",
    "closest_player_dist_change",
    "n_players_near_ball",
    "n_unique_players_in_frame",
    "e.x_meters_absolute",
    "e.y_meters_absolute",
]

BALL_COLUMNS = ["t.ball_x", "t.ball_y", "t.ball_z"]
EVENT_COLUMN = "e.event.event_type_name"
EVENT_DISTANCE_COLUMN = "nearest_timestamp_distance_sec"
EVENT_X_COLUMN = "e.x"
EVENT_Y_COLUMN = "e.y"
POSSESSION_COLUMN = "e.possession_id"
MATCH_COLUMN = "t.match_id"
PERIOD_COLUMN = "t.period"
FRAME_COLUMN = "t.frame"
SPLIT_COLUMN = "data_split"
WINDOW_SIZE = 5
WINDOW_SECONDS = 0.5
UNKNOWN_TEAM = "unknown"
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
NEAR_BALL_DISTANCE_M = 5.0


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


def convert_event_coordinates_to_absolute_meters(
    event_x: float,
    event_y: float,
    period: int,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> tuple[float, float]:
    """Convert 0-100 event coordinates to absolute field meters."""
    event_x = _to_float(event_x)
    event_y = _to_float(event_y)
    if pd.isna(event_x) or pd.isna(event_y):
        return np.nan, np.nan

    x_meters = event_x * (pitch_length_m / 100.0)
    y_meters = event_y * (pitch_width_m / 100.0)
    if int(period) == 1:
        x_meters_absolute = x_meters
    else:
        x_meters_absolute = pitch_length_m - x_meters
    return float(x_meters_absolute), float(y_meters)


def determine_attacking_direction(event_period: int) -> int:
    """Return 1 when the period attacks x=105, else 0."""
    return int(int(event_period) == 1)


def compute_window_features(
    window_frames: pd.DataFrame,
    window_time: float,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
) -> dict[str, Any]:
    """Compute pass-classification features for one 5-frame window."""
    first_frame = window_frames.iloc[0]
    start_player = _closest_visible_player(first_frame)
    end_player = _closest_visible_player(window_frames.iloc[-1])
    primary_event, secondary_events, primary_index = _window_event_summary(
        window_frames,
    )
    event_x_abs, event_y_abs = _primary_event_coordinates_absolute(
        window_frames,
        primary_index,
        first_frame[PERIOD_COLUMN],
        pitch_length_m,
        pitch_width_m,
    )
    player_density = _player_density_features(window_frames, 1)

    return {
        "t.match_id": str(first_frame[MATCH_COLUMN]),
        "t.period": first_frame[PERIOD_COLUMN],
        "window_time": float(window_time),
        "data_split": first_frame[SPLIT_COLUMN],
        "is_attacking_direction": determine_attacking_direction(
            first_frame[PERIOD_COLUMN],
        ),
        "primary_event": primary_event,
        "is_pass": int(primary_event == "PASS"),
        "secondary_events": secondary_events,
        "ball_x_avg": pd.to_numeric(
            window_frames["t.ball_x"],
            errors="coerce",
        ).mean(),
        "ball_y_avg": pd.to_numeric(
            window_frames["t.ball_y"],
            errors="coerce",
        ).mean(),
        "ball_z_avg": pd.to_numeric(
            window_frames["t.ball_z"],
            errors="coerce",
        ).mean(),
        "ball_speed_avg": _ball_speed_avg(window_frames),
        "ball_speed_change": _ball_speed_change(window_frames),
        "ball_direction_x": _ball_direction(window_frames, axis="x"),
        "ball_direction_y": _ball_direction(window_frames, axis="y"),
        "e.x_meters_absolute": event_x_abs,
        "e.y_meters_absolute": event_y_abs,
        "closest_player_dist_start": start_player["distance"],
        "closest_player_team_start": start_player["team_id"],
        "closest_player_dist_end": end_player["distance"],
        "closest_player_team_end": end_player["team_id"],
        "closest_player_dist_change": _distance_change(
            start_player["distance"],
            end_player["distance"],
        ),
        "n_players_near_ball": player_density["n_players_near_ball"][0],
        "n_unique_players_in_frame": (
            player_density["n_unique_players_in_frame"][0]
        ),
        "team_changed": _team_changed(
            window_frames,
            primary_index,
            start_player,
            end_player,
        ),
    }


def build_training_table_for_split(
    master_join: pd.DataFrame,
    split_name: str,
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
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
            ball_speed_changes = _window_ball_speed_changes(window_ball)
            ball_direction_x, ball_direction_y = _window_ball_directions(
                window_ball,
            )

        closest_players = _closest_visible_players(complete_period)
        player_density = _player_density_features(
            complete_period,
            window_count,
        )
        event_types = (
            complete_period[EVENT_COLUMN].fillna("no event").to_numpy()
        )
        event_distances = pd.to_numeric(
            complete_period[EVENT_DISTANCE_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)
        event_x = pd.to_numeric(
            complete_period[EVENT_X_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)
        event_y = pd.to_numeric(
            complete_period[EVENT_Y_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)
        event_possessions = (
            complete_period[POSSESSION_COLUMN].astype("object").to_numpy()
            if POSSESSION_COLUMN in complete_period
            else np.full(len(complete_period), None, dtype=object)
        )
        match_id = str(complete_period[MATCH_COLUMN].iloc[0])
        period = complete_period[PERIOD_COLUMN].iloc[0]
        data_split = complete_period[SPLIT_COLUMN].iloc[0]
        attacking_direction = determine_attacking_direction(period)

        for window_idx in range(window_count):
            if all_nan_windows[window_idx]:
                skipped_all_nan += 1
                continue
            start_idx = window_idx * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            end_frame_idx = end_idx - 1
            event_summary = _window_event_summary_from_arrays(
                event_types[start_idx:end_idx],
                event_distances[start_idx:end_idx],
            )
            primary_event = event_summary["primary_event"]
            secondary_events = event_summary["secondary_events"]
            primary_local_index = event_summary["primary_local_index"]
            primary_abs_index = (
                start_idx + primary_local_index
                if primary_local_index is not None
                else None
            )
            event_x_abs, event_y_abs = _event_coordinates_for_primary(
                event_x,
                event_y,
                primary_abs_index,
                period,
                pitch_length_m,
                pitch_width_m,
            )
            start_player = _player_from_arrays(closest_players, start_idx)
            end_player = _player_from_arrays(closest_players, end_frame_idx)
            window_number = window_idx + 1
            window_time = window_number * WINDOW_SECONDS
            team_changed = _team_changed_from_arrays(
                event_types[start_idx:end_idx],
                event_possessions[start_idx:end_idx],
                primary_local_index,
                start_player,
                end_player,
            )
            records.append(
                {
                    "t.match_id": match_id,
                    "t.period": period,
                    "window_time": float(window_time),
                    "data_split": data_split,
                    "is_attacking_direction": attacking_direction,
                    "primary_event": primary_event,
                    "is_pass": int(primary_event == "PASS"),
                    "secondary_events": secondary_events,
                    "ball_x_avg": ball_means[window_idx, 0],
                    "ball_y_avg": ball_means[window_idx, 1],
                    "ball_z_avg": ball_means[window_idx, 2],
                    "ball_speed_avg": ball_speeds[window_idx],
                    "ball_speed_change": ball_speed_changes[window_idx],
                    "ball_direction_x": ball_direction_x[window_idx],
                    "ball_direction_y": ball_direction_y[window_idx],
                    "e.x_meters_absolute": event_x_abs,
                    "e.y_meters_absolute": event_y_abs,
                    "closest_player_dist_start": start_player["distance"],
                    "closest_player_team_start": start_player["team_id"],
                    "closest_player_dist_end": end_player["distance"],
                    "closest_player_team_end": end_player["team_id"],
                    "closest_player_dist_change": _distance_change(
                        start_player["distance"],
                        end_player["distance"],
                    ),
                    "n_players_near_ball": (
                        player_density["n_players_near_ball"][window_idx]
                    ),
                    "n_unique_players_in_frame": (
                        player_density[
                            "n_unique_players_in_frame"
                        ][window_idx]
                    ),
                    "team_changed": team_changed,
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
    normalize: bool = True,
    scaler_output_path: Path | None = None,
) -> dict[str, Any]:
    """Build and save train, validation, and test training tables."""
    master_join_path = master_join_path.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    master_join, config = load_training_inputs(master_join_path, config_path)
    pitch_length_m, pitch_width_m = _pitch_dimensions(config)
    outputs: dict[str, Any] = {}

    for split_name in match_splits.SPLIT_NAMES:
        training_table = build_training_table_for_split(
            master_join,
            split_name,
            pitch_length_m=pitch_length_m,
            pitch_width_m=pitch_width_m,
        )
        table_path = output_dir / f"training_table_{split_name}.parquet"
        summary_path = output_dir / f"training_table_summary_{split_name}.csv"
        training_table.to_parquet(table_path, index=False)
        _summarize_training_table(training_table, split_name).to_csv(
            summary_path,
            index=False,
        )
        outputs[split_name] = training_table
        LOGGER.info(
            "Wrote %s windows for %s to %s",
            len(training_table),
            split_name,
            table_path,
        )

    if normalize:
        table_paths = {
            split_name: output_dir / f"training_table_{split_name}.parquet"
            for split_name in match_splits.SPLIT_NAMES
        }
        scaler_path = (
            scaler_output_path.expanduser().resolve()
            if scaler_output_path is not None
            else TRAINED_MODELS_DIR / "feature_scaler.pkl"
        )
        normalize_result = normalize_training_tables(
            train_path=table_paths["train"],
            validation_path=table_paths["validation"],
            test_path=table_paths["test"],
            scaler_output_path=scaler_path,
        )
        outputs = {
            split_name: pd.read_parquet(path)
            for split_name, path in table_paths.items()
        }
        outputs["normalization"] = normalize_result

    return outputs


def normalize_training_tables(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    scaler_output_path: Path,
) -> dict[str, Any]:
    """Normalize continuous features using train-split statistics only."""
    train_path = train_path.expanduser().resolve()
    validation_path = validation_path.expanduser().resolve()
    test_path = test_path.expanduser().resolve()
    scaler_output_path = scaler_output_path.expanduser().resolve()

    train_df = pd.read_parquet(train_path)
    validation_df = pd.read_parquet(validation_path)
    test_df = pd.read_parquet(test_path)
    _validate_normalization_columns(train_df, "train")
    _validate_normalization_columns(validation_df, "validation")
    _validate_normalization_columns(test_df, "test")

    scaler = StandardScaler()
    scaler.fit(train_df[CONTINUOUS_FEATURES].fillna(0))

    _apply_normalization(train_df, scaler)
    _apply_normalization(validation_df, scaler)
    _apply_normalization(test_df, scaler)

    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    test_df.to_parquet(test_path, index=False)

    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_output_path)

    LOGGER.info(
        "Normalized %s continuous features using train split statistics.",
        len(CONTINUOUS_FEATURES),
    )
    LOGGER.info("Saved fitted feature scaler to %s", scaler_output_path)

    return {
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "scaler_path": str(scaler_output_path),
        "normalized_features": CONTINUOUS_FEATURES.copy(),
    }


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def _pitch_dimensions(config: dict[str, Any]) -> tuple[float, float]:
    pitch_config = config.get("pitch", {})
    if not isinstance(pitch_config, dict):
        return PITCH_LENGTH_M, PITCH_WIDTH_M
    return (
        float(pitch_config.get("length_m", PITCH_LENGTH_M)),
        float(pitch_config.get("width_m", PITCH_WIDTH_M)),
    )


def _validate_required_columns(table: pd.DataFrame) -> None:
    required_columns = [
        MATCH_COLUMN,
        PERIOD_COLUMN,
        FRAME_COLUMN,
        *BALL_COLUMNS,
        EVENT_COLUMN,
        EVENT_DISTANCE_COLUMN,
        EVENT_X_COLUMN,
        EVENT_Y_COLUMN,
    ]
    missing = [column for column in required_columns if column not in table]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_normalization_columns(
    table: pd.DataFrame,
    split_name: str,
) -> None:
    missing = [
        column for column in CONTINUOUS_FEATURES
        if column not in table
    ]
    if missing:
        raise ValueError(
            f"Missing normalization columns in {split_name}: {missing}",
        )


def _apply_normalization(
    table: pd.DataFrame,
    scaler: StandardScaler,
) -> None:
    normalized_values = scaler.transform(
        table[CONTINUOUS_FEATURES].fillna(0),
    )
    for index, column in enumerate(CONTINUOUS_FEATURES):
        table[column] = normalized_values[:, index]


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


def _ball_speed_change(window_frames: pd.DataFrame) -> float:
    ball_values = window_frames[BALL_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    deltas = np.diff(ball_values, axis=0)
    distances = np.sqrt(np.sum(deltas**2, axis=1))
    valid_distances = distances[~np.isnan(distances)]
    if len(valid_distances) < 2:
        return np.nan
    return float(valid_distances[-1] - valid_distances[0])


def _ball_direction(window_frames: pd.DataFrame, axis: str) -> float:
    ball_values = window_frames[BALL_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if np.isnan(ball_values[0]).any() or np.isnan(ball_values[-1]).any():
        return np.nan
    axis_index = 0 if axis == "x" else 1
    return float(ball_values[-1, axis_index] - ball_values[0, axis_index])


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


def _window_ball_speed_changes(window_ball: np.ndarray) -> np.ndarray:
    deltas = np.diff(window_ball, axis=1)
    distances = np.sqrt(np.sum(deltas**2, axis=2))
    changes = np.full(len(window_ball), np.nan)
    for window_idx, window_distances in enumerate(distances):
        valid_distances = window_distances[~np.isnan(window_distances)]
        if len(valid_distances) >= 2:
            changes[window_idx] = valid_distances[-1] - valid_distances[0]
    return changes


def _window_ball_directions(
    window_ball: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_endpoints = (
        ~np.isnan(window_ball[:, 0, :]).any(axis=1)
        & ~np.isnan(window_ball[:, -1, :]).any(axis=1)
    )
    direction_x = np.full(len(window_ball), np.nan)
    direction_y = np.full(len(window_ball), np.nan)
    direction_x[valid_endpoints] = (
        window_ball[valid_endpoints, -1, 0]
        - window_ball[valid_endpoints, 0, 0]
    )
    direction_y[valid_endpoints] = (
        window_ball[valid_endpoints, -1, 1]
        - window_ball[valid_endpoints, 0, 1]
    )
    return direction_x, direction_y


def _distance_change(start_distance: Any, end_distance: Any) -> float:
    start = _to_float(start_distance)
    end = _to_float(end_distance)
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return float(end - start)


def _player_density_features(
    period_frames: pd.DataFrame,
    window_count: int,
) -> dict[str, np.ndarray]:
    prefixes = _player_prefixes(period_frames.columns)
    near_counts = np.zeros(window_count, dtype=int)
    unique_counts = np.zeros(window_count, dtype=int)
    if not prefixes:
        return {
            "n_players_near_ball": near_counts,
            "n_unique_players_in_frame": unique_counts,
        }

    row_count = window_count * WINDOW_SIZE
    period_frames = period_frames.iloc[:row_count]
    id_columns = [f"{prefix}_id" for prefix in prefixes]
    x_columns = [f"{prefix}_x" for prefix in prefixes]
    y_columns = [f"{prefix}_y" for prefix in prefixes]
    visible_columns = [f"{prefix}_visible" for prefix in prefixes]

    player_ids = period_frames[id_columns].astype("object").to_numpy()
    player_x = period_frames[x_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    player_y = period_frames[y_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    visible = period_frames[visible_columns].apply(
        lambda column: column.map(_is_visible),
    ).to_numpy(dtype=bool)
    ball_x = pd.to_numeric(
        period_frames["t.ball_x"],
        errors="coerce",
    ).to_numpy(dtype=float)
    ball_y = pd.to_numeric(
        period_frames["t.ball_y"],
        errors="coerce",
    ).to_numpy(dtype=float)

    slot_count = len(prefixes)
    player_ids = player_ids.reshape(window_count, WINDOW_SIZE, slot_count)
    visible = visible.reshape(window_count, WINDOW_SIZE, slot_count)
    player_x = player_x.reshape(window_count, WINDOW_SIZE, slot_count)
    player_y = player_y.reshape(window_count, WINDOW_SIZE, slot_count)
    ball_x = ball_x.reshape(window_count, WINDOW_SIZE, 1)
    ball_y = ball_y.reshape(window_count, WINDOW_SIZE, 1)

    valid_ids = ~pd.isna(player_ids)
    distances = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
    valid_near = (
        visible
        & valid_ids
        & ~np.isnan(distances)
        & (distances < NEAR_BALL_DISTANCE_M)
    )
    valid_visible = visible & valid_ids

    for window_idx in range(window_count):
        near_ids = player_ids[window_idx][valid_near[window_idx]]
        visible_ids = player_ids[window_idx][valid_visible[window_idx]]
        near_counts[window_idx] = len(
            {str(player_id) for player_id in near_ids},
        )
        unique_counts[window_idx] = len(
            {str(player_id) for player_id in visible_ids},
        )

    return {
        "n_players_near_ball": near_counts,
        "n_unique_players_in_frame": unique_counts,
    }


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


def _window_events(window_frames: pd.DataFrame) -> tuple[str, str]:
    primary_event, secondary_events, _ = _window_event_summary(window_frames)
    return primary_event, secondary_events


def _window_event_summary(
    window_frames: pd.DataFrame,
) -> tuple[str, str, Any | None]:
    event_rows = window_frames.loc[
        window_frames[EVENT_COLUMN].fillna("no event") != "no event",
        [EVENT_COLUMN, EVENT_DISTANCE_COLUMN],
    ].copy()
    if event_rows.empty:
        return "no event", "", None

    event_rows["_distance_for_sort"] = pd.to_numeric(
        event_rows[EVENT_DISTANCE_COLUMN],
        errors="coerce",
    ).fillna(np.inf)
    primary_index = event_rows["_distance_for_sort"].idxmin()
    primary_event = str(event_rows.loc[primary_index, EVENT_COLUMN])
    secondary_events = (
        event_rows.drop(index=primary_index)[EVENT_COLUMN].astype(str)
    )
    return primary_event, ",".join(secondary_events.tolist()), primary_index


def _window_events_from_arrays(
    event_types: np.ndarray,
    event_distances: np.ndarray,
) -> tuple[str, str]:
    event_summary = _window_event_summary_from_arrays(
        event_types,
        event_distances,
    )
    return (
        event_summary["primary_event"],
        event_summary["secondary_events"],
    )


def _window_event_summary_from_arrays(
    event_types: np.ndarray,
    event_distances: np.ndarray,
) -> dict[str, Any]:
    event_mask = event_types != "no event"
    if not event_mask.any():
        return {
            "primary_event": "no event",
            "secondary_events": "",
            "primary_local_index": None,
        }

    event_indices = np.flatnonzero(event_mask)
    distances = event_distances[event_indices]
    distances = np.where(np.isnan(distances), np.inf, distances)
    primary_position = int(np.argmin(distances))
    primary_index = event_indices[primary_position]
    primary_event = str(event_types[primary_index])
    secondary_indices = np.delete(event_indices, primary_position)
    secondary_events = [str(event_types[index]) for index in secondary_indices]
    return {
        "primary_event": primary_event,
        "secondary_events": ",".join(secondary_events),
        "primary_local_index": int(primary_index),
    }


def _primary_event_coordinates_absolute(
    window_frames: pd.DataFrame,
    primary_index: Any | None,
    period: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> tuple[float, float]:
    if primary_index is None:
        return np.nan, np.nan
    return convert_event_coordinates_to_absolute_meters(
        window_frames.loc[primary_index, EVENT_X_COLUMN],
        window_frames.loc[primary_index, EVENT_Y_COLUMN],
        period,
        pitch_length_m,
        pitch_width_m,
    )


def _event_coordinates_for_primary(
    event_x: np.ndarray,
    event_y: np.ndarray,
    primary_abs_index: int | None,
    period: int,
    pitch_length_m: float,
    pitch_width_m: float,
) -> tuple[float, float]:
    if primary_abs_index is None:
        return np.nan, np.nan
    return convert_event_coordinates_to_absolute_meters(
        event_x[primary_abs_index],
        event_y[primary_abs_index],
        period,
        pitch_length_m,
        pitch_width_m,
    )


def _team_changed(
    window_frames: pd.DataFrame,
    primary_index: Any | None,
    start_player: dict[str, Any],
    end_player: dict[str, Any],
) -> int:
    if POSSESSION_COLUMN not in window_frames:
        return _closest_player_team_changed(start_player, end_player)
    if primary_index is None:
        return _closest_player_team_changed(start_player, end_player)

    primary_possession = window_frames.loc[primary_index, POSSESSION_COLUMN]
    if not _is_usable_possession(primary_possession):
        return _closest_player_team_changed(start_player, end_player)

    event_rows = window_frames.loc[
        window_frames[EVENT_COLUMN].fillna("no event") != "no event",
        POSSESSION_COLUMN,
    ]
    for possession in event_rows:
        if not _is_usable_possession(possession):
            continue
        if str(possession) != str(primary_possession):
            return 1
    return 0


def _team_changed_from_arrays(
    event_types: np.ndarray,
    event_possessions: np.ndarray,
    primary_local_index: int | None,
    start_player: dict[str, Any],
    end_player: dict[str, Any],
) -> int:
    if primary_local_index is None:
        return _closest_player_team_changed(start_player, end_player)

    primary_possession = event_possessions[primary_local_index]
    if not _is_usable_possession(primary_possession):
        return _closest_player_team_changed(start_player, end_player)

    for event_type, possession in zip(event_types, event_possessions):
        if event_type == "no event":
            continue
        if not _is_usable_possession(possession):
            continue
        if str(possession) != str(primary_possession):
            return 1
    return 0


def _is_usable_possession(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip().lower() == "no event":
        return False
    return True


def _closest_player_team_changed(
    start_player: dict[str, Any],
    end_player: dict[str, Any],
) -> int:
    start_team = start_player["team_id"]
    end_team = end_player["team_id"]
    if start_team == UNKNOWN_TEAM or end_team == UNKNOWN_TEAM:
        return 0
    return int(start_team != end_team)


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
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Write raw feature tables without StandardScaler normalization.",
    )
    parser.add_argument(
        "--scaler-output-path",
        type=Path,
        default=TRAINED_MODELS_DIR / "feature_scaler.pkl",
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
        normalize=not args.no_normalize,
        scaler_output_path=project_path(args.scaler_output_path),
    )


if __name__ == "__main__":
    main()

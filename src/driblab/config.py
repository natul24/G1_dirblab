"""Shared project configuration and path constants.

This module loads `config.yaml`, resolves project-relative paths, and exposes
commonly used constants such as raw/processed data directories, artifact
directories, pitch dimensions, and the default match ID. Pipeline modules
import these constants instead of hard-coding paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_project_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the central project config."""
    path = (config_path or CONFIG_PATH).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    with path.open() as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def project_path(path_value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


PROJECT_CONFIG = load_project_config()
PATHS = PROJECT_CONFIG["paths"]
PITCH = PROJECT_CONFIG["pitch"]
ETL = PROJECT_CONFIG["etl"]

DATA_DIR = project_path("data")
RAW_DATA_DIR = project_path(PATHS["raw_data_dir"])
INTERIM_DATA_DIR = project_path(PATHS["interim_data_dir"])
PROCESSED_DATA_DIR = project_path(PATHS["processed_data_dir"])
ARTIFACTS_DIR = project_path(PATHS["artifacts_dir"])
MODEL_BASE_DATA_DIR = project_path(PATHS["model_base_dir"])
TRAINED_MODELS_DIR = project_path(PATHS["trained_models_dir"])
MODELS_DIR = PROJECT_ROOT / "src" / "driblab" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_MATCH_ID = str(ETL["default_match_id"])
PITCH_LENGTH_M = float(PITCH["length_m"])
PITCH_WIDTH_M = float(PITCH["width_m"])

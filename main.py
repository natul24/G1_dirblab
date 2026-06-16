"""Command-line entrypoint for the Driblab pipeline.

This file contains the terminal interface for running each project stage:
ETL checks and Step 2 master join creation. It reads `config.yaml`, converts
config values into each stage's dataclass, calls the stage runner, and prints
the key output paths and summary metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from driblab.config import CONFIG_PATH  # noqa: E402
from driblab.config import load_project_config  # noqa: E402
from driblab.config import project_path  # noqa: E402
from driblab.etl.master_join import Step2BatchConfig  # noqa: E402
from driblab.etl.master_join import run_step2_batch  # noqa: E402
from driblab.etl.pipeline import run_pipeline as run_etl_pipeline  # noqa: E402


def _path(config: dict[str, Any], key: str) -> Path:
    return project_path(config["paths"][key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Driblab project stages.",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        type=Path,
        help="Central YAML config file.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    etl = subparsers.add_parser("etl")
    etl.add_argument("--match-id")
    etl.add_argument("--max-rows", type=int)

    step2 = subparsers.add_parser("step2")
    step2.add_argument("--match-id")
    step2.add_argument("--all-matches", action="store_true")

    return parser.parse_args()


def run_etl(args: argparse.Namespace, config: dict[str, Any]) -> None:
    etl_config = config["etl"]
    run_etl_pipeline(
        data_dir=_path(config, "raw_data_dir"),
        match_id=args.match_id or str(etl_config["default_match_id"]),
        max_rows=args.max_rows or int(etl_config["max_rows"]),
    )


def run_step2(args: argparse.Namespace, config: dict[str, Any]) -> None:
    step_config = config["step2"]
    all_matches = args.all_matches or (
        args.match_id is None
        and bool(step_config["all_matches"])
    )
    batch_config = Step2BatchConfig(
        data_dir=_path(config, "raw_data_dir"),
        output_dir=_path(config, "model_base_dir"),
        model_base_dir=_path(config, "model_base_dir"),
        match_id=args.match_id or str(step_config["match_id"]),
        all_matches=all_matches,
        max_ball_gap_frames=int(step_config["max_ball_gap_frames"]),
        possession_distance_m=float(step_config["possession_distance_m"]),
        possession_max_ball_speed_mps=float(
            step_config["possession_max_ball_speed_mps"]
        ),
        max_sync_tolerance_sec=float(step_config["max_sync_tolerance_sec"]),
        direction_score_tolerance_sec=float(
            step_config["direction_score_tolerance_sec"]
        ),
        max_speed_dt_sec=float(step_config["max_speed_dt_sec"]),
        event_type_names=tuple(step_config.get("event_type_names", [])),
    )
    result = run_step2_batch(batch_config)
    print("\nStep 2 complete")
    print(f"Matches: {len(result['match_ids']):,}")
    print(f"Rows: {result['rows']:,}")
    print("Outputs:")
    for name, path in result["outputs"].items():
        print(f"  {name}: {path}")


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)

    if args.stage == "etl":
        run_etl(args, config)
    elif args.stage == "step2":
        run_step2(args, config)


if __name__ == "__main__":
    main()

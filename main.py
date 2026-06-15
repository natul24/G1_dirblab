"""Command-line entrypoint for the Driblab pipeline.

This file contains the terminal interface for running each project stage:
ETL checks, Step 2 master join creation, Step 3 possession sequence building,
Step 4 rule-based event detection, and the binary pass classifier. It reads
`config.yaml`, converts config values into each stage's dataclass, calls the
stage runner, and prints the key output paths and summary metrics.
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
from driblab.features.possession_sequence import Step3Config  # noqa: E402
from driblab.features.possession_sequence import run_step3  # noqa: E402
from driblab.models.pass_classifier import PassModelConfig  # noqa: E402
from driblab.models.pass_classifier import run_pass_model  # noqa: E402
from driblab.models.rule_based_detector import Step4Config  # noqa: E402
from driblab.models.rule_based_detector import run_step4  # noqa: E402


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

    step3 = subparsers.add_parser("step3")
    step3.add_argument("--match-id")

    step4 = subparsers.add_parser("step4")
    step4.add_argument(
        "--evaluation-split",
        choices=["train", "validation", "test"],
    )

    pass_model = subparsers.add_parser("pass_model")
    pass_model.add_argument("--threshold", type=float)

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
    )
    result = run_step2_batch(batch_config)
    print("\nStep 2 complete")
    print(f"Matches: {len(result['match_ids']):,}")
    print(f"Rows: {result['rows']:,}")
    print("Outputs:")
    for name, path in result["outputs"].items():
        print(f"  {name}: {path}")


def run_possession_sequence(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    step_config = config["step3"]
    result = run_step3(
        Step3Config(
            input_table=_path(config, "master_join_table"),
            output_dir=_path(config, "possession_sequence_dir"),
            match_splits=Path(args.config),
            match_id=args.match_id,
            max_gap_frames=int(step_config["max_gap_frames"]),
            min_stable_frames=int(step_config["min_stable_frames"]),
        )
    )
    summary = result["summary"]
    print("\nStep 3 complete")
    print(f"Rows: {len(result['table']):,}")
    print(
        "Smoothed possession frames: "
        f"{int(summary['smoothed_possession_frames'].sum()):,}"
    )
    print(f"Possession changes: {int(summary['possession_changes'].sum()):,}")
    print("Outputs:")
    for name, path in result["outputs"].items():
        print(f"  {name}: {path}")


def run_rule_detector(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    step_config = config["step4"]
    result = run_step4(
        Step4Config(
            input_table=(
                _path(config, "possession_sequence_dir")
                / "possession_sequence_table.parquet"
            ),
            output_dir=_path(config, "rule_based_detection_dir"),
            evaluation_split=(
                args.evaluation_split or step_config["evaluation_split"]
            ),
            shot_min_speed_mps=float(step_config["shot_min_speed_mps"]),
            shot_min_attacking_x=float(step_config["shot_min_attacking_x"]),
            shot_min_dx_attacking=float(step_config["shot_min_dx_attacking"]),
            interception_min_ball_speed_mps=float(
                step_config["interception_min_ball_speed_mps"]
            ),
            boundary_margin=float(step_config["boundary_margin"]),
            corner_y_margin=float(step_config["corner_y_margin"]),
            rule_classes=tuple(step_config["rule_classes"]),
            label_groups={
                key: tuple(value)
                for key, value in step_config["label_groups"].items()
            },
        )
    )
    summary = result["summary"].iloc[0]
    print("\nStep 4 complete")
    print(f"Evaluation split: {summary['evaluation_split']}")
    print(f"Rows: {int(summary['rows']):,}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1']:.4f}")
    print("Outputs:")
    for name, path in result["outputs"].items():
        print(f"  {name}: {path}")


def run_binary_pass_model(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    model_config = config["pass_model"]
    result = run_pass_model(
        PassModelConfig(
            input_table=_path(config, "master_join_table"),
            metrics_dir=_path(config, "pass_classifier_dir"),
            model_dir=_path(config, "pass_classifier_model_dir"),
            match_splits=Path(args.config),
            threshold=(
                args.threshold
                if args.threshold is not None
                else float(model_config["threshold"])
            ),
            c_value=float(model_config["c_value"]),
            max_iter=int(model_config["max_iter"]),
            solver=str(model_config["solver"]),
            random_state=int(model_config["random_state"]),
            class_weight=model_config["class_weight"],
            positive_labels=tuple(model_config["positive_labels"]),
            feature_columns=tuple(model_config["feature_columns"]),
        )
    )
    print("\nPass logistic regression complete")
    print(f"Features: {len(result['feature_columns']):,}")
    print("Metrics:")
    print(result["metrics"].to_string(index=False))
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
    elif args.stage == "step3":
        run_possession_sequence(args, config)
    elif args.stage == "step4":
        run_rule_detector(args, config)
    elif args.stage == "pass_model":
        run_binary_pass_model(args, config)


if __name__ == "__main__":
    main()

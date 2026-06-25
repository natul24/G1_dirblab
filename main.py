"""Pipeline entrypoint for Driblab.

Run stages in this order:
  1. master-join      Build master_join_table.parquet from all raw match files
  2. pre-training     Build pre_training_table.parquet from master join
  3. training-table   Build training_table_{train,validation,test}.parquet
  4. pass-detector    Train and save the XGBoost pass detector

Usage:
  python main.py master-join
  python main.py pre-training
  python main.py training-table
  python main.py pass-detector
  python main.py all            # runs all four stages in sequence
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from driblab.config import MODEL_BASE_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from driblab.etl.master_join import Step2BatchConfig, run_step2_batch  # noqa: E402
from driblab.features.pre_training_table import main as _pre_training_main  # noqa: E402
from driblab.features.training_table import main as _training_table_main  # noqa: E402
from driblab.models.pass_detector import train_pass_detector  # noqa: E402


def run_master_join() -> None:
    config = Step2BatchConfig(
        data_dir=RAW_DATA_DIR,
        output_dir=MODEL_BASE_DATA_DIR,
        model_base_dir=MODEL_BASE_DATA_DIR,
        all_matches=True,
    )
    result = run_step2_batch(config)
    print(
        f"\nMaster join complete: {result['rows']:,} rows"
        f" across {len(result['match_ids'])} matches"
    )


def run_pre_training() -> None:
    master_join_path = MODEL_BASE_DATA_DIR / "master_join_table.parquet"
    if not master_join_path.exists():
        sys.exit(
            f"Missing {master_join_path}\n"
            "Run: python main.py master-join"
        )
    _pre_training_main()


def run_training_table() -> None:
    pre_training_path = MODEL_BASE_DATA_DIR / "pre_training_table.parquet"
    if not pre_training_path.exists():
        sys.exit(
            f"Missing {pre_training_path}\n"
            "Run: python main.py pre-training"
        )
    _training_table_main()


def run_pass_detector() -> None:
    train_pass_detector()


STAGES = {
    "master-join": run_master_join,
    "pre-training": run_pre_training,
    "training-table": run_training_table,
    "pass-detector": run_pass_detector,
}

USAGE = """\
Usage: python main.py <stage>

Stages:
  master-join      Build master_join_table.parquet from all raw match files
  pre-training     Build pre_training_table.parquet from master join
  training-table   Build training_table_*.parquet
  pass-detector    Train and save the XGBoost pass detector
  all              Run all four stages in sequence
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {*STAGES, "all"}:
        print(USAGE)
        sys.exit(1)

    stage = sys.argv[1]

    if stage == "all":
        for name, fn in STAGES.items():
            print(f"\n{'─' * 50}")
            print(f"  {name}")
            print(f"{'─' * 50}\n")
            fn()
    else:
        STAGES[stage]()


if __name__ == "__main__":
    main()

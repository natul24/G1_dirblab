"""Pipeline entrypoint for Driblab.

Run stages in this order:
  1. master-join      Build master_join_table.parquet from all raw match files
  2. pre-training     [notebook only] notebooks/pre_training_table.ipynb
  3. training-table   Build training_table_{train,validation,test}.parquet
  4. pass-detector    Train and save the XGBoost pass detector

Usage:
  python main.py master-join
  python main.py training-table
  python main.py pass-detector
  python main.py all            # runs 1, 3, 4 — stage 2 must be run manually
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from driblab.config import MODEL_BASE_DATA_DIR, RAW_DATA_DIR
from driblab.etl.master_join import Step2BatchConfig, run_step2_batch
from driblab.features.training_table import main as _training_table_main
from driblab.models.pass_detector import train_pass_detector


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


def run_training_table() -> None:
    pre_training_path = MODEL_BASE_DATA_DIR / "pre_training_table.parquet"
    if not pre_training_path.exists():
        sys.exit(
            f"Missing {pre_training_path}\n"
            "Run notebooks/pre_training_table.ipynb first."
        )
    _training_table_main()


def run_pass_detector() -> None:
    train_pass_detector()


STAGES = {
    "master-join"    : run_master_join,
    "training-table" : run_training_table,
    "pass-detector"  : run_pass_detector,
}

USAGE = """\
Usage: python main.py <stage>

Stages:
  master-join      Build master_join_table.parquet from all raw match files
  training-table   Build training_table_*.parquet (requires pre_training_table.ipynb first)
  pass-detector    Train and save the XGBoost pass detector
  all              Run master-join → training-table → pass-detector in sequence
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

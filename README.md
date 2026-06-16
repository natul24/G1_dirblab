# Driblab Event Detection Pipeline

This repository is structured as a staged machine-learning project. The current
work covers Step 1 ETL and Step 2 master join tables. Later steps can add
event-detection models without reshuffling the project again.

## Fresh Clone Reproduction Steps

Follow these steps when opening the repo on a new computer.

1. Clone the GitHub repository and enter the project folder.

```bash
git clone <repo-url>
cd Driblab
git status
```

Replace `<repo-url>` with the GitHub URL for this project.

2. Create and activate the conda environment.

```bash
conda env create -f environment.yml
conda activate driblabvenv
python -c "from driblab.config import PROJECT_ROOT; print(PROJECT_ROOT)"
```

If the environment already exists, update it instead:

```bash
conda env update -f environment.yml --prune
conda activate driblabvenv
```

3. Add the raw Driblab files locally.

Raw data is ignored by Git, so it will not come from GitHub. Download the shared
raw data file from
[Google Drive](https://drive.google.com/file/d/1cWG2Yly2w1boaDFIX_S076lvqiHS_Yde/view?usp=sharing),
extract it locally if needed, and put the raw files into the existing
`data/raw/` folder using these names:

```text
data/raw/<match_id>_events.json
data/raw/<match_id>_tracking_data.jsonl
```

The ETL check also expects this file in `data/raw/`:

```text
data/raw/dim_event_type.csv
```

If the shared raw folder does not include it, create the local copy from the
tracked reference file:

```bash
cp data/reference/dim_event_type.csv data/raw/dim_event_type.csv
```

4. Confirm that the raw files are visible to the project.

```bash
python -c "from pathlib import Path; print('events', len(list(Path('data/raw').glob('*_events.json')))); print('tracking', len(list(Path('data/raw').glob('*_tracking_data.jsonl')))); print('event types', Path('data/raw/dim_event_type.csv').exists())"
```

For the current class data, the expected inventory is `34` event files, `33`
tracking files, and one event-only match. Step 2 uses only matches that have
both events and tracking.

5. Run the pipeline to recreate generated local outputs.

```bash
python main.py etl --max-rows 5
python main.py step2
```

These commands recreate:

```text
data/processed/model_base/master_join_table.parquet
data/processed/model_base/master_join_summary.csv
```

6. Open the notebooks with the same environment.

```bash
python -m ipykernel install --user --name driblabvenv --display-name driblabvenv
jupyter lab
```

Use the `driblabvenv` kernel and run notebooks in this order:

```text
notebooks/ETL.ipynb
notebooks/step2_3_match_clock_join.ipynb
```

7. Read the markdown documentation.

The markdown files in `docs/` are tracked source documentation, not generated
outputs. They explain the pipeline logic and the columns produced by the
commands above:

```text
docs/step2_foundation.md
docs/data_dictionary.md
```

8. Optional code-quality check.

```bash
python -m flake8 .
```

## Project Layout

```text
.
├── .flake8
├── .gitignore
├── README.md
├── config.yaml
├── environment.yml
├── main.py
├── pyproject.toml
├── artifacts/
│   └── models/
├── data/
│   ├── interim/
│   ├── processed/
│   │   ├── model_base/
│   ├── raw/
│   └── reference/
├── docs/
│   ├── data_dictionary.md
│   ├── step2_foundation.md
├── notebooks/
│   ├── ETL.ipynb
│   ├── data_exploration.ipynb
│   └── step2_3_match_clock_join.ipynb
├── reports/
│   └── figures/
├── src/
│   └── driblab/
│       ├── config.py
│       ├── validation.py
│       ├── etl/
│       │   ├── master_join.py
│       │   └── pipeline.py
│       ├── features/
│       │   └── match_splits.py
│       └── models/
└── tests/
```

## Module Map

| Module | Project stage | What it contains |
| --- | --- | --- |
| `src/driblab/config.py` | Shared configuration | Loads `config.yaml` and exposes project paths, pitch dimensions, and raw-data defaults. |
| `src/driblab/validation.py` | Shared validation | Small checks used across stages, such as required columns, match splits, and binary targets. |
| `src/driblab/etl/pipeline.py` | Step 1 ETL checks | Raw event/tracking loaders plus coordinate, asset, camera, ball, and consistency diagnostics. |
| `src/driblab/etl/master_join.py` | Step 2 foundation | Builds the tracking-first master join table from raw events and tracking data. |
| `src/driblab/features/match_splits.py` | Split management | Assigns complete matches to `train`, `validation`, and `test` without row-level leakage. |

## Data Inventory

Current local raw data under `data/raw/`:

- event files: `34`
- tracking files: `33`
- matched event/tracking pairs usable for Step 2: `33`
- event-only match: `683231`

Current generated processed outputs and model artifacts:

- all-match master join table: `data/processed/model_base/master_join_table.parquet`
- all-match summary: `data/processed/model_base/master_join_summary.csv`
- optional single-match sample files:
  `data/processed/model_base/master_join_table_679026.parquet` and
  `data/processed/model_base/master_join_summary_679026.csv`

## Local Ignored Files

Some files are intentionally ignored by Git. They will not appear when a
classmate clones the repository. The folder structure itself is tracked with
`.gitkeep` placeholder files, so classmates do not need to create these folders
manually. Each person only needs to add the ignored raw files locally and
regenerate processed outputs and model artifacts if needed.

| Local path | Why contents are ignored | How to get contents locally |
| --- | --- | --- |
| `data/raw/` | Original provider data can be large or private. | Download the shared raw data from [Google Drive](https://drive.google.com/file/d/1cWG2Yly2w1boaDFIX_S076lvqiHS_Yde/view?usp=sharing), then copy the files into this folder. |
| `data/interim/` | Temporary scratch outputs are not part of the modelling contract. | Recreate only if a future stage needs them. |
| `data/processed/` | Generated Parquet, CSV, and JSON outputs can be recreated from raw data. | Run `python main.py step2`. |
| `artifacts/models/` | Reserved for future trained model files. | No current command writes model artifacts. |
| `reports/figures/` | Generated plots can be recreated from notebooks or scripts. | Re-run the relevant notebook or reporting code. |
| `docs/DRIBLAB_CAPSTONE_EXECUTIVE_SUMMARY.pdf`, `docs/Student Kickoff Guide - Event Detection.pdf` | Local course/reference PDFs are not needed to run the pipeline. | Keep local copies outside Git if needed. |
| `.matplotlib_cache/`, `__pycache__/`, `.ipynb_checkpoints/` | Local runtime/cache files. | Created automatically by Python, Matplotlib, or Jupyter. |

After cloning the repo, copy the raw files into the existing `data/raw/` folder
using these names:

```text
data/raw/<match_id>_events.json
data/raw/<match_id>_tracking_data.jsonl
data/raw/dim_event_type.csv
```

If `dim_event_type.csv` is not included with the shared raw files, copy it from
the tracked reference folder:

```bash
cp data/reference/dim_event_type.csv data/raw/dim_event_type.csv
```

Once raw files are in place, run the project pipeline from the terminal to
refresh processed tables.

Rerunning a stage overwrites that stage's fixed output files in place. It does
not create duplicate timestamped files. For example, `python main.py step2`
rewrites the all-match master join table and summary.

## Environment

Use the project conda environment before running scripts or notebooks. Do not
run this project from the base conda environment.

### First-Time Setup

1. Install Anaconda or Miniconda if `conda` is not already available.

Check from the terminal:

```bash
conda --version
```

2. Open a terminal at the project root, the folder that contains
   `environment.yml`, `config.yaml`, and `main.py`.

```bash
cd path/to/Driblab
```

3. Create the environment from `environment.yml`.

```bash
conda env create -f environment.yml
```

This creates a conda environment named:

```text
driblabvenv
```

4. Activate the environment.

```bash
conda activate driblabvenv
```

5. Verify that the project package imports correctly.

```bash
python -c "from driblab.config import PROJECT_ROOT; print(PROJECT_ROOT)"
```

If this prints the project path, the environment is ready.

6. Use the same environment in Jupyter.

Start Jupyter from the activated environment:

```bash
jupyter lab
```

Then select the `driblabvenv` kernel when opening notebooks.

If the kernel does not appear, register it once:

```bash
python -m ipykernel install --user --name driblabvenv --display-name driblabvenv
```

### Updating an Existing Environment

If the environment already exists and `environment.yml` changes, update it with:

```bash
conda env update -f environment.yml --prune
```

The environment installs the local project package in editable mode, so imports
from `src/driblab/` work from `main.py` and notebooks.

## Run the Project From Terminal

Run all commands from the project root, after activating the environment:

```bash
conda activate driblabvenv
```

The raw provider files must be available locally under `data/raw/`. They are not
committed to Git. The expected file patterns are:

```text
data/raw/<match_id>_events.json
data/raw/<match_id>_tracking_data.jsonl
data/raw/dim_event_type.csv
```

Check that the raw files are present:

```bash
python -c "from pathlib import Path; print('events', len(list(Path('data/raw').glob('*_events.json')))); print('tracking', len(list(Path('data/raw').glob('*_tracking_data.jsonl')))); print('event types', Path('data/raw/dim_event_type.csv').exists())"
```

Optional quick ETL sanity check on a small sample:

```bash
python main.py etl --max-rows 5
```

Build the all-match Step 2 master join table:

```bash
python main.py step2
```

Expected main output:

```text
data/processed/model_base/master_join_table.parquet
```

To run the full current pipeline from raw data through Step 2:

```bash
conda activate driblabvenv
python main.py step2
```

Detailed Step 2 logic is documented in
[`docs/step2_foundation.md`](docs/step2_foundation.md).
The project column dictionary is in
[`docs/data_dictionary.md`](docs/data_dictionary.md).
The Step 2 notebook walkthrough is in
[`notebooks/step2_3_match_clock_join.ipynb`](notebooks/step2_3_match_clock_join.ipynb).

All pipeline paths, match splits, and Step 2 event labels are configured in
`config.yaml`.

## Coordinate Handling

Step 1 ETL validates the two coordinate systems before Step 2 joins the data.
Events already arrive on the provider's normalized `0-100` attacking-direction
scale, so ETL does not rescale event coordinates. Tracking arrives in physical
meters on a `105 x 68` pitch, so ETL creates normalized tracking x/y columns for
comparison.

| Source | Remained the same | Normalized / added |
| --- | --- | --- |
| Events | `x`, `y`, `x_start`, `y_start`, `x_end`, `y_end` stay as provided on the `0-100` attacking-direction scale | None in ETL |
| Tracking ball | `ball_x_raw_m`, `ball_y_raw_m`, `ball_z_raw_m` keep the original meter values in the ETL notebook tables | `ball_x_norm`, `ball_y_norm` convert x/y meters to `0-100` |
| Tracking players | `player_x_raw_m`, `player_y_raw_m` keep the original meter values in the ETL notebook tables | `player_x_norm`, `player_y_norm` convert x/y meters to `0-100` |
| Ball height | `ball_z_raw_m` remains meters | Not normalized |

In Step 2, event coordinates are still kept as event columns. Tracking ball x/y
columns are the ones converted for modelling, including attacking-perspective
tracking columns such as `ball_x_attacking` and `ball_y_attacking`.

The modelling convention is `x = 0` at the reference team's own-goal side and
`x = 100` at the goal the reference team is attacking. Event coordinates already
use that provider convention. Tracking coordinates start in meters, so Step 2
normalizes saved tracking x/y fields with:

```text
tracking_x_0_100 = clip(tracking_x_m / 105 * 100, 0, 100)
tracking_y_0_100 = clip(tracking_y_m / 68 * 100, 0, 100)
```

`ball_x_attacking` is a tracking-derived coordinate read from the reference
team's attacking direction. A value of `0` means close to that team's own goal,
`100` means close to the goal that team is attacking, and `20` means the ball is
about 20% up the pitch from that team's own goal. The reference team is chosen
in Step 2 as event team first, then possession team, then nearest team to the
ball; the chosen source is stored in `tracking_reference_source`.

## Current Outputs

Default single-match output:

```text
data/processed/model_base/master_join_table_<match_id>.parquet
```

For model training across all available matches, use:

```text
data/processed/model_base/master_join_table.parquet
```

That combined Parquet table has one row per tracking frame across every match,
with tracking/ball/possession/player aggregate features and the matched event
columns joined onto frames where events occur. Event columns are prefixed with
`event_`; frames without a matched event are labelled `no event` in
`event_label` and `event_type_name`. Step 2 currently labels only `PASS`,
`BALL TOUCH`, `AERIAL`, `TACKLE`, `BALL RECOVERY`, `FOUL`, and `TAKEON`;
other provider events remain `no event` rows in this modelling table. Events
are joined to tracking using match-clock time only: `period_id + min/sec/milisec`
matched to the closest tracking `period + match_clock + frame_index/FPS` row
within `0.5` seconds. This nearest-frame sync is the timestamp drift-correction
step.
Field x/y coordinates are kept on a normalized `0-100` scale for modelling;
raw tracking meter coordinates are converted and clipped into that range.

## Next ML Stages

Current and future code should stay staged:

- `src/driblab/features/` for match splits and later supervised training windows
- `src/driblab/models/` for future model training, saving, and inference
- `reports/` for model performance outputs

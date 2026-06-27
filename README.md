# Driblab Event Detection Pipeline

Machine-learning pipeline for detecting pass events from football tracking and event data. Covers ETL, master join, feature engineering, and an XGBoost pass detector.

---

## Setup

**1. Clone and enter the project**

```bash
git clone <repo-url>
cd Driblab
```

**2. Create and activate the conda environment**

```bash
conda env create -f environment.yml
conda activate driblabvenv
```

If the environment already exists, update it:

```bash
conda env update -f environment.yml --prune
conda activate driblabvenv
```

Verify the install:

```bash
python -c "from driblab.config import PROJECT_ROOT; print(PROJECT_ROOT)"
```

---

## Raw Data

Raw files are not committed to Git. Download from [Google Drive](https://drive.google.com/file/d/1cWG2Yly2w1boaDFIX_S076lvqiHS_Yde/view?usp=sharing) and place them in `data/raw/`:

```text
data/raw/<match_id>_events.json
data/raw/<match_id>_tracking_data.jsonl
data/raw/dim_event_type.csv
```

If `dim_event_type.csv` is not in the shared download, copy it from the reference folder:

```bash
cp data/reference/dim_event_type.csv data/raw/dim_event_type.csv
```

Expected inventory: **34** event files, **33** tracking files (match `683231` is event-only).

---

## Run the Pipeline

All commands run from the project root with `driblabvenv` active:

```bash
python main.py master-join      # builds master_join_table.parquet
python main.py pre-training     # builds pre_training_table.parquet
python main.py training-table   # builds training_table_{train,validation,test}.parquet
python main.py pass-detector    # trains XGBoost model and writes reports
```

Or run all stages in sequence:

```bash
python main.py all
```
## Event Detection Results

The project includes both a binary pass detector and a multi-class event detector. The binary pass detector improved after validation-based threshold tuning, while additional features gave only marginal improvements. The multi-class detector runs end-to-end, but minority event classes remain difficult due to class imbalance and likely label noise.

A detailed summary of the modelling results, NMS tuning, and main limitations is available here:

[`docs/event_detector_results_summary.md`](docs/event_detector_results_summary.md)
---

## Notebooks

Register the kernel once, then launch Jupyter:

```bash
python -m ipykernel install --user --name driblabvenv --display-name driblabvenv
jupyter lab
```

Run notebooks with the `driblabvenv` kernel in this order:

```text
notebooks/ETL.ipynb
notebooks/master_join_walkthrough.ipynb
notebooks/pre_training_table.ipynb
notebooks/training_table_walkthrough.ipynb
notebooks/xgboost_pass_detector.ipynb
```

---

## Project Layout

```text
.
├── config.yaml
├── environment.yml
├── main.py
├── pyproject.toml
├── artifacts/
│   └── models/
│       ├── feature_encoders.pkl
│       ├── feature_scaler.pkl
│       ├── pass_detector.json
│       └── pass_detector_metadata.json
├── data/
│   ├── raw/                    ← ignored by Git, download from Drive
│   ├── reference/
│   │   └── dim_event_type.csv
│   ├── interim/
│   └── processed/
│       └── model_base/
├── docs/
│   ├── data_dictionary.md
│   ├── master_join_walkthrough.md
│   ├── training_table_walkthrough.md
│   └── xgboost_model_guide.md
├── notebooks/
│   ├── ETL.ipynb
│   ├── data_exploration.ipynb
│   ├── master_join_walkthrough.ipynb
│   ├── pre_training_table.ipynb
│   ├── training_table_walkthrough.ipynb
│   └── xgboost_pass_detector.ipynb
├── reports/
│   ├── model_evaluation_results.json
│   ├── training_table_summary_train.csv
│   ├── training_table_summary_validation.csv
│   ├── training_table_summary_test.csv
│   └── figures/
│       ├── confusion_matrices.png
│       ├── feature_importance.png
│       └── roc_curve.png
├── src/
│   └── driblab/
│       ├── config.py
│       ├── validation.py
│       ├── etl/
│       │   ├── master_join.py
│       │   └── pipeline.py
│       ├── features/
│       │   ├── match_splits.py
│       │   ├── pre_training_table.py
│       │   └── training_table.py
│       └── models/
│           └── pass_detector.py
└── tests/
```

---

## Module Map

| Module | Stage | Description |
|---|---|---|
| `src/driblab/config.py` | Shared | Loads `config.yaml`; exposes project paths and pitch dimensions. |
| `src/driblab/validation.py` | Shared | Column, split, and target checks used across stages. |
| `src/driblab/etl/pipeline.py` | Step 1 ETL | Raw event/tracking loaders and diagnostics. |
| `src/driblab/etl/master_join.py` | Step 2 | Builds the tracking-first master join table. |
| `src/driblab/features/match_splits.py` | Splits | Assigns matches to train/validation/test without row-level leakage. |
| `src/driblab/features/pre_training_table.py` | Labeling | Assigns nearest event label within a 1-second window. |
| `src/driblab/features/training_table.py` | Features | Samples rows, computes ball speed, finds closest player, creates `is_pass`. |
| `src/driblab/models/pass_detector.py` | Model | Trains XGBoost classifier; writes model, metadata, metrics, and figures. |

---

## Docs

- [docs/master_join_walkthrough.md](docs/master_join_walkthrough.md)
- [docs/data_dictionary.md](docs/data_dictionary.md)
- [docs/training_table_walkthrough.md](docs/training_table_walkthrough.md)
- [docs/xgboost_model_guide.md](docs/xgboost_model_guide.md)

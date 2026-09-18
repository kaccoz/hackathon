# NebulaX PS3 — Rail Corrugation Baseline

This repository is the team's shared starting point for the Rail Corrugation part of Problem Statement 3.

It turns each 10,000-row sensor recording into a small row of useful signal summaries, trains a class-weighted model, checks it with stratified cross-validation, predicts the test files, and creates the required `rail_predictions.csv`.

## What the model does

```text
Raw CSV recording
  -> validate its 129-column sensor layout
  -> calculate time- and frequency-domain features
  -> compare Side I with Side II
  -> Extra Trees classifier
  -> Normal / Side I / Side II
```

The model is a conventional machine-learning classifier trained on the supplied labels. Gemini is not used to decide the rail condition.

## 1. First-time setup

Install Python 3.11 or newer, then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## 2. Add the supplied data

Do not commit the large organizer datasets to Git. Copy the Rail Corrugation folder into this layout:

```text
data/raw/rail/
├── Train/
│   ├── Train1.csv
│   └── ...
├── Test/
│   ├── Test1.csv
│   └── ...
└── Train_Labels.csv
```

## 3. Inspect the data

```bash
rail-inspect \
  --train-dir data/raw/rail/Train \
  --labels data/raw/rail/Train_Labels.csv
```

This checks file-label matching, prints the class counts, validates a sample CSV, and saves beginner-friendly plots in `outputs/inspection/`.

## 4. Train and validate

```bash
rail-train \
  --train-dir data/raw/rail/Train \
  --labels data/raw/rail/Train_Labels.csv \
  --artifact-dir artifacts
```

Training creates:

- `artifacts/rail_model.joblib` — fitted feature/model pipeline.
- `artifacts/cross_validation_predictions.csv` — honest validation predictions.
- `artifacts/metrics.json` — macro F1 and per-class results.
- `data/processed/rail_features.csv` — one feature row per recording.

The headline metric is macro F1. Always inspect the individual Side I and Side II scores too.

## 5. Produce test predictions

```bash
rail-predict \
  --test-dir data/raw/rail/Test \
  --model artifacts/rail_model.joblib \
  --output outputs/rail_predictions.csv
```

The command validates that every prediction is exactly `Normal`, `Side I`, or `Side II`, and writes the required columns:

```csv
file_id,prediction
Test1.csv,Normal
Test2.csv,Side II
```

## 6. Run the app

```bash
streamlit run app.py
```

Upload one or more Rail CSV files, view predictions, and download `rail_predictions.csv`. The same saved model is used by both the command line and the app.

## Team workflow

Read [`../docs/team-workflow.md`](../docs/team-workflow.md) before the first group coding session.

Short version:

1. Protect `main`; it should always run.
2. Each task gets a short branch such as `feature/speed-features`.
3. Pull requests must include the validation result before and after the change.
4. Never tune using the unlabeled Test folder.
5. Never commit datasets, trained models, API keys, or `.env` files.

## Project layout

```text
app.py                         Streamlit user interface
src/rail_cdm/io.py             Loading and validation
src/rail_cdm/features.py       Signal-to-feature conversion
src/rail_cdm/inspect_data.py   Dataset report and plots
src/rail_cdm/train.py          Cross-validation and final training
src/rail_cdm/predict.py        Required CSV generation
tests/                         Fast checks with synthetic data
docs/team-workflow.md          Four-person collaboration process
```

## Baseline limitations

This is deliberately a clear baseline rather than a claimed winning model. Likely improvement areas are speed-normalized frequency features, better frequency bands, consistency across cars, feature selection, and model comparison. Every improvement should be accepted only when repeated cross-validation supports it.

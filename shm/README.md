# SHM subsystem workspace

Owner task: predict one numeric cumulative-fatigue-damage value for every file.

Required output:

```csv
file_id,prediction
test01.csv,0.12345
```

Suggested internal layout:

```text
shm/
├── data/raw/            Local organizer data; do not commit
├── notebooks/           Exploration only
├── src/shm_cdm/         Reusable loading, feature and regression code
├── tests/               Synthetic and format checks
├── artifacts/           Saved models; do not commit
└── outputs/             Generated predictions; do not commit
```

First milestone: compare low- and high-damage files, then build file-level stress/rainflow features and a cross-validated regression baseline measured using MAPE.


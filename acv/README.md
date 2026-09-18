# ACV subsystem workspace

Owner task: rank all eight cars from most to least likely to have a refrigerant leak.

Required output:

```csv
file_id,ranked_cars
acv_test_case.xlsx,03|01|05|02|04|06|07|08
```

Suggested internal layout:

```text
acv/
├── data/raw/            Local organizer data; do not commit
├── notebooks/           Exploration only
├── src/acv_cdm/         Reusable loading, scoring and ranking code
├── tests/               Synthetic and format checks
├── artifacts/           Saved parameters/models; do not commit
└── outputs/             Generated predictions; do not commit
```

First milestone: dynamically identify the columns belonging to each car, plot car temperatures together, and create a peer-comparison anomaly score. With only six labelled cases, start with leave-one-case-out validation rather than a large model.


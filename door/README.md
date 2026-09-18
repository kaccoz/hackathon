# Door subsystem workspace

Owner task: detect door-cycle start/end boundaries in a continuous stream and classify every detected cycle as `Normal` or `Abnormal resistance`.

Required output:

```csv
start_time,end_time,prediction
```

Suggested internal layout:

```text
door/
├── data/raw/            Local organizer data; do not commit
├── notebooks/           Exploration only
├── src/door_cdm/        Reusable loading, segmentation and classification code
├── tests/               Synthetic and format checks
├── artifacts/           Saved models; do not commit
└── outputs/             Generated predictions; do not commit
```

First milestone: plot door position/current with the supplied true segment boundaries and implement an end-to-end validation score that includes both segmentation and classification.


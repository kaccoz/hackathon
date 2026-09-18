# NebulaX PS3 Team Repository

One shared repository contains four independent subsystem workspaces. Each member can develop and test their own pipeline without repeatedly editing the same files.

```text
hackathon/
├── door/                 Door cycle segmentation and classification
├── acv/                  Refrigerant-leak car ranking
├── rail_corrugation/     Rail Side I / Side II classification
├── shm/                  Cumulative fatigue-damage regression
├── integrated_app/       Final shared upload and results application
├── shared/               Agreements used by all four pipelines
└── docs/                 Team process and Git workflow
```

## Ownership

| Workspace | Main owner | Required output |
|---|---|---|
| `door/` | Door member | `door_predictions.csv` |
| `acv/` | ACV member | `acv_predictions.csv` |
| `rail_corrugation/` | Rail member | `rail_predictions.csv` |
| `shm/` | SHM member | `shm_predictions.csv` |
| `integrated_app/` | Shared near the end | One app calling all completed pipelines |

Each owner decides the internal modelling approach but must follow [`shared/PIPELINE_CONTRACT.md`](shared/PIPELINE_CONTRACT.md). This is what makes later integration predictable.

## Getting started

1. Clone the repository.
2. Read [`docs/team-workflow.md`](docs/team-workflow.md).
3. Enter your assigned directory.
4. Follow that directory's README.
5. Work on a branch and open a pull request rather than editing `main` directly.

The Rail workspace already contains a complete beginner-friendly baseline. The other workspaces begin with their data contract and task checklist so their owners can add code independently.

## Repository rules

- Do not commit organizer datasets; every member copies them into their own ignored `data/raw/` directory.
- Do not commit trained models, generated outputs, `.env` files, or credentials.
- Do not use hidden test answers to select features or models.
- Keep the exact organizer filenames and prediction labels.
- A subsystem is integration-ready only when its command can turn an input path into the exact required CSV.


# Shared pipeline contract

Every subsystem must expose one reproducible prediction command before integration.

## Required behavior

The command accepts:

- An input file or folder supplied by a non-technical user.
- A destination for the prediction CSV.
- Any trained-model location through configuration, not a hard-coded personal path.

The command must:

1. Validate the uploaded data.
2. Use the same preprocessing used during training.
3. Run without opening a notebook.
4. Preserve source filenames where `file_id` is required.
5. Generate the exact official column names and allowed values.
6. Fail with a readable message when the input is invalid.

## Required outputs

| Subsystem | Filename | Columns |
|---|---|---|
| Door | `door_predictions.csv` | `start_time,end_time,prediction` |
| ACV | `acv_predictions.csv` | `file_id,ranked_cars` |
| Rail | `rail_predictions.csv` | `file_id,prediction` |
| SHM | `shm_predictions.csv` | `file_id,prediction` |

## Integration handoff

Each subsystem owner hands the app integrator:

- A tested prediction function or command.
- A saved model and its version.
- A small, non-sensitive example input or synthetic test fixture.
- The expected output CSV.
- A list of Python dependencies.
- A short explanation of what the UI can safely display.

The integrated app must call these pipelines. It must not duplicate their feature-extraction logic.


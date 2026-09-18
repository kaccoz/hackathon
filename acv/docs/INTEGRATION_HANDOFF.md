# ACV integration handoff

## Selected method

Use `acv/predict.py`, which calls `src/baseline.py`. The selected method is original-positive-gap-v1, the sample average of positive indoor-minus-control differences. It requires no fitted model artifact, labels, external services, or plotting dependencies at prediction time. Pressure, valves, peer comparisons and time-weighted area experiments are not part of the selected ranking.

## Invocation

From the repository root:

```bash
acv/.venv/bin/python acv/predict.py --input acv/data/raw/train --output acv/outputs/check_all_training/acv_predictions.csv
```

Supply one workbook or a flat folder of workbooks. `create_predictions(input_path)` is also exported from `predict.py` for in-process integration and returns a pandas DataFrame. Do not duplicate feature calculations in the app. The function emits warnings for provisional rich-layout mapping and cars without usable readings; display those warnings to the user. CLI failures return a nonzero exit status and a readable message on stderr.

Output must contain only `file_id,ranked_cars`, with original filenames and all eight header car identifiers once each, separated by pipes. Input files are read only. A destination CSV is overwritten on successful output writing.

## Safe presentation

Describe the result as a suggested inspection order. Scores are average temperature gaps, not leak probabilities. Unscored cars are placed last by ID solely to complete the ordering; this does not mean they are healthy. No new model weights should be selected from test visualisations.

## Evidence and limitations

The user ran single-file and full six-training-file CLI checks and supplied matching CSV outputs. The all-six-case development comparison yielded five first-place matches and one second-place match, with mean official score 0.9792. This is not independent validation or measured test performance. A clean-environment install and automated regression suite have not been run. No synthetic integration fixture has been provided yet.

Case 04 has provisional field interpretations and no established information-valid equivalent; Cars 05-08 lack usable measurements. Control semantics and temperature units remain incompletely documented. Pressure investigation supports diagnostic discussion but is based on only one pressure-equipped training case.

The final test output has not been verified in this handoff. The official main specification also requires an app, demo video and prediction ZIP, with predictions generated through the app. These deliverables are not completed by this commit.

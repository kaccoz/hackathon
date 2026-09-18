# Official sources checked on 18 September 2026

Downloaded unchanged from https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/tree/main:

- `PS3/01_Problem_Statement_3_Specifications.md`
- `PS3/03_References/ACV/ACV_Subsystem_Info_Kit.md`
- First training workbook: `PS3/02_Datasets/ACV/Train/acv_case_01.xlsx`

Only the first training workbook was downloaded for lesson 1. Later lessons downloaded all six training workbooks and `Train_Labels.csv` from the same official ACV dataset directory. After selecting the original baseline, the user downloaded the test workbook. Plotting commands were extended to accept it for display; test data must not be used to tune or select the rule. No test performance is known.

## Confirmed requirements

The kit describes six labelled training cases, eight cars per case, one faulty car per case, and nominal 30-second sampling. Read each file's headers because parameter sets differ. Rank every car using its exact two-digit header identifier. The output has only `file_id` and `ranked_cars`, with pipe-separated identifiers. The official per-case score is `(n - (r - 1)) / n`, where r is the true faulty car's rank.

## Conflicts and missing details

The kit section 3 describes mandatory development code, a trained model, `predict.py`, a write-up, and a specified command-line interface. Main specification section 4 instead requires an app, a demo video of at most three minutes, and `predictions.zip`; development code/models and the write-up are optional. Recommendation: prepare the main specification's required deliverables and retain reusable prediction code; confirm the submission discrepancy with the organisers before packaging.

The kit does not explicitly define temperature units/scaling, sensor placement, or the detailed interpretation of control/status fields. Lesson 1 retains the recorded numeric scale and the original indoor/outdoor names. Do not infer refrigerant pressure or other unprovided sensor meanings.

## Lesson 1 choices (recommendations, not requirements)

Plot raw indoor and outdoor temperature readings separately for the first training case, with one line per car. Do not smooth, fill missing values, or make predictions yet. Later, hold out whole cases when validating, as requested by the learner.

## Case 04 exploration (lesson 6)

Case 04 has 483 columns. The kit's nominal 30-second sampling description conflicts with the 10-second intervals observed at the beginning of this case. Use recorded timestamps.

The learner agreed to explore passenger cabin temperature minus target temperature during Full Cooling or Half Cooling, excluding missing numeric readings. These are provisional interpretations based on original field names, not an officially confirmed equivalence with the other cases. Case 04 has no `ACV Information Valid` field. `ACV Operating Mode = Invalid` is not treated as an equivalent validity flag; its meaning remains unresolved. Missing values are not filled and zero values are not automatically discarded. Lesson 6 plots this difference only, without ranking or reading case 04's label. Cases 01 and 04 have now informed development and should not be described as untouched holdouts.

# Participant pack readiness audit

Source reviewed: `[Nebula X Hackathon] Participants Information Pack[22].pdf`, dated September 2026.

## Hard schedule

| Event | Time |
|---|---|
| Registration opens | 18 September, 4:30 pm |
| Opening programme | 18 September, 5:30 pm |
| Google workshop and mentor consultation begin | 18 September, 7:00 pm |
| Mentor consultation resumes | 19 September, 9:00 am |
| Submission counter opens | 19 September, 2:30 pm |
| Hard submission deadline | 19 September, 4:00 pm |
| Top 10 announced | 20 September, 10:30 am |
| Finalist presentations | 20 September, 1:30 pm and 3:00 pm; 10 minutes per team |

The effective build window is short. A working prediction pipeline and hosted prototype must exist early enough to leave time for packaging, video recording, and upload verification.

## PS3 submission checklist

| Requirement in participant pack | Current status | Required action |
|---|---|---|
| GitHub repository and README | In progress | Repository exists; update the root README with the final problem, architecture, setup, results, and team contributions |
| Hosted prototype URL | Not complete | Integrate at least one real pipeline, deploy to Cloud Run, then add the URL to the README |
| 2-3 minute video pitch | Not started | Write a short script, record the real upload-to-download flow, and verify the link before submission |
| Prediction output folder | Rail baseline complete locally; team package incomplete | Collect the exact CSV from every attempted subsystem |
| Short write-up covering solution, uniqueness, and technology stack | Not started | Draft alongside development; do not leave it for the final hour |

## Packaging ambiguity to clarify

The participant pack asks for a folder containing `*_predictions.csv`. The PS3 specification asks for those files at the top level of `predictions.zip`.

Safest preparation:

```text
predictions/
├── door_predictions.csv       if attempted
├── acv_predictions.csv        if attempted
├── rail_predictions.csv
└── shm_predictions.csv        if attempted

predictions.zip                same files at the ZIP root, no nested folder
```

Ask a PS3 mentor or the submission counter which portal field expects the folder/ZIP. Preparing both avoids a last-minute conversion.

## Attendance and access checks

- At least one member must register physically on 18 September and receive the team passkey.
- At least one member must physically sign in for submission on 19 September.
- All members of a shortlisted Top 10 team must confirm attendance and be physically present for the final presentation.
- Confirm the assigned Google Cloud project appears after login.
- Download Google Antigravity before the workshop if the team is attending it, then wait for the workshop login instructions.
- Keep the team passkey and cloud credentials private and out of Git.

## Does the current technical plan fit the pack?

Yes, with two priority changes:

1. Treat the hosted prototype, video, and write-up as required deliverables, not optional polish.
2. Deploy a thin working application early; continue model improvement behind the stable interface.

The four separate subsystem workspaces plus one `integrated_app/` are still appropriate. Each subsystem must generate its official CSV independently so integration cannot block submission.

## Current Rail status

- Official data downloaded locally into the Git-ignored data folder.
- Inventory confirmed: 272 training files and 68 test files.
- Training class distribution confirmed: 234 Normal, 14 Side I, 24 Side II.
- Data shape confirmed: 10,000 rows and 129 columns per inspected file.
- Baseline feature extraction, five-fold validation, final training, and test inference completed.
- Initial Extra Trees macro F1: 0.6181.
- Class-weighted Random Forest baseline macro F1: 0.7191 using the same five folds.
- Moderate fold-safe SMOTE plus a fixed Side I probability adjustment improved macro F1 to 0.7791 and Side I recall from the original 35.7% to 64.3%; Side II recall is 79.2% on the dashboard split.
- Across five repeated fold shuffles, the promoted model achieved average macro F1 0.7948, average Side I precision 47.4%, and average Side I recall 64.3%.
- A schema-valid 68-row `outputs/rail_predictions.csv` has been generated locally.

The next Rail priority is error analysis and deployment while continuing to monitor Side I precision/recall—not rebuilding the project structure.

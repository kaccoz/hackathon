# Four-person team workflow

## Shared definition of done

A change is complete only when:

1. The code runs from a fresh checkout.
2. Tests pass with `pytest`.
3. The author reports cross-validated macro F1 before and after a modelling change.
4. No training data, test data, model artifacts, credentials, or environment files are committed.
5. Another member has reviewed the pull request.

## Suggested ownership

### Person 1 — Data and validation

- Own `io.py`, inventory checks, dataset documentation, and cross-validation discipline.
- Confirm that no row or test-set leakage enters the pipeline.
- Maintain the experiment-results table.

### Person 2 — Signal features

- Own `features.py` experiments.
- Research speed-aware frequency features and cross-car consistency.
- Keep every feature deterministic and usable on unseen uploaded files.

### Person 3 — Models and evaluation

- Own `train.py` experiments.
- Compare class-weighted models using the same folds.
- Track macro F1, per-class F1, and confusion matrices.

### Person 4 — App and release

- Own `app.py`, Cloud Run deployment, exact output validation, and demo flow.
- Ensure the app calls the same feature/model code as the command-line predictor.
- Package and test `predictions.zip`.

Ownership means responsibility, not exclusive access. Pair on unfamiliar work and review each other's changes.

## Git workflow

Create a GitHub repository and push this scaffold to `main`. Protect `main` if your GitHub plan allows it.

For each task:

```bash
git switch main
git pull
git switch -c feature/short-description
```

Commit a small, reviewable change, push the branch, and open a pull request. Suggested branch names:

```text
feature/speed-normalization
feature/frequency-bands
model/random-forest-comparison
app/result-explanation
fix/test-file-ordering
```

Avoid four people editing the same file simultaneously. Discuss interface changes before coding them.

## Experiment log

Maintain a shared sheet with:

| Date | Branch | Features | Model | CV scheme | Macro F1 | Side I F1 | Side II F1 | Decision |
|---|---|---|---|---|---:|---:|---:|---|

Do not replace the current model because one split improved. Prefer repeated, consistent evidence.

## Recommended milestones

1. **Data understood:** inventory checks pass and the team can explain one CSV.
2. **Baseline complete:** cached features, cross-validation, saved model, and prediction CSV work.
3. **App complete:** upload-to-download journey works locally.
4. **Improvements frozen:** best reproducible model selected without examining test answers.
5. **Submission rehearsed:** a teammate performs the full demo from a clean environment.


"""Evaluate the fixed baseline on four whole training cases not used to design it.

This is not fitted leave-one-case-out cross-validation: the baseline has no
learned parameters. Reusing these results to change it makes them development
evidence, not a fresh independent evaluation of the changed rule.
"""
from pathlib import Path
import pandas as pd
from baseline import rank_file

ACV = Path(__file__).resolve().parents[1]
cases = [f"acv_case_{number:02d}.xlsx" for number in (2, 3, 5, 6)]

# Produce every ranking first. Labels never enter the ranking function.
rankings = {}
for filename in cases:
    print(f"Ranking {filename} with the unchanged rule...", flush=True)
    rankings[filename] = rank_file(ACV / "data/raw/train" / filename)

# The answer key is used only to grade completed rankings.
labels = pd.read_csv(ACV / "data/raw/Train_Labels.csv", dtype=str)
results = []
for filename, table in rankings.items():
    answer = labels.loc[labels["filename"].eq(filename), "faulty_car"]
    if len(answer) != 1:
        raise ValueError(f"Expected exactly one label for {filename}.")
    faulty = answer.iloc[0]
    cars = table["car"].tolist()
    if len(cars) != len(set(cars)) or faulty not in cars:
        raise ValueError(f"Incomplete or duplicate car identifiers for {filename}.")
    rank = cars.index(faulty) + 1
    score = (len(cars) - (rank - 1)) / len(cars)
    results.append({"file_id": filename, "ranked_cars": "|".join(cars),
                    "faulty_car": faulty, "faulty_car_rank": rank,
                    "official_rank_score": score,
                    "unscored_cars": "|".join(table.loc[table["evidence"].eq("unavailable"), "car"])})

report = pd.DataFrame(results)
print("\nWhole-case evaluation of the fixed rule (cases 02, 03, 05, 06):")
print(report.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"\nMean official rank score: {report['official_rank_score'].mean():.4f}")
print(f"First-place matches: {report['faulty_car_rank'].eq(1).sum()} of {len(report)} cases")
print("Cases 01 and 04 are excluded because they informed development.")
print("No fitting, model selection, or test data were used in this evaluation.")
destination = ACV / "outputs/baseline_evaluation_cases_02_03_05_06.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
report.to_csv(destination, index=False)
print(f"Saved evaluation report: {destination}")

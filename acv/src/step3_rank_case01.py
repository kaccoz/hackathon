"""An exploratory baseline for training case 01 only; no test data or fitting."""
from pathlib import Path
import re
import pandas as pd

ACV = Path(__file__).resolve().parents[1]
data = pd.read_excel(ACV / "data/raw/train/acv_case_01.xlsx")
cars = sorted({match.group(1) for column in data.columns
               if (match := re.fullmatch(r"Car (\d{2}) - Indoor Average Temperature", column))})
if not cars:
    raise ValueError("No car temperature columns found.")

results = []
for car in cars:
    prefix = f"Car {car} - "
    indoor = pd.to_numeric(data[prefix + "Indoor Average Temperature"], errors="raise")
    control = pd.to_numeric(data[prefix + "ACV Control Temperature (Cooling)"], errors="raise")
    usable = (data[prefix + "ACV Information Valid"].eq("Valid")
              & data[prefix + "ACV Running Mode"].eq("Automatic Cooling")
              & indoor.notna() & control.notna())
    if not usable.any():
        raise ValueError(f"Car {car} has no usable readings; cannot score it.")

    # Negative differences become zero. Zero differences remain in the average.
    positive_gap = (indoor[usable] - control[usable]).clip(lower=0)
    score = positive_gap.mean()
    results.append({"car": car, "usable_rows": int(usable.sum()),
                    "average_positive_gap": score})

# Use full precision to sort; car ID breaks exact ties reproducibly.
ranking = pd.DataFrame(results).sort_values(
    ["average_positive_gap", "car"], ascending=[False, True])
ranking.insert(0, "rank", range(1, len(ranking) + 1))
print("Training case 01: provisional baseline ranking")
print("Score = average of max(indoor - cooling control, 0) over usable rows")
print(ranking.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("\nRanking:", "|".join(ranking["car"]))
print("These scores are not probabilities. No labels or validation were used here.")

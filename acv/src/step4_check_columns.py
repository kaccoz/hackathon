"""Check training headers only; no scores, labels, or test files are read."""
from pathlib import Path
import re
import pandas as pd

ACV = Path(__file__).resolve().parents[1]
required = {
    "Indoor Average Temperature",
    "ACV Control Temperature (Cooling)",
    "ACV Information Valid",
    "ACV Running Mode",
}
files = sorted((ACV / "data/raw/train").glob("*.xlsx"))
if not files:
    raise ValueError("No training workbooks found.")

for path in files:
    # nrows=0 reads column headings without loading measurement rows.
    columns = pd.read_excel(path, nrows=0).columns
    per_car = {}
    for column in columns:
        match = re.fullmatch(r"Car (\d{2}) - (.+)", str(column))
        if match:
            per_car.setdefault(match.group(1), set()).add(match.group(2))
    print(f"\n{path.name}: {len(columns)} columns; {len(per_car)} cars")
    print("Car IDs:", ", ".join(sorted(per_car)))
    compatible = bool(per_car)
    for car, parameters in sorted(per_car.items()):
        missing = required - parameters
        if missing:
            compatible = False
            print(f"  Car {car} missing: {', '.join(sorted(missing))}")
    if compatible:
        print("All four baseline column names exist for every car.")
    else:
        print("Needs inspection before the baseline can be applied.")
        print("Available temperature/status/mode fields (union across cars):")
        parameters = set().union(*per_car.values()) if per_car else set()
        for parameter in sorted(parameters):
            if any(word in parameter.lower() for word in ["temperature", "valid", "mode"]):
                print(" -", parameter)

print("\nMatching headers alone does not confirm status values or usable data.")

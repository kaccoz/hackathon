"""Explain exclusions in case 04 without changing the filter or ranking cars."""
from pathlib import Path
import pandas as pd

ACV = Path(__file__).resolve().parents[1]
cars = [f"{number:02d}" for number in range(1, 9)]
fields = ["Passenger Cabin Temperature Detected Value", "Target Temperature Value",
          "ACV Running Mode"]
columns = [f"Car {car} - {field}" for car in cars for field in fields]
data = pd.read_excel(ACV / "data/raw/train/acv_case_04.xlsx", usecols=columns)
for car in cars:
    prefix = f"Car {car} - "
    cabin = pd.to_numeric(data[prefix + fields[0]], errors="raise")
    target = pd.to_numeric(data[prefix + fields[1]], errors="raise")
    mode = data[prefix + fields[2]]
    temperatures_present = cabin.notna() & target.notna()
    mode_matches = mode.isin(["Full Cooling", "Half Cooling"])
    print(f"\nCar {car} ({len(data):,} total rows)")
    print(f"Both temperature fields present: {temperatures_present.sum():,}")
    print(f"Mode matches Full Cooling or Half Cooling: {mode_matches.sum():,}")
    print(f"Both conditions together: {(temperatures_present & mode_matches).sum():,}")
    print("Recorded running-mode values and counts:")
    for value, count in mode.value_counts(dropna=False).items():
        print(f"  {value!r}: {count:,}")
print("\nThis report explains exclusions only. Missing readings are not zero gaps.")

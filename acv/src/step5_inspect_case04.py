"""Inspect a small sample using original field names; no mapping or predictions."""
from pathlib import Path
import pandas as pd

ACV = Path(__file__).resolve().parents[1]
fields = [
    "Car 01 - Passenger Cabin Temperature Detected Value",
    "Car 01 - Target Temperature Value",
    "Car 01 - ACV Running Mode",
    "Car 01 - ACV Operating Mode",
    "Car 01 - ACV Control Mode",
]
# A sample keeps this first inspection small. It cannot describe the whole file.
data = pd.read_excel(ACV / "data/raw/train/acv_case_04.xlsx",
                     usecols=["Time"] + fields, nrows=200)
print(f"Case 04: first {len(data)} rows only; Car 01 only")
print(f"Sample time range: {data['Time'].iloc[0]} to {data['Time'].iloc[-1]}")
for field in fields:
    print(f"\n{field}")
    print("First five recorded entries:", data[field].head().tolist())
    print("Missing entries in sample:", int(data[field].isna().sum()))
    if "Temperature" in field:
        numeric = pd.to_numeric(data[field], errors="raise")
        print("Sample minimum / maximum:", numeric.min(), "/", numeric.max())
    else:
        print("Sample value counts (including missing):")
        print(data[field].value_counts(dropna=False).to_string())
print("\nNames and values alone do not establish sensor equivalence or validity.")
print("No training labels, rankings, or test files were used.")

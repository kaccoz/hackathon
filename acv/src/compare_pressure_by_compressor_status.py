"""Compare case 04 pressure imbalance within raw compressor-status pairs.

Status meanings and compressor/system correspondence are unconfirmed. No code
is mapped to running or stopped. Includes finite pressure zeros. No labels,
model fitting, test inputs, or leak predictions are used.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ACV = Path(__file__).resolve().parents[1]
FIELDS = ["Refrigeration System 1 High Pressure Value",
          "Refrigeration System 2 High Pressure Value",
          "Compressor 1 Running", "Compressor 2 Running"]


def raw_label(value):
    # Include the type to distinguish text '1' from numeric 1.0.
    return "<missing>" if pd.isna(value) else f"{type(value).__name__}:{value!r}"


def main():
    source = ACV / "data/raw/train/acv_case_04.xlsx"
    print(f"Reading {source.name}; please allow about a minute...", flush=True)
    headers = pd.read_excel(source, nrows=0).columns
    cars = sorted({m.group(1) for c in headers if (m := re.fullmatch(r"Car (\d{2}) - .+", str(c)))})
    if not cars:
        raise ValueError("No car columns found.")
    columns = [f"Car {car} - {field}" for car in cars for field in FIELDS]
    missing = set(columns) - set(headers)
    if missing:
        raise ValueError(f"Expected columns missing: {sorted(missing)}")
    data = pd.read_excel(source, usecols=columns)
    records = []
    for car in cars:
        prefix = f"Car {car} - "
        high1 = pd.to_numeric(data[prefix + FIELDS[0]], errors="raise")
        high2 = pd.to_numeric(data[prefix + FIELDS[1]], errors="raise")
        paired = np.isfinite(high1) & np.isfinite(high2)
        rows = pd.DataFrame({
            "compressor_1_raw": data[prefix + FIELDS[2]].map(raw_label),
            "compressor_2_raw": data[prefix + FIELDS[3]].map(raw_label),
            "imbalance": (high1 - high2).abs().where(paired),
            "high1": high1.where(paired), "high2": high2.where(paired),
            "pair_with_zero": paired & (high1.eq(0) | high2.eq(0)),
        })
        for (status1, status2), group in rows.groupby(["compressor_1_raw", "compressor_2_raw"], sort=True):
            records.append({
                "car": car, "compressor_1_raw": status1, "compressor_2_raw": status2,
                "status_rows": len(group), "paired_pressure_rows": int(group["imbalance"].notna().sum()),
                "pairs_with_zero": int(group["pair_with_zero"].sum()),
                "median_high1": group["high1"].median(), "median_high2": group["high2"].median(),
                "median_absolute_difference": group["imbalance"].median(),
                "mean_absolute_difference": group["imbalance"].mean(),
            })
    report = pd.DataFrame(records)
    print("\nGroups use recorded status values, not interpreted operating states.")
    for (status1, status2), group in report.groupby(["compressor_1_raw", "compressor_2_raw"], sort=True):
        print(f"\nCOMPRESSOR 1 = {status1}; COMPRESSOR 2 = {status2}")
        print(group.drop(columns=["compressor_1_raw", "compressor_2_raw"]).to_string(
            index=False, na_rep="unavailable", float_format=lambda value: f"{value:.3f}"), flush=True)
    destination = ACV / "outputs/case_04_pressure_by_compressor_status.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False, na_rep="unavailable")
    print(f"\nSaved table: {destination}")
    print("Counts are rows, not durations. Missing status remains its own group.")
    print("All pressure zeros retained; units, status semantics, and system pairing unconfirmed.")
    print("Matching status pairs across cars may occur at different times and cooling demands.")
    print("No validated leak conclusions follow from this descriptive comparison alone.")


if __name__ == "__main__":
    main()

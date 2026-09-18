"""Inspect raw pressure readings in training case 04; no physical thresholds.

Pressure units, scaling, and status meanings remain unconfirmed. Zeros stay in
the summaries. No labels, test data, leak scores, or predictions are used.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ACV = Path(__file__).resolve().parents[1]
FIELDS = [f"Refrigeration System {system} {side} Pressure Value"
          for system in (1, 2) for side in ("High", "Low")]


def main():
    source = ACV / "data/raw/train/acv_case_04.xlsx"
    columns = [f"Car {car:02d} - {field}" for car in range(1, 9) for field in FIELDS]
    print(f"Reading {source.name}; please allow about a minute...", flush=True)
    headers = pd.read_excel(source, nrows=0).columns
    missing = sorted(set(columns) - set(headers))
    if missing:
        raise ValueError(f"Pressure headers missing: {missing}")
    data = pd.read_excel(source, usecols=columns)
    records = []
    for car in range(1, 9):
        for field in FIELDS:
            raw = data[f"Car {car:02d} - {field}"]
            numeric = pd.to_numeric(raw, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            records.append({
                "car": f"{car:02d}", "field": field,
                "numeric_rows": len(finite), "missing_rows": int(raw.isna().sum()),
                "non_numeric_rows": int((raw.notna() & numeric.isna()).sum()),
                "infinite_rows": int(np.isinf(numeric).sum()),
                "zero_rows": int(finite.eq(0).sum()),
                "minimum": finite.min(), "median": finite.median(), "maximum": finite.max(),
            })
    report = pd.DataFrame(records)
    for car in report["car"].unique():
        print(f"\nCAR {car} - raw values; units and scaling unconfirmed")
        print(report.loc[report["car"].eq(car)].drop(columns="car").to_string(
            index=False, na_rep="unavailable", float_format=lambda value: f"{value:.3f}"), flush=True)
    destination = ACV / "outputs/case_04_pressure_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False, na_rep="unavailable")
    print(f"\nSaved: {destination}")
    print("All recorded operating modes are included; zeros have not been removed.")
    print("These summaries are not evidence of low refrigerant without field definitions and operating context.")


if __name__ == "__main__":
    main()

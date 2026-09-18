"""Compare case 04 pressure imbalance on jointly observed (1,1) status rows.

Uses Cars 01-04, which have pressure data in prior inspection. Cars 05-08 remain
unavailable. Status 1 semantics, system/compressor pairing, and units/scaling
are unconfirmed. No labels, test data, or fault thresholds are used.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ACV = Path(__file__).resolve().parents[1]
CARS = ("01", "02", "03", "04")
FIELDS = (
    "Refrigeration System 1 High Pressure Value",
    "Refrigeration System 2 High Pressure Value",
    "Compressor 1 Running",
    "Compressor 2 Running",
)


def main():
    source = ACV / "data/raw/train/acv_case_04.xlsx"
    columns = [f"Car {car} - {field}" for car in CARS for field in FIELDS]
    print(f"Reading {source.name}; please allow about a minute...", flush=True)
    data = pd.read_excel(source, usecols=["Time"] + columns)
    time = pd.to_datetime(data["Time"], errors="raise")
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Inspect missing, duplicate, or out-of-order timestamps first.")
    eligible, high1, high2, gaps = {}, {}, {}, {}
    for car in CARS:
        prefix = f"Car {car} - "
        high1[car] = pd.to_numeric(data[prefix + FIELDS[0]], errors="raise")
        high2[car] = pd.to_numeric(data[prefix + FIELDS[1]], errors="raise")
        status1 = data[prefix + FIELDS[2]]
        status2 = data[prefix + FIELDS[3]]
        # Exact numeric code match. No inference that code 1 means running.
        eligible[car] = (status1.eq(1) & status2.eq(1)
                         & np.isfinite(high1[car]) & np.isfinite(high2[car]))
        gaps[car] = (high1[car] - high2[car]).abs()
        print(f"Car {car}: {eligible[car].sum():,} eligible rows before time matching", flush=True)
    common = pd.DataFrame(eligible).all(axis=1)
    count = int(common.sum())
    print(f"\nShared eligible timestamps: {count:,} of {len(data):,} total rows")
    rows = []
    for car in CARS:
        gap = gaps[car][common]
        rows.append({"car": car, "individual_eligible_rows": int(eligible[car].sum()),
                     "shared_rows": count,
                     "pairs_with_zero": int((common & (high1[car].eq(0) | high2[car].eq(0))).sum()),
                     "median_high1": high1[car][common].median(),
                     "median_high2": high2[car][common].median(),
                     "median_absolute_difference": gap.median(),
                     "mean_absolute_difference": gap.mean()})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, na_rep="unavailable", float_format=lambda x: f"{x:.3f}"))
    if count:
        print(f"First shared timestamp: {time[common].iloc[0]}")
        print(f"Last shared timestamp: {time[common].iloc[-1]}")
        print("These endpoints do not imply continuously observed or eligible operation between them.")
    else:
        print("No shared eligible rows. No comparison is possible under this rule.")
    output = ACV / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "case_04_pressure_matched_times_summary.csv"
    summary.to_csv(summary_path, index=False, na_rep="unavailable")
    # Save exact selected timestamps and per-car differences for review.
    matched = pd.DataFrame({"Time": time[common]})
    for car in CARS:
        matched[f"Car {car} - Absolute High Pressure Difference"] = gaps[car][common]
    detail_path = output / "case_04_pressure_matched_times_readings.csv"
    matched.to_csv(detail_path, index=False)
    print(f"\nSaved summary: {summary_path}\nSaved matched readings: {detail_path}")
    print("All four cars use identical timestamps. Numeric zeros retained; missing values excluded.")
    print("Counts are rows, not durations. Other operating conditions may still differ.")
    print("Exploratory comparison only, not an independent validation or leak prediction.")


if __name__ == "__main__":
    main()

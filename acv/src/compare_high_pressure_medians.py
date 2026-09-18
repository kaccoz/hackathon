"""Compare case 04 high-pressure medians across cars, separately by system.

Descriptive comparison only: no physical thresholds, fault classification,
labels, or test data. Includes all operating modes and finite numeric zeros.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ACV = Path(__file__).resolve().parents[1]


def main():
    source = ACV / "data/raw/train/acv_case_04.xlsx"
    print(f"Reading {source.name}; please allow about a minute...", flush=True)
    cars = [f"{number:02d}" for number in range(1, 9)]
    columns = [f"Car {car} - Refrigeration System {system} High Pressure Value"
               for system in (1, 2) for car in cars]
    data = pd.read_excel(source, usecols=columns)
    reports = []
    for system in (1, 2):
        rows = []
        for car in cars:
            raw = data[f"Car {car} - Refrigeration System {system} High Pressure Value"]
            numeric = pd.to_numeric(raw, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            rows.append({"car": car, "system": system,
                         "numeric_rows": len(finite),
                         "missing_rows": int(raw.isna().sum()),
                         "non_numeric_rows": int((raw.notna() & numeric.isna()).sum()),
                         "infinite_rows": int(np.isinf(numeric).sum()),
                         "zero_rows": int(finite.eq(0).sum()),
                         "median_high_pressure": finite.median()})
        report = pd.DataFrame(rows)
        for index, row in report.iterrows():
            # Every other car contributes one median, regardless of row count.
            others = report.loc[report["car"].ne(row["car"]), "median_high_pressure"].dropna()
            reference = others.median()
            report.loc[index, "available_peer_cars"] = len(others)
            report.loc[index, "other_cars_median"] = reference
            report.loc[index, "difference_from_peers"] = row["median_high_pressure"] - reference
            report.loc[index, "percent_below_peers"] = (
                100 * (reference - row["median_high_pressure"]) / reference
                if pd.notna(reference) and reference > 0 else np.nan)
        report["available_peer_cars"] = report["available_peer_cars"].astype(int)
        report = report.sort_values(["median_high_pressure", "car"], na_position="last")
        print(f"\nSYSTEM {system}: lowest car median first (not a leak ranking)")
        print(report[["car", "numeric_rows", "zero_rows", "median_high_pressure",
                      "available_peer_cars", "other_cars_median", "difference_from_peers",
                      "percent_below_peers"]].to_string(
                          index=False, na_rep="unavailable", float_format=lambda x: f"{x:.3f}"), flush=True)
        reports.append(report)
    output = ACV / "outputs/case_04_high_pressure_median_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(reports, ignore_index=True).to_csv(output, index=False, na_rep="unavailable")
    print(f"\nSaved: {output}")
    print("Negative difference_from_peers means lower than the other cars' median reference.")
    print("Positive percent_below_peers means below that reference; it is not a probability.")
    print("Units/scaling unconfirmed. All modes and finite zeros included; missing values excluded.")
    print("Car medians may cover different times and operating conditions.")
    print("No threshold for 'unusually low' has been established.")


if __name__ == "__main__":
    main()

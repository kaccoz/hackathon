"""Descriptive valve/temperature alignment in case 04, Cars 01 and 04.

Warm means cabin minus target > 0, an exploratory definition, not a fault
threshold. Counts are recorded samples, not durations. No causal claims.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ACV = Path(__file__).resolve().parents[1]


def main():
    cars = ["01", "04"]
    valves = [f"Fresh Air Valve {name} Closed" for name in ("A1", "A2", "B1", "B2")]
    fields = valves + ["Passenger Cabin Temperature Detected Value", "Target Temperature Value",
                       "Fresh Air Temperature Detected Value", "ACV Running Mode",
                       "Compressor 1 Running", "Compressor 2 Running"]
    print("Reading case 04...", flush=True)
    data = pd.read_excel(ACV / "data/raw/train/acv_case_04.xlsx",
                        usecols=["Time"] + [f"Car {car} - {field}" for car in cars for field in fields])
    time = pd.to_datetime(data["Time"], errors="raise")
    result = pd.DataFrame({"Time": time})
    masks = []
    for car in cars:
        prefix = f"Car {car} - "
        statuses = data[[prefix + field for field in valves]]
        known = statuses.isin(["Open", "Closed"]).all(axis=1)
        cabin = pd.to_numeric(data[prefix + "Passenger Cabin Temperature Detected Value"], errors="raise")
        target = pd.to_numeric(data[prefix + "Target Temperature Value"], errors="raise")
        fresh = pd.to_numeric(data[prefix + "Fresh Air Temperature Detected Value"], errors="raise")
        result[f"{car}_gap"] = cabin - target
        result[f"{car}_open"] = statuses.eq("Open").sum(axis=1).where(known)
        result[f"{car}_fresh"] = fresh
        masks.append(known & np.isfinite(cabin) & np.isfinite(target) & np.isfinite(fresh)
                     & data[prefix + "ACV Running Mode"].eq("Full Cooling")
                     & data[prefix + "Compressor 1 Running"].eq(1)
                     & data[prefix + "Compressor 2 Running"].eq(1))
    common = masks[0] & masks[1]
    selected = result.loc[common].copy()
    print(f"Shared eligible timestamps for Cars 01 and 04: {len(selected):,}")
    print("Warm definition: recorded cabin-minus-target gap > 0.")
    summaries = []
    for car in cars:
        for label, mask in [("above_target", selected[f"{car}_gap"].gt(0)),
                            ("at_or_below_target", selected[f"{car}_gap"].le(0))]:
            group = selected.loc[mask]
            row = {"car": car, "period": label, "rows": len(group),
                   "mean_open_valves": group[f"{car}_open"].mean(),
                   "fraction_with_3_or_4_open": group[f"{car}_open"].ge(3).mean(),
                   "mean_gap": group[f"{car}_gap"].mean()}
            summaries.append(row)
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nCar 04 gap by open count on shared eligible rows:")
    print(selected.groupby("04_open")["04_gap"].agg(["size", "mean", "median"]).to_string())
    windows = []
    for start, end in [("2023-08-16 12:00", "2023-08-16 20:00"),
                       ("2023-08-17 12:00", "2023-08-17 18:00"),
                       ("2023-08-18 12:00", "2023-08-18 20:00")]:
        group = selected.loc[selected.Time.ge(start) & selected.Time.lt(end)]
        row = {"start": start, "end_exclusive": end, "rows": len(group)}
        for car in cars:
            row[f"{car}_mean_open"] = group[f"{car}_open"].mean()
            row[f"{car}_mean_gap"] = group[f"{car}_gap"].mean()
        row["fraction_car04_more_open"] = group["04_open"].gt(group["01_open"]).mean()
        windows.append(row)
    print("\nPreviously discussed windows, shared eligible rows only:")
    print(pd.DataFrame(windows).to_string(index=False))
    print("\nDuring Car 04 above-target rows:")
    warm = selected.loc[selected["04_gap"].gt(0)]
    print("Open valve count distribution:", warm["04_open"].value_counts().sort_index().to_dict())
    print("Fraction Car04 more open than Car01:", warm["04_open"].gt(warm["01_open"]).mean())
    print("Fraction equal:", warm["04_open"].eq(warm["01_open"]).mean())
    out = ACV / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out / "case_04_valves_and_warm_periods_readings.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "case_04_valves_and_warm_periods_summary.csv", index=False)
    pd.DataFrame(windows).to_csv(out / "case_04_valves_and_warm_windows.csv", index=False)
    print("Saved readings, summary and window tables. Association only; no airflow or causal inference.")


if __name__ == "__main__":
    main()

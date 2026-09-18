"""Compare within-car high-pressure differences in training case 04.

Descriptive only: all operating modes and finite zeros retained. No labels,
test data, physical thresholds, or leak ranking. Units/scaling unconfirmed.
"""
from pathlib import Path
import os
import re

ACV = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ACV / "artifacts/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def main():
    source = ACV / "data/raw/train/acv_case_04.xlsx"
    print(f"Reading {source.name}; please allow about a minute...", flush=True)
    headers = pd.read_excel(source, nrows=0).columns
    cars = sorted({m.group(1) for c in headers if (m := re.fullmatch(r"Car (\d{2}) - .+", str(c)))})
    if not cars:
        raise ValueError("No car identifiers found.")
    columns = [f"Car {car} - Refrigeration System {system} High Pressure Value"
               for car in cars for system in (1, 2)]
    missing = set(["Time"] + columns) - set(headers)
    if missing:
        raise ValueError(f"Missing headers: {sorted(missing)}")
    data = pd.read_excel(source, usecols=["Time"] + columns)
    time = pd.to_datetime(data["Time"], errors="raise")
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Inspect missing, duplicate, or out-of-order timestamps first.")
    intervals = time.diff().dropna()
    if intervals.empty:
        raise ValueError("At least two timestamps are needed.")
    typical_interval = intervals.mode().min()
    breaks = time.loc[time.diff() > typical_interval] - typical_interval
    plot_times = pd.DatetimeIndex(time).union(pd.DatetimeIndex(breaks)).sort_values()
    differences, records = {}, []
    for car in cars:
        high1 = pd.to_numeric(data[f"Car {car} - Refrigeration System 1 High Pressure Value"], errors="raise")
        high2 = pd.to_numeric(data[f"Car {car} - Refrigeration System 2 High Pressure Value"], errors="raise")
        paired = np.isfinite(high1) & np.isfinite(high2)
        # Take each simultaneous difference BEFORE summarising, so reversals
        # in which system has higher pressure cannot cancel each other out.
        gap = (high1 - high2).abs().where(paired)
        differences[car] = gap
        records.append({"car": car, "paired_rows": int(paired.sum()),
                        "pairs_with_a_zero": int((paired & (high1.eq(0) | high2.eq(0))).sum()),
                        "median_absolute_difference": gap.median(),
                        "mean_absolute_difference": gap.mean()})
    gaps = pd.DataFrame(differences)
    report = pd.DataFrame(records)

    # For a fairer side-by-side summary, also use the same rows for all cars
    # that have any pressure pairs. Entirely unobserved cars remain unavailable.
    observed_cars = [car for car in cars if gaps[car].notna().any()]
    common = gaps[observed_cars].notna().all(axis=1) if observed_cars else pd.Series(False, index=data.index)
    for index, row in report.iterrows():
        car = row["car"]
        usable_common = common & gaps[car].notna()
        report.loc[index, "common_rows"] = int(usable_common.sum())
        report.loc[index, "median_difference_common_rows"] = gaps.loc[usable_common, car].median()
    report["common_rows"] = report["common_rows"].astype(int)
    for index, row in report.iterrows():
        peers = report.loc[report["car"].ne(row["car"]), "median_difference_common_rows"].dropna()
        reference = peers.median()
        report.loc[index, "other_cars_median_difference"] = reference
        report.loc[index, "excess_over_other_cars"] = row["median_difference_common_rows"] - reference
    print("\nPressure imbalance = absolute value of (high pressure 1 - high pressure 2)")
    print("Common rows: same timestamps across observed cars " + ", ".join(observed_cars))
    print(report.to_string(index=False, na_rep="unavailable", float_format=lambda x: f"{x:.3f}"))

    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    styles = ["-", "--", "-.", ":"]
    for index, car in enumerate(cars):
        if car not in observed_cars:
            continue
        values = pd.Series(gaps[car].to_numpy(), index=time).reindex(plot_times)
        ax.plot(values.index, values, label=f"Car {car}", color=plt.get_cmap("tab10")(index),
                linestyle=styles[index % len(styles)], linewidth=0.9, alpha=0.8)
    ax.set_title("Case 04: within-car high-pressure imbalance compared across cars\n"
                 "Absolute difference between Systems 1 and 2 at each timestamp")
    ax.set_ylabel("Absolute pressure difference (units/scaling unconfirmed)")
    ax.set_xlabel("Recorded time in 2023 (month-day hour:minute)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(alpha=0.2)
    if observed_cars:
        ax.legend(ncol=4)
    else:
        ax.text(0.5, 0.5, "No paired pressure readings", transform=ax.transAxes, ha="center")
    absent = [car for car in cars if car not in observed_cars]
    note = "All operating modes; zeros retained; raw differences without smoothing."
    if absent:
        note += "\nUnavailable cars: " + ", ".join(absent) + ". Missing values are not zero differences."
    fig.supxlabel(note, fontsize=9)
    output = ACV / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    chart = output / "case_04_high_pressure_imbalance.png"
    table = output / "case_04_high_pressure_imbalance.csv"
    fig.savefig(chart, dpi=150)
    plt.close(fig)
    report.to_csv(table, index=False, na_rep="unavailable")
    print(f"\nSaved chart: {chart}\nSaved table: {table}")
    print("Larger differences are descriptive, not validated leak predictions.")
    print("Common timestamps do not establish equivalent operating states.")


if __name__ == "__main__":
    main()

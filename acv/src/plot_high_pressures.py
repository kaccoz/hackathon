"""Plot System 1 and System 2 high pressures within each car in case 04.

Raw recorded values only. Units and scaling are unconfirmed; all operating modes
and numeric zeros are retained. No labels, rankings, or test data are used.
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
    cars = sorted({match.group(1) for column in headers
                   if (match := re.fullmatch(r"Car (\d{2}) - .+", str(column)))})
    if not cars:
        raise ValueError("No car identifiers found in headers.")
    columns = [f"Car {car} - Refrigeration System {system} High Pressure Value"
               for car in cars for system in (1, 2)]
    missing = sorted(set(["Time"] + columns) - set(headers))
    if missing:
        raise ValueError(f"Expected columns missing: {missing}")
    data = pd.read_excel(source, usecols=["Time"] + columns)
    time = pd.to_datetime(data["Time"], errors="raise")
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Inspect missing, duplicate, or out-of-order timestamps first.")
    intervals = time.diff().dropna()
    if intervals.empty:
        raise ValueError("At least two timestamps are needed.")
    typical_interval = intervals.mode().min()
    # Break lines across absent timestamps without filling in any pressure data.
    breaks = time.loc[time.diff() > typical_interval] - typical_interval
    plot_times = pd.DatetimeIndex(time).union(pd.DatetimeIndex(breaks)).sort_values()
    fig, axes = plt.subplots(len(cars), 1, figsize=(13, 15), sharex=True,
                             sharey=True, squeeze=False, layout="constrained")
    for index, car in enumerate(cars):
        ax = axes[index, 0]
        unavailable = []
        for system, color, style in [(1, "#1769aa", "-"), (2, "#d97706", "--")]:
            column = f"Car {car} - Refrigeration System {system} High Pressure Value"
            numeric = pd.to_numeric(data[column], errors="raise")
            values = numeric.where(np.isfinite(numeric))
            print(f"Car {car}, System {system}: {values.notna().sum():,} finite readings; "
                  f"{values.eq(0).sum():,} zeros retained", flush=True)
            readings = pd.Series(values.to_numpy(), index=time).reindex(plot_times)
            ax.plot(readings.index, readings, color=color, linestyle=style,
                    linewidth=0.9, alpha=0.85, label=f"System {system} high pressure")
            if not values.notna().any():
                unavailable.append(str(system))
        ax.set_ylabel(f"Car {car}")
        ax.grid(alpha=0.2)
        if unavailable:
            ax.text(0.5, 0.75, "No finite readings: System " + ", ".join(unavailable),
                    transform=ax.transAxes, ha="center", fontsize=9)
    axes[0, 0].legend(loc="upper left", ncol=2, fontsize=9)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1, 0].set_xlabel("Recorded time in 2023 (month-day hour:minute)")
    fig.supylabel("Recorded high-pressure value (units and scaling unconfirmed)")
    fig.suptitle("Case 04: System 1 versus System 2 high pressure within each car\n"
                 "Raw readings; all operating modes; zeros retained; missing readings left blank")
    destination = ACV / "outputs/case_04_high_pressures_by_car.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    print(f"Saved plot: {destination}")
    print("Pressure differences alone do not establish a leak or equivalent operating conditions.")


if __name__ == "__main__":
    main()

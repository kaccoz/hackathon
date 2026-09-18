"""Explore case 04 with explicitly provisional field interpretations."""
from pathlib import Path
import os
import re

ACV = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ACV / "artifacts/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

source = ACV / "data/raw/train/acv_case_04.xlsx"
parameters = {"Passenger Cabin Temperature Detected Value", "Target Temperature Value",
              "ACV Running Mode"}
headers = pd.read_excel(source, nrows=0).columns
car_fields = {}
for column in headers:
    match = re.fullmatch(r"Car (\d{2}) - (.+)", str(column))
    if match:
        car_fields.setdefault(match.group(1), {})[match.group(2)] = column
if not car_fields:
    raise ValueError("No car columns found.")
selected = ["Time"]
for car, fields in sorted(car_fields.items()):
    missing = parameters - fields.keys()
    if missing:
        raise ValueError(f"Car {car} missing {sorted(missing)}")
    selected.extend(fields[name] for name in sorted(parameters))
data = pd.read_excel(source, usecols=selected)
time = pd.to_datetime(data["Time"], errors="raise")
if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
    raise ValueError("Inspect missing, duplicate, or out-of-order timestamps first.")
print(f"File: {source.name}; rows: {len(data):,}")
print("Most common recorded time gaps:")
print(time.diff().value_counts().head().to_string())
# Use the most frequent observed interval ONLY to break lines across missing times.
# Preserve every original timestamp and never fill in temperature readings.
interval = time.diff().dropna().mode().iloc[0]
breaks = time.loc[time.diff() > interval] - interval
plot_times = pd.DatetimeIndex(time).union(pd.DatetimeIndex(breaks)).sort_values()
print("Working assumptions: cabin-minus-target; Full Cooling or Half Cooling only.")
print("No Information Valid field is available; Operating Mode is not a substitute.")

cars = sorted(car_fields)
fig, axes = plt.subplots(len(cars), 1, figsize=(12, 14), sharex=True,
                         sharey=True, squeeze=False, layout="constrained")
for index, car in enumerate(cars):
    fields = car_fields[car]
    cabin = pd.to_numeric(data[fields["Passenger Cabin Temperature Detected Value"]], errors="raise")
    target = pd.to_numeric(data[fields["Target Temperature Value"]], errors="raise")
    running = data[fields["ACV Running Mode"]]
    keep = running.isin(["Full Cooling", "Half Cooling"]) & cabin.notna() & target.notna()
    print(f"Car {car}: {keep.sum():,} of {len(data):,} rows kept")
    gap = (cabin - target).where(keep)
    readings = pd.Series(gap.to_numpy(), index=time).reindex(plot_times)
    ax = axes[index, 0]
    ax.plot(readings.index, readings, color=plt.get_cmap("tab10")(index), linewidth=0.9)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.set_ylabel(f"Car {car}")
    ax.grid(alpha=0.2)
    if not keep.any():
        ax.text(0.5, 0.6, "No readings pass this filter - gap unknown", transform=ax.transAxes,
                ha="center", fontsize=10, color="#8b3800")
axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
axes[-1, 0].set_xlabel("Recorded time in 2023 (month-day hour:minute)")
fig.supylabel("Cabin minus target (recorded units, unconfirmed)")
fig.suptitle("Case 04: exploratory cabin-minus-target difference\nFull/Half Cooling only; field interpretation provisional; validity unconfirmed")
destination = ACV / "outputs/case_04_cooling_gaps.png"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=140)
plt.close(fig)
print(f"Saved plot: {destination}")
print("No rankings, training labels, or test data used.")

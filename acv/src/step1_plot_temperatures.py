"""Lesson 1: inspect and plot one training case; no predictions or fitting."""
from pathlib import Path
import os
import re

# Keep plotting caches inside this workspace.
ACV = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ACV / "artifacts" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# A DataFrame is a table: rows are observations, columns are measurements.
source = ACV / "data" / "raw" / "train" / "acv_case_01.xlsx"
data = pd.read_excel(source, sheet_name="Sheet1")
print(f"File: {source.name}")
print(f"Rows: {len(data):,}; columns: {len(data.columns)}")
time = pd.to_datetime(data["Time"], errors="raise")
if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
    raise ValueError("Missing or out-of-order timestamps: inspect before plotting.")
print(f"Time range: {time.min()} to {time.max()}")
print("Most common time gaps:")
print(time.diff().value_counts().head().to_string())
# Empty positions break the plotted lines across absent timestamps.
# No temperature values are filled in.
plot_times = pd.date_range(time.min(), time.max(), freq="30s").union(pd.DatetimeIndex(time))

# Match the names themselves; columns are not grouped in car order.
groups = {}
for parameter in ["Indoor Average Temperature", "Outdoor Average Temperature"]:
    columns = {}
    for column in data.columns:
        match = re.fullmatch(r"Car (\d{2}) - " + re.escape(parameter), column)
        if match:
            columns[match.group(1)] = column
    if not columns:
        raise ValueError(f"No columns found for {parameter}")
    groups[parameter] = dict(sorted(columns.items()))
print("Indoor car IDs:", ", ".join(groups["Indoor Average Temperature"]))

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
colors = plt.get_cmap("tab10")
for ax, (parameter, columns) in zip(axes, groups.items()):
    for index, (car, column) in enumerate(columns.items()):
        values = pd.to_numeric(data[column], errors="raise")
        print(f"{car} {parameter}: {values.isna().sum()} missing values")
        readings = pd.Series(values.to_numpy(), index=time).reindex(plot_times)
        ax.plot(readings.index, readings, label=f"Car {car}", color=colors(index), linewidth=0.9, alpha=0.8)
    ax.set_title(parameter)
    # The kit does not explicitly specify temperature units or scaling.
    ax.set_ylabel("Recorded value (unit unconfirmed)")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, fontsize=8)
axes[-1].set_xlabel("Time recorded in workbook")
fig.suptitle("Training case 01: temperatures over time\nRaw readings, without smoothing or fault predictions")
destination = ACV / "outputs" / "case_01_temperatures.png"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=160)
plt.close(fig)
print(f"Saved plot: {destination}")

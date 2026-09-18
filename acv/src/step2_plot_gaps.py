"""Plot an exploratory difference, not a leak prediction or confirmed target error."""
from pathlib import Path
import os
import re

ACV = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ACV / "artifacts" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 1. Read only the first training case. Leave its original values untouched.
data = pd.read_excel(ACV / "data/raw/train/acv_case_01.xlsx")
time = pd.to_datetime(data["Time"], errors="raise")
if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
    raise ValueError("Check missing, duplicate, or out-of-order timestamps first.")
cars = sorted({m.group(1) for c in data.columns
               if (m := re.fullmatch(r"Car (\d{2}) - Indoor Average Temperature", c))})
if not cars:
    raise ValueError("No indoor temperature columns found.")

# Keep gaps in time visible instead of connecting across absent readings.
plot_times = pd.date_range(time.min(), time.max(), freq="30s").union(pd.DatetimeIndex(time))
fig, axes = plt.subplots(len(cars), 1, figsize=(12, 14), sharex=True,
                         sharey=True, squeeze=False, layout="constrained")
print("Exploratory gap = indoor reading - cooling control value")
print("Kept: Valid + Automatic Cooling + both numeric readings present")
for index, car in enumerate(cars):
    prefix = f"Car {car} - "
    indoor = pd.to_numeric(data[prefix + "Indoor Average Temperature"], errors="raise")
    control = pd.to_numeric(data[prefix + "ACV Control Temperature (Cooling)"], errors="raise")

    # 2. Each car has its own filter. A missing status does not pass it.
    keep = (data[prefix + "ACV Information Valid"].eq("Valid")
            & data[prefix + "ACV Running Mode"].eq("Automatic Cooling")
            & indoor.notna() & control.notna())

    # 3. Subtract, then hide excluded readings in this derived series only.
    gap = (indoor - control).where(keep)
    print(f"Car {car}: {keep.sum():,} of {len(data):,} rows kept")
    series = pd.Series(gap.to_numpy(), index=time).reindex(plot_times)

    # 4. Use one panel per car, with the same vertical scale for comparison.
    ax = axes[index, 0]
    ax.plot(series.index, series, color=plt.get_cmap("tab10")(index), linewidth=0.9)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_ylabel(f"Car {car}")
    ax.grid(alpha=0.2)

axes[-1, 0].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
axes[-1, 0].set_xlabel("Recorded time in 2023 (month-day hour:minute)")
fig.supylabel("Indoor minus cooling control (recorded units, unconfirmed)")
fig.suptitle("Case 01: indoor-minus-control difference\nValid, Automatic Cooling readings only; no smoothing or predictions")
destination = ACV / "outputs/case_01_cooling_gaps.png"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=140)
plt.close(fig)
print(f"Saved plot: {destination}")

"""Align case 04 pressure and raw compressor statuses without interpreting codes.

Compressor-to-refrigeration-system correspondence is unconfirmed. Numeric status
codes are displayed as categories, not mapped to assumed running/stopped states.
No labels, rankings, interpolation of missing values, or test data are used.
"""
from pathlib import Path
import argparse
import os

ACV = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ACV / "artifacts/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--car", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    car = f"{args.car:02d}"
    source = ACV / "data/raw/train/acv_case_04.xlsx"
    prefix = f"Car {car} - "
    pressures = [prefix + f"Refrigeration System {system} High Pressure Value" for system in (1, 2)]
    statuses = [prefix + f"Compressor {compressor} Running" for compressor in (1, 2)]
    columns = ["Time"] + pressures + statuses
    print(f"Reading {source.name}, Car {car}; please allow about a minute...", flush=True)
    headers = pd.read_excel(source, nrows=0).columns
    missing = sorted(set(columns) - set(headers))
    if missing:
        raise ValueError(f"Expected columns missing: {missing}")
    data = pd.read_excel(source, usecols=columns)
    time = pd.to_datetime(data["Time"], errors="raise")
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Inspect missing, duplicate, or out-of-order timestamps first.")
    intervals = time.diff().dropna()
    if intervals.empty:
        raise ValueError("At least two timestamps are needed.")
    typical_interval = intervals.mode().min()
    breaks = time.loc[time.diff() > typical_interval] - typical_interval
    plot_times = pd.DatetimeIndex(time).union(pd.DatetimeIndex(breaks)).sort_values()

    # Use one shared category mapping so identical raw values align in both panels.
    # repr distinguishes numeric codes from text codes; it assigns no meaning.
    labels = {column: data[column].map(lambda value: None if pd.isna(value) else repr(value))
              for column in statuses}
    categories = sorted({value for series in labels.values() for value in series.dropna()})
    category_positions = {value: index for index, value in enumerate(categories)}
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3, 1.3]}, layout="constrained")
    for column, system, color, style in zip(pressures, (1, 2), ("#1769aa", "#d97706"), ("-", "--")):
        numeric = pd.to_numeric(data[column], errors="raise")
        values = numeric.where(np.isfinite(numeric))
        series = pd.Series(values.to_numpy(), index=time).reindex(plot_times)
        axes[0].plot(series.index, series, label=f"System {system} high pressure",
                     color=color, linestyle=style, linewidth=0.9, alpha=0.85)
        print(f"System {system} pressure: {values.notna().sum():,} finite readings; "
              f"{values.eq(0).sum():,} zeros retained", flush=True)
    axes[0].set_ylabel("Recorded pressure\n(units/scaling unconfirmed)")
    axes[0].legend(loc="upper left")
    if not any(np.isfinite(pd.to_numeric(data[column], errors="raise")).any() for column in pressures):
        axes[0].text(0.5, 0.5, "No finite pressure readings", transform=axes[0].transAxes, ha="center")
    for panel, (column, compressor) in enumerate(zip(statuses, (1, 2)), start=1):
        ax = axes[panel]
        positions = labels[column].map(category_positions)
        # Mark recorded samples only: no implied status continuation across gaps.
        ax.scatter(time, positions, marker=".", s=5, color="#475569", alpha=0.65)
        ax.set_yticks(list(category_positions.values()), list(category_positions.keys()))
        ax.set_ylim(-0.5, max(0, len(categories) - 1) + 0.5)
        ax.set_ylabel(f"Compressor {compressor}\nraw running status")
        if not positions.notna().any():
            ax.text(0.5, 0.5, "No recorded status", transform=ax.transAxes, ha="center")
        print(f"\nCompressor {compressor}: recorded status counts")
        print(labels[column].fillna("<missing>").value_counts().to_string())
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1].set_xlabel("Recorded time in 2023 (month-day hour:minute)")
    fig.suptitle(f"Case 04, Car {car}: pressure and compressor status at the same times\n"
                 "Status codes shown as recorded; system/compressor pairing unconfirmed")
    fig.supxlabel("All operating modes included. Missing readings left blank. Status dots show observed samples only.", fontsize=9)
    output = ACV / f"outputs/case_04_car_{car}_pressure_and_compressors.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {output}")
    print("Alignment supports investigation; it does not establish a cause or a refrigerant leak.")


if __name__ == "__main__":
    main()

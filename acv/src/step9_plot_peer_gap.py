"""Display per-car control and peer gaps without fitting or changing rankings."""
from pathlib import Path
import argparse
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
from baseline import STANDARD, RICH


def peer_gaps(indoor, minimum_peers=3):
    """Exclude the subject car from its own simultaneous peer reference."""
    gaps, counts = {}, {}
    for car in indoor:
        others = indoor.drop(columns=car)
        counts[car] = others.notna().sum(axis=1)
        reference = others.median(axis=1).where(counts[car] >= minimum_peers)
        gaps[car] = indoor[car] - reference
    return pd.DataFrame(gaps), pd.DataFrame(counts)


def load_gaps(source):
    """Shared calculation for plots and summaries; no labels or fitting."""
    print(f"Reading {source.name}; please allow about a minute...", flush=True)
    data = pd.read_excel(source)
    time = pd.to_datetime(data["Time"], errors="raise")
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Inspect missing, duplicate, or out-of-order timestamps first.")
    fields = {}
    for column in data:
        match = re.fullmatch(r"Car (\d{2}) - (.+)", str(column))
        if match:
            fields.setdefault(match.group(1), set()).add(match.group(2))
    indoor, control = {}, {}
    for car, parameters in sorted(fields.items()):
        if set(STANDARD) <= parameters:
            names = STANDARD
        elif set(RICH) <= parameters:
            names = RICH
            print(f"Car {car}: rich layout interpretation provisional; validity unconfirmed")
        else:
            raise ValueError(f"Car {car}: unrecognised fields.")
        prefix = f"Car {car} - "
        temperature = pd.to_numeric(data[prefix + names[0]], errors="raise")
        target = pd.to_numeric(data[prefix + names[1]], errors="raise")
        mode = data[prefix + names[2]]
        keep = np.isfinite(temperature) & np.isfinite(target)
        if names == STANDARD:
            keep &= mode.eq("Automatic Cooling") & data[prefix + names[3]].eq("Valid")
        else:
            keep &= mode.isin(["Full Cooling", "Half Cooling"])
        indoor[car] = temperature.where(keep)
        control[car] = (temperature - target).where(keep)
    if not indoor:
        raise ValueError("No cars found.")
    indoor, control = pd.DataFrame(indoor), pd.DataFrame(control)
    peer, counts = peer_gaps(indoor)
    return time, indoor, control, peer, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", type=int, choices=range(1, 7))
    selection.add_argument("--input", type=Path, help="Case workbook to visualise")
    parser.add_argument("--hide-total", action="store_true", help="Show only control and peer gaps")
    args = parser.parse_args()
    case = args.case or 1
    source = args.input if args.input is not None else ACV / f"data/raw/train/acv_case_{case:02d}.xlsx"
    title = source.name if args.input is not None else f"Case {case:02d}"
    time, indoor, control, peer, counts = load_gaps(source)
    report = pd.DataFrame({"car": indoor.columns,
                           "usable_rows": indoor.notna().sum().to_numpy(),
                           "peer_comparison_rows": peer.notna().sum().to_numpy()})
    print("\nPeer reference: median of at least 3 OTHER usable cars at the same timestamp.")
    print(report.to_string(index=False))
    print("Display only: no scores, labels, fitting, or ranking changes.")

    interval = time.diff().dropna().mode().iloc[0]
    breaks = time.loc[time.diff() > interval] - interval
    plot_times = pd.DatetimeIndex(time).union(pd.DatetimeIndex(breaks)).sort_values()
    fig, axes = plt.subplots(len(indoor.columns), 1, figsize=(13, 15), sharex=True,
                             sharey=True, squeeze=False, layout="constrained")
    for index, car in enumerate(indoor.columns):
        ax = axes[index, 0]
        # Add signed gaps at the same timestamp. If either is missing, the sum
        # stays missing. This is a visual comparison, not a fitted ranking rule.
        total = control[car] + peer[car]
        lines = [(control[car], "Indoor minus control/target", "#d97706"),
                 (peer[car], "Indoor minus other cars' median", "#1769aa")]
        if not args.hide_total:
            lines.append((total, "Sum of both gaps", "#9333aa"))
        for values, label, color in lines:
            series = pd.Series(values.to_numpy(), index=time).reindex(plot_times)
            ax.plot(series.index, series, label=label, color=color, linewidth=0.8, alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_ylabel(f"Car {car}")
        ax.grid(alpha=0.2)
        if peer[car].notna().sum() == 0:
            ax.text(0.5, 0.7, "Peer comparison unavailable", transform=ax.transAxes, ha="center")
    axes[0, 0].legend(loc="upper left", fontsize=9, ncol=2 if args.hide_total else 3)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1, 0].set_xlabel("Recorded time (month-day hour:minute)")
    fig.supylabel("Difference in recorded units (units unconfirmed)")
    description = "Control/target gap and peer gap by car" if args.hide_total else "Temperature gaps and their sum"
    fig.suptitle(f"{title}: {description}\n"
                 "Same-timestamp comparison; at least 3 other usable cars; negative differences retained")
    stem = source.stem if args.input is not None else f"case_{case:02d}"
    suffix = "target_and_peer_gaps" if args.hide_total else "peer_gap"
    destination = ACV / f"outputs/{stem}_{suffix}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    print(f"Saved plot: {destination}")


if __name__ == "__main__":
    main()

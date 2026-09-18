"""Explore control-gap differences between cars; no scores or fitting."""
import argparse
from pathlib import Path
from step9_plot_peer_gap import ACV, load_gaps, peer_gaps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", type=int, choices=range(1, 7))
    selection.add_argument("--input", type=Path, help="Workbook to visualise without fitting")
    args = parser.parse_args()
    case = args.case or 1
    source = args.input if args.input is not None else ACV / f"data/raw/train/acv_case_{case:02d}.xlsx"
    title = source.name if args.input is not None else f"Case {case:02d}"
    time, _, control, _, _ = load_gaps(source)
    # Pass CONTROL GAPS to the same leave-one-car-out median calculation.
    # Negative control gaps remain signed in both the subject and reference.
    relative, counts = peer_gaps(control, minimum_peers=3)
    print("Relative control gap = own control gap - median control gap of other usable cars")
    print("Minimum: 3 OTHER cars at the same timestamp. Negative values retained.")
    print(pd.DataFrame({"car": control.columns,
                        "usable_rows": control.notna().sum().to_numpy(),
                        "relative_comparison_rows": relative.notna().sum().to_numpy()}).to_string(index=False))
    interval = time.diff().dropna().mode().iloc[0]
    breaks = time.loc[time.diff() > interval] - interval
    plot_times = pd.DatetimeIndex(time).union(pd.DatetimeIndex(breaks)).sort_values()
    fig, axes = plt.subplots(len(control.columns), 1, figsize=(13, 15), sharex=True,
                             sharey=True, squeeze=False, layout="constrained")
    for index, car in enumerate(control.columns):
        ax = axes[index, 0]
        for values, label, color in [
            (control[car], "Own control gap", "#d97706"),
            (relative[car], "Control gap minus other cars' median control gap", "#0f766e"),
        ]:
            series = pd.Series(values.to_numpy(), index=time).reindex(plot_times)
            ax.plot(series.index, series, label=label, color=color, linewidth=0.9, alpha=0.85)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
        ax.set_ylabel(f"Car {car}")
        ax.grid(alpha=0.2)
        if not relative[car].notna().any():
            ax.text(0.5, 0.7, "Relative comparison unavailable", transform=ax.transAxes, ha="center")
    axes[0, 0].legend(loc="upper left", ncol=2, fontsize=9)
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1, 0].set_xlabel("Recorded time (month-day hour:minute)")
    fig.supylabel("Difference in recorded units (units unconfirmed)")
    fig.suptitle(f"{title}: own and relative control gaps\n"
                 "At least 3 other usable cars; provisional control interpretation; no accumulation or ranking")
    stem = source.stem if args.input is not None else f"case_{case:02d}"
    destination = ACV / f"outputs/{stem}_relative_control_gap.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    print(f"Saved plot: {destination}")
    print("Display only: no labels, fitting, or ranking changes.")


if __name__ == "__main__":
    main()

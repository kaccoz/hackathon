"""Visualise existing per-timestamp gap formulas for a case, without fitting."""
import argparse
from pathlib import Path
from step9_plot_peer_gap import ACV, load_gaps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_case(case=None, input_path=None):
    source = Path(input_path) if input_path is not None else ACV / f"data/raw/train/acv_case_{case:02d}.xlsx"
    title = source.name if input_path is not None else f"Case {case:02d}"
    time, _, control, peer, _ = load_gaps(source)
    total = control + peer
    interval = time.diff().dropna().mode().iloc[0]
    breaks = time.loc[time.diff() > interval] - interval
    plot_times = pd.DatetimeIndex(time).union(pd.DatetimeIndex(breaks)).sort_values()
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    unavailable = []
    for index, car in enumerate(total):
        if total[car].notna().sum() == 0:
            unavailable.append(car)
            continue
        readings = pd.Series(total[car].to_numpy(), index=time).reindex(plot_times)
        ax.plot(readings.index, readings, label=f"Car {car}",
                color=plt.get_cmap("tab10")(index), linestyle=styles[index % len(styles)],
                linewidth=1.1, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_xlabel("Recorded time (month-day hour:minute)")
    ax.set_ylabel("Total gap (recorded units, unconfirmed)")
    ax.set_title(f"{title}: total gap compared across cars\n"
                 "Total = control gap + peer gap at each timestamp; signed values retained")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    note = "Display only; no smoothing or accumulation. This combined gap is not the baseline ranking score."
    if unavailable:
        note += "\nNo total available for Cars " + ", ".join(unavailable) + "; they are not plotted as zero."
    fig.supxlabel(note, fontsize=9)
    stem = source.stem if input_path is not None else f"case_{case:02d}"
    destination = ACV / f"outputs/{stem}_total_gap_across_cars.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", type=int, choices=range(1, 7))
    selection.add_argument("--all", action="store_true", help="Generate one chart for each training case")
    selection.add_argument("--input", type=Path, help="Specific case workbook to visualise without changing the model")
    args = parser.parse_args()
    if args.input is not None:
        plot_case(input_path=args.input)
    else:
        for case in (range(1, 7) if args.all else [args.case or 4]):
            plot_case(case)

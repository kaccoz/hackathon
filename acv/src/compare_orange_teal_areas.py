"""Measure positive orange/control and teal/relative gap areas in training data.

Both colours use identical accepted intervals for each car. Gaps longer than
the most common recording interval, or missing endpoints, contribute no time.
Areas approximate each interval using the mean of nonnegative endpoint values.
No labels, combined scores, prediction changes, or test inputs are used.
"""
import argparse
import numpy as np
import pandas as pd
from step9_plot_peer_gap import ACV, load_gaps, peer_gaps


def measure(time, control, relative):
    seconds = time.diff().dt.total_seconds()
    positive_intervals = seconds[seconds.gt(0)]
    if positive_intervals.empty:
        raise ValueError("At least two distinct timestamps are required.")
    typical_seconds = float(positive_intervals.mode().min())
    short = seconds.gt(0) & seconds.le(typical_seconds)
    minutes = seconds / 60
    records = []
    for car in control:
        valid = np.isfinite(control[car]) & np.isfinite(relative[car])
        accepted = short & valid & valid.shift(1, fill_value=False)
        duration = float(minutes[accepted].sum())
        row = {"car": car, "accepted_intervals": int(accepted.sum()),
               "usable_minutes": duration}
        for colour, gap in [("orange", control[car]), ("teal", relative[car])]:
            positive = gap.clip(lower=0)
            height = (positive.shift(1) + positive) / 2
            area = float((height[accepted] * minutes[accepted]).sum()) if duration > 0 else np.nan
            row[f"{colour}_gap_minutes"] = area
            row[f"{colour}_time_weighted_mean"] = area / duration if duration > 0 else np.nan
        records.append(row)
    return pd.DataFrame(records), typical_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--case", type=int, choices=range(1, 7))
    choice.add_argument("--all", action="store_true", help="Compare all six training cases")
    args = parser.parse_args()
    reports = []
    for case in (range(1, 7) if args.all else [args.case or 1]):
        filename = f"acv_case_{case:02d}.xlsx"
        time, _, control, _, _ = load_gaps(ACV / "data/raw/train" / filename)
        relative, _ = peer_gaps(control, minimum_peers=3)
        report, maximum_interval = measure(time, control, relative)
        print(f"\nCASE {case:02d}: orange and teal areas on identical intervals per car")
        print(f"Maximum counted interval: {maximum_interval:g} seconds; at least 3 other usable cars")
        print(report.to_string(index=False, na_rep="unavailable", float_format=lambda value: f"{value:.4f}"), flush=True)
        print("Orange = own control gap; teal = control gap minus median OTHER-car control gap.")
        report.insert(0, "file_id", filename)
        reports.append(report)
    output = ACV / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    name = "all_training_orange_teal_areas.csv" if args.all else f"case_{args.case or 1:02d}_orange_teal_areas.csv"
    destination = output / name
    pd.concat(reports, ignore_index=True).to_csv(destination, index=False, na_rep="unavailable")
    print(f"\nSaved: {destination}")
    print("Positive parts measured separately; negative gaps do not cancel positive gaps.")
    print("No usable intervals means unavailable, not zero. Different cars may retain different intervals.")
    print("Orange uses peer-available intervals here and can differ from the original baseline.")
    print("No labels or test data used; original ranking unchanged.")


if __name__ == "__main__":
    main()

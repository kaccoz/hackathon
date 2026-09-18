"""Candidate ranking: time-weighted positive relative control gap.

No labels, fitting, or test data are read. Field interpretations and the minimum
of three peers follow our exploratory plots. This is not a validated model.
"""
import argparse
import numpy as np
import pandas as pd
from step9_plot_peer_gap import ACV, load_gaps, peer_gaps


def rank_relative_gaps(time, relative):
    """Accumulate short observed intervals; never bridge missing readings.

    Interval rule: consecutive original rows, valid relative gaps at both ends,
    and elapsed time no greater than the file's most common positive interval.
    Estimate interval area by averaging the two nonnegative endpoint gaps.
    This is an approximation between observations, not continuous measurement.
    """
    time = pd.Series(pd.to_datetime(time, errors="raise")).reset_index(drop=True)
    relative = relative.reset_index(drop=True)
    if len(time) != len(relative):
        raise ValueError("Timestamp and gap row counts differ.")
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Timestamps must be present, unique, and increasing.")
    seconds = time.diff().dt.total_seconds()
    observed = seconds[seconds > 0]
    if observed.empty:
        raise ValueError("At least two timestamps are needed to measure elapsed time.")
    # On an exact frequency tie, use the shortest modal interval conservatively.
    typical_seconds = float(observed.mode().min())
    short_interval = seconds.gt(0) & seconds.le(typical_seconds)
    minutes = seconds / 60
    records = []
    for car in relative:
        gap = pd.to_numeric(relative[car], errors="raise")
        finite = pd.Series(np.isfinite(gap), index=gap.index)
        accepted = short_interval & finite & finite.shift(1, fill_value=False)
        positive = gap.clip(lower=0)
        endpoint_average = (positive.shift(1) + positive) / 2
        usable_minutes = float(minutes[accepted].sum())
        area = float((endpoint_average[accepted] * minutes[accepted]).sum()) if usable_minutes > 0 else np.nan
        score = area / usable_minutes if usable_minutes > 0 else np.nan
        records.append({
            "car": car,
            "accepted_intervals": int(accepted.sum()),
            "usable_minutes": usable_minutes,
            "positive_gap_minutes": area,
            "time_weighted_score": score,
            "evidence": "scored" if usable_minutes > 0 else "unavailable",
        })
    ranking = pd.DataFrame(records).sort_values(
        ["time_weighted_score", "car"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking, typical_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=range(1, 7), default=1,
                        help="Training case number (1 to 6)")
    args = parser.parse_args()
    filename = f"acv_case_{args.case:02d}.xlsx"
    time, _, control, _, _ = load_gaps(ACV / "data/raw/train" / filename)
    relative, _ = peer_gaps(control, minimum_peers=3)
    ranking, interval = rank_relative_gaps(time, relative)
    print(f"\nCandidate method for {filename}")
    print("Relative gap = own control gap - median control gap of at least 3 OTHER usable cars.")
    print(f"Only adjacent valid endpoints at most {interval:g} seconds apart contribute.")
    print("Area uses the average of the two positive endpoint gaps times elapsed minutes.")
    print("Score = accumulated positive gap-minutes / usable comparison minutes.")
    print(ranking.to_string(index=False, na_rep="unavailable", float_format=lambda x: f"{x:.4f}"))
    print("\nRanking:", "|".join(ranking["car"]))
    print("Unscored cars follow by ID only; this does not indicate they are healthy.")
    print("No label checking or validation performed. Scores are not probabilities.")
    destination = ACV / f"outputs/case_{args.case:02d}_time_weighted_relative_scores.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(destination, index=False, na_rep="unavailable")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

"""Inspect valve codes, then optionally compare fresh-air readings by open count.

No code meanings are assumed. Supply --open-code and --closed-code only after
confirmation. Counts describe four recorded valve indicators, not airflow.
All comparisons retain the same Full Cooling/(1,1) filter used previously.
No labels, test inputs, or model fitting are used.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ACV = Path(__file__).resolve().parents[1]
CARS = ("01", "02", "03", "04")
VALVES = [f"Fresh Air Valve {name} Closed" for name in ("A1", "A2", "B1", "B2")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-open", action="store_true", help="Count the observed literal Open/Closed labels")
    parser.add_argument("--open-code", type=float, help="Confirmed numeric value meaning open in these Closed fields")
    parser.add_argument("--closed-code", type=float, help="Confirmed numeric value meaning closed")
    args = parser.parse_args()
    if args.count_open and (args.open_code is not None or args.closed_code is not None):
        parser.error("Use --count-open for text labels OR supply numeric codes, not both.")
    if (args.open_code is None) != (args.closed_code is None):
        parser.error("Supply both confirmed codes, or neither to inspect raw values.")
    if args.open_code is not None and (not np.isfinite(args.open_code)
            or not np.isfinite(args.closed_code) or args.open_code == args.closed_code):
        parser.error("Codes must be finite and different.")
    fields = VALVES + ["Fresh Air Temperature Detected Value", "ACV Running Mode",
                       "Compressor 1 Running", "Compressor 2 Running"]
    print("Reading case 04; please allow about a minute...", flush=True)
    data = pd.read_excel(ACV / "data/raw/train/acv_case_04.xlsx",
                        usecols=["Time"] + [f"Car {car} - {field}" for car in CARS for field in fields])
    times = pd.to_datetime(data["Time"], errors="raise")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Inspect missing, duplicate, or out-of-order timestamps first.")
    for car in CARS:
        print(f"\nCar {car}: raw valve values and row counts")
        for valve in VALVES:
            print(valve)
            print(data[f"Car {car} - {valve}"].value_counts(dropna=False).to_string())
    if args.open_code is None and not args.count_open:
        print("\nInspection only. Open counts NOT calculated because code meanings are unconfirmed.")
        print("After confirming meanings, rerun with --open-code VALUE --closed-code VALUE.")
        print("For literal Open/Closed labels, use --count-open instead.")
        return

    open_value = "Open" if args.count_open else args.open_code
    closed_value = "Closed" if args.count_open else args.closed_code
    temperatures, open_counts, eligible = {}, {}, {}
    for car in CARS:
        prefix = f"Car {car} - "
        valves = data[[prefix + field for field in VALVES]]
        if not args.count_open:
            valves = valves.apply(pd.to_numeric, errors="raise")
        known = valves.isin([open_value, closed_value]).all(axis=1)
        # All four must be known: a missing status is never silently 'closed'.
        open_counts[car] = valves.eq(open_value).sum(axis=1).where(known)
        temp = pd.to_numeric(data[prefix + "Fresh Air Temperature Detected Value"], errors="raise")
        temperatures[car] = temp.where(np.isfinite(temp))
        eligible[car] = (data[prefix + "ACV Running Mode"].eq("Full Cooling")
                         & data[prefix + "Compressor 1 Running"].eq(1)
                         & data[prefix + "Compressor 2 Running"].eq(1)
                         & np.isfinite(temp))
    counts = pd.DataFrame(open_counts)
    temps = pd.DataFrame(temperatures)
    common_cooling = pd.DataFrame(eligible).all(axis=1)
    matched_valves = common_cooling & counts.notna().all(axis=1)
    print(f"\nShared cooling timestamps with all four valve statuses known in every car: {matched_valves.sum():,}")
    print("Mean number of open valves on those identical timestamps:")
    print(counts.loc[matched_valves].mean().rename("mean_open_valves").to_string())
    if matched_valves.any():
        delta_count = counts.loc[matched_valves, "04"] - counts.loc[matched_valves, "01"]
        print(f"Car 04 has more open valves than Car 01 in {delta_count.gt(0).mean():.1%} of these rows")
        print(f"Same number in {delta_count.eq(0).mean():.1%}; fewer in {delta_count.lt(0).mean():.1%}")
    summaries = []
    for car in CARS:
        for number_open in range(5):
            mask = common_cooling & counts[car].eq(number_open)
            summaries.append({"car": car, "open_valves": number_open, "rows": int(mask.sum()),
                              "mean_fresh_air": temps.loc[mask, car].mean(),
                              "median_fresh_air": temps.loc[mask, car].median()})
    summary = pd.DataFrame(summaries)
    print("\nFresh-air readings by each car's open count during shared cooling operation:")
    print(summary.to_string(index=False, na_rep="unavailable", float_format=lambda x: f"{x:.3f}"))
    print("Different count groups may contain different timestamps; use the matched comparison below.")
    comparisons = []
    for number_open in range(5):
        mask = common_cooling & counts["01"].eq(number_open) & counts["04"].eq(number_open)
        delta = temps.loc[mask, "04"] - temps.loc[mask, "01"]
        comparisons.append({"same_open_count": number_open, "matched_rows": int(mask.sum()),
                            "car01_mean_fresh_air": temps.loc[mask, "01"].mean(),
                            "car04_mean_fresh_air": temps.loc[mask, "04"].mean(),
                            "mean_car04_minus_car01": delta.mean(),
                            "median_car04_minus_car01": delta.median()})
    comparison = pd.DataFrame(comparisons)
    print("\nCars 01 and 04: same timestamps AND same open-valve count:")
    print(comparison.to_string(index=False, na_rep="unavailable", float_format=lambda x: f"{x:.3f}"))
    detail = pd.DataFrame({"Time": times, "all_cars_full_cooling_and_11": common_cooling})
    for car in CARS:
        detail[f"car_{car}_open_valves"] = counts[car]
        detail[f"car_{car}_fresh_air"] = temps[car]
    output = ACV / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "case_04_fresh_air_by_open_valve_count.csv", index=False, na_rep="unavailable")
    comparison.to_csv(output / "case_04_fresh_air_matched_valve_counts.csv", index=False, na_rep="unavailable")
    detail.to_csv(output / "case_04_open_valves_over_time.csv", index=False, na_rep="unavailable")
    (output / "case_04_valve_code_assumptions.txt").write_text(
        f"Selected mapping: open={open_value!r}, closed={closed_value!r}\n"
        "Four Closed fields counted equally; physical airflow not established.\n", encoding="utf-8")
    print(f"\nSaved tables and supplied code mapping in {output}")
    print("Zero open valves is retained as context, not treated as measured fresh-air entry.")
    print("Equal counts do not prove equal airflow; valve identity, geometry and fan operation can differ.")
    print("No temperature-times-valve-count score or leak prediction is calculated.")


if __name__ == "__main__":
    main()

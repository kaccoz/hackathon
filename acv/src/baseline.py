"""Fixed exploratory ranking rule. Reads no labels and learns no parameters.

Rich-layout field mapping and missing-car placement are working assumptions.
Scores average samples, not elapsed time, and are not probabilities.
"""
from pathlib import Path
import argparse
import re
import sys
import warnings
import numpy as np
import pandas as pd

STANDARD = ("Indoor Average Temperature", "ACV Control Temperature (Cooling)",
            "ACV Running Mode", "ACV Information Valid")
RICH = ("Passenger Cabin Temperature Detected Value", "Target Temperature Value",
        "ACV Running Mode")


def rank_table(data):
    """Return every header car, with unavailable scores distinct from zero."""
    fields = {}
    for column in data.columns:
        match = re.fullmatch(r"Car (\d{2}) - (.+)", str(column))
        if match:
            fields.setdefault(match.group(1), set()).add(match.group(2))
    if not fields:
        raise ValueError("No car identifiers found in column headers.")
    results = []
    for car, parameters in sorted(fields.items()):
        if set(STANDARD) <= parameters:
            names, layout = STANDARD, "standard"
        elif set(RICH) <= parameters:
            names, layout = RICH, "rich (provisional)"
        else:
            raise ValueError(f"Car {car}: unrecognised layout; inspect its headers first.")
        prefix = f"Car {car} - "
        indoor = pd.to_numeric(data[prefix + names[0]], errors="raise")
        control = pd.to_numeric(data[prefix + names[1]], errors="raise")
        mode = data[prefix + names[2]]
        usable = np.isfinite(indoor) & np.isfinite(control)
        if layout == "standard":
            usable &= mode.eq("Automatic Cooling") & data[prefix + names[3]].eq("Valid")
        else:
            usable &= mode.isin(["Full Cooling", "Half Cooling"])
        count = int(usable.sum())
        score = float((indoor[usable] - control[usable]).clip(lower=0).mean()) if count else np.nan
        results.append({"car": car, "layout": layout, "usable_rows": count,
                        "average_positive_gap": score,
                        "evidence": "scored" if count else "unavailable"})
    # Missing scores go last. Exact score ties and unscored cars use car ID.
    ranking = pd.DataFrame(results).sort_values(
        ["average_positive_gap", "car"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking


def rank_file(path):
    # Only load relevant fields, but retain a column for every header car so no
    # car disappears silently if its schema is unfamiliar.
    headers = pd.read_excel(path, nrows=0).columns
    if "Time" not in headers:
        raise ValueError("Missing Time column.")
    relevant, seen = [], set()
    for column in headers:
        match = re.fullmatch(r"Car (\d{2}) - (.+)", str(column))
        if match:
            car, parameter = match.groups()
            if car not in seen or parameter in set(STANDARD + RICH):
                relevant.append(column)
            seen.add(car)
    if len(seen) != 8:
        raise ValueError(f"Expected eight car identifiers; found {len(seen)}.")
    data = pd.read_excel(path, usecols=["Time"] + relevant)
    if data.empty:
        raise ValueError("Workbook has no measurement rows.")
    time = pd.to_datetime(data["Time"], errors="raise")
    if time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing:
        raise ValueError("Timestamps must be present, unique, and increasing.")
    return rank_table(data)


def create_predictions(input_path):
    """Return the official two-column table for a file or non-recursive folder.

    Reads no labels; learns no parameters. Errors abort rather than skip cases.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        paths = sorted((p for p in input_path.iterdir()
                        if p.is_file() and p.suffix.lower() == ".xlsx"
                        and not p.name.startswith("~$")), key=lambda p: p.name)
        if not paths:
            raise ValueError("Input folder contains no .xlsx case files.")
    elif input_path.is_file() and input_path.suffix.lower() == ".xlsx":
        paths = [input_path]
    else:
        raise ValueError("Input must be an existing .xlsx case or a folder of .xlsx cases.")
    records = []
    for path in paths:
        try:
            ranking = rank_file(path)
            if not ranking["evidence"].eq("scored").any():
                raise ValueError("No car has usable readings; cannot produce an evidence-based ranking.")
            cars = ranking["car"].tolist()
            if len(cars) != 8 or len(set(cars)) != 8 or any(not re.fullmatch(r"\d{2}", car) for car in cars):
                raise ValueError("Ranking must contain eight distinct two-digit header identifiers.")
            unavailable = ranking.loc[ranking["evidence"].eq("unavailable"), "car"].tolist()
            if unavailable:
                warnings.warn(f"{path.name}: no usable readings for {', '.join(unavailable)}; "
                              "placed last by ID, not evidence of health.", stacklevel=2)
            if ranking["layout"].eq("rich (provisional)").any():
                warnings.warn(f"{path.name}: rich-layout mapping provisional; no confirmed validity flag.",
                              stacklevel=2)
            records.append({"file_id": path.name, "ranked_cars": "|".join(cars)})
        except Exception as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    output = pd.DataFrame(records, columns=["file_id", "ranked_cars"])
    if output["file_id"].duplicated().any():
        raise ValueError("Duplicate input filenames.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One .xlsx case or folder of cases")
    parser.add_argument("--output", type=Path, required=True, help="Destination acv_predictions.csv")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must be a .csv file path.")
    print("Generating predictions; each large workbook may take about a minute...", flush=True)
    try:
        output = create_predictions(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
    except Exception as exc:
        print(f"ACV prediction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Saved {len(output)} prediction row(s) to {args.output}")


if __name__ == "__main__":
    main()

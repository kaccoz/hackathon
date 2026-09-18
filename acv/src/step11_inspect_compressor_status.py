"""Inspect case 04 compressor status values without interpreting their codes.

Counts are numbers of rows, not durations. This script does not score faults,
read labels, filter by an assumed status meaning, or access test data.
"""
from pathlib import Path
import pandas as pd

ACV = Path(__file__).resolve().parents[1]
CARS = ("01", "02", "03", "04")
FIELDS = (
    "Compressor 1 Running",
    "Compressor 2 Running",
    "Compressor 1 Fault",
    "Compressor 2 Fault",
)


def main():
    source = ACV / "data/raw/train/acv_case_04.xlsx"
    columns = [f"Car {car} - {field}" for car in CARS for field in FIELDS]
    print(f"Reading {source.name}; please allow about a minute...", flush=True)
    headers = pd.read_excel(source, nrows=0).columns
    missing = sorted(set(["Time"] + columns) - set(headers))
    if missing:
        raise ValueError(f"Expected headers not found: {missing}")
    data = pd.read_excel(source, usecols=["Time"] + columns)
    time = pd.to_datetime(data["Time"], errors="raise")
    print(f"Rows: {len(data):,}; recorded range: {time.min()} to {time.max()}")
    records = []
    for car in CARS:
        print(f"\nCAR {car}")
        for field in FIELDS:
            values = data[f"Car {car} - {field}"]
            print(f"  {field}:")
            # repr preserves distinctions such as numeric 1 versus text '1'.
            for value, count in values.value_counts(dropna=False).items():
                displayed = "<missing>" if pd.isna(value) else repr(value)
                kind = "missing" if pd.isna(value) else type(value).__name__
                print(f"    {displayed}: {count:,} rows")
                records.append({"car": car, "field": field,
                                "recorded_value": displayed, "value_type": kind,
                                "row_count": int(count)})
    output = ACV / "outputs/case_04_compressor_status_counts.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"\nSaved: {output}")
    print("These are raw value counts, not running minutes or leak predictions.")
    print("Numeric codes require confirmation; missing status does not mean stopped.")


if __name__ == "__main__":
    main()

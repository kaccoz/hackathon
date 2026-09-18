"""Side-by-side positive-gap averages on matching rows; no combined model."""
import pandas as pd
from step9_plot_peer_gap import ACV, load_gaps


def summarise(control, peer):
    records = []
    for car in control:
        # Both factors use the same accepted timestamps for this car.
        shared = control[car].notna() & peer[car].notna()
        records.append({
            "car": car,
            "usable_rows": int(control[car].notna().sum()),
            "shared_rows": int(shared.sum()),
            "mean_positive_control_gap": control.loc[shared, car].clip(lower=0).mean(),
            "mean_positive_peer_gap": peer.loc[shared, car].clip(lower=0).mean(),
        })
    return pd.DataFrame(records)


def main():
    reports = []
    for case in range(1, 7):
        filename = f"acv_case_{case:02d}.xlsx"
        _, _, control, peer, _ = load_gaps(ACV / "data/raw/train" / filename)
        report = summarise(control, peer)
        print(f"\nCase {case:02d}: averages over matching rows (zeros included)")
        print(report.to_string(index=False, na_rep="unavailable", float_format=lambda x: f"{x:.4f}"), flush=True)
        report.insert(0, "file_id", filename)
        reports.append(report)
    destination = ACV / "outputs/training_factor_comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(reports, ignore_index=True).to_csv(destination, index=False, na_rep="unavailable")
    print(f"\nSaved: {destination}")
    print("No labels, test data, combined scores, or validation results used here.")
    print("Baseline unchanged. Shared-row control averages can differ from the original baseline.")


if __name__ == "__main__":
    main()

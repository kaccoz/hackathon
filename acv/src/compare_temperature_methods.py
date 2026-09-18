"""Development comparison of two fixed temperature rules across whole cases.

All six cases have already informed exploration. These results are not a new
independent holdout estimate or fitted leave-one-case-out cross-validation.
Pressure and valve information do not enter either ranking. No test data used.
"""
import pandas as pd
from step9_plot_peer_gap import ACV, load_gaps, peer_gaps
from time_weighted_relative_baseline import rank_relative_gaps


def main():
    rankings = {}
    details = []
    for case in range(1, 7):
        filename = f"acv_case_{case:02d}.xlsx"
        time, _, control, _, _ = load_gaps(ACV / "data/raw/train" / filename)
        # Identical original rule: mean positive gap on all usable rows.
        baseline = pd.DataFrame({"car": control.columns,
                                 "score": control.clip(lower=0).mean().to_numpy()})
        baseline = baseline.sort_values(["score", "car"], ascending=[False, True],
                                        na_position="last").reset_index(drop=True)
        relative, _ = peer_gaps(control, minimum_peers=3)
        challenger, _ = rank_relative_gaps(time, relative)
        challenger = challenger.rename(columns={"time_weighted_score": "score"})
        rankings[filename] = {"original": baseline, "relative_time_weighted": challenger}
        for method, table in rankings[filename].items():
            detail = table[["car", "score"]].copy()
            detail.insert(0, "rank", range(1, len(detail) + 1))
            detail.insert(0, "method", method)
            detail.insert(0, "file_id", filename)
            details.append(detail)
        print(f"Completed both rankings for {filename}", flush=True)

    # Answer key only grades completed rankings; it never enters either method.
    labels = pd.read_csv(ACV / "data/raw/Train_Labels.csv", dtype=str)
    rows = []
    for filename, methods in rankings.items():
        answer = labels.loc[labels["filename"].eq(filename), "faulty_car"]
        if len(answer) != 1:
            raise ValueError(f"Expected one label for {filename}.")
        faulty = answer.iloc[0]
        row = {"file_id": filename, "faulty_car": faulty}
        for method, table in methods.items():
            cars = table["car"].tolist()
            if faulty not in cars or len(cars) != len(set(cars)):
                raise ValueError(f"Invalid car list for {filename}, {method}.")
            position = cars.index(faulty) + 1
            row[f"{method}_top_car"] = cars[0]
            row[f"{method}_faulty_rank"] = position
            row[f"{method}_official_score"] = (len(cars) - position + 1) / len(cars)
            row[f"{method}_unscored_cars"] = "|".join(table.loc[table["score"].isna(), "car"])
        old = row["original_faulty_rank"]
        new = row["relative_time_weighted_faulty_rank"]
        row["challenger_result"] = "better" if new < old else "worse" if new > old else "same"
        rows.append(row)
    report = pd.DataFrame(rows)
    display_columns = ["file_id", "faulty_car", "original_top_car", "original_faulty_rank",
                       "relative_time_weighted_top_car", "relative_time_weighted_faulty_rank",
                       "challenger_result"]
    print("\nDEVELOPMENT COMPARISON (not new independent validation)")
    print(report[display_columns].to_string(index=False))
    for method in ("original", "relative_time_weighted"):
        score = report[f"{method}_official_score"].mean()
        first = int(report[f"{method}_faulty_rank"].eq(1).sum())
        print(f"{method}: mean official score {score:.4f}; first-place matches {first}/{len(report)}")
    output = ACV / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "temperature_methods_comparison.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(output / "temperature_methods_car_scores.csv",
                                                index=False, na_rep="unavailable")
    print(f"Saved comparison and individual car scores in {output}")
    print("Methods are unchanged. No weights or thresholds were fitted or selected.")
    print("Unscored cars are placed last by ID only; their fault likelihood is unknown.")


if __name__ == "__main__":
    main()

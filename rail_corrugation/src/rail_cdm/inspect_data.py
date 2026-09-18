from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rail_cdm.io import read_labels, read_sensor_csv, validate_training_inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the Rail training dataset.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inspection"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = read_labels(args.labels)
    paths = validate_training_inventory(args.train_dir, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nClass counts")
    print(labels["label"].value_counts().reindex(["Normal", "Side I", "Side II"], fill_value=0))

    sample_path = paths[0]
    sample = read_sensor_csv(sample_path)
    print(f"\nSample: {sample_path.name}")
    print(f"Rows: {sample.shape[0]:,}")
    print(f"Columns: {sample.shape[1]}")
    print(f"Missing values: {int(sample.isna().sum().sum())}")

    sensor = sample.iloc[:, 1:].to_numpy(dtype=float)
    rms = np.sqrt(np.mean(sensor**2, axis=0))
    vibration_rms = rms[0::2].reshape(8, 8)
    shock_rms = rms[1::2].reshape(8, 8)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for axis, matrix, title in zip(
        axes, (vibration_rms, shock_rms), ("Vibration RMS", "Shock RMS"), strict=True
    ):
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_title(f"{sample_path.name}: {title}")
        axis.set_xlabel("Axle-box position")
        axis.set_ylabel("Car")
        axis.set_xticks(range(8), range(1, 9))
        axis.set_yticks(range(8), range(1, 9))
        figure.colorbar(image, ax=axis)
    heatmap_path = args.output_dir / "sample_sensor_rms_heatmap.png"
    figure.savefig(heatmap_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    excerpt = min(1_000, len(sample))
    axis.plot(sample.iloc[:excerpt, 1], label=sample.columns[1], linewidth=0.8)
    axis.plot(sample.iloc[:excerpt, 3], label=sample.columns[3], linewidth=0.8)
    axis.set_title(f"{sample_path.name}: first {excerpt} readings")
    axis.set_xlabel("Reading number")
    axis.set_ylabel("Acceleration")
    axis.legend(fontsize=7)
    signal_path = args.output_dir / "sample_signal_excerpt.png"
    figure.savefig(signal_path, dpi=160)
    plt.close(figure)

    print(f"\nSaved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

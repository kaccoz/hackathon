from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

SAMPLE_RATE_HZ = 10_000
FREQUENCY_BANDS_HZ = (
    (0, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1_000),
    (1_000, 2_500),
    (2_500, 5_001),
)


def _summarize_channels(prefix: str, values: np.ndarray) -> dict[str, float]:
    """Summarize a group of sensor channels without cancelling signed signals."""
    rms = np.sqrt(np.mean(np.square(values), axis=0))
    std = np.std(values, axis=0)
    peak = np.max(np.abs(values), axis=0)
    channel_kurtosis = np.nan_to_num(kurtosis(values, axis=0, fisher=False), nan=0.0)

    summaries: dict[str, float] = {}
    channel_metrics = {
        "rms": rms,
        "std": std,
        "peak": peak,
        "kurtosis": channel_kurtosis,
    }
    for metric_name, metric_values in channel_metrics.items():
        summaries[f"{prefix}_{metric_name}_mean"] = float(np.mean(metric_values))
        summaries[f"{prefix}_{metric_name}_median"] = float(np.median(metric_values))
        summaries[f"{prefix}_{metric_name}_max"] = float(np.max(metric_values))
        summaries[f"{prefix}_{metric_name}_spread"] = float(np.std(metric_values))

    centered = values - np.mean(values, axis=0, keepdims=True)
    spectrum_power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[0], d=1 / SAMPLE_RATE_HZ)
    total_power = np.sum(spectrum_power, axis=0) + 1e-12

    for low, high in FREQUENCY_BANDS_HZ:
        mask = (frequencies >= low) & (frequencies < high)
        band_fraction = np.sum(spectrum_power[mask], axis=0) / total_power
        band_name = f"band_{low}_{high}hz"
        summaries[f"{prefix}_{band_name}_mean"] = float(np.mean(band_fraction))
        summaries[f"{prefix}_{band_name}_max"] = float(np.max(band_fraction))
    return summaries


def extract_features(frame: pd.DataFrame) -> dict[str, float]:
    """Convert one 129-column, one-second recording into model features.

    Sensor columns alternate vibration and shock. Within each car, odd positions
    belong to Side I and even positions belong to Side II.
    """
    values = frame.to_numpy(dtype=np.float64)
    speed = values[:, 0]
    sensor_values = values[:, 1:]

    features: dict[str, float] = {
        "n_rows": float(len(frame)),
        "speed_mean": float(np.mean(speed)),
        "speed_std": float(np.std(speed)),
        "speed_transition_rate": float(np.mean(np.diff(speed) != 0)) if len(speed) > 1 else 0.0,
    }

    groups: dict[tuple[str, str], list[int]] = {
        ("side1", "vibration"): [],
        ("side1", "shock"): [],
        ("side2", "vibration"): [],
        ("side2", "shock"): [],
    }
    for sensor_column in range(sensor_values.shape[1]):
        position = (sensor_column // 2) % 8 + 1
        side = "side1" if position % 2 == 1 else "side2"
        signal_type = "vibration" if sensor_column % 2 == 0 else "shock"
        groups[(side, signal_type)].append(sensor_column)

    for (side, signal_type), indices in groups.items():
        features.update(_summarize_channels(f"{side}_{signal_type}", sensor_values[:, indices]))

    comparison_metrics = ("rms_mean", "std_mean", "peak_mean", "kurtosis_mean")
    for signal_type in ("vibration", "shock"):
        for metric in comparison_metrics:
            side1 = features[f"side1_{signal_type}_{metric}"]
            side2 = features[f"side2_{signal_type}_{metric}"]
            name = f"{signal_type}_{metric}"
            features[f"{name}_side1_minus_side2"] = side1 - side2
            features[f"{name}_side1_over_side2"] = side1 / (side2 + 1e-12)

    return features


def extract_feature_table(paths: list, *, progress: bool = True) -> pd.DataFrame:
    from rail_cdm.io import read_sensor_csv

    rows: list[dict[str, float | str]] = []
    total = len(paths)
    for number, path in enumerate(paths, start=1):
        if progress:
            print(f"[{number:>3}/{total}] Extracting {path.name}")
        rows.append({"filename": path.name, **extract_features(read_sensor_csv(path))})
    return pd.DataFrame(rows)

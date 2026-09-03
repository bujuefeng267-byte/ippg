#!/usr/bin/env python3
"""Detailed waveform- and heart-rate comparison for UBFC rPPG outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, coherence, sosfiltfilt, welch


def bandpass_zscore(x: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(3, [0.7, 3.5], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    return (y - np.mean(y)) / (np.std(y) + 1e-12)


def paired_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair x[t] with y[t + lag]. Positive lag means reference is later."""
    if lag > 0:
        return x[:-lag], y[lag:]
    if lag < 0:
        return x[-lag:], y[:lag]
    return x, y


def best_alignment(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, int, float]:
    best = (0, 1, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        a, b = paired_at_lag(x, y, lag)
        corr = float(np.corrcoef(a, b)[0, 1])
        if abs(corr) > best[2]:
            best = (lag, 1 if corr >= 0 else -1, abs(corr))
    return best


def spectral_peak_bpm(x: np.ndarray, fs: float) -> float:
    f, p = welch(x, fs=fs, nperseg=min(len(x), 1024), nfft=8192)
    mask = (f >= 0.7) & (f <= 3.5)
    return float(f[mask][np.argmax(p[mask])] * 60.0)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def method_metrics(
    name: str,
    waveform_path: Path,
    hr_path: Path,
    gt_ppg: np.ndarray,
    gt_hr: np.ndarray,
    output: Path,
) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
    wave = pd.read_csv(waveform_path)
    hr = pd.read_csv(hr_path)
    time = wave["time_s"].to_numpy(float)
    fs = float(1.0 / np.median(np.diff(time)))
    n = min(len(wave), len(gt_ppg), len(gt_hr))
    time = time[:n]
    x_full = wave["rppg_normalized"].to_numpy(float)[:n]
    y_full = bandpass_zscore(gt_ppg[:n], fs)

    # Remove filter and POS overlap-add edge transients before waveform scoring.
    edge = int(round(3.0 * fs))
    x = x_full[edge : n - edge]
    y = y_full[edge : n - edge]
    lag, polarity, aligned_abs_r = best_alignment(x, y, int(round(2.0 * fs)))
    xa, ya = paired_at_lag(x * polarity, y, lag)
    xa = (xa - np.mean(xa)) / (np.std(xa) + 1e-12)
    ya = (ya - np.mean(ya)) / (np.std(ya) + 1e-12)

    zero_r = safe_corr(x, y)
    rmse = float(np.sqrt(np.mean((xa - ya) ** 2)))
    f_coh, cxy = coherence(xa, ya, fs=fs, nperseg=min(len(xa), 512))
    gt_peak = spectral_peak_bpm(ya, fs)
    rppg_peak = spectral_peak_bpm(xa, fs)
    peak_hz = gt_peak / 60.0
    coh_mask = np.abs(f_coh - peak_hz) <= 0.15
    fundamental_coh = float(np.mean(cxy[coh_mask])) if np.any(coh_mask) else float("nan")

    win = int(round(10.0 * fs))
    step = int(round(1.0 * fs))
    local_rows = []
    for start in range(0, len(xa) - win + 1, step):
        stop = start + win
        local_rows.append(
            {
                "time_s": float((edge + start + win / 2) / fs),
                "waveform_r": safe_corr(xa[start:stop], ya[start:stop]),
            }
        )
    local = pd.DataFrame(local_rows)

    half_window = 5.0
    hr_rows = []
    for row in hr.itertuples(index=False):
        center = float(row.time_s)
        mask = (time >= center - half_window) & (time < center + half_window)
        if not np.any(mask):
            continue
        ref = float(np.median(gt_hr[:n][mask]))
        est = float(row.bpm)
        hr_rows.append(
            {
                "time_s": center,
                "reference_bpm": ref,
                "estimated_bpm": est,
                "error_bpm": est - ref,
                "absolute_error_bpm": abs(est - ref),
                "snr_db": float(row.snr_db),
            }
        )
    windows = pd.DataFrame(hr_rows)
    errors = windows["error_bpm"].to_numpy(float)
    ref_values = windows["reference_bpm"].to_numpy(float)
    est_values = windows["estimated_bpm"].to_numpy(float)
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    error_sd = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

    metrics = {
        "method": name.upper(),
        "samples_compared": int(n),
        "sampling_rate_hz": fs,
        "edge_excluded_s": 3.0,
        "waveform_zero_lag_r": zero_r,
        "waveform_best_abs_r": aligned_abs_r,
        "waveform_polarity": polarity,
        "reference_lag_samples": int(lag),
        "reference_lag_s": float(lag / fs),
        "waveform_zscore_rmse": rmse,
        "fundamental_coherence": fundamental_coh,
        "reference_spectral_hr_bpm": gt_peak,
        "rppg_spectral_hr_bpm": rppg_peak,
        "spectral_hr_absolute_error_bpm": abs(rppg_peak - gt_peak),
        "local_waveform_r_median": float(local["waveform_r"].median()),
        "local_waveform_r_min": float(local["waveform_r"].min()),
        "local_waveform_fraction_r_ge_0_5": float((local["waveform_r"] >= 0.5).mean()),
        "hr_windows": int(len(windows)),
        "hr_mae_bpm": mae,
        "hr_rmse_bpm": float(np.sqrt(np.mean(errors**2))),
        "hr_mape_percent": float(np.mean(np.abs(errors) / ref_values) * 100.0),
        "hr_bias_bpm": bias,
        "hr_error_sd_bpm": error_sd,
        "bland_altman_lower_bpm": bias - 1.96 * error_sd,
        "bland_altman_upper_bpm": bias + 1.96 * error_sd,
        "hr_correlation_r": safe_corr(est_values, ref_values),
        "hr_fraction_within_5_bpm": float((np.abs(errors) <= 5.0).mean()),
        "median_snr_db": float(windows["snr_db"].median()),
        "worst_windows": windows.nlargest(5, "absolute_error_bpm").round(4).to_dict("records"),
    }
    windows.insert(0, "method", name.upper())
    local.insert(0, "method", name.upper())
    windows.to_csv(output / f"{name.lower()}_hr_window_comparison.csv", index=False)
    local.to_csv(output / f"{name.lower()}_waveform_window_comparison.csv", index=False)
    series = {"time": time, "rppg": x_full * polarity, "reference": y_full, "local": local, "hr": windows}
    return metrics, windows, series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--pos-dir", type=Path, required=True)
    parser.add_argument("--chrom-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = [
        np.fromstring(line, sep=" ")
        for line in args.ground_truth.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gt_ppg, gt_hr = rows[0], rows[1]
    methods = [("pos", args.pos_dir)]
    if args.chrom_dir:
        methods.append(("chrom", args.chrom_dir))

    all_metrics, all_windows, all_series = [], [], {}
    for name, directory in methods:
        metrics, windows, series = method_metrics(
            name,
            directory / "rppg_waveform.csv",
            directory / "heart_rate.csv",
            gt_ppg,
            gt_hr,
            args.output,
        )
        all_metrics.append(metrics)
        all_windows.append(windows)
        all_series[name] = series

    pd.concat(all_windows, ignore_index=True).to_csv(args.output / "hr_window_comparison.csv", index=False)
    (args.output / "comparison_metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    first = all_series[methods[0][0]]
    mask = (first["time"] >= 3) & (first["time"] <= 23)
    axes[0].plot(first["time"][mask], first["reference"][mask], label="Contact PPG (filtered)", lw=1.6)
    for name, _ in methods:
        data = all_series[name]
        axes[0].plot(data["time"][mask], data["rppg"][mask], label=f"{name.upper()} rPPG", alpha=0.8)
    axes[0].set(title="Waveform comparison (polarity corrected; lag reported separately)", ylabel="z-score")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)

    for name, _ in methods:
        local = all_series[name]["local"]
        axes[1].plot(local["time_s"], local["waveform_r"], label=name.upper())
    axes[1].axhline(0.5, color="black", ls="--", lw=1, label="r = 0.5")
    axes[1].set(title="10-second local waveform correlation", ylabel="Pearson r", ylim=(-1.05, 1.05))
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    ref_drawn = False
    for name, _ in methods:
        h = all_series[name]["hr"]
        if not ref_drawn:
            axes[2].plot(h["time_s"], h["reference_bpm"], color="black", lw=2, label="Reference HR")
            ref_drawn = True
        axes[2].plot(h["time_s"], h["estimated_bpm"], label=f"{name.upper()} HR")
    axes[2].set(title="10-second sliding-window heart rate", xlabel="Time (s)", ylabel="BPM")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.savefig(args.output / "detailed_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

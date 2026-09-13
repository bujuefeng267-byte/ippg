#!/usr/bin/env python3
"""Offline facial-video rPPG baseline using MediaPipe plus POS/CHROM.

The classical signal extraction follows the open-source pyVHR CPU pipeline:
https://github.com/phuselab/pyVHR (GPL-3.0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an estimated rPPG waveform and heart rate from a face video."
    )
    parser.add_argument("video", type=Path, help="Input MP4/AVI/MOV video")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--method", choices=("pos", "chrom"), default="pos")
    parser.add_argument("--window", type=float, default=10.0, help="BPM window (seconds)")
    parser.add_argument("--step", type=float, default=1.0, help="BPM step (seconds)")
    parser.add_argument("--min-bpm", type=float, default=42.0)
    parser.add_argument("--max-bpm", type=float, default=210.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    return parser.parse_args()


def roi_mean_rgb(frame_rgb: np.ndarray, landmarks) -> np.ndarray | None:
    """Average three conservative skin patches defined relative to the face mesh."""
    height, width = frame_rgb.shape[:2]
    xs = np.array([p.x for p in landmarks.landmark]) * width
    ys = np.array([p.y for p in landmarks.landmark]) * height
    x0, x1 = np.percentile(xs, [2, 98])
    y0, y1 = np.percentile(ys, [2, 98])
    fw, fh = x1 - x0, y1 - y0
    if fw < 40 or fh < 40:
        return None

    relative_boxes = (
        (0.32, 0.12, 0.68, 0.30),  # forehead
        (0.16, 0.46, 0.40, 0.70),  # left cheek
        (0.60, 0.46, 0.84, 0.70),  # right cheek
    )
    samples = []
    for rx0, ry0, rx1, ry1 in relative_boxes:
        xa = max(0, min(width - 1, int(x0 + rx0 * fw)))
        xb = max(xa + 1, min(width, int(x0 + rx1 * fw)))
        ya = max(0, min(height - 1, int(y0 + ry0 * fh)))
        yb = max(ya + 1, min(height, int(y0 + ry1 * fh)))
        patch = frame_rgb[ya:yb, xa:xb]
        if patch.size:
            # Reject very dark/saturated pixels while retaining broad skin-tone coverage.
            valid = np.all((patch > 20) & (patch < 245), axis=2)
            pixels = patch[valid]
            if len(pixels) >= 25:
                samples.append(np.median(pixels, axis=0))
    if not samples:
        return None
    return np.mean(samples, axis=0)


def read_rgb_trace(video: Path, max_seconds: float | None):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 5:
        raise RuntimeError(f"Invalid/too-low frame rate reported by video: {fps}")
    limit = math.inf if max_seconds is None else max(1, int(max_seconds * fps))

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rgb_values: list[np.ndarray] = []
    detected = 0
    frame_count = 0
    try:
        while frame_count < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(frame_rgb)
            value = None
            if result.multi_face_landmarks:
                value = roi_mean_rgb(frame_rgb, result.multi_face_landmarks[0])
            if value is None:
                rgb_values.append(np.array([np.nan, np.nan, np.nan]))
            else:
                rgb_values.append(value.astype(float))
                detected += 1
            frame_count += 1
    finally:
        mesh.close()
        capture.release()

    if frame_count == 0:
        raise RuntimeError("The video contains no readable frames.")
    detection_rate = detected / frame_count
    if detected < max(int(6 * fps), 30) or detection_rate < 0.35:
        raise RuntimeError(
            f"Face/skin ROI was found in only {detection_rate:.1%} of frames; "
            "use a frontal, well-lit video with the full face visible."
        )

    trace = pd.DataFrame(rgb_values, columns=["r", "g", "b"])
    trace = trace.interpolate(limit_direction="both").to_numpy(dtype=float)
    return trace, fps, detection_rate


def pos_signal(rgb: np.ndarray, fps: float) -> np.ndarray:
    """POS overlap-add implementation used by the classical pyVHR method."""
    signal = rgb.T[None, :, :]
    estimators, _, frames = signal.shape
    window = max(2, int(round(1.6 * fps)))
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    projected = np.stack([projection for _ in range(estimators)], axis=0)
    output = np.zeros((estimators, frames), dtype=float)
    eps = 1e-9
    for end in range(window - 1, frames):
        start = end - window + 1
        current = signal[:, :, start : end + 1]
        current = current / (np.mean(current, axis=2, keepdims=True) + eps)
        components = np.matmul(projected, current)
        first, second = components[:, 0, :], components[:, 1, :]
        alpha = np.std(first, axis=1, keepdims=True) / (
            np.std(second, axis=1, keepdims=True) + eps
        )
        pulse = first + alpha * second
        pulse -= np.mean(pulse, axis=1, keepdims=True)
        output[:, start : end + 1] += pulse
    return output[0]


def chrom_signal(rgb: np.ndarray, fps: float) -> np.ndarray:
    frames = len(rgb)
    window = max(2, int(round(1.6 * fps)))
    output = np.zeros(frames, dtype=float)
    weights = np.zeros(frames, dtype=float)
    eps = 1e-9
    for start in range(0, max(1, frames - window + 1), max(1, window // 2)):
        stop = min(frames, start + window)
        current = rgb[start:stop] / (np.mean(rgb[start:stop], axis=0) + eps)
        x = 3.0 * current[:, 0] - 2.0 * current[:, 1]
        y = 1.5 * current[:, 0] + current[:, 1] - 1.5 * current[:, 2]
        pulse = x - (np.std(x) / (np.std(y) + eps)) * y
        pulse -= pulse.mean()
        taper = np.hanning(len(pulse)) if len(pulse) > 3 else np.ones(len(pulse))
        output[start:stop] += pulse * taper
        weights[start:stop] += taper
    weights[weights < eps] = 1.0
    return output / weights


def bandpass(signal: np.ndarray, fps: float, min_bpm: float, max_bpm: float) -> np.ndarray:
    low = min_bpm / 60.0
    high = min(max_bpm / 60.0, fps * 0.48)
    if not 0 < low < high < fps / 2:
        raise RuntimeError("Heart-rate band is incompatible with the video frame rate.")
    sos = butter(3, [low, high], btype="bandpass", fs=fps, output="sos")
    filtered = sosfiltfilt(sos, signal)
    std = np.std(filtered)
    return (filtered - np.mean(filtered)) / (std + 1e-9)


def estimate_bpm(
    signal: np.ndarray,
    fps: float,
    window_seconds: float,
    step_seconds: float,
    min_bpm: float,
    max_bpm: float,
) -> pd.DataFrame:
    window = max(int(round(window_seconds * fps)), int(round(6 * fps)))
    step = max(1, int(round(step_seconds * fps)))
    rows = []
    for start in range(0, len(signal) - window + 1, step):
        segment = signal[start : start + window]
        frequencies, power = welch(
            segment,
            fs=fps,
            window="hann",
            nperseg=len(segment),
            nfft=max(2048, 2 ** int(np.ceil(np.log2(len(segment))))),
            detrend="constant",
        )
        band = (frequencies >= min_bpm / 60.0) & (frequencies <= max_bpm / 60.0)
        if not np.any(band):
            continue
        band_f = frequencies[band]
        band_p = power[band]
        peak_index = int(np.argmax(band_p))
        peak_hz = float(band_f[peak_index])
        signal_band = np.abs(band_f - peak_hz) <= 0.10
        wanted = float(np.sum(band_p[signal_band]))
        unwanted = float(np.sum(band_p[~signal_band]))
        snr_db = 10.0 * np.log10((wanted + 1e-12) / (unwanted + 1e-12))
        rows.append(
            {
                "time_s": (start + window / 2) / fps,
                "bpm": peak_hz * 60.0,
                "snr_db": snr_db,
            }
        )
    if not rows:
        raise RuntimeError(
            f"Video is too short for a {window_seconds:g}-second BPM window."
        )
    return pd.DataFrame(rows)


def analyze(video: Path, output: Path, method: str, args: argparse.Namespace) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rgb, fps, detection_rate = read_rgb_trace(video, args.max_seconds)
    raw = pos_signal(rgb, fps) if method == "pos" else chrom_signal(rgb, fps)
    rppg = bandpass(raw, fps, args.min_bpm, args.max_bpm)
    bpm = estimate_bpm(
        rppg, fps, args.window, args.step, args.min_bpm, args.max_bpm
    )
    times = np.arange(len(rgb)) / fps
    waveform = pd.DataFrame(
        {
            "time_s": times,
            "r": rgb[:, 0],
            "g": rgb[:, 1],
            "b": rgb[:, 2],
            "rppg_normalized": rppg,
        }
    )
    waveform.to_csv(output / "rppg_waveform.csv", index=False)
    bpm.to_csv(output / "heart_rate.csv", index=False)

    summary = {
        "input_video": str(video.resolve()),
        "method": method.upper(),
        "fps": fps,
        "frames": int(len(rgb)),
        "duration_s": len(rgb) / fps,
        "face_detection_rate": detection_rate,
        "median_bpm": float(bpm["bpm"].median()),
        "mean_bpm": float(bpm["bpm"].mean()),
        "bpm_std": float(bpm["bpm"].std(ddof=0)),
        "median_snr_db": float(bpm["snr_db"].median()),
        "warning": "Camera-derived rPPG is an estimate, not contact-PPG ground truth.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(times, rppg, linewidth=0.8, color="#b2182b")
    axes[0].set(title=f"Estimated rPPG waveform ({method.upper()})", ylabel="Normalized amplitude")
    axes[0].grid(alpha=0.25)
    axes[1].plot(bpm["time_s"], bpm["bpm"], marker="o", markersize=2.5, color="#2166ac")
    axes[1].axhline(summary["median_bpm"], linestyle="--", color="#4d4d4d", label=f"median {summary['median_bpm']:.1f} bpm")
    axes[1].set(xlabel="Time (s)", ylabel="Heart rate (bpm)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(video.name)
    fig.savefig(output / "rppg_report.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> int:
    args = parse_args()
    video = args.video.expanduser()
    if not video.is_file():
        print(f"ERROR: input video does not exist: {video}", file=sys.stderr)
        return 2
    output = args.output or Path("results") / video.stem / args.method
    try:
        summary = analyze(video, output, args.method, args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results written to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

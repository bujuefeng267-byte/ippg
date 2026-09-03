#!/usr/bin/env python3
"""Scan a face video for motion-heavy intervals before running rPPG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return np.clip((values - med) / (1.4826 * mad + 1e-9), -3.0, 8.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=4.0)
    parser.add_argument("--window", type=float, default=30.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / args.sample_hz)))
    sample_dt = stride / fps

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rows = []
    previous_gray = None
    previous_face = None
    frame_index = 0
    try:
        while frame_index < total:
            ok = cap.grab()
            if not ok:
                break
            if frame_index % stride:
                frame_index += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            scale = 640.0 / frame.shape[1]
            small = cv2.resize(frame, (640, max(1, int(frame.shape[0] * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            result = mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            face = None
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                xs = np.array([p.x for p in lm]) * small.shape[1]
                ys = np.array([p.y for p in lm]) * small.shape[0]
                x0, x1 = np.percentile(xs, [2, 98])
                y0, y1 = np.percentile(ys, [2, 98])
                face = np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])

            center_speed = np.nan
            scale_speed = np.nan
            flow = np.nan
            if face is not None and previous_face is not None:
                norm = max(1.0, (face[3] + previous_face[3]) / 2)
                center_speed = float(np.linalg.norm(face[:2] - previous_face[:2]) / norm / sample_dt)
                scale_speed = float(abs(np.log((face[3] + 1e-9) / (previous_face[3] + 1e-9))) / sample_dt)
            if previous_gray is not None:
                dense = cv2.calcOpticalFlowFarneback(
                    previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.linalg.norm(dense, axis=2)
                if face is not None:
                    cx, cy, fw, fh = face
                    xa = max(0, int(cx - 0.4 * fw)); xb = min(gray.shape[1], int(cx + 0.4 * fw))
                    ya = max(0, int(cy - 0.4 * fh)); yb = min(gray.shape[0], int(cy + 0.4 * fh))
                    region = magnitude[ya:yb, xa:xb]
                    flow = float(np.median(region)) if region.size else np.nan
                else:
                    flow = float(np.median(magnitude))

            rows.append({
                "time_s": frame_index / fps,
                "face_found": int(face is not None),
                "center_speed_face_heights_s": center_speed,
                "scale_change_s": scale_speed,
                "face_flow_px_sample": flow,
                "blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            })
            previous_gray = gray
            previous_face = face
            frame_index += 1
    finally:
        mesh.close()
        cap.release()

    data = pd.DataFrame(rows)
    for column in ("center_speed_face_heights_s", "scale_change_s", "face_flow_px_sample"):
        data[column] = data[column].interpolate(limit_direction="both")
    data["motion_score"] = (
        robust_z(data["center_speed_face_heights_s"].to_numpy())
        + 0.5 * robust_z(data["scale_change_s"].to_numpy())
        + robust_z(data["face_flow_px_sample"].to_numpy())
    ) / 2.5
    n = max(1, int(round(args.window / sample_dt)))
    data["motion_score_window"] = data["motion_score"].rolling(n, center=True, min_periods=max(2, n // 2)).median()
    best_i = int(data["motion_score_window"].fillna(-np.inf).idxmax())
    best_center = float(data.loc[best_i, "time_s"])
    best_start = max(0.0, best_center - args.window / 2)
    duration = total / fps
    best_start = min(best_start, max(0.0, duration - args.window))
    best_stop = min(duration, best_start + args.window)

    summary = {
        "input_video": str(args.video.resolve()),
        "fps": fps,
        "duration_s": duration,
        "sample_hz_actual": 1.0 / sample_dt,
        "samples": len(data),
        "face_detection_rate": float(data["face_found"].mean()),
        "median_center_speed_face_heights_s": float(data["center_speed_face_heights_s"].median()),
        "p95_center_speed_face_heights_s": float(data["center_speed_face_heights_s"].quantile(0.95)),
        "median_face_flow_px_sample": float(data["face_flow_px_sample"].median()),
        "p95_face_flow_px_sample": float(data["face_flow_px_sample"].quantile(0.95)),
        "best_motion_window_start_s": best_start,
        "best_motion_window_stop_s": best_stop,
    }
    data.to_csv(args.output / "motion_timeseries.csv", index=False)
    (args.output / "motion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(data.time_s, data.center_speed_face_heights_s, lw=0.8)
    axes[0].set_ylabel("Face center speed\n(face heights/s)")
    axes[1].plot(data.time_s, data.face_flow_px_sample, lw=0.8, color="tab:orange")
    axes[1].set_ylabel("Median face flow\n(px/sample)")
    axes[2].plot(data.time_s, data.motion_score, lw=0.5, alpha=0.4)
    axes[2].plot(data.time_s, data.motion_score_window, lw=1.4, color="tab:red")
    axes[2].axvspan(best_start, best_stop, color="tab:red", alpha=0.15)
    axes[2].set_ylabel("Motion score")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(args.output / "motion_scan.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

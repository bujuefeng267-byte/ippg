#!/usr/bin/env python3
"""Auditable offline POS/CHROM motion experiment; original baseline is unchanged.

NLMS idea: Zhu et al., doi:10.1109/TBME.2024.3442785.
The dynamic-programming ridge below is NOT a reproduction of their AMTC.
Signal extraction imports the existing project baseline (pyVHR-derived).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import welch

from analyze_rppg import bandpass, chrom_signal, pos_signal, roi_mean_rgb


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_short_gaps(values, max_frames):
    """Fill only bounded internal gaps. Never extrapolate or partly fill long gaps."""
    out = np.array(values, dtype=float, copy=True)
    valid = np.isfinite(out).all(axis=1)
    filled = np.zeros(len(out), bool)
    for a, b in runs(~valid):
        if a > 0 and b < len(out) and b - a <= max_frames:
            alpha = np.arange(1, b - a + 1)[:, None] / (b - a + 1)
            out[a:b] = out[a - 1] + alpha * (out[b] - out[a - 1])
            filled[a:b] = True
    return out, filled


def mesh_object(points, width, height):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x / width, y=y / height)
                                     for x, y in points])


def flow_affine(previous, current, points):
    """Forward/backward LK + robust partial affine; rejects unreliable bridges."""
    if points is None or previous is None:
        return None
    p = points[::5].astype(np.float32).reshape(-1, 1, 2)
    q, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, p, None,
                                     winSize=(21, 21), maxLevel=3)
    if q is None:
        return None
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(current, previous, q, None,
                                         winSize=(21, 21), maxLevel=3)
    if back is None:
        return None
    good = ok.ravel().astype(bool) & ok2.ravel().astype(bool)
    good &= np.linalg.norm(back[:, 0] - p[:, 0], axis=1) < 1.5
    good &= np.isfinite(q[:, 0]).all(axis=1)
    if good.sum() < 20 or good.mean() < 0.6:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(p[good, 0], q[good, 0],
                                                method=cv2.RANSAC,
                                                ransacReprojThreshold=2.0)
    if matrix is None or inliers.mean() < 0.7:
        return None
    scale = np.sqrt(np.linalg.det(matrix[:, :2]))
    if not 0.90 <= scale <= 1.10:
        return None
    projected = cv2.transform(points[None].astype(np.float32), matrix)[0]
    h, w = current.shape
    inside = ((projected[:, 0] >= 0) & (projected[:, 0] < w) &
              (projected[:, 1] >= 0) & (projected[:, 1] < h))
    if inside.mean() < 0.95:
        return None
    return projected


def extract(video, frontend, max_seconds=None, detection_width=960, max_bridge_s=0.2, qa_dir=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps < 5:
        raise ValueError("Invalid fps")
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
    fallback = (mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                              min_detection_confidence=0.5)
                if frontend == "robust" else None)
    previous, previous_points = None, None
    bridge_age, rows, index, qa_tracked = 0, [], 0, 0
    limit = float("inf") if max_seconds is None else round(max_seconds * fps)
    try:
        while index < limit:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            ratio = min(1.0, detection_width / w) if frontend == "robust" else 1.0
            small = cv2.resize(rgb, (round(w * ratio), round(h * ratio))) if ratio < 1 else rgb
            sh, sw = small.shape[:2]
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            prediction = flow_affine(previous, gray, previous_points) if frontend == "robust" else None
            result = mesh.process(small)
            source, points = "missing", None
            if result.multi_face_landmarks:
                source = "mesh"
            elif fallback is not None:
                result = fallback.process(small)
                if result.multi_face_landmarks:
                    source = "redetected"
            if result.multi_face_landmarks:
                points = np.array([(p.x * sw, p.y * sh)
                                   for p in result.multi_face_landmarks[0].landmark])
                bridge_age = 0
            elif prediction is not None and bridge_age < round(max_bridge_s * fps):
                points, source = prediction, "flow_tracked"
                bridge_age += 1
            else:
                bridge_age += 1
            value = None
            if points is not None:
                full = points * np.array([w / sw, h / sh])
                value = roi_mean_rgb(rgb, mesh_object(full, w, h))
            motion = np.array([np.nan, np.nan])
            if previous_points is not None:
                if prediction is not None:
                    motion = np.median(prediction - previous_points, axis=0) / sh
                elif frontend == "baseline" and points is not None:
                    motion = (np.median(points, axis=0) - np.median(previous_points, axis=0)) / sh
            bbox = [np.nan] * 4
            if points is not None:
                lo = np.percentile(points, 2, axis=0) / [sw, sh]
                hi = np.percentile(points, 98, axis=0) / [sw, sh]
                bbox = [lo[0], lo[1], hi[0], hi[1]]
                if qa_dir and (index % 300 == 0 or (source == "flow_tracked" and qa_tracked < 6)):
                    qa_dir.mkdir(exist_ok=True)
                    overlay = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                    for x, y in points[::5]:
                        cv2.circle(overlay, (round(x), round(y)), 1, (0, 255, 0), -1)
                    cv2.putText(overlay, f"frame {index} {source}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imwrite(str(qa_dir / f"{index:05d}_{source}.jpg"), overlay)
                    if source == "flow_tracked":
                        qa_tracked += 1
            rows.append(dict(frame=index, time_s=index / fps, source=source,
                             rgb_valid=value is not None,
                             r=np.nan if value is None else value[0],
                             g=np.nan if value is None else value[1],
                             b=np.nan if value is None else value[2],
                             motion_x=motion[0], motion_y=motion[1],
                             face_x0=bbox[0], face_y0=bbox[1], face_x1=bbox[2], face_y1=bbox[3]))
            previous, previous_points = gray, points
            index += 1
            if index % 300 == 0:
                print(f"{frontend}: extracted {index} frames", flush=True)
    finally:
        cap.release()
        mesh.close()
        if fallback is not None:
            fallback.close()
    if not rows:
        raise ValueError("No frames decoded")
    return pd.DataFrame(rows), fps


def nlms_residual(signal, motion, taps=8, mu=0.1):
    """Causal adaptive noise canceller, zero-initialized per valid run.

    Missing motion clears tap history and passes the signal unchanged. It is NOT
    proof of pulse preservation when motion and pulse frequencies coincide.
    """
    if taps < 1 or not 0 < mu < 2:
        raise ValueError("Invalid NLMS settings")
    history = np.zeros((taps, 2))
    weights = np.zeros(taps * 2)
    out = np.asarray(signal, float).copy()
    for i, (sample, m) in enumerate(zip(signal, motion)):
        if not np.isfinite(m).all():
            history[:] = 0
            continue
        history[1:] = history[:-1]
        history[0] = m
        reference = history.ravel()
        out[i] = sample - weights @ reference
        energy = reference @ reference
        if energy > 1e-12:
            weights += mu * out[i] * reference / (energy + 1e-9)
    return out


def gap_frame_limit(gap_s, fps):
    """A duration limit must not round upward past the requested seconds."""
    if not (np.isfinite(gap_s) and gap_s >= 0 and np.isfinite(fps) and fps > 0):
        raise ValueError('Invalid gap duration or FPS')
    return int(np.floor(gap_s * fps + 1e-9))


def make_waveforms(trace, fps, method, gap_s, min_bpm, max_bpm):
    rgb, interpolated = fill_short_gaps(trace[["r", "g", "b"]].to_numpy(), gap_frame_limit(gap_s, fps))
    motion = trace[["motion_x", "motion_y"]].to_numpy()
    finite = np.isfinite(rgb).all(axis=1)
    raw_out, filtered_out = np.full(len(rgb), np.nan), np.full(len(rgb), np.nan)
    # Do not filter across unobserved gaps or score filter/overlap-add edges.
    edge = round(1.6 * fps)
    for a, b in runs(finite):
        if b - a < max(round(4 * fps), 2 * edge + 1):
            continue
        raw = (pos_signal if method == "pos" else chrom_signal)(rgb[a:b], fps)
        base = bandpass(raw, fps, min_bpm, max_bpm)
        m = motion[a:b].copy()
        # Reference filtering restricted to observed motion runs; no zero filling.
        for c, d in runs(np.isfinite(m).all(axis=1)):
            if d - c < round(2 * fps):
                m[c:d] = np.nan
            else:
                for axis in range(2):
                    if np.std(m[c:d, axis]) > 1e-10:
                        m[c:d, axis] = bandpass(m[c:d, axis], fps, min_bpm, max_bpm)
                    else:
                        m[c:d, axis] = 0
        residual = nlms_residual(base, m)
        raw_out[a + edge:b - edge] = base[edge:-edge]
        filtered_out[a + edge:b - edge] = residual[edge:-edge]
    return raw_out, filtered_out, interpolated


def ridge_path(spectra, bpm_grid, step_s, max_rate=12.0, penalty=0.05):
    """Offline DP ridge with bounded transitions; resets across rejected windows.

    Uses future windows (not real-time). Not AMTC and not a confidence estimator.
    """
    if len(spectra) == 0:
        return np.array([], int)
    spectra = np.asarray(spectra)
    if not np.isfinite(spectra).all() or np.any(spectra < 0):
        raise ValueError("Ridge spectra must be finite and nonnegative")
    logpower = np.log(np.maximum(spectra / np.maximum(spectra.max(axis=1, keepdims=True), 1e-30), 1e-12))
    delta = bpm_grid[:, None] - bpm_grid[None, :]
    transition = -penalty * (delta / max(step_s, 1e-6)) ** 2
    transition[np.abs(delta) > max_rate * step_s + 1e-9] = -np.inf
    score = logpower[0].copy()
    back = np.zeros(spectra.shape, int)
    for i in range(1, len(spectra)):
        candidate = score[:, None] + transition
        back[i] = np.argmax(candidate, axis=0)
        score = candidate[back[i], np.arange(len(score))] + logpower[i]
    path = np.zeros(len(spectra), int)
    path[-1] = np.argmax(score)
    for i in range(len(path) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def estimate(signal, trace, interpolated, fps, window_s=10, step_s=1,
             min_bpm=42, max_bpm=210):
    window, step = round(window_s * fps), round(step_s * fps)
    if window < 6 * fps or step < 1:
        raise ValueError("Use windows >= 6 seconds and a positive step")
    rows, powers = [], []
    grid = np.arange(min_bpm, max_bpm + 0.01, 1.0)
    for a in range(0, len(signal) - window + 1, step):
        b = a + window
        segment = signal[a:b]
        valid_fraction = float(trace.rgb_valid.iloc[a:b].mean())
        filled_fraction = float(interpolated[a:b].mean())
        reason = "candidate_only"
        p = np.zeros(len(grid))
        peak, concentration = np.nan, np.nan
        if not np.isfinite(segment).all():
            reason = "gap_or_filter_edge"
        elif valid_fraction < 0.9 or filled_fraction > 0.1:
            reason = "insufficient_observed_rgb"
        elif np.std(segment) < 1e-8:
            reason = "flat_signal"
        else:
            f, power = welch(segment, fs=fps, nperseg=window, nfft=max(2048, window), detrend="constant")
            p = np.interp(grid / 60, f, power)
            k = np.argmax(p)
            peak = grid[k]
            concentration = p[np.abs(grid - peak) <= 6].sum() / max(p.sum(), 1e-30)
            if concentration < 0.12:
                reason = "diffuse_spectrum"
        rows.append(dict(time_s=(a + window / 2) / fps,
                         observed_fraction=valid_fraction, interpolated_fraction=filled_fraction,
                         spectral_peak_bpm=peak, peak_concentration=concentration,
                         accepted=reason == "candidate_only", status=reason, ridge_bpm=np.nan))
        powers.append(p)
    table = pd.DataFrame(rows, columns=["time_s", "observed_fraction", "interpolated_fraction",
                                      "spectral_peak_bpm", "peak_concentration", "accepted", "status", "ridge_bpm"])
    for a, b in runs(table.accepted.to_numpy(bool)):
        path = ridge_path(powers[a:b], grid, step / fps)
        table.loc[a:b - 1, "ridge_bpm"] = grid[path]
    return table


def reference_metrics(table, path, window_s):
    """UBFC provided HR, paired at video elapsed time; no fitted temporal shift."""
    gt = np.loadtxt(path)
    if gt.ndim != 2 or gt.shape[0] != 3:
        raise ValueError("Expected UBFC ground_truth.txt with 3 rows: PPG, HR, time")
    t = gt[2] - gt[2, 0]
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError("Reference timestamps must be finite and nondecreasing")
    # A duplicate timestamp exists in UBFC subject1. Merge simultaneous samples.
    reference = pd.DataFrame(dict(time=t, hr=gt[1])).groupby("time", sort=True).hr.mean()
    t, hr = reference.index.to_numpy(), reference.to_numpy()
    targets = []
    for center in table.time_s:
        a, b = center - window_s / 2, center + window_s / 2
        selection = (t >= a) & (t < b) & np.isfinite(hr) & (hr > 0)
        covered = a >= t[0] and b <= t[-1] + np.median(np.diff(t))
        targets.append(float(np.mean(hr[selection])) if covered and selection.sum() > 1 else np.nan)
    table["reference_bpm"] = targets
    metrics = {}
    for column in ("spectral_peak_bpm", "ridge_bpm"):
        mask = table.accepted & np.isfinite(table[column]) & np.isfinite(table.reference_bpm)
        difference = table.loc[mask, column] - table.loc[mask, "reference_bpm"]
        metrics[column] = dict(n=int(mask.sum()),
                               mae_bpm=float(np.abs(difference).mean()) if len(difference) else None,
                               rmse_bpm=float(np.sqrt(np.mean(difference ** 2))) if len(difference) else None)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frontend", choices=["baseline", "robust"], default="robust")
    parser.add_argument("--cache", type=Path, help="Reuse frame trace produced by this exact version")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-gap", type=float, default=0.1)
    parser.add_argument("--window", type=float, default=10)
    parser.add_argument("--step", type=float, default=1)
    parser.add_argument("--min-bpm", type=float, default=42)
    parser.add_argument("--max-bpm", type=float, default=210)
    parser.add_argument("--reference-ubfc", type=Path)
    parser.add_argument("--nlms", action="store_true", help="Also export experimental NLMS variants (off by default)")
    args = parser.parse_args()
    if (args.max_gap < 0 or args.window < 6 or args.step <= 0 or
            not 0 < args.min_bpm < args.max_bpm or
            (args.max_seconds is not None and args.max_seconds <= 0)):
        parser.error("Invalid gap, window, step, BPM range or duration")
    args.output.mkdir(parents=True, exist_ok=False)
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = dict(video=str(args.video.resolve()), size=args.video.stat().st_size,
                    mtime_ns=args.video.stat().st_mtime_ns, frontend=args.frontend,
                    max_seconds=args.max_seconds, source_sha256=version)
    if args.cache:
        cache_meta = json.loads(args.cache.with_suffix(".json").read_text())
        if cache_meta["identity"] != identity:
            raise ValueError("Cache does not match input, frontend or code version")
        trace, fps = pd.read_csv(args.cache), cache_meta["fps"]
    else:
        trace, fps = extract(args.video, args.frontend, args.max_seconds, qa_dir=args.output / "qa")
    trace.to_csv(args.output / "frame_trace.csv", index=False)
    (args.output / "frame_trace.json").write_text(json.dumps(dict(identity=identity, fps=fps), indent=2))
    summary = dict(identity=identity, fps=fps, frames=len(trace),
                   observed_rgb_fraction=float(trace.rgb_valid.mean()),
                   source_counts=trace.source.value_counts().to_dict(),
                   config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   warning="Experimental offline estimates. Coverage/continuity are not accuracy. DP uses future windows. NLMS may suppress pulse correlated with motion.",
                   variants={})
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for method in ("pos", "chrom"):
        base, cleaned, filled = make_waveforms(trace, fps, method, args.max_gap, args.min_bpm, args.max_bpm)
        pd.DataFrame(dict(time_s=trace.time_s, base=base, nlms=cleaned,
                          interpolated=filled, observed=trace.rgb_valid)).to_csv(args.output / f"{method}_waveform.csv", index=False)
        variants = [(method, base)]
        if args.nlms:
            variants.append((method + "_nlms", cleaned))
        for label, waveform in variants:
            table = estimate(waveform, trace, filled, fps, args.window, args.step, args.min_bpm, args.max_bpm)
            info = dict(total_windows=len(table), accepted_windows=int(table.accepted.sum()),
                        accepted_fraction=float(table.accepted.mean()) if len(table) else 0,
                        reference=None)
            if args.reference_ubfc:
                info["reference"] = reference_metrics(table, args.reference_ubfc, args.window)
            table.to_csv(args.output / f"{label}_heart_rate.csv", index=False)
            summary["variants"][label] = info
            accepted_peak = table.spectral_peak_bpm.where(table.accepted)
            axes[1].plot(table.time_s, accepted_peak, label=label, linewidth=1)
            axes[2].plot(table.time_s, table.ridge_bpm, label=label, linewidth=1)
    if args.reference_ubfc:
        for ax in axes[1:]:
            ax.plot(table.time_s, table.reference_bpm, color="black", linestyle="--", label="UBFC reference")
    axes[0].plot(trace.time_s, trace.rgb_valid.astype(int), linewidth=0.7)
    axes[0].set(ylabel="Observed RGB", ylim=(-0.05, 1.05), title=args.frontend + " / " + args.video.name)
    for ax, title in zip(axes[1:], ["Local spectral peak (quality-screened)", "Offline DP ridge (not AMTC)"]):
        ax.set(ylabel="Estimated bpm", title=title, ylim=(args.min_bpm, args.max_bpm))
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.2)
    axes[2].set_xlabel("Video elapsed time (s)")
    fig.savefig(args.output / "comparison.png", dpi=150)
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

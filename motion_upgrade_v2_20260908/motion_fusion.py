"""Auditable multi-ROI spectral consensus and forward HR gating.

Input signals are already filtered/edge-trimmed offline by the frozen frontend.
No reference HR/PPG is accepted or read. ``ridge_bpm`` is a compatibility name
for a forward continuity-gated candidate, NOT the V1 future-aware DP ridge.
Each HR row uses its complete window and past decisions; it is only available
at ``window_end_s`` (not its center timestamp). Appending identical-prefix
preprocessed inputs cannot rewrite earlier HR rows. This is not an end-to-end
causal pipeline: preprocessing and waveform overlap-add remain offline.

The waveform retains actual broadband input samples. Its polarity and amplitude
are arbitrary, and it is not a claimed reconstruction of pressure morphology.
Only accepted windows contribute; samples covered by no accepted window are
NaN. An accepted overlapping window can cover part of a rejected window, but
rejected HR rows are always NaN, never held or interpolated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch


@dataclass(frozen=True)
class FusionConfig:
    """Defaults fixed using synthetic cases only; quality is not probability."""

    roi_names: tuple[str, ...] = ("forehead", "left_cheek", "right_cheek")
    methods: tuple[str, ...] = ("pos", "chrom")
    min_rois: int = 2
    min_observed_fraction: float = 0.90
    min_roi_quality: float = 0.20
    min_signal_std: float = 1e-8
    grid_step_bpm: float = 1.0
    support_tolerance_bpm: float = 6.0
    max_peaks_per_channel: int = 4
    min_relative_power: float = 0.20
    min_peak_concentration: float = 0.15
    min_prominence_ratio: float = 0.08
    single_method_factor: float = 0.65
    min_consensus_quality: float = 0.20
    ambiguity_ratio: float = 0.80
    max_motion_penalty: float = 0.25
    max_slew_bpm_per_s: float = 12.0
    rejected_windows_to_reset: int = 2
    min_waveform_correlation: float = 0.15
    normalize_clip: float = 8.0
    huber_delta: float = 2.5

    def __post_init__(self):
        if (len(set(self.roi_names)) != len(self.roi_names) or
                len(set(self.methods)) != len(self.methods) or
                not self.roi_names or not self.methods):
            raise ValueError("ROI and method names must be unique and nonempty")
        if not 2 <= self.min_rois <= len(self.roi_names):
            raise ValueError("min_rois must require at least two distinct ROIs")
        for name in ("min_observed_fraction", "min_roi_quality",
                     "min_relative_power", "min_peak_concentration",
                     "min_prominence_ratio", "single_method_factor",
                     "min_consensus_quality", "ambiguity_ratio",
                     "min_waveform_correlation"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.max_motion_penalty < 1:
            raise ValueError("motion penalty must remain soft (<1)")
        for name in ("min_signal_std", "grid_step_bpm", "support_tolerance_bpm",
                     "max_slew_bpm_per_s", "normalize_clip", "huber_delta"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_peaks_per_channel < 2 or self.rejected_windows_to_reset < 1:
            raise ValueError("at least two peaks and a positive reset count required")


DEFAULT_CONFIG = FusionConfig()

TABLE_COLUMNS = [
    "time_s", "window_start_s", "window_end_s", "accepted", "status",
    "spectral_peak_bpm", "ridge_bpm", "roi_count", "quality_proxy", "state",
    "available_roi_count", "method_agreement", "motion_overlap",
    "consensus_score", "runner_up_ratio", "candidate_group_count",
    "waveform_roi_count",
]
DIAGNOSTIC_COLUMNS = [
    "window_index", "time_s", "channel", "roi", "method", "channel_status",
    "observed_fraction", "roi_quality", "signal_std", "candidate_rank",
    "candidate_bpm", "relative_power", "peak_concentration", "prominence_ratio",
    "candidate_weight", "motion_overlap", "motion_weight",
    "selected_consensus", "output_accepted", "waveform_weight",
]


def _numeric_column(trace, name):
    values = pd.to_numeric(trace[name], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    # Infinite metadata must remain missing, not become an enormous quality
    # weight through np.nan_to_num in the window summary.
    values = values.copy()
    values[~np.isfinite(values)] = np.nan
    return values


def _power(segment, fps, grid):
    # A single full-window Hann periodogram via Welch; padding refines the grid,
    # it does not increase the physical resolution of a ten-second window.
    nfft = max(2048, 2 ** int(np.ceil(np.log2(len(segment) * 4))))
    freq, power = welch(segment, fs=fps, window="hann", nperseg=len(segment),
                        noverlap=0, nfft=nfft, detrend="constant")
    return np.interp(grid / 60.0, freq, power)


def _motion_profile(trace, start, stop, fps, grid, config):
    profiles = []
    for name in ("motion_x", "motion_y"):
        if name not in trace:
            continue
        x = _numeric_column(trace.iloc[start:stop], name)
        # Unknown motion is not filled or treated as evidence against pulse.
        if not np.isfinite(x).all() or np.std(x) < config.min_signal_std:
            continue
        p = _power(x, fps, grid)
        if p.max() > 0:
            profiles.append(p / p.max())
    return np.max(profiles, axis=0) if profiles else np.zeros(len(grid))


def _peaks(segment, fps, grid, config):
    p = _power(segment, fps, grid)
    maximum, total = float(p.max()), float(p.sum())
    if maximum <= 0 or total <= 0:
        return []
    normalized = p / maximum
    # Padding permits legitimate boundary peaks; no preference for low BPM.
    indices, properties = find_peaks(
        np.r_[0.0, normalized, 0.0],
        distance=max(1, round(config.support_tolerance_bpm / config.grid_step_bpm)),
        height=config.min_relative_power, prominence=config.min_prominence_ratio)
    candidates = []
    for j, padded_index in enumerate(indices):
        k = padded_index - 1
        if not 0 <= k < len(grid):
            continue
        concentration = float(p[np.abs(grid - grid[k]) <=
                                config.support_tolerance_bpm].sum() / total)
        if concentration < config.min_peak_concentration:
            continue
        candidates.append(dict(candidate_bpm=float(grid[k]),
                               relative_power=float(normalized[k]),
                               peak_concentration=concentration,
                               prominence_ratio=float(properties["prominences"][j])))
    candidates.sort(key=lambda x: (-x["relative_power"], x["candidate_bpm"]))
    return candidates[:config.max_peaks_per_channel]


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    weights = np.asarray(weights)[order]
    return float(np.asarray(values)[order][
        np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def _best_members(candidates, center, tolerance):
    members = {}
    for c in candidates:
        if abs(c["candidate_bpm"] - center) > tolerance + 1e-9:
            continue
        key = (c["roi"], c["method"])
        if key not in members or c["candidate_weight"] > members[key]["candidate_weight"]:
            members[key] = c
    return list(members.values())


def _groups(candidates, config):
    """Hypothesize frequencies; cap support to one vote per anatomical ROI."""
    groups, seen = [], set()
    for seed in candidates:
        center = seed["candidate_bpm"]
        members = _best_members(candidates, center, config.support_tolerance_bpm)
        center = _weighted_median([c["candidate_bpm"] for c in members],
                                  [c["candidate_weight"] for c in members])
        members = _best_members(candidates, center, config.support_tolerance_bpm)
        identity = tuple(sorted(c["diag_id"] for c in members))
        if identity in seen:
            continue
        seen.add(identity)
        rois = {}
        for c in members:
            rois.setdefault(c["roi"], []).append(c)
        if len(rois) < config.min_rois:
            continue
        roi_scores = {}
        for roi, channels in rois.items():
            agreement = (len(channels) - 1) / max(1, len(config.methods) - 1)
            factor = config.single_method_factor + (1 - config.single_method_factor) * agreement
            roi_scores[roi] = max(c["candidate_weight"] for c in channels) * factor
        base_score = float(sum(roi_scores.values()))
        quality = base_score / len(rois)
        if quality < config.min_consensus_quality:
            continue
        motion = float(np.average([c["motion_overlap"] for c in members],
                                 weights=[c["candidate_weight"] for c in members]))
        groups.append(dict(bpm=center, members=members, rois=rois,
                           roi_scores=roi_scores, roi_count=len(rois), quality=quality,
                           base_score=base_score, motion=motion,
                           score=base_score * (1 - config.max_motion_penalty * motion),
                           method_agreement=float(np.mean([
                               len(v) / len(config.methods) for v in rois.values()]))))
    groups.sort(key=lambda g: (-g["score"], -g["roi_count"], g["bpm"]))
    distinct = []
    for group in groups:
        # Do not call nearby hypotheses within one spectral lobe competitors.
        if all(abs(group["bpm"] - g["bpm"]) > 2 * config.support_tolerance_bpm
               for g in distinct):
            distinct.append(group)
    return distinct


def _normalize(x, config):
    y = np.asarray(x, float) - np.median(x)
    scale = 1.4826 * np.median(np.abs(y))
    if scale < config.min_signal_std:
        scale = np.std(y)
    return np.clip(y / max(scale, config.min_signal_std),
                   -config.normalize_clip, config.normalize_clip)


def _correlation(a, b):
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-15 else 0.0


def _fuse_waveform(signals, start, stop, group, config):
    roi_signals, roi_weights, used_channels = {}, {}, {}
    for roi, channels in group["rois"].items():
        ordered = sorted(channels, key=lambda c: (-c["candidate_weight"], c["channel"]))
        anchor = _normalize(signals[ordered[0]["channel"]][start:stop], config)
        z, weights, used = [], [], []
        for c in ordered:
            value = _normalize(signals[c["channel"]][start:stop], config)
            corr = _correlation(value, anchor)
            if abs(corr) < config.min_waveform_correlation:
                continue
            z.append(value if corr >= 0 else -value)
            weights.append(c["candidate_weight"])
            used.append(c)
        if z:
            roi_signals[roi] = np.average(z, axis=0, weights=weights)
            roi_weights[roi] = group["roi_scores"][roi]
            used_channels[roi] = used
    if not roi_signals:
        return None, 0, {}
    anchor_roi = sorted(roi_signals, key=lambda r: (-roi_weights[r], r))[0]
    anchor = roi_signals[anchor_roi]
    z, weights, used = [], [], {}
    for roi in sorted(roi_signals):
        corr = _correlation(roi_signals[roi], anchor)
        if abs(corr) < config.min_waveform_correlation:
            continue
        z.append(roi_signals[roi] if corr >= 0 else -roi_signals[roi])
        weights.append(roi_weights[roi])
        total = sum(c["candidate_weight"] for c in used_channels[roi])
        for c in used_channels[roi]:
            used[c["diag_id"]] = roi_weights[roi] * c["candidate_weight"] / total
    if len(z) < config.min_rois:
        # HR evidence can agree in frequency yet have incoherent wave shapes.
        return None, len(z), {}
    stack = np.asarray(z)
    residual = np.abs(stack - np.median(stack, axis=0))
    robust = np.minimum(1.0, config.huber_delta / np.maximum(residual, 1e-12))
    w = np.asarray(weights)[:, None] * robust
    return np.sum(stack * w, axis=0) / np.sum(w, axis=0), len(z), used


def fuse_windows(signals: dict[str, np.ndarray], trace: pd.DataFrame, fps: float,
                 window_s=10, step_s=1, min_bpm=42, max_bpm=210, *,
                 config: FusionConfig | None = None):
    """Return ``(HR table, actual-signal fused waveform, candidate diagnostics)``.

    Signal names are ``<roi>_<method>``; missing channels are permitted. Trace
    ``<roi>_valid`` and ``<roi>_quality`` columns must represent per-frame values
    in [0,1]. A missing metadata column makes its ROI unavailable. Each channel
    must be finite throughout a window: this module does not interpolate gaps.
    Extra trace columns (including any reference columns) are never read.
    ``quality_proxy`` is an uncalibrated consensus score, never an accuracy or
    confidence probability. ``roi_count`` is support for the selected hypothesis;
    it can be nonzero for an explicitly rejected ambiguous/jumping hypothesis.
    """
    config = DEFAULT_CONFIG if config is None else config
    if not isinstance(config, FusionConfig):
        raise TypeError("config must be FusionConfig")
    if not (np.isfinite(fps) and fps > 0 and np.isfinite(window_s) and
            window_s >= 6 and np.isfinite(step_s) and step_s > 0 and
            np.isfinite(min_bpm) and np.isfinite(max_bpm) and
            0 < min_bpm < max_bpm < fps * 30):
        raise ValueError("Use a >=6s window, positive FPS/step and BPM below Nyquist")
    window, step = round(window_s * fps), round(step_s * fps)
    if step < 1 or window < 2:
        raise ValueError("window and step must contain at least one sample")
    n = len(trace)
    expected = {f"{r}_{m}": (r, m) for r in config.roi_names for m in config.methods}
    unknown = set(signals) - expected.keys()
    if unknown:
        raise ValueError(f"Unknown channels: {sorted(unknown)}")
    inputs = {}
    for key, value in signals.items():
        value = np.asarray(value, dtype=float)
        if value.ndim != 1 or len(value) != n:
            raise ValueError(f"{key}: signal must be 1D and match len(trace)")
        inputs[key] = value
    metadata = {}
    for roi in config.roi_names:
        names = (f"{roi}_valid", f"{roi}_quality")
        if not all(name in trace for name in names):
            continue
        valid, quality = (_numeric_column(trace, name) for name in names)
        for values in (valid, quality):
            finite = values[np.isfinite(values)]
            if np.any((finite < 0) | (finite > 1)):
                raise ValueError(f"{roi}: valid and quality values must be in [0,1]")
        metadata[roi] = valid, quality
    grid = np.arange(min_bpm, max_bpm + config.grid_step_bpm / 2,
                     config.grid_step_bpm)
    rows, diagnostics = [], []
    wave_sum, wave_weight = np.zeros(n), np.zeros(n)
    taper = np.hanning(window + 2)[1:-1]
    previous_bpm, previous_time = None, None
    rejected, ever_locked = 0, False
    for wi, start in enumerate(range(0, n - window + 1, step)):
        stop = start + window
        center = (start + window / 2) / fps
        motion = _motion_profile(trace, start, stop, fps, grid, config)
        candidates, available_rois, window_diag_ids = [], set(), []
        for channel, (roi, method) in expected.items():
            status, observed, quality, std = "usable", 0.0, 0.0, np.nan
            segment = inputs.get(channel)
            if roi not in metadata:
                status = "missing_roi_metadata"
            else:
                valid, q = metadata[roi]
                observed = float(np.mean(np.nan_to_num(valid[start:stop], nan=0.0)))
                quality = float(np.mean(np.nan_to_num(q[start:stop], nan=0.0)))
                if observed < config.min_observed_fraction:
                    status = "insufficient_observed_roi"
                elif quality < config.min_roi_quality:
                    status = "poor_roi_quality"
            if status == "usable":
                if segment is None:
                    status = "missing_channel"
                elif not np.isfinite(segment[start:stop]).all():
                    status = "gap_or_filter_edge"
                else:
                    std = float(np.std(segment[start:stop]))
                    if std < config.min_signal_std:
                        status = "flat_signal"
            peaks = []
            if status == "usable":
                available_rois.add(roi)
                peaks = _peaks(segment[start:stop], fps, grid, config)
                if not peaks:
                    status = "diffuse_spectrum"
            base = dict(window_index=wi, time_s=center, channel=channel, roi=roi,
                        method=method, channel_status=status, observed_fraction=observed,
                        roi_quality=quality, signal_std=std, candidate_rank=0,
                        candidate_bpm=np.nan, relative_power=0.0, peak_concentration=0.0,
                        prominence_ratio=0.0, candidate_weight=0.0, motion_overlap=0.0,
                        motion_weight=1.0, selected_consensus=False,
                        output_accepted=False, waveform_weight=0.0)
            for rank, peak in enumerate(peaks or [None], 1):
                d = base.copy()
                if peak is not None:
                    d.update(peak)
                    d["candidate_rank"] = rank
                    d["candidate_weight"] = quality * observed * np.sqrt(
                        peak["peak_concentration"] * peak["relative_power"])
                    d["motion_overlap"] = float(np.interp(peak["candidate_bpm"], grid, motion))
                    d["motion_weight"] = 1 - config.max_motion_penalty * d["motion_overlap"]
                    candidates.append({**d, "diag_id": len(diagnostics)})
                window_diag_ids.append(len(diagnostics))
                diagnostics.append(d)
        groups = _groups(candidates, config)
        best = groups[0] if groups else None
        row = dict(time_s=center, window_start_s=start / fps, window_end_s=stop / fps,
                   accepted=False, status="insufficient_rois", spectral_peak_bpm=np.nan,
                   ridge_bpm=np.nan, roi_count=0, quality_proxy=0.0, state="searching",
                   available_roi_count=len(available_rois), method_agreement=0.0,
                   motion_overlap=0.0, consensus_score=0.0, runner_up_ratio=0.0,
                   candidate_group_count=len(groups), waveform_roi_count=0)
        if best is not None:
            ratio = max((g["base_score"] / best["base_score"] for g in groups[1:]), default=0.0)
            row.update(roi_count=best["roi_count"], quality_proxy=best["quality"],
                       method_agreement=best["method_agreement"], motion_overlap=best["motion"],
                       consensus_score=best["score"], runner_up_ratio=ratio)
            for c in best["members"]:
                diagnostics[c["diag_id"]]["selected_consensus"] = True
            # Ambiguity is checked before any temporal prior. A past lock cannot
            # turn two equally supported spectral explanations into certainty.
            if ratio >= config.ambiguity_ratio:
                row["status"] = "ambiguous_consensus"
            elif previous_bpm is not None and abs(best["bpm"] - previous_bpm) > (
                    config.max_slew_bpm_per_s * (center - previous_time)):
                row["status"] = "tracking_jump"
            else:
                row.update(accepted=True, status="accepted_consensus",
                           spectral_peak_bpm=best["bpm"], ridge_bpm=best["bpm"],
                           state=("tracking" if previous_bpm is not None else
                                  "reacquired" if ever_locked else "acquired"))
        elif len(available_rois) >= config.min_rois:
            row["status"] = "no_frequency_consensus"
        if row["accepted"]:
            previous_bpm, previous_time = best["bpm"], center
            rejected, ever_locked = 0, True
            for idx in window_diag_ids:
                diagnostics[idx]["output_accepted"] = True
            fused, count, used = _fuse_waveform(inputs, start, stop, best, config)
            row["waveform_roi_count"] = count
            if fused is not None:
                existing = wave_weight[start:stop] > 0
                if existing.sum() >= round(fps):
                    prior = wave_sum[start:stop][existing] / wave_weight[start:stop][existing]
                    if _correlation(fused[existing], prior) < 0:
                        fused = -fused
                weights = taper * best["quality"]
                wave_sum[start:stop] += fused * weights
                wave_weight[start:stop] += weights
                for idx, weight in used.items():
                    diagnostics[idx]["waveform_weight"] = float(weight)
        else:
            rejected += 1
            if rejected >= config.rejected_windows_to_reset:
                previous_bpm, previous_time = None, None
            row["state"] = ("uncertain" if previous_bpm is not None else
                            "lost" if ever_locked else "searching")
        rows.append(row)
    waveform = np.full(n, np.nan)
    np.divide(wave_sum, wave_weight, out=waveform, where=wave_weight > 0)
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    table["accepted"] = table["accepted"].astype(bool)
    diag = pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS)
    for col in ("selected_consensus", "output_accepted"):
        diag[col] = diag[col].astype(bool)
    table.attrs.update(config=asdict(config), hr_tracker="forward_consensus_gate_not_DP",
                       quality_proxy_is_probability=False, waveform_is_offline=True)
    return table, waveform, diag

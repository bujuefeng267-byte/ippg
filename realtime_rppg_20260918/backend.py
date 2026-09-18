"""Past-only 10-second rPPG snapshots, separate from the retained offline path.

The current window uses POS and zero-phase filtering *inside the received
buffer*. A snapshot can therefore describe earlier samples, but it never reads
a future frame or revises a previously emitted snapshot. Overlapping snapshot
samples may differ: they must not be exported as a single immutable waveform.
HR is measured from the exact snapshot waveform returned to the caller.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd

ROIS = ("forehead", "left_cheek", "right_cheek")
VARIANT = "realtime_pos3_guarded_v28_evidence_forward_v1"


@dataclass(frozen=True)
class StreamingConfig:
    window_s: float = 10.0
    step_s: float = 1.0
    sample_hz: float = 30.0
    max_gap_s: float = 0.1
    min_source_hz: float = 8.0
    max_buffer_frames: int = 5000

    def __post_init__(self):
        if self.window_s != 10.0 or self.step_s != 1.0:
            raise ValueError("This evaluated variant uses fixed 10 s / 1 s windows")
        if self.sample_hz != 30.0 or self.max_gap_s != .1:
            raise ValueError("This variant uses fixed 30 Hz resampling and 0.1 s gaps")
        if not np.isfinite(self.min_source_hz) or self.min_source_hz < 8:
            raise ValueError("Source minimum must be at least 8 Hz")
        if not isinstance(self.max_buffer_frames, int) or self.max_buffer_frames < 600:
            raise ValueError("Buffer must retain at least 600 samples")


def _number(value, default=np.nan):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (ValueError, TypeError, OverflowError):
        return default


def _nullable(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def motion_to_grid_units(displacement, capture_dt, sample_hz, max_gap_s=.1):
    """Convert camera-frame displacement into displacement per resampled frame.

    The reused V28 motion provider multiplies displacement by its sampling Hz.
    Scaling here preserves measured velocity when camera cadence is irregular
    or differs from the 30 Hz processing grid.
    """
    displacement, capture_dt = np.asarray(displacement, float), np.asarray(capture_dt, float)
    out = np.full(displacement.shape, np.nan)
    valid = np.isfinite(displacement) & np.isfinite(capture_dt) & (capture_dt > 0) & (capture_dt <= max_gap_s + 1e-7)
    out[valid] = displacement[valid] / capture_dt[valid] / sample_hz
    return out


def resample_values(times, values, grid, nominal_dt, max_gap=.1):
    """Interpolate only bracketed measurements with a bounded real-time gap.

    Regular clock resampling is distinguished from replacing missed frames.
    ``observed`` means direct/regularly bracketed measured support; it does not
    claim a camera frame exists at every synthetic 30 Hz grid timestamp.
    """
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    good = np.isfinite(values).all(axis=1)
    out = np.full((len(grid), values.shape[1]), np.nan)
    observed, interpolated = np.zeros(len(grid), bool), np.zeros(len(grid), bool)
    indices = np.flatnonzero(good)
    if not len(indices):
        return out, observed, interpolated
    valid_t = times[indices]
    right = np.searchsorted(valid_t, grid, side="left")
    for j, r in enumerate(right):
        if r < len(valid_t) and abs(valid_t[r] - grid[j]) <= 1e-7:
            out[j] = values[indices[r]]
            observed[j] = True
            continue
        if r == 0 or r == len(valid_t):
            continue
        left_i, right_i = indices[r - 1], indices[r]
        span = times[right_i] - times[left_i]
        if span > max_gap + 1e-7:
            continue
        weight = (grid[j] - times[left_i]) / span
        out[j] = values[left_i] * (1 - weight) + values[right_i] * weight
        filled = right_i != left_i + 1 or span > 1.75 * nominal_dt + 1e-7
        interpolated[j] = filled
        observed[j] = not filled
    return out, observed, interpolated


class StreamingBackend:
    """Append physical-ROI trace rows in strictly increasing capture time.

    ``append(row)`` returns None before an update is due, otherwise a JSON-safe
    event. Values omitted due to poor sampling/signal remain null, never zero.
    It retains at most about 12 seconds / max_buffer_frames raw trace rows.
    Only physical ROI, geometry and tracking diagnostic fields are read; no
    reference HR, filename or per-video identity enters inference.
    """

    def __init__(self, project_root=None, config=None):
        self.config = config or StreamingConfig()
        if not isinstance(self.config, StreamingConfig):
            raise TypeError("config must be StreamingConfig")
        root = Path(project_root or os.environ.get("RPPG_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
        module_dir = root / "motion_upgrade_v28_20260912"
        if not (module_dir / "evidence_hr.py").is_file():
            raise FileNotFoundError(f"Retained V28 kernels unavailable: {module_dir}")
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        from analyze_rppg import pos_signal, bandpass
        from evidence_hr import DEFAULT_CONFIG, score_candidates
        from guarded_fusion import GuardConfig, guarded_fuse
        from legacy_motion import estimate
        from motion_evidence import motion_evidence
        from motion_fusion import FusionConfig
        self.pos_signal, self.bandpass = pos_signal, bandpass
        self.score_candidates, self.evidence_config = score_candidates, DEFAULT_CONFIG
        self.guarded_fuse, self.guard_config = guarded_fuse, GuardConfig(routing_mode="legacy")
        self.fusion_config = FusionConfig(methods=("pos",), motion_evidence_mode="legacy")
        self.estimate, self.motion_evidence = estimate, motion_evidence
        self.grid = np.arange(42., 211.)
        self.rows = deque(maxlen=self.config.max_buffer_frames)
        self.first_time = self.last_time = self.next_end = None
        self.previous_scores = None
        self.last_emission_end = None
        self.events_emitted = 0

    @property
    def configuration(self):
        return dict(variant=VARIANT, **asdict(self.config), reference_used=False,
                    full_offline_v28_v32_equivalent=False,
                    waveform_kind="independent_past_buffer_window_snapshots")

    def append(self, row):
        current = _number(row.get("time_s"))
        if not np.isfinite(current) or current < 0:
            raise ValueError("Capture timestamp must be finite and nonnegative")
        if self.last_time is not None and current <= self.last_time:
            raise ValueError("Capture timestamps must be strictly increasing")
        # Copy only the inference fields to prevent retaining arbitrary payloads.
        copied = {"time_s": current,
                  "capture_dt_s": current - self.last_time if self.last_time is not None else np.nan}
        for field in ("motion_x", "motion_y", "face_x0", "face_y0", "face_x1", "face_y1"):
            copied[field] = _number(row.get(field))
        for roi in ROIS:
            copied[f"{roi}_quality"] = float(np.clip(_number(row.get(f"{roi}_quality"), 0.), 0., 1.))
            copied[f"{roi}_valid"] = row.get(f"{roi}_valid") in (True, 1, 1.)
            copied[f"{roi}_pixel_source"] = str(row.get(f"{roi}_pixel_source", "missing"))
            for c in "rgb":
                field = f"{roi}_{c}"
                copied[field] = _number(row.get(field))
                copied[f"baseline_{field}"] = _number(row.get(f"baseline_{field}", row.get(field)))
        self.rows.append(copied)
        self.last_time = current
        if self.first_time is None:
            self.first_time = current
            self.next_end = current + self.config.window_s
        cutoff = current - self.config.window_s - 2 * self.config.step_s
        while len(self.rows) > 2 and self.rows[1]["time_s"] < cutoff:
            self.rows.popleft()
        if current + 1e-7 < self.next_end:
            return None
        skipped = max(0, int(np.floor((current - self.next_end + 1e-7) / self.config.step_s)))
        end = self.next_end + skipped * self.config.step_s
        self.next_end = end + self.config.step_s
        if skipped:
            self.previous_scores = None
        started = time.perf_counter()
        result = self._infer(end)
        result.update(emitted_at_source_s=current, input_delay_s=max(0., current - end),
                      skipped_update_count=skipped, processing_ms=1000 * (time.perf_counter() - started),
                      variant=VARIANT, reference_used=False, event_index=self.events_emitted)
        self.events_emitted += 1
        self.last_emission_end = end
        return result

    def _infer(self, end):
        cfg = self.config
        start = end - cfg.window_s
        times = np.asarray([r["time_s"] for r in self.rows], float)
        rows = list(self.rows)
        take = times <= end + 1e-7
        rows = [r for r, use in zip(rows, take) if use]
        times = times[take]
        grid_times = start + np.arange(round(cfg.window_s * cfg.sample_hz)) / cfg.sample_hz
        local = pd.DataFrame({"time_s": np.arange(len(grid_times)) / cfg.sample_hz})
        intervals = np.diff(times[(times >= start - .1) & (times <= end)])
        nominal_dt = float(np.median(intervals)) if len(intervals) else float("inf")
        source_hz = 1 / nominal_dt if np.isfinite(nominal_dt) and nominal_dt > 0 else 0.
        result = dict(window_start_s=start, window_end_s=end, time_s=(start + end) / 2,
                      hr_bpm=None, accepted=False, status="insufficient_sampling",
                      observed_fraction=0., interpolated_fraction=0., source_hz=source_hz,
                      selected_branch="none", contributing_rois=[],
                      waveform={"time_s": grid_times.tolist(), "values": [None] * len(grid_times),
                                "covered": [False] * len(grid_times),
                                "observed": [False] * len(grid_times),
                                "interpolated": [False] * len(grid_times)},
                      waveform_coverage_fraction=0., evidence_candidates=[])
        if source_hz < cfg.min_source_hz:
            self.previous_scores = None
            return result
        nearest = np.searchsorted(times, grid_times, side="left")
        nearest = np.clip(nearest, 0, len(times) - 1)
        previous = np.maximum(0, nearest - 1)
        nearest = np.where(np.abs(times[previous] - grid_times) <= np.abs(times[nearest] - grid_times), previous, nearest)
        provenance, signals = {}, {"baseline": {}, "tracked": {}}
        for field in ("motion_x", "motion_y", "face_x0", "face_y0", "face_x1", "face_y1"):
            values = np.asarray([r[field] for r in rows], float)
            if field in ("motion_x", "motion_y"):
                values = motion_to_grid_units(values, [r["capture_dt_s"] for r in rows], cfg.sample_hz, cfg.max_gap_s)
            local[field] = resample_values(times, values, grid_times, nominal_dt, cfg.max_gap_s)[0][:, 0]
        for roi in ROIS:
            valid = np.asarray([r[f"{roi}_valid"] for r in rows], bool)
            quality = np.asarray([r[f"{roi}_quality"] for r in rows], float)
            local[f"{roi}_quality"] = quality[nearest]
            sources = np.asarray([r[f"{roi}_pixel_source"] for r in rows], object)
            local[f"{roi}_pixel_source"] = sources[nearest]
            both_observed = []
            for branch, prefix in (("baseline", "baseline_"), ("tracked", "")):
                rgb = np.asarray([[r[f"{prefix}{roi}_{c}"] for c in "rgb"] for r in rows], float)
                rgb[~valid | ~np.isfinite(rgb).all(axis=1) | (rgb <= 0).any(axis=1)] = np.nan
                sampled, observed, interpolated = resample_values(times, rgb, grid_times, nominal_dt, cfg.max_gap_s)
                finite = np.isfinite(sampled).all(axis=1)
                signal = np.full(len(grid_times), np.nan)
                for a, b in _runs(finite):
                    if b - a < round(4 * cfg.sample_hz):
                        continue
                    # Original POS projection, but only the received 10 s window
                    # and no original 1.6 s edge masking: this is a new variant.
                    raw = self.pos_signal(sampled[a:b], cfg.sample_hz)
                    signal[a:b] = self.bandpass(raw, cfg.sample_hz, 42., 210.)
                signals[branch][f"{roi}_pos"] = signal
                provenance[(branch, roi)] = observed, interpolated
                both_observed.append(observed)
            local[f"{roi}_valid"] = np.logical_and.reduce(both_observed)
            local.loc[~local[f"{roi}_valid"], f"{roi}_quality"] = 0.
        proposals, wave, diagnostics, routing, _ = self.guarded_fuse(
            signals["baseline"], signals["tracked"], local, cfg.sample_hz,
            config=self.fusion_config, guard=self.guard_config)
        branch = str(routing.iloc[0].selected_branch)
        contributing = sorted(set(diagnostics.loc[diagnostics.waveform_weight > 0, "roi"].tolist()))
        result.update(selected_branch=branch, contributing_rois=contributing,
                      fusion_status=str(proposals.iloc[0].status),
                      waveform_coverage_fraction=float(np.isfinite(wave).mean()))
        if branch == "none" or not contributing:
            result["status"] = "no_qualified_roi_consensus"
            self.previous_scores = None
            return result
        observed = np.logical_and.reduce([provenance[(branch, roi)][0] for roi in contributing])
        interpolated = np.logical_or.reduce([provenance[(branch, roi)][1] for roi in contributing])
        covered = np.isfinite(wave)
        observed &= covered
        interpolated &= covered
        result["waveform"] = dict(time_s=grid_times.tolist(), values=_nullable(wave),
                                  covered=covered.tolist(), observed=observed.tolist(),
                                  interpolated=interpolated.tolist())
        result.update(observed_fraction=float(observed.mean()), interpolated_fraction=float(interpolated.mean()))
        gate = self.estimate(wave, pd.DataFrame({"rgb_valid": observed}), interpolated,
                             cfg.sample_hz).iloc[0]
        result["status"] = str(gate.status)
        result["peak_concentration"] = _number(gate.peak_concentration, None)
        if not bool(gate.accepted):
            self.previous_scores = None
            return result
        motion = self.motion_evidence(local, 0, len(local), cfg.sample_hz, self.grid)
        candidates, scores = self.score_candidates(wave, cfg.sample_hz, self.grid, motion)
        result["evidence_candidates"] = candidates
        if not candidates or not np.isfinite(scores).any():
            result["status"] = "no_supported_candidate"
            self.previous_scores = None
            return result
        local_scores = scores.copy()
        if self.previous_scores is not None:
            dt = max(cfg.step_s, end - self.last_emission_end)
            delta = self.grid[:, None] - self.grid[None, :]
            cost = np.where(np.abs(delta) <= self.evidence_config.max_rate_bpm_per_s * dt,
                            -self.evidence_config.transition_penalty * (delta / dt) ** 2, -np.inf)
            transition = np.maximum(cost, -self.evidence_config.reacquisition_cost)
            scores += np.max(self.previous_scores[:, None] + transition, axis=0)
        # Forward filtering only: unlike full offline DP, never backtrack or
        # replace any historical output when the next window arrives.
        self.previous_scores = scores - np.max(scores)
        state = int(np.argmax(scores))
        result.update(hr_bpm=float(self.grid[state]), accepted=True, status="candidate_only",
                      local_evidence_bpm=float(self.grid[np.argmax(local_scores)]),
                      selected_current_evidence=float(local_scores[state]),
                      motion_available=bool(motion["available"]))
        return result

"""Bounded-past V28 component readout with a fixed 2 s measurement delay.

All inputs are already received when an event is emitted. Zero-phase filters,
overlap fusion and dynamic programming operate on at most the last 22 seconds.
The final 10 second observation ends 2 seconds before that received boundary;
it is published once, never revised. This is not equivalent to full-video V28
or V32. It deliberately retains the original edge and quality rejection rules.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

ROIS = ("forehead", "left_cheek", "right_cheek")
VARIANT = "realtime_retained_v28_22s_history_2s_delay_v2"


@dataclass(frozen=True)
class StreamingConfig:
    window_s: float = 10.
    step_s: float = 1.
    sample_hz: float = 30.
    max_gap_s: float = .1
    min_source_hz: float = 8.
    max_buffer_frames: int = 5000
    history_s: float = 22.
    fixed_delay_s: float = 2.

    def __post_init__(self):
        if (self.window_s, self.step_s, self.sample_hz, self.max_gap_s,
                self.history_s, self.fixed_delay_s) != (10., 1., 30., .1, 22., 2.):
            raise ValueError("Evaluated retained streaming settings are fixed: 22 s history / 2 s delay / 10 s window / 1 s update / 30 Hz")
        if not np.isfinite(self.min_source_hz) or self.min_source_hz < 8:
            raise ValueError("Source minimum must be at least 8 Hz")
        if not isinstance(self.max_buffer_frames, int) or self.max_buffer_frames < 700:
            raise ValueError("Buffer must retain at least 700 samples")


def _number(value, default=np.nan):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (ValueError, TypeError, OverflowError):
        return default


def _nullable(values):
    return [float(x) if np.isfinite(x) else None for x in values]


class StreamingBackend:
    def __init__(self, project_root=None, config=None):
        self.config = config or StreamingConfig()
        if not isinstance(self.config, StreamingConfig):
            raise TypeError("config must be retained_backend.StreamingConfig")
        root = Path(project_root or os.environ.get("RPPG_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
        directory = root / "motion_upgrade_v28_20260912"
        sys.path.insert(0, str(directory)) if str(directory) not in sys.path else None
        import analyze_components_v26 as components
        from evidence_hr import estimate_evidence
        from direct_guard_router_v28 import route_components
        from guarded_fusion import guarded_fuse, GuardConfig
        from motion_fusion import FusionConfig
        from waveform_hr import estimate_fused_waveform
        helper_path = root / "realtime_rppg_20260918" / "backend.py"
        spec = importlib.util.spec_from_file_location("_retained_stream_resampling", helper_path)
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        self.components = components
        self.estimate_evidence = estimate_evidence
        self.route_components = route_components
        self.guarded_fuse = guarded_fuse
        self.fusion_config = FusionConfig(motion_evidence_mode="legacy")
        self.guard_config = GuardConfig(routing_mode="legacy")
        self.estimate_fused_waveform = estimate_fused_waveform
        self.resample_values = helper.resample_values
        self.motion_to_grid_units = helper.motion_to_grid_units
        self.rows = deque(maxlen=self.config.max_buffer_frames)
        self.first_time = self.last_time = self.next_end = None
        self.events_emitted = 0

    @property
    def configuration(self):
        return dict(variant=VARIANT, **asdict(self.config), reference_used=False,
                    full_offline_v28_v32_equivalent=False,
                    earliest_event_s=12., earliest_possible_accepted_s=14.,
                    waveform_kind="independent_fixed_delay_measured_window_snapshots",
                    online_policy="received_22s_buffer_only_immutable_emissions",
                    quality_gates="original_V28_edges_and_evidence_gates")

    def append(self, row):
        current = _number(row.get("time_s"))
        if not np.isfinite(current) or current < 0:
            raise ValueError("Capture timestamp must be finite and nonnegative")
        if self.last_time is not None and current <= self.last_time:
            raise ValueError("Capture timestamps must strictly increase")
        copied = dict(time_s=current, capture_dt_s=current-self.last_time if self.last_time is not None else np.nan)
        for field in ("motion_x", "motion_y", "face_x0", "face_y0", "face_x1", "face_y1"):
            copied[field] = _number(row.get(field))
        for roi in ROIS:
            copied[f"{roi}_valid"] = row.get(f"{roi}_valid") in (True, 1, 1.)
            copied[f"{roi}_quality"] = float(np.clip(_number(row.get(f"{roi}_quality"), 0.), 0., 1.))
            source = row.get(f"{roi}_pixel_source", "missing")
            copied[f"{roi}_pixel_source"] = source if source in (
                "tracked_ratio", "baseline_reset", "numerical_reset", "missing", "baseline_ratio_fallback") else "missing"
            for c in "rgb":
                copied[f"{roi}_{c}"] = _number(row.get(f"{roi}_{c}"))
                copied[f"baseline_{roi}_{c}"] = _number(row.get(f"baseline_{roi}_{c}"))
        self.rows.append(copied)
        self.last_time = current
        if self.first_time is None:
            self.first_time = current
            self.next_end = current + self.config.window_s + self.config.fixed_delay_s
        cutoff = current - self.config.history_s - 2.
        while len(self.rows) > 2 and self.rows[1]["time_s"] < cutoff:
            self.rows.popleft()
        if current + 1e-7 < self.next_end:
            return None
        skipped = max(0, int(np.floor((current-self.next_end+1e-7)/self.config.step_s)))
        received_end = self.next_end + skipped*self.config.step_s
        self.next_end = received_end + self.config.step_s
        started = time.perf_counter()
        result = self._infer(received_end)
        result.update(emitted_at_source_s=current,
                      input_delay_s=current-result["window_end_s"],
                      skipped_update_count=skipped,
                      processing_ms=1000*(time.perf_counter()-started),
                      variant=VARIANT, reference_used=False, event_index=self.events_emitted)
        self.events_emitted += 1
        return result

    def _resample(self, received_end):
        cfg = self.config
        start = max(self.first_time, received_end-cfg.history_s)
        grid_times = start + np.arange(round((received_end-start)*cfg.sample_hz))/cfg.sample_hz
        rows = [r for r in self.rows if r["time_s"] <= received_end+1e-7]
        times = np.array([r["time_s"] for r in rows])
        window_end = received_end-cfg.fixed_delay_s
        window_start = window_end-cfg.window_s
        intervals = np.diff(times[(times >= window_start-.1) & (times <= received_end)])
        nominal_dt = float(np.median(intervals)) if len(intervals) else np.inf
        source_hz = 1./nominal_dt if np.isfinite(nominal_dt) and nominal_dt > 0 else 0.
        trace = pd.DataFrame(dict(frame=np.arange(len(grid_times)), time_s=np.arange(len(grid_times))/cfg.sample_hz))
        nearest = np.clip(np.searchsorted(times, grid_times, side="left"), 0, len(times)-1)
        previous = np.maximum(0, nearest-1)
        nearest = np.where(np.abs(times[previous]-grid_times) <= np.abs(times[nearest]-grid_times), previous, nearest)
        for field in ("motion_x", "motion_y", "face_x0", "face_y0", "face_x1", "face_y1"):
            values = np.asarray([r[field] for r in rows], float)
            if field in ("motion_x", "motion_y"):
                values = self.motion_to_grid_units(values, [r["capture_dt_s"] for r in rows], cfg.sample_hz, cfg.max_gap_s)
            trace[field] = self.resample_values(times, values, grid_times, nominal_dt, cfg.max_gap_s)[0][:,0]
        for roi in ROIS:
            valid = np.asarray([r[f"{roi}_valid"] for r in rows], bool)
            all_sampled, all_observed = [], []
            for prefix in ("", "baseline_"):
                values = np.asarray([[r[f"{prefix}{roi}_{c}"] for c in "rgb"] for r in rows])
                values[~valid | ~np.isfinite(values).all(axis=1) | (values <= 0).any(axis=1)] = np.nan
                sampled, observed, _ = self.resample_values(times, values, grid_times, nominal_dt, cfg.max_gap_s)
                all_sampled.append(sampled)
                all_observed.append(observed)
            physical_observed = np.logical_and.reduce(all_observed)
            # Missing camera support remains missing here. Frozen V28 may fill
            # only its pre-existing bounded gaps and records their provenance.
            for prefix, sampled in zip(("", "baseline_"), all_sampled):
                sampled[~physical_observed] = np.nan
                for j, c in enumerate("rgb"):
                    trace[f"{prefix}{roi}_{c}"] = sampled[:,j]
            trace[f"{roi}_valid"] = physical_observed
            quality = np.asarray([r[f"{roi}_quality"] for r in rows])[nearest].copy()
            quality[~physical_observed] = 0.
            trace[f"{roi}_quality"] = quality
            sources = np.asarray([r[f"{roi}_pixel_source"] for r in rows], object)[nearest].copy()
            sources[~physical_observed] = "missing"
            trace[f"{roi}_pixel_source"] = sources
        for prefix in ("", "baseline_"):
            stack = np.asarray([trace[[f"{prefix}{roi}_{c}" for c in "rgb"]].to_numpy() for roi in ROIS])
            finite = np.isfinite(stack).all(axis=2)
            count = finite.sum(axis=0)
            merged = np.full((len(trace),3),np.nan)
            np.divide(np.where(finite[:,:,None],stack,0).sum(axis=0),count[:,None],out=merged,where=count[:,None]>0)
            for j,c in enumerate("rgb"):
                trace[f"{prefix}{c}"] = merged[:,j]
        trace["rgb_valid"] = np.isfinite(trace[["r","g","b"]]).all(axis=1)
        return trace, grid_times, source_hz, window_start, window_end

    def _pipeline(self, trace):
        """Frozen V28 kernels in memory, using shared prepared ROI channels."""
        fps = self.config.sample_hz
        channels, tables = self.components.prepare_roi_channels(trace, fps)
        signals = {branch: {f"{roi}_{method}": channels[f"{branch}/{roi}/{method}"]
                   for roi in ROIS for method in ("pos","chrom")}
                   for branch in ("baseline","tracked")}
        proposals, values, diagnostics, routing, _ = self.guarded_fuse(
            signals["baseline"], signals["tracked"], trace, fps,
            config=self.fusion_config, guard=self.guard_config)
        proposals["waveform_generated"] = proposals.accepted & (proposals.waveform_roi_count >= self.fusion_config.min_rois)
        old_hr, old_wave = self.estimate_fused_waveform(values, trace, tables["tracked"], proposals,
                                                       diagnostics, fps, hr_mode="evidence")
        old_wave.insert(1,"base",values)
        module = self.components.load_variant("harmonic")
        candidate, component_proposals, _ = module.fuse_components(channels,trace,tables["baseline"],fps)
        wave, decisions = self.route_components(old_wave,old_hr,candidate,component_proposals,trace,fps)
        hr = self.estimate_evidence(wave.base.to_numpy(float),pd.DataFrame({"rgb_valid":wave.observed}),
                wave.interpolated.to_numpy(bool),fps,motion_trace=trace)
        return wave, hr, decisions, routing

    def _infer(self, received_end):
        trace, times, source_hz, start, end = self._resample(received_end)
        target = round((start-times[0])*self.config.sample_hz)
        width = round(self.config.window_s*self.config.sample_hz)
        span = slice(target,target+width)
        grid_times = times[span]
        result = dict(window_start_s=start,window_end_s=end,time_s=(start+end)/2,
            hr_bpm=None,accepted=False,status="insufficient_sampling",source_hz=source_hz,
            observed_fraction=0.,interpolated_fraction=0.,selected_branch="none",contributing_rois=[],
            waveform=dict(time_s=grid_times.tolist(),values=[None]*width,covered=[False]*width,
                          observed=[False]*width,interpolated=[False]*width),
            waveform_coverage_fraction=0.,evidence_candidates=[],buffer_start_s=float(times[0]),
            buffer_end_s=received_end,fixed_delay_s=self.config.fixed_delay_s)
        if source_hz < self.config.min_source_hz:
            return result
        wave, hr, decisions, routing = self._pipeline(trace)
        index = round(target/self.config.sample_hz)
        row = hr.iloc[index]
        segment = wave.iloc[span]
        covered = segment.covered.to_numpy(bool)
        result.update(hr_bpm=_number(row.ridge_bpm,None) if bool(row.accepted) else None,
            accepted=bool(row.accepted),status=str(row.status),
            observed_fraction=float(segment.observed.mean()),interpolated_fraction=float(segment.interpolated.mean()),
            waveform_coverage_fraction=float(covered.mean()),
            peak_concentration=_number(row.peak_concentration,None),
            selected_branch=str(routing.iloc[index].selected_branch),
            component_route_eligible=bool(decisions.iloc[index].route_eligible),
            component_route_reason=str(decisions.iloc[index].reason),
            waveform=dict(time_s=grid_times.tolist(),values=_nullable(segment.base),covered=covered.tolist(),
                observed=segment.observed.to_numpy(bool).tolist(),interpolated=segment.interpolated.to_numpy(bool).tolist()))
        if "evidence_candidates_json" in row:
            import json
            result["evidence_candidates"] = json.loads(row.evidence_candidates_json)
        return result

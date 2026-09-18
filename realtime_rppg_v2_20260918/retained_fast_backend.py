"""Compute-only acceleration of the frozen 22 s / 2 s streaming candidate.

The used POS/CHROM base waveforms, short-gap filling, 1.6 s edge masks and all
downstream routing/readout are retained. Legacy preparation also computes an
NLMS residual that it immediately discards; this adapter never calculates that
unused product. No shared legacy module is mutated.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fast_kernels import pos_projection
from retained_backend import ROIS, StreamingConfig, StreamingBackend as FrozenBackend


class StreamingBackend(FrozenBackend):
    def __init__(self, project_root=None, config=None):
        super().__init__(project_root, config)
        import legacy_motion
        self._legacy = legacy_motion
        self._original_components = self.components
        self.components = SimpleNamespace(
            prepare_roi_channels=self._prepare_roi_channels,
            load_variant=self._original_components.load_variant)

    @property
    def configuration(self):
        return dict(super().configuration,
                    compute_implementation="vectorized_POS_and_omit_unused_NLMS_residual",
                    equivalent_to_frozen_retained_candidate=True)

    def _prepare_roi_channels(self, trace, fps):
        channels, tables = {}, {}
        edge = round(1.6 * fps)
        for branch, prefix in (("baseline", "baseline_"), ("tracked", "")):
            table = pd.DataFrame({"time_s": trace.time_s})
            for roi in ROIS:
                measured = trace[[f"{prefix}{roi}_{c}" for c in "rgb"]].to_numpy()
                rgb, filled = self._legacy.fill_short_gaps(
                    measured, self._legacy.gap_frame_limit(.1, fps))
                finite = np.isfinite(rgb).all(axis=1)
                for method in ("pos", "chrom"):
                    wave = np.full(len(trace), np.nan)
                    for a,b in self._legacy.runs(finite):
                        if b-a < max(round(4*fps),2*edge+1):
                            continue
                        raw = (pos_projection(rgb[a:b],fps,normalize_overlap=False)
                               if method == "pos" else self._legacy.chrom_signal(rgb[a:b],fps))
                        filtered = self._legacy.bandpass(raw,fps,42.,210.)
                        wave[a+edge:b-edge] = filtered[edge:-edge]
                    channels[f"{branch}/{roi}/{method}"] = wave
                    table[f"{roi}_{method}"] = wave
                table[f"{roi}_observed"] = trace[f"{roi}_valid"]
                table[f"{roi}_interpolated"] = filled
            tables[branch] = table
        for roi in ROIS:
            for flag in ("observed","interpolated"):
                np.testing.assert_array_equal(tables["baseline"][f"{roi}_{flag}"],
                                              tables["tracked"][f"{roi}_{flag}"])
        return channels,tables

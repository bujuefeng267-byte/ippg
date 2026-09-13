"""Synthetic-only frozen-default tests. No real reference data are loaded."""
from dataclasses import FrozenInstanceError
import unittest

import numpy as np
import pandas as pd

try:
    from .motion_fusion import DEFAULT_CONFIG, FusionConfig, fuse_windows
except ImportError:
    from motion_fusion import DEFAULT_CONFIG, FusionConfig, fuse_windows


FPS = 30.0
ROIS = ("forehead", "left_cheek", "right_cheek")


def make_case(seconds=32, bpm=84, seed=123):
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * FPS)) / FPS
    pulse = np.sin(2 * np.pi * bpm / 60 * t) + 0.25 * np.sin(4 * np.pi * bpm / 60 * t + .2)
    trace = pd.DataFrame({"time_s": t, "motion_x": np.zeros(len(t)),
                          "motion_y": np.zeros(len(t))})
    signals = {}
    for roi in ROIS:
        trace[f"{roi}_valid"] = True
        trace[f"{roi}_quality"] = 0.95
        for method in ("pos", "chrom"):
            signals[f"{roi}_{method}"] = pulse + .02 * rng.standard_normal(len(t))
    return t, pulse, signals, trace


class FusionSyntheticTests(unittest.TestCase):
    def test_consensus_survives_local_pollution_and_polarity(self):
        t, pulse, signals, trace = make_case()
        signals["left_cheek_chrom"] *= -1
        for method in ("pos", "chrom"):
            signals[f"right_cheek_{method}"] += 3.0 * np.sin(2 * np.pi * 2.2 * t)
        table, wave, diag = fuse_windows(signals, trace, FPS)
        self.assertGreater(table.accepted.mean(), .95)
        self.assertLess(np.max(np.abs(table.loc[table.accepted, "ridge_bpm"] - 84)), 1.5)
        self.assertTrue(table.loc[table.accepted, "roi_count"].between(2, 3).all())
        valid = np.isfinite(wave)
        self.assertGreater(abs(np.corrcoef(wave[valid], pulse[valid])[0, 1]), .95)
        self.assertTrue((diag.candidate_weight >= 0).all())
        self.assertTrue((diag.loc[diag.waveform_weight > 0, "selected_consensus"]).all())

    def test_equal_competing_frequencies_are_rejected(self):
        t, _, signals, trace = make_case()
        mixture = np.sin(2 * np.pi * 1.2 * t) + np.sin(2 * np.pi * 2.2 * t)
        signals = {name: mixture.copy() for name in signals}
        table, wave, diag = fuse_windows(signals, trace, FPS)
        self.assertFalse(table.accepted.any())
        self.assertTrue((table.status == "ambiguous_consensus").all())
        self.assertTrue(table.ridge_bpm.isna().all())
        self.assertTrue(np.isnan(wave).all())
        self.assertTrue((diag.candidate_rank >= 2).any())
        self.assertTrue((table.runner_up_ratio >= DEFAULT_CONFIG.ambiguity_ratio).all())

    def test_missing_channels_produce_explicit_rejection(self):
        _, _, signals, trace = make_case()
        signals = {name: np.full_like(x, np.nan) for name, x in signals.items()}
        table, wave, diag = fuse_windows(signals, trace, FPS)
        self.assertFalse(table.accepted.any())
        self.assertEqual(table.accepted.dtype, np.dtype(bool))
        self.assertTrue(table.ridge_bpm.isna().all())
        self.assertTrue((table.roi_count == 0).all())
        self.assertTrue(np.isnan(wave).all())
        self.assertEqual(len(diag), len(table) * 6)

    def test_same_frequency_motion_never_automatically_cancels_pulse(self):
        t, pulse, signals, trace = make_case(bpm=90)
        trace["motion_x"] = 100 * np.sin(2 * np.pi * 1.5 * t)
        trace["motion_y"] = 50 * np.cos(2 * np.pi * 1.5 * t)
        table, wave, diag = fuse_windows(signals, trace, FPS)
        self.assertTrue(table.accepted.all())
        self.assertLess(np.max(np.abs(table.ridge_bpm - 90)), 1)
        self.assertGreater(table.motion_overlap.min(), .95)
        self.assertGreaterEqual(diag.motion_weight.min(), 1 - DEFAULT_CONFIG.max_motion_penalty)
        valid = np.isfinite(wave)
        self.assertGreater(abs(np.corrcoef(wave[valid], pulse[valid])[0, 1]), .98)

    def test_loss_then_reacquisition_does_not_fill_rejected_hr(self):
        t, _, signals, trace = make_case(seconds=60, bpm=72)
        missing = (t >= 20) & (t < 35)
        for name in signals:
            signals[name][missing] = np.nan
            signals[name][t >= 35] = np.sin(2 * np.pi * 2.4 * t[t >= 35])
        for roi in ROIS:
            trace.loc[missing, f"{roi}_valid"] = False
        table, wave, _ = fuse_windows(signals, trace, FPS)
        rejected = ~table.accepted
        self.assertTrue(rejected.any())
        self.assertTrue(table.loc[rejected, ["ridge_bpm", "spectral_peak_bpm"]].isna().all().all())
        self.assertTrue((table.state == "lost").any())
        reacquired = table[table.state == "reacquired"]
        self.assertEqual(len(reacquired), 1)
        self.assertAlmostEqual(float(reacquired.ridge_bpm.iloc[0]), 144, delta=1)
        self.assertTrue(np.isnan(wave[missing]).all())

    def test_append_future_does_not_rewrite_already_emitted_hr(self):
        t, _, signals, trace = make_case(seconds=55)
        end = round(25 * FPS)
        prefix, _, prefix_diag = fuse_windows(
            {k: v[:end] for k, v in signals.items()}, trace.iloc[:end], FPS)
        for name in signals:
            signals[name][end:] = np.sin(2 * np.pi * 2.6 * t[end:])
        # A tempting future/reference column must not influence the computation.
        trace["reference_bpm"] = np.linspace(40, 205, len(t))
        full, _, full_diag = fuse_windows(signals, trace, FPS)
        pd.testing.assert_frame_equal(prefix, full.iloc[:len(prefix)].reset_index(drop=True))
        pd.testing.assert_frame_equal(prefix_diag, full_diag.iloc[:len(prefix_diag)].reset_index(drop=True))
        self.assertEqual(full.attrs["hr_tracker"], "forward_consensus_gate_not_DP")
        self.assertTrue(full.attrs["waveform_is_offline"])

    def test_two_methods_from_one_roi_do_not_form_quorum(self):
        _, _, signals, trace = make_case()
        signals = {k: v for k, v in signals.items() if k.startswith("forehead_")}
        table, wave, _ = fuse_windows(signals, trace, FPS)
        self.assertFalse(table.accepted.any())
        self.assertTrue((table.available_roi_count == 1).all())
        self.assertTrue(np.isnan(wave).all())

    def test_waveform_keeps_pulse_harmonics_not_sine_reconstruction(self):
        t, _, signals, trace = make_case(bpm=72)
        pulse = np.sin(2 * np.pi * 1.2 * t) + .4 * np.sin(2 * np.pi * 2.4 * t + .5)
        signals = {name: pulse.copy() for name in signals}
        table, wave, _ = fuse_windows(signals, trace, FPS)
        self.assertTrue(table.accepted.all())
        valid = np.isfinite(wave)
        # Compare to the actual nonsinusoidal input, with arbitrary global sign.
        self.assertGreater(abs(np.corrcoef(wave[valid], pulse[valid])[0, 1]), .99)
        basis = np.column_stack([np.sin(2 * np.pi * 1.2 * t[valid]),
                                 np.cos(2 * np.pi * 1.2 * t[valid])])
        residual = wave[valid] - basis @ np.linalg.lstsq(basis, wave[valid], rcond=None)[0]
        self.assertGreater(np.std(residual) / np.std(wave[valid]), .25)

    def test_invalid_contract_and_short_input(self):
        _, _, signals, trace = make_case(seconds=4)
        table, wave, diag = fuse_windows(signals, trace, FPS)
        self.assertTrue(table.empty and diag.empty)
        self.assertEqual(len(wave), len(trace))
        self.assertTrue(np.isnan(wave).all())
        with self.assertRaises(ValueError):
            fuse_windows(signals, trace, 0)
        with self.assertRaises(ValueError):
            fuse_windows({"forehead_pos": np.ones(3)}, trace, FPS)
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_CONFIG.ambiguity_ratio = .9
        with self.assertRaises(ValueError):
            FusionConfig(min_rois=1)


if __name__ == "__main__":
    unittest.main()

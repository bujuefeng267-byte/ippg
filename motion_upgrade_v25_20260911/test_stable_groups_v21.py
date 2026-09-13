"""Grouping invariants; no reference HR or reference waveform is loaded.

The one recorded regression fixture reads only the frozen UBFC candidate
diagnostics, not evaluation outputs or ground truth. Synthetic cases determine
the invariant expectations; no fusion threshold is changed by these tests.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

try:
    from .motion_fusion import DEFAULT_CONFIG
    from .stable_groups import group_candidates
except ImportError:
    from motion_fusion import DEFAULT_CONFIG
    from stable_groups import group_candidates


ROOT = Path(__file__).resolve().parent / "test_fixtures"
ROIS = ("forehead", "left_cheek", "right_cheek")


def candidate(roi, method, bpm, weight, diag_id, motion=0.2):
    return dict(roi=roi, method=method, channel=f"{roi}_{method}",
                candidate_bpm=float(bpm), candidate_weight=float(weight),
                motion_overlap=float(motion), diag_id=diag_id)


def physical_signature(groups):
    """Exclude tracing IDs so changing IDs cannot disguise semantic changes."""
    return [(
        g["bpm"], g["base_score"], g["quality"], g["motion"], g["score"],
        g["roi_count"], g["method_agreement"], tuple(g["roi_scores"].items()),
        tuple((c["roi"], c["method"], c["candidate_bpm"],
               c["candidate_weight"], c["motion_overlap"]) for c in g["members"]))
        for g in groups]


def assert_consistent(test, groups, candidates):
    tolerance = DEFAULT_CONFIG.support_tolerance_bpm
    frequencies = {c["candidate_bpm"] for c in candidates}
    for g in groups:
        test.assertIn(g["bpm"], frequencies)
        members = g["members"]
        test.assertEqual(len({(c["roi"], c["method"]) for c in members}), len(members))
        test.assertGreaterEqual(len({c["roi"] for c in members}), DEFAULT_CONFIG.min_rois)
        test.assertTrue(all(abs(c["candidate_bpm"] - g["bpm"]) <= tolerance + 1e-9
                            for c in members))
        # Median characterization checks the invariant independently of the
        # implementation: <center has less than half; <=center has at least half.
        total = math.fsum(c["candidate_weight"] for c in members)
        below = math.fsum(c["candidate_weight"] for c in members if c["candidate_bpm"] < g["bpm"])
        through = math.fsum(c["candidate_weight"] for c in members if c["candidate_bpm"] <= g["bpm"])
        test.assertLess(below, total / 2)
        test.assertGreaterEqual(through, total / 2)
        for roi, score in g["roi_scores"].items():
            test.assertTrue(math.isfinite(score))
            test.assertTrue(all(c["roi"] == roi for c in g["rois"][roi]))
    for a, b in itertools.combinations(groups, 2):
        test.assertGreater(abs(a["bpm"] - b["bpm"]), 2 * tolerance)


class StableGroupTests(unittest.TestCase):
    def test_empty_and_single_roi_reject_without_fabricating_support(self):
        self.assertEqual(group_candidates([], DEFAULT_CONFIG), [])
        single = [candidate("forehead", "pos", 90, .6, 0),
                  candidate("forehead", "chrom", 90, .6, 1)]
        self.assertEqual(group_candidates(single, DEFAULT_CONFIG), [])

    def test_original_roi_scoring_and_motion_penalty_are_preserved(self):
        rows = [candidate("forehead", "pos", 100, .5, 0, .2),
                candidate("forehead", "chrom", 100, .4, 1, .6),
                candidate("left_cheek", "pos", 100, .6, 2, 1)]
        g, = group_candidates(rows, DEFAULT_CONFIG)
        self.assertAlmostEqual(g["roi_scores"]["forehead"], .5)
        self.assertAlmostEqual(g["roi_scores"]["left_cheek"], .6 * .65)
        self.assertAlmostEqual(g["base_score"], .89)
        self.assertAlmostEqual(g["quality"], .445)
        self.assertAlmostEqual(g["motion"], (.5 * .2 + .4 * .6 + .6) / 1.5)
        self.assertAlmostEqual(g["score"], .89 * (1 - .25 * g["motion"]))
        self.assertAlmostEqual(g["method_agreement"], .75)

    def test_per_channel_weight_tie_uses_center_distance_then_semantics(self):
        rows = [candidate("forehead", "pos", 84, .6, 0),
                candidate("forehead", "pos", 80, .6, 1),
                candidate("left_cheek", "pos", 80, .9, 2),
                candidate("right_cheek", "chrom", 80, .9, 3)]
        baseline = physical_signature(group_candidates(rows, DEFAULT_CONFIG))
        for permuted in itertools.permutations(rows):
            groups = group_candidates(permuted, DEFAULT_CONFIG)
            self.assertEqual(physical_signature(groups), baseline)
            chosen = [c for c in groups[0]["members"] if c["roi"] == "forehead"]
            self.assertEqual(chosen[0]["candidate_bpm"], 80)
        # Equal distance is also a tie, rather than an instruction to use the
        # candidate encountered first. This case has stable centers 83 and 86.
        tied = [candidate("forehead", "pos", 80, .5, 0),
                candidate("forehead", "pos", 86, .5, 1),
                candidate("left_cheek", "pos", 83, .6, 2),
                candidate("right_cheek", "chrom", 86, .5, 3)]
        baseline = physical_signature(group_candidates(tied, DEFAULT_CONFIG))
        for permuted in itertools.permutations(tied):
            self.assertEqual(physical_signature(group_candidates(permuted, DEFAULT_CONFIG)), baseline)

    def test_exact_half_uses_lower_weighted_median(self):
        rows = [candidate("forehead", "pos", 84, .5, 0),
                candidate("left_cheek", "pos", 90, .5, 1)]
        g, = group_candidates(rows, DEFAULT_CONFIG)
        self.assertEqual(g["bpm"], 84)
        assert_consistent(self, [g], rows)

    def test_membership_boundary_and_original_nms_radius(self):
        rows = [candidate("forehead", "pos", 72, .4, 0),
                candidate("left_cheek", "pos", 78, .7, 1)]
        self.assertEqual(group_candidates(rows, DEFAULT_CONFIG)[0]["bpm"], 78)
        rows[0]["candidate_bpm"] = 71.9999
        self.assertEqual(group_candidates(rows, DEFAULT_CONFIG), [])
        for gap, expected in [(12, [72]), (13, [72, 85])]:
            rows = [candidate(roi, "pos", bpm, .5, i * 2 + j)
                    for i, bpm in enumerate((72, 72 + gap))
                    for j, roi in enumerate(ROIS[:2])]
            self.assertEqual([g["bpm"] for g in group_candidates(rows, DEFAULT_CONFIG)], expected)

    def test_duplicate_semantic_peaks_and_conflicting_ids_fail_explicitly(self):
        a = candidate("forehead", "pos", 84, .6, 0)
        for b in (dict(a, diag_id=1), dict(a), dict(a, candidate_weight=.7, diag_id=2)):
            with self.assertRaisesRegex(ValueError, "Duplicate semantic peak"):
                group_candidates([a, b], DEFAULT_CONFIG)
        with self.assertRaisesRegex(ValueError, "Duplicate diag_id"):
            group_candidates([a, candidate("left_cheek", "pos", 84, .6, 0)], DEFAULT_CONFIG)

    def test_nonfinite_invalid_and_zero_weight_candidates_fail_explicitly(self):
        a = candidate("forehead", "pos", 84, .6, 0)
        for key, value in (("candidate_bpm", math.nan), ("candidate_weight", math.inf),
                           ("candidate_weight", 0), ("motion_overlap", -0.1),
                           ("motion_overlap", 1.1), ("diag_id", True),
                           ("channel", "mismatched")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                group_candidates([dict(a, **{key: value})], DEFAULT_CONFIG)

    def test_random_synthetic_permutations_id_relabelling_and_stability(self):
        rng = random.Random(20260909)
        for trial in range(35):
            rows = []
            for roi in ROIS:
                for method in DEFAULT_CONFIG.methods:
                    for bpm in rng.sample(range(48, 157), 4):
                        rows.append(candidate(roi, method, bpm,
                                              rng.choice((.3, .5, .5, .8)), len(rows),
                                              rng.choice((0, .2, .8, 1))))
            original = copy.deepcopy(rows)
            baseline_groups = group_candidates(rows, DEFAULT_CONFIG)
            baseline = physical_signature(baseline_groups)
            assert_consistent(self, baseline_groups, rows)
            for _ in range(12):
                permuted = copy.deepcopy(rows)
                rng.shuffle(permuted)
                for index, c in enumerate(permuted):
                    c["diag_id"] = 1000 + index
                groups = group_candidates(permuted, DEFAULT_CONFIG)
                self.assertEqual(physical_signature(groups), baseline, msg=f"trial {trial}")
                assert_consistent(self, groups, permuted)
            self.assertEqual(rows, original)

    def test_recorded_candidates_reproduce_old_defect_and_new_invariance(self):
        path = ROOT / "rppg_motion_v2/validation/ubfc/fusion_diagnostics.csv"
        rows = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for diag_id, row in enumerate(csv.DictReader(stream)):
                if int(row["window_index"]) == 27 and int(row["candidate_rank"]) > 0:
                    rows.append(candidate(row["roi"], row["method"],
                                          float(row["candidate_bpm"]), float(row["candidate_weight"]),
                                          diag_id, float(row["motion_overlap"])))
        self.assertEqual(len(rows), 13)
        spec = importlib.util.spec_from_file_location(
            "_frozen_v2_groups_regression", ROOT / "rppg_motion_v2/motion_fusion.py")
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        try:
            spec.loader.exec_module(old)
            self.assertEqual(old._groups(rows, old.DEFAULT_CONFIG)[0]["bpm"], 126)
            self.assertEqual(old._groups(list(reversed(rows)), old.DEFAULT_CONFIG)[0]["bpm"], 124)
        finally:
            del sys.modules[spec.name]
        baseline_groups = group_candidates(rows, DEFAULT_CONFIG)
        self.assertEqual(baseline_groups[0]["bpm"], 124)
        self.assertEqual({c["diag_id"] for c in baseline_groups[0]["members"]},
                         {267, 270, 272, 274, 277})
        baseline = physical_signature(baseline_groups)
        rng = random.Random(31_762)
        permutations = [rows, list(reversed(rows)), sorted(rows, key=lambda c: c["channel"])]
        for _ in range(100):
            shuffled = copy.deepcopy(rows)
            rng.shuffle(shuffled)
            for index, c in enumerate(shuffled):
                c["diag_id"] = 9000 + index
            permutations.append(shuffled)
        for permuted in permutations:
            groups = group_candidates(permuted, DEFAULT_CONFIG)
            self.assertEqual(physical_signature(groups), baseline)
            assert_consistent(self, groups, permuted)


if __name__ == "__main__":
    unittest.main()

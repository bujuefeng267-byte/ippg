"""Deterministic, self-consistent replacement for the V2 candidate grouping.

Only grouping mechanics change. ROI/method weights, quality gates, soft motion
penalty and the final 2*tolerance suppression are the existing config values.
No reference signal, HR label, or temporal state is accepted or consulted.

Termination: a lower weighted median is always one of the member frequencies.
Consequently every self-consistent hypothesis has a center in the finite set
of input frequencies. We check each of its U <= N distinct values exactly once,
select members, and retain it only when their weighted median is that center.
There is no re-centering loop, convergence limit, or arbitrary starting seed.
If no fixed-point hypothesis meets the original gates, the result is empty.

Per-channel equal-weight ties prefer the candidate closest to the hypothesis
center, then a canonical semantic key. This last tie-break is a reproducibility
convention, not evidence that a lower frequency is physiologically preferable.
diag_id is retained for diagnostics only, never used to rank or deduplicate a
hypothesis. Duplicate peak identities or duplicate tracing IDs are rejected
explicitly instead of allowing an arbitrary row to decide provenance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral


_BOUNDARY_EPSILON = 1e-9  # Same membership roundoff allowance as frozen V2.


def _semantic_key(candidate):
    return (candidate["roi"], candidate["method"], candidate["channel"],
            candidate["candidate_bpm"])


def _validated_candidates(candidates, config):
    normalized, peaks, tracing_ids = [], set(), set()
    required = ("roi", "method", "channel", "candidate_bpm",
                "candidate_weight", "motion_overlap", "diag_id")
    for original in candidates:
        if not isinstance(original, Mapping):
            raise ValueError("Each candidate must be a mapping")
        missing = [key for key in required if key not in original]
        if missing:
            raise ValueError(f"Candidate is missing fields: {missing}")
        c = dict(original)
        if c["roi"] not in config.roi_names or c["method"] not in config.methods:
            raise ValueError("Candidate ROI and method must be configured")
        if c["channel"] != f"{c['roi']}_{c['method']}":
            raise ValueError("Candidate channel must agree with ROI and method")
        for key in ("candidate_bpm", "candidate_weight", "motion_overlap"):
            try:
                c[key] = float(c[key])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Candidate {key} must be finite") from exc
            if not math.isfinite(c[key]):
                raise ValueError(f"Candidate {key} must be finite")
        if c["candidate_bpm"] <= 0 or c["candidate_weight"] <= 0:
            raise ValueError("Candidate BPM and weight must be positive")
        if not 0 <= c["motion_overlap"] <= 1:
            raise ValueError("Candidate motion_overlap must be in [0, 1]")
        diag_id = c["diag_id"]
        if isinstance(diag_id, bool) or not isinstance(diag_id, Integral) or diag_id < 0:
            raise ValueError("Candidate diag_id must be a nonnegative integer")
        c["diag_id"] = int(diag_id)
        key = _semantic_key(c)
        if key in peaks:
            raise ValueError("Duplicate semantic peak (ROI/method/frequency)")
        if c["diag_id"] in tracing_ids:
            raise ValueError("Duplicate diag_id cannot identify distinct candidates")
        peaks.add(key)
        tracing_ids.add(c["diag_id"])
        normalized.append(c)
    return sorted(normalized, key=_semantic_key)


def _members_at(candidates, center, tolerance):
    members = {}
    for c in candidates:
        if abs(c["candidate_bpm"] - center) > tolerance + _BOUNDARY_EPSILON:
            continue
        channel = (c["roi"], c["method"])
        priority = (-c["candidate_weight"], abs(c["candidate_bpm"] - center),
                    _semantic_key(c))
        previous = members.get(channel)
        if previous is None or priority < previous[0]:
            members[channel] = priority, c
    return sorted((entry[1] for entry in members.values()), key=_semantic_key)


def _member_median(members):
    # Group equal frequencies before accumulating, with canonical summation.
    # The lower-median convention matches V2's searchsorted(..., side='left').
    by_frequency = {}
    for c in members:
        by_frequency.setdefault(c["candidate_bpm"], []).append(c["candidate_weight"])
    frequencies = sorted(by_frequency)
    masses = [math.fsum(by_frequency[f]) for f in frequencies]
    halfway = math.fsum(masses) / 2.0
    for index, frequency in enumerate(frequencies):
        if math.fsum(masses[:index + 1]) >= halfway:
            return frequency
    raise AssertionError("A positive finite candidate set must have a median")


def group_candidates(candidates, config):
    """Return V2-compatible groups with deterministic, stable membership.

    All members are within tolerance (plus the unchanged 1e-9 boundary
    roundoff allowance) of the returned center; that center is exactly their
    lower weighted median. Each ROI/method contributes at most one member.
    Candidate dictionaries are copied, so neither input order nor input data
    are mutated. Returned ``members`` retain their original tracing diag_id.
    """
    candidates = _validated_candidates(candidates, config)
    centers = sorted({c["candidate_bpm"] for c in candidates})
    groups, identities = [], set()
    for center in centers:
        members = _members_at(candidates, center, config.support_tolerance_bpm)
        if not members or _member_median(members) != center:
            continue
        identity = tuple(_semantic_key(c) for c in members)
        if identity in identities:
            continue
        identities.add(identity)
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
        base_score = math.fsum(roi_scores.values())
        quality = base_score / len(rois)
        if quality < config.min_consensus_quality:
            continue
        motion = (math.fsum(c["candidate_weight"] * c["motion_overlap"] for c in members)
                  / math.fsum(c["candidate_weight"] for c in members))
        penalty = (config.reliable_motion_penalty if config.motion_evidence_mode == 'reliable'
                   else config.max_motion_penalty)
        groups.append(dict(
            bpm=center, members=members, rois=rois, roi_scores=roi_scores,
            roi_count=len(rois), quality=quality, base_score=base_score,
            motion=motion, score=base_score * (1 - penalty * motion),
            method_agreement=math.fsum(len(v) / len(config.methods)
                                       for v in rois.values()) / len(rois)))
    groups.sort(key=lambda g: (-g["score"], -g["roi_count"], g["bpm"],
                               tuple(_semantic_key(c) for c in g["members"])))
    distinct = []
    for group in groups:
        if all(abs(group["bpm"] - existing["bpm"]) > 2 * config.support_tolerance_bpm
               for existing in distinct):
            distinct.append(group)
    return distinct

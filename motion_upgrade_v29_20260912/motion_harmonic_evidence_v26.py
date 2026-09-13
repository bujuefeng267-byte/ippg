"""Bounded, reference-free motion evidence at f, 2f and f/2.

The fixed relation weights are engineering choices made before evaluating this
revision on human videos. A spectral relation is only a soft interference cue:
motion can share the pulse frequency without making the pulse an artifact.
Nothing here estimates heart rate, removes a waveform, or fills a missing gap.

The existing motion estimator supplies physical-speed scaling, contiguous-run
support and honest missingness. Its spectrum is evaluated and normalized ONCE
over the extended band, so the three relations share a common amplitude scale.
Its profile already contains strength * reliability; neither is multiplied again.
"""
from __future__ import annotations

import numpy as np

from motion_evidence import motion_evidence


DIRECT_WEIGHT = 1.0
DOUBLE_WEIGHT = 0.8
HALF_WEIGHT = 0.5
EXTENDED_GRID_STEP_BPM = 0.5


def motion_evidence_harmonics(trace, a, b, fps, grid):
    """Return compatible motion metadata and a bounded soft harmonic profile.

    ``grid`` is candidate BPM, not an inferred or reference heart rate. The
    return ``profile`` is max(direct, .8 * double, .5 * half), where all three
    terms are sampled from one extended motion spectrum. Diagnostic relation
    profiles include the estimator's absolute strength/reliability but exclude
    the corresponding relation weight. Unsupported frequencies (at or above
    Nyquist) contribute neutral zero and have explicit False support masks.

    Unknown motion has available=False, strength/speed=NaN and neutral zeros;
    this is not evidence that motion was absent. Observed flat motion is distinct.
    """
    grid = np.asarray(grid, dtype=float)
    if (not np.isfinite(fps) or fps <= 0 or
            not isinstance(a, (int, np.integer)) or
            not isinstance(b, (int, np.integer)) or not 0 <= a < b <= len(trace) or
            grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or
            np.any(grid <= 0) or np.any(np.diff(grid) <= 0) or grid[-1] >= fps * 30):
        raise ValueError('Invalid motion window, FPS or ordered sub-Nyquist BPM grid')

    nyquist_bpm = float(fps * 30)
    queries = {'direct': grid, 'double': 2.0 * grid, 'half': 0.5 * grid}
    support = {name: values < nyquist_bpm for name, values in queries.items()}
    lower = float(0.5 * grid[0])
    upper = float(min(2.0 * grid[-1], np.nextafter(nyquist_bpm, 0.0)))
    # The fixed fine grid captures peaks between candidate queries, avoiding a
    # separate normalization that would promote a weak tail to full strength.
    # Exact query points are included as well; Nyquist-invalid queries are never
    # clamped to a different, supported frequency.
    extended_grid = np.unique(np.concatenate([
        np.arange(lower, upper, EXTENDED_GRID_STEP_BPM),
        np.array([lower, upper]),
        *[values[support[name]] for name, values in queries.items()],
    ]))
    evidence = motion_evidence(trace, a, b, fps, extended_grid)
    extended_profile = evidence['profile'].copy()
    relation_profiles = {}
    for name, values in queries.items():
        profile = np.zeros(len(grid), dtype=float)
        valid = support[name]
        profile[valid] = np.interp(values[valid], extended_grid, extended_profile)
        relation_profiles[name] = profile

    weights = {'direct': DIRECT_WEIGHT, 'double': DOUBLE_WEIGHT, 'half': HALF_WEIGHT}
    weighted = np.stack([weights[name] * relation_profiles[name]
                         for name in ('direct', 'double', 'half')])
    profile = np.clip(np.max(weighted, axis=0), 0.0, 1.0)
    dominant = np.array(['direct', 'double', 'half'], dtype='<U7')[np.argmax(weighted, axis=0)]
    dominant[profile <= 0] = 'none'
    if not evidence['available']:
        dominant[:] = 'unknown'

    result = dict(evidence)
    result.update(
        profile=profile,
        method='extended_motion_soft_harmonics_v26',
        profile_direct=relation_profiles['direct'],
        profile_double=relation_profiles['double'],
        profile_half=relation_profiles['half'],
        frequency_support_direct=support['direct'],
        frequency_support_double=support['double'],
        frequency_support_half=support['half'],
        dominant_relation=dominant,
        harmonic_weights=weights,
        extended_grid_bpm=extended_grid,
        extended_profile=extended_profile,
        nyquist_bpm=nyquist_bpm,
        interpretation='soft_interference_evidence_not_artifact_probability',
    )
    return result

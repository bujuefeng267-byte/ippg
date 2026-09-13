"""Reference-free local candidate routing with immutable V28 protection.

All runtime decisions use video evidence and original V28 outputs. Reference
measurements and video identity are neither arguments nor opened here.
"""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

WINDOW_S = 10.
STEP_S = 1.
JUMP_BPM = 12.


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf-8')


def jump_info(hr):
    accepted = hr.accepted.to_numpy(bool)
    y = hr.ridge_bpm.to_numpy(float)
    adjacent = accepted[1:] & accepted[:-1]
    delta = np.where(adjacent, np.abs(np.diff(y)), 0.)
    return dict(count=int((delta > JUMP_BPM+1e-9).sum()),
                maximum=float(delta.max()) if len(delta) else 0.,
                delta=delta)


def write_wave_preserving_protected_text(path, wave, old_path):
    """Keep original protected decimal fields to avoid CSV reparse drift.

The computational router already preserves these doubles. Keeping the original
decimal spelling also preserves default pandas parsing at protected samples.
"""
    original = pd.read_csv(old_path, dtype=str, keep_default_na=False)
    assert len(original) == len(wave)
    protected = wave.protected_sample.to_numpy(bool)
    export = wave.copy(deep=True)
    export['time_s'] = original.time_s.to_numpy()
    export['base'] = export.base.astype(object)
    export.loc[protected, 'base'] = original.loc[protected, 'base'].to_numpy()
    export.to_csv(path, index=False)
    reread, old = pd.read_csv(path), pd.read_csv(old_path)
    np.testing.assert_array_equal(reread.base[protected], old.base[protected])
    np.testing.assert_array_equal(reread.time_s, old.time_s)


def withdraw(eligible, starts, width, failed_windows):
    """Withdraw intersecting admissions, with no reference or parameter search."""
    eligible = np.asarray(eligible, bool)
    remove = np.zeros(len(eligible), bool)
    for failed in failed_windows:
        a, b = starts[failed], starts[failed]+width
        remove |= (starts < b) & (starts+width > a)
    result = eligible & ~remove
    if np.array_equal(result, eligible) and eligible.any():
        # A global path effect can be outside local admission intervals. The
        # safe terminating response is to restore the complete V28 path.
        result[:] = False
    assert not np.any(result & ~eligible)
    return result


def infer_protected(old_wave_path, old_hr_path, candidate_wave_path,
                    candidate_hr_path, trace, fps, out, mode):
    from proposal_evidence_v31 import build_proposals, ProposalConfig
    from protected_waveform_router import route_protected_waveform, read_saved_protected_hr, ProtectionError
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    old_wave, old_hr = pd.read_csv(old_wave_path), pd.read_csv(old_hr_path)
    candidate_wave, candidate_hr = pd.read_csv(candidate_wave_path), pd.read_csv(candidate_hr_path)
    proposals = build_proposals(old_wave, old_hr, candidate_wave, candidate_hr,
                                trace, fps, config=ProposalConfig(mode=mode))
    proposals.to_csv(out/'proposal_evidence.csv', index=False)
    initial = proposals.proposal_eligible.to_numpy(bool)
    eligible = initial.copy()
    old_ok = old_hr.accepted.to_numpy(bool)
    assert not (eligible & ~old_ok).any()
    width, step = round(WINDOW_S*fps), round(STEP_S*fps)
    starts = np.arange(0, len(trace)-width+1, step, dtype=int)
    assert len(starts) == len(eligible) == len(old_hr)
    old_jumps = jump_info(old_hr)
    trials = out/'guard_attempts'
    trials.mkdir()
    history = []
    for iteration in range(int(initial.sum())+2):
        trial = trials/f'{iteration:03d}'
        trial.mkdir()
        # A requested window with no actual sample/provenance change must not
        # silently become a free HR state. Demote it and rebuild protection.
        while True:
            wave, decisions = route_protected_waveform(old_wave, old_hr, candidate_wave,
                                                        trace, fps, eligible)
            effective = decisions.eligible.to_numpy(bool) & decisions.effective_changed_window.to_numpy(bool)
            if np.array_equal(effective, eligible):
                break
            assert not np.any(effective & ~eligible)
            eligible = effective
        write_wave_preserving_protected_text(trial/'waveform.csv', wave, old_wave_path)
        try:
            hr = read_saved_protected_hr(trial/'waveform.csv', trace, fps, old_hr, eligible)
        except ProtectionError as exc:
            failed = sorted(set(exc.window_indices))
            diagnostic = dict(iteration=iteration, eligible_windows=int(eligible.sum()),
                status='withdraw_after_protection_error', error=str(exc), failed_windows=failed)
            history.append(diagnostic)
            save_json(trial/'protection_error.json', diagnostic)
            if not eligible.any():
                raise
            eligible = withdraw(eligible, starts, width, failed)
            continue
        hr.to_csv(trial/'heart_rate.csv', index=False)
        decisions.to_csv(trial/'routing_decisions.csv', index=False)
        assert not np.any(hr.accepted.to_numpy(bool) & ~old_ok)
        rejected = np.flatnonzero(old_ok & ~hr.accepted.to_numpy(bool))
        new_jumps = jump_info(hr)
        excessive_jumps = (new_jumps['count'] > old_jumps['count'] or
                           new_jumps['maximum'] > max(JUMP_BPM, old_jumps['maximum'])+1e-9)
        failed = set(rejected.tolist())
        if excessive_jumps:
            candidate_edges = np.flatnonzero((new_jumps['delta'] > JUMP_BPM+1e-9) &
                (new_jumps['delta'] > old_jumps['delta']+1e-9))
            for edge in candidate_edges:
                failed.update((int(edge), int(edge+1)))
        record = dict(iteration=iteration, eligible_windows=int(eligible.sum()),
            accepted_windows=int(hr.accepted.sum()), quality_failures=rejected.tolist(),
            large_jump_count=new_jumps['count'], maximum_jump_bpm=new_jumps['maximum'],
            excess_jump_guard=bool(excessive_jumps), failed_windows=sorted(failed))
        history.append(record)
        if not len(rejected) and not excessive_jumps:
            record['status'] = 'accepted'
            break
        record['status'] = 'withdraw_and_recompute'
        if not eligible.any():
            raise AssertionError('Complete V28 fallback must reproduce its acceptance and jumps')
        eligible = withdraw(eligible, starts, width, sorted(failed))
    else:
        raise AssertionError('Monotone protection rollback did not terminate')
    shutil.copyfile(trial/'waveform.csv', out/'waveform.csv')
    final_hr = read_saved_protected_hr(out/'waveform.csv', trace, fps, old_hr, eligible)
    np.testing.assert_array_equal(final_hr.accepted, old_hr.accepted)
    np.testing.assert_allclose(final_hr.ridge_bpm, hr.ridge_bpm, rtol=0, atol=0, equal_nan=True)
    final_hr.to_csv(out/'heart_rate.csv', index=False)
    decisions['initial_eligible'] = initial
    decisions['withdrawn_by_runtime_guard'] = initial & ~eligible
    for key in proposals.columns:
        if key not in decisions:
            decisions[key] = proposals[key].to_numpy()
    np.testing.assert_array_equal(decisions.eligible, eligible)
    decisions.to_csv(out/'routing_decisions.csv', index=False)
    save_json(out/'guard_history.json', history)
    final_wave = pd.read_csv(out/'waveform.csv')
    fixed = ~eligible
    np.testing.assert_allclose(final_hr.ridge_bpm[fixed], old_hr.ridge_bpm[fixed],
                               rtol=0, atol=0, equal_nan=True)
    for i in np.flatnonzero(fixed):
        a, b = starts[i], starts[i]+width
        np.testing.assert_array_equal(final_wave.base.iloc[a:b], old_wave.base.iloc[a:b])
    np.testing.assert_array_equal(np.isfinite(final_wave.base), np.isfinite(old_wave.base))
    changed = np.isfinite(old_wave.base) & (final_wave.base != old_wave.base)
    audit = dict(reference_used=False, mode=mode, offline=True,
        initial_eligible_windows=int(initial.sum()), final_eligible_windows=int(eligible.sum()),
        withdrawn_windows=int((initial & ~eligible).sum()), guard_attempts=len(history),
        modified_samples=int(changed.sum()), modified_seconds=float(changed.sum()/fps),
        protected_samples=int(final_wave.protected_sample.sum()),
        old_accepted_mask_preserved=True, old_waveform_finite_mask_preserved=True,
        protected_waveform_and_HR_preserved=True,
        old_jump_count=old_jumps['count'], new_jump_count=jump_info(final_hr)['count'],
        old_maximum_jump_bpm=old_jumps['maximum'], new_maximum_jump_bpm=jump_info(final_hr)['maximum'],
        preservation_scope='All final noneligible complete HR windows are immutable; reference correctness is evaluated separately.',
        runtime_guard='Monotone withdrawal on old accepted HR loss or excessive large-jump count/amplitude, followed by complete waveform and constrained-HR recomputation.')
    save_json(out/'preservation_audit.json', audit)
    return final_wave, final_hr, decisions, audit

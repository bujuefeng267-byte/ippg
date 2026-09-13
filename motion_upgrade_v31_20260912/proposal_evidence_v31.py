"""Local, reference-free admission of saved CDF candidates against saved V28.

This module proposes windows only. It never modifies a waveform, fills a gap,
chooses a video-specific method, or reads reference heart rates. Root-level
sample protection and final saved-waveform readout are separate operations.
"""
from dataclasses import asdict, dataclass
import json
import numpy as np
import pandas as pd

from component_harmonics_v26 import ComponentConfig
from evidence_hr import DEFAULT_CONFIG as EVIDENCE_CONFIG, score_candidates
from motion_harmonic_evidence_v26 import motion_evidence_harmonics
from direct_guard_router_v28 import MIN_OLD_MOTION_RISK, MIN_MOTION_RISK_DECREASE

ROIS = ('forehead', 'left_cheek', 'right_cheek')
BRANCHES = ('baseline', 'tracked')
METHODS = ('pos', 'chrom')
WINDOW_S, STEP_S, MIN_BPM, MAX_BPM = 10., 1., 42., 210.
ROI_CONFIG = ComponentConfig()


@dataclass(frozen=True)
class ProposalConfig:
    mode: str = 'direct_motion'

    def __post_init__(self):
        if self.mode not in ('direct_motion', 'raw_consensus'):
            raise ValueError('Only the two prospectively declared evidence modes are supported')


def protocol_description():
    return dict(modes=['direct_motion', 'raw_consensus'], window_s=WINDOW_S, step_s=STEP_S,
        min_bpm=MIN_BPM, max_bpm=MAX_BPM, roi_config=asdict(ROI_CONFIG),
        candidate_support=dict(radius_bpm=EVIDENCE_CONFIG.candidate_radius_bpm,
            min_relative_power=EVIDENCE_CONFIG.min_relative_power, min_prominence=EVIDENCE_CONFIG.min_prominence),
        material_disagreement_bpm=EVIDENCE_CONFIG.peak_distance_bpm,
        minimum_direct_old_motion_risk=MIN_OLD_MOTION_RISK,
        minimum_direct_risk_decrease=MIN_MOTION_RISK_DECREASE,
        common_conditions='Both saved HRs accepted, both saved 10s waveforms entirely finite, in-band supported HR, disagreement strictly greater than 6bpm, candidate has actual support in >=2 original physical ROIs.',
        direct_motion='Use profile_direct from the unchanged extended-band motion provider; old risk >=.50 and old-candidate risk >=.30. Half/double relation risks never authorize admission.',
        raw_consensus='Direct-motion condition OR >=2 physical ROIs where original baseline POS and original baseline CHROM each have their strongest PSD candidate within 3bpm of CDF HR and farther than 6bpm from V28 HR.',
        strongest_definition='Maximum relative_power among original score_candidates peaks; ties use smaller BPM. No periodic/harmonic evidence score ranking is used for this test.',
        source_policy='ROI signals must derive from the original un-CDF trace; tracking and methods never create additional physical-region votes.',
        limitation='Motion overlap and agreement of strong original peaks are engineering evidence, not proof of physiological correctness.',
        reference_used=False, waveform_modified=False, coverage_expansion=False)


def _number(table, column):
    if column not in table:
        raise ValueError('Missing required column: '+column)
    return pd.to_numeric(table[column], errors='coerce').to_numpy(float)


def _mask(table, column):
    values = _number(table, column)
    if not np.isfinite(values).all() or not np.isin(values, [0., 1.]).all():
        raise ValueError('Explicit Boolean mask required: '+column)
    return values.astype(bool)


def _wave(table, times, name):
    if len(table) != len(times) or not np.allclose(_number(table, 'time_s'), times, atol=1e-8, rtol=0):
        raise ValueError(name+' waveform clock mismatch')
    signal = _number(table, 'base')
    if np.isinf(signal).any() or not np.array_equal(np.isfinite(signal), _mask(table, 'covered')):
        raise ValueError(name+' waveform finite mask mismatch')
    for flag in ('observed', 'interpolated'):
        if (_mask(table, flag) & ~np.isfinite(signal)).any():
            raise ValueError(name+' sampling flag outside waveform coverage')
    return signal


def _hr(table, starts, width, fps, name):
    if len(table) != len(starts):
        raise ValueError(name+' HR has an incomplete window plan')
    expected = dict(time_s=(starts+width/2)/fps, window_start_s=starts/fps,
                    window_end_s=(starts+width)/fps)
    for key, value in expected.items():
        if not np.allclose(_number(table, key), value, atol=1e-8, rtol=0):
            raise ValueError(name+' HR clock mismatch: '+key)
    values, accepted = _number(table, 'ridge_bpm'), _mask(table, 'accepted')
    if not np.isfinite(values[accepted]).all() or not np.isnan(values[~accepted]).all():
        raise ValueError(name+' accepted/NaN HR mismatch')
    return values, accepted


def _roi_inputs(trace, fps, channels, tables):
    if (channels is None) != (tables is None):
        raise ValueError('Supply both original ROI channels and tables, or neither')
    if channels is None:
        from analyze_components_v26 import validate_component_trace, prepare_roi_channels
        validate_component_trace(trace, fps)
        channels, tables = prepare_roi_channels(trace, fps)
    required = {f'{branch}/{roi}/{method}' for branch in BRANCHES for roi in ROIS for method in METHODS}
    if set(channels) != required or set(tables) != set(BRANCHES):
        raise ValueError('Exactly original baseline/tracked POS/CHROM for three physical ROIs is required')
    observations, qualities, sources = {}, {}, {}
    for roi in ROIS:
        observations[roi] = _mask(trace, roi+'_valid')
        quality = _number(trace, roi+'_quality')
        if not np.isfinite(quality).all() or np.any((quality < 0)|(quality > 1)):
            raise ValueError('Original ROI quality must be finite in [0,1]')
        qualities[roi] = quality
        if roi+'_pixel_source' not in trace:
            raise ValueError('Original tracking-source diagnostics are required')
        sources[roi] = trace[roi+'_pixel_source'].astype(str).to_numpy()
        if not np.isin(sources[roi], ['missing','baseline_reset','tracked_ratio','baseline_ratio_fallback','numerical_reset']).all():
            raise ValueError('Unknown original tracking-source label')
    for branch in BRANCHES:
        table = tables[branch]
        if len(table) != len(trace) or not np.allclose(_number(table,'time_s'),trace.time_s,atol=1e-8,rtol=0):
            raise ValueError('Original ROI waveform clock mismatch')
        for roi in ROIS:
            if not np.array_equal(_mask(table,roi+'_observed'), observations[roi]):
                raise ValueError('Original ROI observation mask mismatch')
            _mask(table,roi+'_interpolated')
            for method in METHODS:
                signal = np.asarray(channels[f'{branch}/{roi}/{method}'],float)
                if signal.shape != (len(trace),) or np.isinf(signal).any():
                    raise ValueError('Original ROI waveform shape or numeric values invalid')
    return channels, observations, qualities, sources


def _channel_evidence(signal, observed, quality, source, branch, start, stop, fps, grid, old_bpm, new_bpm):
    x = np.asarray(signal[start:stop],float)
    q, ob = float(np.mean(quality[start:stop])), float(np.mean(observed[start:stop]))
    track = float(np.mean(source[start:stop] == 'tracked_ratio'))
    reset = float(np.mean(np.isin(source[start:stop],['baseline_reset','numerical_reset'])))
    row = dict(qualified=False, status='waveform_gap', observed_fraction=ob, mean_quality=q,
        tracked_fraction=track, reset_fraction=reset, candidate_supported=False,
        strongest_peak_bpm=None, strongest_relative_power=None,
        strongest_near_candidate=False, strongest_far_from_old=False, support_peaks=[])
    if not np.isfinite(x).all(): return row
    if ob < ROI_CONFIG.min_observed: row['status']='insufficient_observed'; return row
    if q < ROI_CONFIG.min_quality: row['status']='low_roi_quality'; return row
    if branch == 'tracked' and (track < ROI_CONFIG.min_tracked_fraction or reset > ROI_CONFIG.max_reset_fraction):
        row['status']='tracking_quality'; return row
    candidates,_ = score_candidates(x,fps,grid,motion=None)
    if not candidates: row['status']='no_actual_spectral_candidate'; return row
    strongest = min(candidates,key=lambda item:(-item['relative_power'],item['bpm']))
    support = []
    for candidate in candidates:
        for bpm,power in zip(candidate['supported_bpm'],candidate['supported_relative_power']):
            if abs(float(bpm)-new_bpm) <= 1e-8 and power >= EVIDENCE_CONFIG.min_relative_power:
                support.append(dict(peak_bpm=float(candidate['bpm']),target_relative_power=float(power),
                                    prominence=float(candidate['prominence'])))
                break
    row.update(qualified=True,status='qualified',candidate_supported=bool(support),support_peaks=support,
        strongest_peak_bpm=float(strongest['bpm']),strongest_relative_power=float(strongest['relative_power']),
        strongest_near_candidate=bool(abs(strongest['bpm']-new_bpm) <= EVIDENCE_CONFIG.candidate_radius_bpm),
        strongest_far_from_old=bool(abs(strongest['bpm']-old_bpm) > EVIDENCE_CONFIG.peak_distance_bpm))
    return row


def build_proposals(old_wave, old_hr, candidate_wave, candidate_hr, trace, fps, *,
                    roi_channels=None, roi_tables=None, config=ProposalConfig()):
    """Return one auditable proposal row per original complete 10s/1s window.

    Optional ROI inputs must be verified derivatives of the original trace.
    They are not accepted from CDF preprocessing or another candidate model.
    Extra identity/reference columns in input tables are ignored completely.
    """
    if not np.isfinite(fps) or fps <= MAX_BPM/30 or len(trace) == 0:
        raise ValueError('Positive sufficient FPS and a nonempty trace are required')
    times = _number(trace,'time_s')
    if not np.allclose(times,np.arange(len(trace))/fps,atol=1e-8,rtol=0):
        raise ValueError('Original uniform frame clock required')
    if 'frame' in trace and not np.array_equal(_number(trace,'frame'),np.arange(len(trace))):
        raise ValueError('Original ordered frame indices required')
    width,hop = round(WINDOW_S*fps),round(STEP_S*fps)
    starts = np.arange(0,max(0,len(trace)-width+1),hop,dtype=int)
    old_values,new_values = _wave(old_wave,times,'V28'),_wave(candidate_wave,times,'CDF')
    old_bpms,old_ok = _hr(old_hr,starts,width,fps,'V28')
    new_bpms,new_ok = _hr(candidate_hr,starts,width,fps,'CDF')
    channels,observed,quality,pixel_sources = _roi_inputs(trace,fps,roi_channels,roi_tables)
    grid = np.arange(MIN_BPM,MAX_BPM+.01,1.)
    rows = []
    for index,a in enumerate(starts):
        b = a+width
        old_bpm,new_bpm = float(old_bpms[index]),float(new_bpms[index])
        row = dict(window_index=index,time_s=(a+width/2)/fps,window_start_s=a/fps,window_end_s=b/fps,
            mode=config.mode,old_bpm=old_bpm,candidate_bpm=new_bpm,
            old_accepted=bool(old_ok[index]),candidate_accepted=bool(new_ok[index]),
            old_window_finite=bool(np.isfinite(old_values[a:b]).all()),
            candidate_window_finite=bool(np.isfinite(new_values[a:b]).all()),
            disagreement_bpm=float(abs(new_bpm-old_bpm)),physical_roi_support_count=0,
            physical_roi_support_names='[]',raw_consensus_roi_count=0,raw_consensus_roi_names='[]',
            motion_available=False,motion_status='not_evaluated',old_direct_motion_risk=np.nan,
            candidate_direct_motion_risk=np.nan,direct_motion_risk_decrease=np.nan,
            direct_motion_pass=False,raw_consensus_pass=False,proposal_eligible=False,
            reason='old_hr_not_accepted',roi_evidence_json='{}')
        if not old_ok[index]: rows.append(row); continue
        if not new_ok[index]: row['reason']='candidate_hr_not_accepted'; rows.append(row); continue
        if not row['old_window_finite']: row['reason']='old_waveform_gap'; rows.append(row); continue
        if not row['candidate_window_finite']: row['reason']='candidate_waveform_gap'; rows.append(row); continue
        if not (MIN_BPM <= old_bpm <= MAX_BPM and MIN_BPM <= new_bpm <= MAX_BPM):
            row['reason']='frequency_outside_fixed_band'; rows.append(row); continue
        if row['disagreement_bpm'] <= EVIDENCE_CONFIG.peak_distance_bpm:
            row['reason']='no_material_disagreement'; rows.append(row); continue
        diagnostics = {}
        for key in sorted(channels):
            branch,roi,method = key.split('/')
            diagnostics[key] = _channel_evidence(channels[key],observed[roi],quality[roi],pixel_sources[roi],
                branch,a,b,fps,grid,old_bpm,new_bpm)
        support = [roi for roi in ROIS if any(diagnostics[f'{branch}/{roi}/{method}']['candidate_supported']
                   for branch in BRANCHES for method in METHODS)]
        consensus = [roi for roi in ROIS if all(diagnostics[f'baseline/{roi}/{method}']['qualified'] and
            diagnostics[f'baseline/{roi}/{method}']['strongest_near_candidate'] and
            diagnostics[f'baseline/{roi}/{method}']['strongest_far_from_old'] for method in METHODS)]
        row.update(physical_roi_support_count=len(support),physical_roi_support_names=json.dumps(support),
            raw_consensus_roi_count=len(consensus),raw_consensus_roi_names=json.dumps(consensus),
            raw_consensus_pass=len(consensus) >= ROI_CONFIG.min_rois,
            roi_evidence_json=json.dumps(diagnostics,allow_nan=False,separators=(',',':')))
        if len(support) < ROI_CONFIG.min_rois:
            row['reason']='insufficient_original_physical_roi_support'; rows.append(row); continue
        # Preserve the existing extended-band normalization; only its direct
        # component participates in admission, never the max of harmonic risks.
        motion = motion_evidence_harmonics(trace,int(a),int(b),fps,grid)
        row.update(motion_available=bool(motion['available']),motion_status=motion['status'])
        if motion['available']:
            risk = np.asarray(motion['profile_direct'],float)
            if risk.shape != grid.shape or not np.isfinite(risk).all() or np.any((risk < 0)|(risk > 1)):
                raise ValueError('Invalid inherited direct-motion profile')
            old_risk,new_risk = np.interp([old_bpm,new_bpm],grid,risk)
            drop = float(old_risk-new_risk)
            row.update(old_direct_motion_risk=float(old_risk),candidate_direct_motion_risk=float(new_risk),
                direct_motion_risk_decrease=drop,direct_motion_pass=bool(
                    old_risk >= MIN_OLD_MOTION_RISK and drop >= MIN_MOTION_RISK_DECREASE))
        if row['direct_motion_pass']:
            row.update(proposal_eligible=True,reason='direct_motion_evidence_and_original_roi_support')
        elif config.mode == 'raw_consensus' and row['raw_consensus_pass']:
            row.update(proposal_eligible=True,reason='original_baseline_POS_CHROM_main_peak_consensus')
        else:
            row['reason']='admission_evidence_insufficient'
        rows.append(row)
    result = pd.DataFrame(rows)
    if not rows:
        result = pd.DataFrame(columns=['window_index','time_s','window_start_s','window_end_s','mode',
            'old_bpm','candidate_bpm','proposal_eligible','reason','physical_roi_support_count',
            'direct_motion_pass','raw_consensus_pass','roi_evidence_json'])
    result.attrs.update(protocol_description(),mode=config.mode,reference_used=False)
    return result

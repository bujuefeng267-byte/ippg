"""Reference-free routing of original and bounded tracking waveform proposals.

Branches must independently pass the unchanged multi-ROI waveform gates.
Correlated branches do not count as extra ROIs or independent sensors. The
default legacy mode preserves the original branch on large disagreements.
The optional evidence mode may select either branch or remain unresolved;
no past HR is filled and no reference values enter a decision.
The final HR is subsequently derived from the saved overlap-added waveform.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from motion_fusion import FusionConfig, fuse_windows, _correlation


@dataclass(frozen=True)
class GuardConfig:
    min_tracked_fraction: float = 0.60
    max_reset_fraction: float = 0.05
    switch_score_ratio: float = 1.15
    max_branch_disagreement_bpm: float = 12.0
    routing_mode: str = 'legacy'
    min_evidence_score: float = 0.20
    max_motion_evidence_penalty: float = 0.45
    disagreement_score_ratio: float = 1.35
    min_disagreement_score_margin: float = 0.08

    def __post_init__(self):
        if not 0<self.min_tracked_fraction<=1 or not 0<=self.max_reset_fraction<=1:
            raise ValueError('Tracking/reset fractions must be bounded')
        if not np.isfinite(self.switch_score_ratio) or self.switch_score_ratio<=1:
            raise ValueError('A switch must require stronger evidence')
        if not np.isfinite(self.max_branch_disagreement_bpm) or self.max_branch_disagreement_bpm<=0:
            raise ValueError('Branch disagreement bound must be positive')
        if self.routing_mode not in ('legacy', 'evidence'):
            raise ValueError('routing_mode must be legacy or evidence')
        if not np.isfinite(self.min_evidence_score) or not 0 < self.min_evidence_score <= 1:
            raise ValueError('min_evidence_score must be in (0, 1]')
        if not np.isfinite(self.max_motion_evidence_penalty) or not 0 <= self.max_motion_evidence_penalty < 1:
            raise ValueError('Motion evidence must remain a soft penalty (<1)')
        if not np.isfinite(self.disagreement_score_ratio) or self.disagreement_score_ratio <= 1:
            raise ValueError('Disagreement requires a score ratio greater than one')
        if not np.isfinite(self.min_disagreement_score_margin) or not 0 < self.min_disagreement_score_margin <= 1:
            raise ValueError('Disagreement margin must be in (0, 1]')


ROUTING_COLUMNS=['window_index','time_s','selected_branch','selection_reason',
    'baseline_bpm','tracked_bpm','baseline_score','tracked_score',
    'baseline_generated','tracked_generated','tracking_eligible','minimum_tracked_fraction',
    'second_largest_tracked_fraction','maximum_reset_fraction',
    'baseline_motion_overlap','tracked_motion_overlap']

EVIDENCE_COLUMNS = ['routing_mode', 'baseline_pulse_support', 'tracked_pulse_support',
    'baseline_motion_risk', 'tracked_motion_risk', 'baseline_ambiguity', 'tracked_ambiguity',
    'baseline_contributing_roi_count', 'tracked_contributing_roi_count',
    'motion_available', 'motion_status', 'motion_observed_fraction',
    'motion_speed_face_per_s', 'motion_strength', 'motion_reliability']


def _unit(value, missing=0.0):
    """Bound engineering features; unavailable features never become infinite support."""
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return missing
    return float(np.clip(value, 0., 1.)) if np.isfinite(value) else missing


def _pulse_support(row, diagnostics):
    """Use only actual waveform contributors, counting a physical ROI once.

    POS/CHROM share pixels: their strongest contributing spectral feature is
    retained within an ROI, not added as an independent vote. ROI features are
    averaged without a bonus for the number of correlated branches/regions.
    This is spectral support, not proof that a common peak is physiological.
    """
    actual = diagnostics.loc[diagnostics.waveform_weight > 0]
    strengths = []
    for _, group in actual.groupby('roi', sort=True):
        strengths.append(max(.5 * (_unit(r.peak_concentration) + _unit(r.prominence_ratio))
                             for r in group.itertuples()))
    if not strengths:
        return 0.0, 0
    return .5 * _unit(row.get('quality_proxy')) + .5 * float(np.mean(strengths)), len(strengths)


def _score_evidence(row, branch_evidence, config):
    # motion_evidence.profile already includes normalized spectrum, physical
    # amplitude and reliability. Do NOT multiply strength/reliability again.
    supplied = branch_evidence or {}
    pulse = _unit(supplied.get('pulse_support', row.get('quality_proxy')))
    ambiguity = _unit(row.get('runner_up_ratio'), missing=1.)
    available = bool(supplied.get('motion_available', False))
    risk = _unit(supplied.get('motion_risk')) if available else 0.
    return pulse * (1. - .5 * ambiguity) * (1. - config.max_motion_evidence_penalty * risk)


def _choose_with_evidence(baseline, tracked, baseline_generated, tracked_generated,
                          tracking_ok, config, evidence):
    evidence = evidence or {}
    bs = _score_evidence(baseline, evidence.get('baseline'), config)
    ts = _score_evidence(tracked, evidence.get('tracked'), config)
    def valid_bpm(row):
        try:
            bpm = float(row.get('ridge_bpm', np.nan))
        except (TypeError, ValueError, OverflowError):
            return False
        return np.isfinite(bpm) and bpm > 0
    b_ok = bool(baseline_generated and valid_bpm(baseline) and bs >= config.min_evidence_score)
    t_ok = bool(tracked_generated and tracking_ok and valid_bpm(tracked) and ts >= config.min_evidence_score)
    if not b_ok and not t_ok:
        return 'none', 'insufficient_branch_evidence', bs, ts
    if not t_ok:
        return 'baseline', 'only_baseline_qualified_evidence', bs, ts
    if not b_ok:
        return 'tracked', 'only_tracked_qualified_evidence', bs, ts
    difference = abs(float(tracked['ridge_bpm']) - float(baseline['ridge_bpm']))
    if difference > config.max_branch_disagreement_bpm:
        # Symmetric decision at large disagreement: no low-frequency or
        # baseline preference, and no previous HR/ground truth tie breaker.
        if ts >= config.disagreement_score_ratio * bs and ts - bs >= config.min_disagreement_score_margin:
            return 'tracked', 'disagreement_resolved_tracked_evidence', bs, ts
        if bs >= config.disagreement_score_ratio * ts and bs - ts >= config.min_disagreement_score_margin:
            return 'baseline', 'disagreement_resolved_baseline_evidence', bs, ts
        return 'none', 'unresolved_frequency_disagreement', bs, ts
    if ts >= config.switch_score_ratio * bs:
        return 'tracked', 'stronger_qualified_evidence', bs, ts
    return 'baseline', 'baseline_evidence_retained', bs, ts


def tracking_eligibility(trace,start,stop,config=GuardConfig(),contributing_rois=None):
    fractions=[];resets=[]
    for roi in FusionConfig().roi_names:
        source=trace[f'{roi}_pixel_source'].iloc[start:stop].astype(str)
        fractions.append(float((source=='tracked_ratio').mean()))
        resets.append(float(source.isin(['baseline_reset','numerical_reset']).mean()))
    allowed=set(FusionConfig().roi_names if contributing_rois is None else contributing_rois)
    usable=[roi in allowed and f>=config.min_tracked_fraction and r<=config.max_reset_fraction
            for roi,f,r in zip(FusionConfig().roi_names,fractions,resets)]
    return sum(usable)>=2, fractions, resets


def choose_branch(baseline,tracked,baseline_generated,tracked_generated,tracking_ok,
                  config=GuardConfig(),evidence=None):
    """Quality is an uncalibrated engineering score, never accuracy probability."""
    if config.routing_mode == 'evidence':
        return _choose_with_evidence(baseline, tracked, baseline_generated,
                                    tracked_generated, tracking_ok, config, evidence)
    score=lambda row: float(row['quality_proxy'])*(1-.25*float(row['motion_overlap']))*(1-.5*float(row['runner_up_ratio']))
    bs,ts=score(baseline),score(tracked)
    if not tracked_generated or not tracking_ok:
        return ('baseline','tracking_quality_fallback',bs,ts) if baseline_generated else ('none','no_qualified_branch',bs,ts)
    if not baseline_generated:
        return 'tracked','baseline_unavailable',bs,ts
    difference=abs(float(tracked['ridge_bpm'])-float(baseline['ridge_bpm']))
    if difference>config.max_branch_disagreement_bpm:
        return 'baseline','frequency_disagreement_fallback',bs,ts
    if ts>=config.switch_score_ratio*bs:
        return 'tracked','stronger_qualified_evidence',bs,ts
    return 'baseline','baseline_evidence_retained',bs,ts


def guarded_fuse(baseline_signals,tracked_signals,trace,fps,window_s=10,step_s=1,
                 min_bpm=42,max_bpm=210,config=FusionConfig(),guard=GuardConfig()):
    branches={name:fuse_windows(signals,trace,fps,window_s,step_s,min_bpm,max_bpm,
                                config=config,return_segments=True)
              for name,signals in [('baseline',baseline_signals),('tracked',tracked_signals)]}
    base_rows=branches['baseline'][0];track_rows=branches['tracked'][0]
    n=len(trace);window=round(window_s*fps);step=round(step_s*fps)
    evidence_mode = guard.routing_mode == 'evidence'
    if evidence_mode:
        from motion_evidence import motion_evidence
        grid = np.arange(min_bpm, max_bpm + config.grid_step_bpm / 2., config.grid_step_bpm)
    taper=np.hanning(window+2)[1:-1]
    wave_sum=np.zeros(n);wave_weight=np.zeros(n)
    selected=[];routing=[];diags=[]
    for wi,start in enumerate(range(0,n-window+1,step)):
        stop=start+window
        tracked_diag=branches['tracked'][2]
        contributing_rois=tracked_diag.loc[(tracked_diag.window_index==wi)&
            (tracked_diag.waveform_weight>0),'roi'].unique()
        ok,fractions,resets=tracking_eligibility(trace,start,stop,guard,contributing_rois)
        b,t=base_rows.iloc[wi],track_rows.iloc[wi]
        generated={k:wi in v[3] for k,v in branches.items()}
        evidence = None
        extra = {}
        if evidence_mode:
            motion = motion_evidence(trace, start, stop, fps, grid)
            profile = np.asarray(motion['profile'], dtype=float)
            if profile.shape != grid.shape or not np.isfinite(profile).all() or np.any((profile < 0) | (profile > 1)):
                raise ValueError('Motion evidence profile must match BPM grid and be finite in [0,1]')
            available = bool(motion['available'])
            evidence = {}
            extra = dict(routing_mode=guard.routing_mode, motion_available=available,
                motion_status=motion.get('status', 'available' if available else 'unknown'),
                motion_observed_fraction=motion.get('observed_fraction', np.nan),
                motion_speed_face_per_s=motion.get('speed_face_per_s', np.nan),
                motion_strength=motion.get('strength', np.nan), motion_reliability=motion.get('reliability', 0.))
            for branch, proposal in [('baseline', b), ('tracked', t)]:
                diag = branches[branch][2]
                pulse, count = _pulse_support(proposal, diag.loc[diag.window_index == wi])
                bpm = float(proposal.ridge_bpm)
                # Unavailable motion stays explicitly unknown in audit columns;
                # zero below is only the neutral arithmetic penalty.
                risk = float(np.interp(bpm, grid, profile)) if available and np.isfinite(bpm) else 0.
                evidence[branch] = dict(pulse_support=pulse, motion_risk=risk, motion_available=available)
                extra.update({f'{branch}_pulse_support': pulse,
                    f'{branch}_motion_risk': risk if available and np.isfinite(bpm) else np.nan,
                    f'{branch}_ambiguity': _unit(proposal.runner_up_ratio, missing=1.),
                    f'{branch}_contributing_roi_count': count})
        name,reason,bs,ts=choose_branch(b,t,generated['baseline'],generated['tracked'],ok,guard,evidence)
        row=(t if name=='tracked' else b).copy()
        if name=='none':
            row['accepted']=False;row['waveform_roi_count']=0
            row['ridge_bpm']=row['spectral_peak_bpm']=np.nan
            if evidence_mode:
                row['status']='guard_'+reason
                row['state']='searching'
        selected.append(row)
        routing.append(dict(window_index=wi,time_s=b.time_s,selected_branch=name,selection_reason=reason,
            baseline_bpm=b.ridge_bpm,tracked_bpm=t.ridge_bpm,baseline_score=bs,tracked_score=ts,
            baseline_generated=generated['baseline'],tracked_generated=generated['tracked'],
            tracking_eligible=ok,minimum_tracked_fraction=min(fractions),
            second_largest_tracked_fraction=sorted(fractions)[1],maximum_reset_fraction=max(resets),
            baseline_motion_overlap=b.motion_overlap,tracked_motion_overlap=t.motion_overlap, **extra))
        if name!='none':
            segment=branches[name][3][wi]
            wave=segment['waveform'].copy()
            existing=wave_weight[start:stop]>0
            if existing.sum()>=round(fps):
                prior=wave_sum[start:stop][existing]/wave_weight[start:stop][existing]
                if _correlation(wave[existing],prior)<0:wave=-wave
            weights=taper*segment['quality']
            wave_sum[start:stop]+=wave*weights;wave_weight[start:stop]+=weights
        for branch,(_,_,d,_) in branches.items():
            subset=d.loc[d.window_index==wi].copy()
            subset['branch']=branch
            subset['original_channel']=subset.channel
            subset['method']=branch+'_'+subset.method
            subset['channel']=subset.roi+'_'+subset.method
            if branch!=name:
                subset['waveform_weight']=0.;subset['output_accepted']=False
            diags.append(subset)
    out=np.full(n,np.nan)
    np.divide(wave_sum,wave_weight,out=out,where=wave_weight>0)
    table=pd.DataFrame(selected,columns=base_rows.columns)
    table['accepted']=table.accepted.astype(bool)
    diagnostics=pd.concat(diags,ignore_index=True) if diags else branches['baseline'][2].copy()
    routing=pd.DataFrame(routing,columns=ROUTING_COLUMNS + (EVIDENCE_COLUMNS if evidence_mode else []))
    table.attrs.update(guard_config=asdict(guard),quality_proxy_is_probability=False)
    return table,out,diagnostics,routing,branches

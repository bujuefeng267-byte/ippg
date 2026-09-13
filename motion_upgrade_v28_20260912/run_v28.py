"""One predeclared V26 routing repair; reference-free prediction, separate evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别')
B=P/'results/data1_6_20260911'
V25=P/'results/data1_6_v25_20260911/stage4_preserve_waveform'
V26=P/'results/data1_6_v26_20260911'
ROOT=P/'results/data1_6_v28_20260912'
VARIANT='direct_guard'
SOURCE_NAMES=('analyze_motion_v2.py','analyze_rppg.py','anchored_reconstruction.py',
    'baseline_frontend.py','component_fusion_v26.py','component_harmonics_v26.py',
    'conservative_component_router_v26.py','evidence_hr.py','guarded_fusion.py',
    'legacy_motion.py','motion_evidence.py','motion_harmonic_evidence_v26.py',
    'motion_frontend.py','motion_fusion.py','pixel_tracking.py','signal_candidates_v26.py',
    'stable_groups.py','waveform_hr.py','direct_guard_router_v28.py','run_v28.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def save(path,value):Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def inputs(case):
    return [B/case/'inference/frame_trace.csv',B/case/'inference/frame_trace.json',
        V25/case/'fusion_waveform.csv',V25/case/'fusion_heart_rate.csv',
        V26/'component_harmonics'/case/'waveform.csv',
        V26/'component_harmonics'/case/'component_proposals.csv',
        V26/'component_harmonics'/case/'summary.json']

def freeze():
    from evaluate_v28 import freeze_evaluation_inputs
    for name in SOURCE_NAMES:
        if name not in ('direct_guard_router_v28.py','run_v28.py'):
            assert sha(HERE/name)==sha(P/'motion_upgrade_v26_20260911'/name),f'Changed inherited module: {name}'
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'protocol_before_run.json'
    if path.exists():raise FileExistsError(path)
    entries=[P/name for name in ('run.sh','run_motion_v25.sh','run_motion_v26_experimental.sh','run_motion_v27_experimental.sh')]
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),variants=[VARIANT],
        source_hashes={name:sha(HERE/name) for name in SOURCE_NAMES},
        evaluation_hashes={str(p):sha(p) for p in [HERE/'evaluate_v28.py',P/'batch_analysis_20260911/evaluate_batch.py']},
        evaluation_inputs=freeze_evaluation_inputs(),
        input_hashes={f'data{i}':{str(p):sha(p) for p in inputs(f'data{i}')} for i in range(1,7)},
        fixed_inputs_manifest_sha256=sha(B/'inputs_manifest.json'),
        protected_entries={str(p):sha(p) for p in entries if p.exists()},
        reference_used_in_inference=False,previously_seen_development_data=True,
        maximum_candidates=1,no_post_score_threshold_search=True,
        baseline='V26 conservative_components; original V25 waveform is the fallback',
        structural_change='Require motion profile_direct evaluated at the OLD inferred heart rate >= the existing MIN_OLD_MOTION_RISK=0.50 before any replacement. Keep the original maximum harmonic-risk gate>=0.50 and decrease>=0.30 and all sampling/contributor/gap conditions.',
        no_new_thresholds=True,all_other_modules_unchanged=True,
        inference='Re-use content-bound original traces, V25 output, V26 harmonic component wave/proposals; route actual samples, save and reload the new waveform, then run unchanged V25 estimate_evidence. Never splice heart rates.',
        fixed_windows=dict(window_s=10,step_s=1,min_bpm=42,max_bpm=210),
        comparison='Same original 309 reference windows and estimated alignment, fixed +/-1/2/5 second sensitivity; no offset selection or per-video method selection.',
        retention_scope='Preserve V26 conservative-branch gains when direct motion evidence supports replacement; do not claim to retain every gain of the full V26 harmonic branch.',
        expected_diagnostic_mechanism='Earlier diagnosis showed double-only motion relations can wrongly replace V25 half-frequency estimates. A relation alone is insufficient to authorize replacement.',
        promotion='Original eight V25 gates unchanged; report V26 harmonic and conservative comparators, all six videos and common/new/lost windows.')
    save(path,record);print(json.dumps(dict(protocol=str(path),variants=record['variants'],sources=len(SOURCE_NAMES))),flush=True)

def frozen(case=None):
    fixed=json.loads((ROOT/'protocol_before_run.json').read_text())
    for name,h in fixed['source_hashes'].items():assert sha(HERE/name)==h,name
    for path,h in fixed['protected_entries'].items():assert sha(path)==h,path
    assert sha(B/'inputs_manifest.json')==fixed['fixed_inputs_manifest_sha256']
    if case is not None:
        assert {str(p) for p in inputs(case)}==set(fixed['input_hashes'][case])
        for path,h in fixed['input_hashes'][case].items():assert sha(path)==h,path
    return fixed

def readout(path,trace,fps):
    from evidence_hr import estimate_evidence
    saved=pd.read_csv(path)
    np.testing.assert_allclose(saved.time_s,trace.time_s,atol=1e-8,rtol=0)
    np.testing.assert_array_equal(saved.covered,np.isfinite(saved.base))
    hr=estimate_evidence(saved.base.to_numpy(float),pd.DataFrame({'rgb_valid':saved.observed}),
                         saved.interpolated.to_numpy(bool),fps,motion_trace=trace)
    hr['raw_spectral_peak_bpm']=hr.spectral_peak_bpm
    hr.loc[~hr.accepted,['spectral_peak_bpm','ridge_bpm']]=np.nan
    starts=np.arange(0,len(trace)-round(10*fps)+1,round(fps))
    hr['window_start_s']=starts/fps;hr['window_end_s']=(starts+round(10*fps))/fps
    hr['hr_source']='saved_direct_guard_V26_V25_waveform_offline'
    return hr

def run(case):
    from direct_guard_router_v28 import route_components
    from analyze_motion_v2 import validate_trace
    fixed=frozen(case);started=time.perf_counter()
    out=ROOT/VARIANT/case;out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((B/case/'inference/frame_trace.json').read_text());fps=meta['fps']
    assert sha(B/case/'inference/frame_trace.csv')==meta['trace_sha256']
    trace=pd.read_csv(B/case/'inference/frame_trace.csv');validate_trace(trace,fps)
    assert len(trace)==meta['n_frames']
    old_wave=pd.read_csv(V25/case/'fusion_waveform.csv');old_hr=pd.read_csv(V25/case/'fusion_heart_rate.csv')
    candidate=pd.read_csv(V26/'component_harmonics'/case/'waveform.csv')
    proposals=pd.read_csv(V26/'component_harmonics'/case/'component_proposals.csv')
    wave,decisions=route_components(old_wave,old_hr,candidate,proposals,trace,fps)
    np.testing.assert_array_equal(np.isfinite(old_wave.base),np.isfinite(wave.base))
    wave.to_csv(out/'waveform.csv',index=False)
    hr=readout(out/'waveform.csv',trace,fps)
    hr.to_csv(out/'heart_rate.csv',index=False);decisions.to_csv(out/'routing_decisions.csv',index=False)
    summary=dict(status='complete',case=case,variant=VARIANT,frames=len(trace),fps=fps,
        source_hashes=fixed['source_hashes'],reference_used=False,offline=True,hr_from_saved_waveform=True,
        protocol_sha256=sha(ROOT/'protocol_before_run.json'),input_hashes=fixed['input_hashes'][case],
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        eligible_windows=int(decisions.route_eligible.sum()),planned_windows=len(hr),accepted_windows=int(hr.accepted.sum()),
        waveform_coverage_pct=100*float(wave.covered.mean()),hr_coverage_pct=100*float(hr.accepted.mean()),
        elapsed_s=time.perf_counter()-started,old_finite_mask_preserved=True,
        interpretation='Actual V25 samples mixed with measured V26 filtered components. Not validation of PPG morphology.')
    frozen(case);save(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['case','eligible_windows','planned_windows','accepted_windows','waveform_coverage_pct','hr_coverage_pct']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['freeze','infer'])
    ap.add_argument('--case',choices=[f'data{i}' for i in range(1,7)]);a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:
        if not a.case:ap.error('--case required')
        run(a.case)

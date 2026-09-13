"""Final fixed V26 routing trial; no reference values enter inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, time
import numpy as np
import pandas as pd
from conservative_component_router_v26 import route_components
from evidence_hr import estimate_evidence
from analyze_motion_v2 import source_hashes, validate_trace

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
B = P/'results/data1_6_20260911'
V25 = P/'results/data1_6_v25_20260911/stage4_preserve_waveform'
ROOT = P/'results/data1_6_v26_20260911'
H = ROOT/'component_harmonics'
OUT = ROOT/'conservative_components'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False))
def frozen():
    names = ['conservative_component_router_v26.py', 'test_conservative_component_router_v26.py',
             'run_conservative_components.py', 'motion_harmonic_evidence_v26.py', 'evaluate_v26.py']
    return {**source_hashes(), **{n: sha(HERE/n) for n in names}}

def main():
    fixed = frozen()
    OUT.mkdir()
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=fixed,
        reference_used=False, post_diagnosis_development=True,
        reason='Final bounded revision after harmonic-component gains and regressions: use reference-free motion risk to retain V25 whenever candidate replacement lacks strong geometric evidence.',
        fixed_rule='Old accepted, candidate generated with >=2 physical ROIs and finite full window, old harmonic-motion risk >=0.50 and old-minus-candidate risk >=0.30; no per-video parameters.',
        waveform='Convex combination of actual old and candidate saved samples using eligible/all planned-window Hann ratio; retain exact old finite mask.',
        readout='Re-read saved waveform, then unchanged V25 estimate_evidence. Never copy a candidate or reference HR to the final output.',
        validation='Same frozen 309 windows and estimated offsets; evaluate_v26.py unchanged.',
        previously_observed_reports={name:sha(ROOT/name/'evaluation_summary.json') for name in
            ['log_projection_baseline', 'log_projection_guarded', 'component_consensus', 'neural_efficientphys', 'component_harmonics']},
        input_hashes={f'data{i}':{str(path):sha(path) for path in
            [V25/f'data{i}'/'fusion_waveform.csv', V25/f'data{i}'/'fusion_heart_rate.csv',
             H/f'data{i}'/'waveform.csv', H/f'data{i}'/'component_proposals.csv',
             B/f'data{i}'/'inference/frame_trace.csv', B/f'data{i}'/'inference/frame_trace.json']}
            for i in range(1,7)})
    save(OUT/'protocol_before_run.json', protocol)
    results=[]
    for i in range(1,7):
        case=f'data{i}'; out=OUT/case; out.mkdir(); start=time.perf_counter()
        meta=json.loads((B/case/'inference/frame_trace.json').read_text()); fps=meta['fps']
        trace=pd.read_csv(B/case/'inference/frame_trace.csv'); validate_trace(trace,fps)
        assert sha(B/case/'inference/frame_trace.csv')==meta['trace_sha256']
        old_wave=pd.read_csv(V25/case/'fusion_waveform.csv'); old_hr=pd.read_csv(V25/case/'fusion_heart_rate.csv')
        candidate=pd.read_csv(H/case/'waveform.csv'); proposals=pd.read_csv(H/case/'component_proposals.csv')
        wave, decisions=route_components(old_wave,old_hr,candidate,proposals,trace,fps)
        np.testing.assert_array_equal(np.isfinite(old_wave.base),np.isfinite(wave.base))
        wave.to_csv(out/'waveform.csv',index=False); saved=pd.read_csv(out/'waveform.csv')
        hr=estimate_evidence(saved.base.to_numpy(float),pd.DataFrame({'rgb_valid':saved.observed}),
                             saved.interpolated.to_numpy(bool),fps,motion_trace=trace)
        hr['raw_spectral_peak_bpm']=hr.spectral_peak_bpm
        hr.loc[~hr.accepted,['spectral_peak_bpm','ridge_bpm']]=np.nan
        starts=np.arange(0,len(trace)-round(10*fps)+1,round(fps))
        hr['window_start_s']=starts/fps; hr['window_end_s']=(starts+round(10*fps))/fps
        hr['hr_source']='saved_conservative_component_waveform_offline'
        hr.to_csv(out/'heart_rate.csv',index=False); decisions.to_csv(out/'routing_decisions.csv',index=False)
        summary=dict(case=case,variant='conservative_components',fps=fps,frames=len(trace),source_hashes=fixed,
            reference_used=False,offline=True,hr_from_saved_waveform=True,old_finite_mask_preserved=True,
            waveform_coverage_pct=100*float(wave.covered.mean()),hr_coverage_pct=100*float(hr.accepted.mean()),
            elapsed_s=time.perf_counter()-start)
        save(out/'summary.json',summary);results.append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    assert frozen()==fixed
    save(OUT/'runs.json',results)

if __name__=='__main__': main()

"""Post-diagnosis fixed joint spectral consensus; labels excluded from inference."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import pandas as pd
from component_fusion_v26 import fuse_components,ComponentConfig,ROIS
from evidence_hr import estimate_evidence
from analyze_motion_v2 import validate_trace,source_hashes

HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别');BASE=P/'results/data1_6_20260911';ROOT=P/'results/data1_6_v26_20260911';OUT=ROOT/'component_consensus'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def frozen():return {**source_hashes(),**{n:sha(HERE/n) for n in ['component_fusion_v26.py','run_component_fusion.py','test_component_fusion_v26.py','evaluate_v26.py']}}

inputs=json.loads((BASE/'inputs_manifest.json').read_text());fixed=frozen();OUT.mkdir()
protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=fixed,component_config=asdict(ComponentConfig()),
    reference_used=False,post_diagnosis_development=True,
    reason='ROI diagnostic showed complementary weak supported peaks lost before HR estimation; test joint frequency competition with one vote per physical ROI, then phase-preserving measured-component filtering.',
    source_candidate_and_DP_defaults='Exactly existing V25 evidence_hr defaults; no per-video values or target-frequency input',
    waveform_interpretation='Adaptively band-limited measured component, not PPG morphology validation',
    validation='Same 309 reference/planned windows, original offsets, evaluate_v26.py unchanged',
    previously_observed_development_reports={name:sha(ROOT/name/'evaluation_summary.json') for name in ['log_projection_baseline','log_projection_guarded']},
    input_hashes={c['case']:{name:sha(BASE/c['case']/'inference'/name) for name in ['frame_trace.csv','frame_trace.json','baseline_roi_waveforms.csv','roi_waveforms.csv']} for c in inputs})
save(OUT/'protocol_before_run.json',protocol);results=[]
for item in inputs:
    case=item['case'];d=BASE/case/'inference';out=OUT/case;out.mkdir();start=time.perf_counter()
    trace=pd.read_csv(d/'frame_trace.csv');meta=json.loads((d/'frame_trace.json').read_text());fps=meta['fps'];validate_trace(trace,fps)
    assert sha(d/'frame_trace.csv')==meta['trace_sha256']
    for name,v in meta['identity']['frontend_hashes'].items():assert sha(HERE/name)==v
    channels={};rw=None
    for branch,file in [('baseline','baseline_roi_waveforms.csv'),('tracked','roi_waveforms.csv')]:
        source=pd.read_csv(d/file)
        np.testing.assert_allclose(source.time_s,trace.time_s,atol=1e-8,rtol=0)
        if rw is None:rw=source
        else:
            for roi in ROIS:
                for suffix in ['observed','interpolated']:np.testing.assert_array_equal(rw[f'{roi}_{suffix}'],source[f'{roi}_{suffix}'])
        for roi in ROIS:
            for method in ['pos','chrom']:channels[f'{branch}/{roi}/{method}']=source[f'{roi}_{method}'].to_numpy(float)
    wave,proposals,candidates=fuse_components(channels,trace,rw,fps)
    wave.to_csv(out/'waveform.csv',index=False);saved=pd.read_csv(out/'waveform.csv')
    hr=estimate_evidence(saved.base.to_numpy(float),pd.DataFrame({'rgb_valid':saved.observed}),saved.interpolated.to_numpy(bool),fps,motion_trace=trace)
    hr['raw_spectral_peak_bpm']=hr.spectral_peak_bpm;hr.loc[~hr.accepted,['spectral_peak_bpm','ridge_bpm']]=np.nan
    starts=np.arange(0,len(trace)-round(10*fps)+1,round(fps));hr['window_start_s']=starts/fps;hr['window_end_s']=(starts+round(10*fps))/fps
    hr['component_proposal_bpm']=proposals.proposal_bpm;hr['hr_source']='saved_measured_component_offline'
    hr.to_csv(out/'heart_rate.csv',index=False);proposals.to_csv(out/'component_proposals.csv',index=False);save(out/'all_roi_candidates.json',candidates)
    summary=dict(case=case,variant='component_consensus',fps=fps,frames=len(trace),source_hashes=fixed,
        config=protocol,reference_used=False,offline=True,hr_from_saved_waveform=True,
        waveform_coverage_pct=100*float(wave.covered.mean()),hr_coverage_pct=100*float(hr.accepted.mean()),elapsed_s=time.perf_counter()-start)
    save(out/'summary.json',summary);results.append(dict(case=case,returncode=0,elapsed_s=summary['elapsed_s']))
    print(json.dumps({k:summary[k] for k in ['case','waveform_coverage_pct','hr_coverage_pct','elapsed_s']},ensure_ascii=False),flush=True)
assert frozen()==fixed;save(OUT/'runs.json',results)

"""Frozen six-video local-patch experiment. References are read only by evaluation.

freeze -> extract CASE -> infer CASE -> evaluate VARIANT are explicit stages.
Nothing chooses a different configuration for an individual video.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,time
import numpy as np
import pandas as pd
from prepare_patch_signals_v27 import anchor_patches,prepare_signals,read_saved_signals,REGIONS
from evidence_hr import estimate_evidence,DEFAULT_CONFIG

HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别')
B=P/'results/data1_6_20260911'
ROOT=P/'results/data1_6_v27_20260912'
SPATIAL=('early_median','early_psd_cluster','late_median','late_psd_cluster')
READOUT=('fusion_nomotion_local','fusion_motion_local','fusion_nomotion_dp','fusion_motion_dp')
VARIANTS=SPATIAL+READOUT
INFERENCE_SOURCES=('analyze_motion_v2.py','analyze_rppg.py','legacy_motion.py',
    'motion_frontend.py','motion_fusion.py','stable_groups.py','waveform_hr.py',
    'baseline_frontend.py','pixel_tracking.py','anchored_reconstruction.py',
    'guarded_fusion.py','motion_evidence.py','evidence_hr.py',
    'stable_patch_frontend_v27.py','prepare_patch_signals_v27.py','patch_hr_v27.py','run_v27.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,Path):return str(x)
    return x

def save(path,x):
    Path(path).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def freeze():
    from stable_patch_frontend_v27 import PatchConfig
    from patch_hr_v27 import PatchHRConfig
    from dataclasses import asdict
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'protocol_before_inference.json'
    if path.exists():raise ValueError('Preserve frozen protocol')
    from evaluate_v27 import freeze_evaluation_inputs
    code={name:sha(HERE/name) for name in INFERENCE_SOURCES}
    source_protocol=json.loads((HERE/'development_protocol_v27.json').read_text())
    data=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=code,
        development_protocol=source_protocol,development_protocol_sha256=sha(HERE/'development_protocol_v27.json'),
        frontend_config=asdict(PatchConfig()),patch_hr_config=asdict(PatchHRConfig()),
        evidence_config=asdict(DEFAULT_CONFIG),variants=list(VARIANTS),reference_used_in_inference=False,
        evaluation_hashes={str(p):sha(p) for p in [HERE/'evaluate_v27.py',P/'batch_analysis_20260911/evaluate_batch.py']},
        evaluation_inputs=freeze_evaluation_inputs(),
        fixed_inputs_manifest_sha256=sha(B/'inputs_manifest.json'),
        protected_entries={str(p):sha(p) for p in [P/'run.sh',P/'run_motion_v25.sh']})
    save(path,data);print(json.dumps(dict(frozen=str(path),source_files=len(code),variants=VARIANTS)),flush=True)

def frozen():
    data=json.loads((ROOT/'protocol_before_inference.json').read_text())
    for name,digest in data['source_hashes'].items():
        assert sha(HERE/name)==digest,f'Frozen source changed: {name}'
    assert sha(HERE/'development_protocol_v27.json')==data['development_protocol_sha256']
    assert sha(B/'inputs_manifest.json')==data['fixed_inputs_manifest_sha256']
    for name,digest in data['protected_entries'].items():assert sha(name)==digest
    return data

def extract_case(case):
    from stable_patch_frontend_v27 import extract_patches
    import cv2
    cv2.setNumThreads(2)
    fixed=frozen();items=json.loads((B/'inputs_manifest.json').read_text())
    item=next(row for row in items if row['case']==case);video=Path(item['video']['path'])
    assert sha(video)==item['video']['sha256']
    out=ROOT/'frontend'/case;out.parent.mkdir(parents=True,exist_ok=True)
    metadata=extract_patches(video,out)
    frames=pd.read_csv(out/'frame_trace.csv')
    original=json.loads((B/case/'inference/frame_trace.json').read_text())
    assert len(frames)==original['n_frames']
    assert metadata['frames']==len(frames) and abs(metadata['fps']-original['fps'])<1e-9
    np.testing.assert_allclose(frames.time_s,np.arange(len(frames))/original['fps'],atol=1e-8,rtol=0)
    save(out/'batch_binding.json',dict(case=case,video=video,video_sha256=sha(video),frames=len(frames),fps=original['fps'],
        metadata_sha256=sha(out/'metadata.json'),
        fixed_protocol_sha256=sha(ROOT/'protocol_before_inference.json'),source_hashes=fixed['source_hashes']))
    validate_frontend(case,fixed)
    frozen();print(f'{case}: extraction validated, {len(frames)} original frames',flush=True)

def validate_frontend(case,fixed):
    from stable_patch_frontend_v27 import PatchConfig,source_hashes
    from dataclasses import asdict
    front=ROOT/'frontend'/case
    binding=json.loads((front/'batch_binding.json').read_text())
    assert binding['case']==case
    assert binding['fixed_protocol_sha256']==sha(ROOT/'protocol_before_inference.json')
    assert binding['source_hashes']==fixed['source_hashes']
    assert binding['metadata_sha256']==sha(front/'metadata.json')
    meta=json.loads((front/'metadata.json').read_text())
    item=next(row for row in json.loads((B/'inputs_manifest.json').read_text()) if row['case']==case)
    assert meta['video_sha256']==binding['video_sha256']==item['video']['sha256']
    assert Path(meta['video_path']).resolve()==Path(binding['video']).resolve()==Path(item['video']['path']).resolve()
    assert meta['video_bytes']==item['video']['bytes']
    assert meta['source_hashes']==source_hashes()
    assert all(fixed['source_hashes'][name]==digest for name,digest in meta['source_hashes'].items())
    assert meta['config']==asdict(PatchConfig())==fixed['frontend_config']
    assert meta['reference_used'] is False and meta['max_seconds'] is None
    expected={'patch_trace.csv','frame_trace.csv','patch_anchors.json','landmark_trace.npz'}
    assert set(meta['output_hashes'])==expected
    for name,digest in meta['output_hashes'].items():assert sha(front/name)==digest,f'Corrupt frontend cache: {name}'
    frames=pd.read_csv(front/'frame_trace.csv');patches=pd.read_csv(front/'patch_trace.csv')
    original=json.loads((B/case/'inference/frame_trace.json').read_text())
    assert len(frames)==meta['frames']==binding['frames']==original['n_frames']
    assert abs(meta['fps']-binding['fps'])<1e-9 and abs(meta['fps']-original['fps'])<1e-9
    from prepare_patch_signals_v27 import validate_inputs
    validate_inputs(patches,frames,binding['fps'])
    return binding,frames,patches

def patch_availability(patches,frames):
    valid=patches.pivot(index='frame',columns='patch_id',values='valid').astype(bool)
    parent=np.stack([valid[[name for name in valid if name.startswith(region+'_')]].any(axis=1) for region in REGIONS])
    detected=frames.face_detected.astype(bool) if 'face_detected' in frames else frames.source.isin(['mesh','redetected'])
    return dict(mean_patch_observed_pct=100*float(valid.to_numpy().mean()),
        at_least_two_regions_observed_pct=100*float((parent.sum(0)>=2).mean()),
        face_detection_pct=100*float(detected.mean()),
        per_patch_observed_pct={name:100*float(valid[name].mean()) for name in valid})

def readout(wave,frames,fps,motion=True,temporal=True):
    hr=estimate_evidence(wave.base.to_numpy(float),pd.DataFrame({'rgb_valid':wave.observed.astype(bool)}),
        wave.interpolated.to_numpy(bool),fps,motion_trace=frames if motion else None)
    if not temporal:hr['ridge_bpm']=hr.evidence_local_bpm
    hr['raw_spectral_peak_bpm']=hr.spectral_peak_bpm
    hr.loc[~hr.accepted,['spectral_peak_bpm','ridge_bpm']]=np.nan
    starts=np.arange(0,len(frames)-round(10*fps)+1,round(fps))
    hr['window_start_s']=starts/fps;hr['window_end_s']=(starts+round(10*fps))/fps
    hr['hr_source']='saved_fused_wave_'+('motion' if motion else 'nomotion')+'_'+('dp' if temporal else 'local')
    return hr

def infer_case(case):
    from patch_hr_v27 import infer_patch_hr
    from dataclasses import asdict
    from patch_hr_v27 import PatchHRConfig
    fixed=frozen();start=time.perf_counter();front=ROOT/'frontend'/case
    binding,frames,patches=validate_frontend(case,fixed);fps=binding['fps']
    signals=ROOT/'signals'/case;signals.mkdir(parents=True)
    availability=patch_availability(patches,frames)
    anchored=anchor_patches(patches,frames,fps);anchored.to_csv(signals/'anchored_patch_trace.csv',index=False)
    anchored=pd.read_csv(signals/'anchored_patch_trace.csv')
    source_files={str(p):sha(p) for p in [front/'patch_trace.csv',front/'frame_trace.csv',front/'patch_anchors.json',front/'metadata.json',front/'batch_binding.json',front/'landmark_trace.npz',signals/'anchored_patch_trace.csv']}
    for spatial in ('early','late'):
        _,wave,meta=prepare_signals(anchored,frames,fps,spatial)
        path=signals/f'{spatial}_patch_waveforms.csv';wave.to_csv(path,index=False)
        save(signals/f'{spatial}_signals.json',meta)
        spatial_source_files={**source_files,str(path):sha(path),str(signals/f'{spatial}_signals.json'):sha(signals/f'{spatial}_signals.json')}
        args=read_saved_signals(path,meta)
        for mode in ('median','psd_cluster'):
            name=f'{spatial}_{mode}';out=ROOT/name/case;out.mkdir(parents=True)
            fused,hr,diagnostics=infer_patch_hr(*args,frames,fps,mode=mode)
            fused.to_csv(out/'waveform.csv',index=False);hr.to_csv(out/'heart_rate.csv',index=False)
            if isinstance(diagnostics,pd.DataFrame):diagnostics.to_csv(out/'patch_diagnostics.csv',index=False)
            else:save(out/'patch_diagnostics.json',diagnostics)
            saved=pd.read_csv(out/'waveform.csv');secondary=readout(saved,frames,fps)
            secondary.to_csv(out/'fusion_heart_rate.csv',index=False)
            manifest=dict(case=case,variant=name,source_hashes=fixed['source_hashes'],primary_hr_source='patch_'+mode,
                primary_hr_interpretation='Hierarchical spatial aggregation from the saved independent patch BVPs. Not claimed to be the heart rate of the single displayed fused waveform.',
                patch_availability=availability,patch_waveforms_path=path,patch_waveforms_sha256=sha(path),
                patch_signal_metadata=meta,patch_hr_config=asdict(PatchHRConfig()),
                source_files=spatial_source_files,fps=fps,frames=len(frames),reference_used=False,
                motion_on_primary=False,temporal_correction_on_primary=False,
                secondary_HR='Unchanged V25 motion+DP readout of this saved measured broadband fused waveform',
                output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},status='complete')
            save(out/'manifest.json',manifest)
            print(json.dumps(dict(case=case,variant=name,accepted=int(hr.accepted.sum()),planned=len(hr),
                                  waveform_coverage_pct=100*float(fused.covered.mean()))),flush=True)
    common=ROOT/'late_psd_cluster'/case
    saved=pd.read_csv(common/'waveform.csv');base_manifest=json.loads((common/'manifest.json').read_text())
    for name in READOUT:
        motion='_nomotion_' not in name;temporal=name.endswith('_dp')
        out=ROOT/name/case;out.mkdir(parents=True)
        shutil.copyfile(common/'waveform.csv',out/'waveform.csv')
        shutil.copyfile(common/'fusion_heart_rate.csv',out/'fusion_heart_rate.csv')
        hr=readout(saved,frames,fps,motion,temporal);hr.to_csv(out/'heart_rate.csv',index=False)
        manifest={**base_manifest,'variant':name,'primary_hr_source':'fused_wave_dp' if temporal else 'fused_wave_local',
            'primary_hr_interpretation':'Existing V25 supported-candidate evidence from the identical saved late-PSD-cluster broadband waveform; switches only motion evidence and local/DP selection.',
            'motion_on_primary':motion,'temporal_correction_on_primary':temporal,
            'parent_spatial_manifest':str(common/'manifest.json'),'parent_spatial_manifest_sha256':sha(common/'manifest.json'),
            'evidence_config':asdict(DEFAULT_CONFIG),
            'output_hashes':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
        save(out/'manifest.json',manifest)
        assert sha(out/'waveform.csv')==sha(common/'waveform.csv')
        print(json.dumps(dict(case=case,variant=name,accepted=int(hr.accepted.sum()),planned=len(hr))),flush=True)
    frozen();save(signals/'processing_summary.json',dict(case=case,elapsed_s=time.perf_counter()-start,patch_availability=availability))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['freeze','extract','infer'])
    ap.add_argument('--case',choices=[f'data{i}' for i in range(1,7)])
    a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:
        if a.case is None:ap.error('--case is required')
        if a.stage=='extract':extract_case(a.case)
        else:infer_case(a.case)

if __name__=='__main__':main()

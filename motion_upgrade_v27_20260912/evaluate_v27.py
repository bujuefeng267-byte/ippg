"""Fixed V27 post-inference comparison; spatial HR comes from saved patch BVPs."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别');B=P/'results/data1_6_20260911';V25=P/'results/data1_6_v25_20260911/stage4_preserve_waveform';ROOT=P/'results/data1_6_v27_20260912'
SPATIAL=('early_median','early_psd_cluster','late_median','late_psd_cluster')
READOUT=('fusion_nomotion_local','fusion_motion_local','fusion_nomotion_dp','fusion_motion_dp')
spec=importlib.util.spec_from_file_location('reference_core',P/'batch_analysis_20260911/evaluate_batch.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(core.clean(x),ensure_ascii=False,indent=2,allow_nan=False))

def freeze_evaluation_inputs():
    """Bind existing reference/baseline files before inference; hash, never fit."""
    manifest=json.loads((B/'inputs_manifest.json').read_text())
    assert [row['case'] for row in manifest]==[f'data{i}' for i in range(1,7)]
    paths=[B/'inputs_manifest.json',B/'evaluation_protocol.json',V25/'stage_evaluation.json']
    for item in manifest:
        case=item['case'];folder=B/case
        alignment=json.loads((folder/'evaluation/alignment.json').read_text())
        reference=Path(alignment['reference_path'])
        matches=[row for row in item['files'] if Path(row['path'])==reference]
        assert len(matches)==1 and matches[0]['sha256']==alignment['reference_sha256']
        assert sha(reference)==alignment['reference_sha256']
        paths += [folder/'inference/summary.json',folder/'inference/frame_trace.json',
            folder/'evaluation/paired_windows.csv',folder/'evaluation/alignment.json',
            folder/'evaluation/evaluation_provenance.json',reference,
            V25/case/'fusion_heart_rate.csv',V25/case/'fusion_waveform.csv']
    return {str(path.resolve()):sha(path) for path in paths}

def validate_protocol():
    path=ROOT/'protocol_before_inference.json';fixed=json.loads(path.read_text())
    assert fixed['reference_used_in_inference'] is False
    assert fixed['variants']==list(SPATIAL+READOUT)
    assert fixed['development_protocol']['fixed_windows']==dict(window_s=10,step_s=1,min_bpm=42,max_bpm=210,max_gap_s=.1)
    assert sha(B/'inputs_manifest.json')==fixed['fixed_inputs_manifest_sha256']
    assert sha(HERE/'development_protocol_v27.json')==fixed['development_protocol_sha256']
    assert fixed['source_hashes'] and fixed['evaluation_hashes'] and fixed['evaluation_inputs']
    for name,digest in fixed['source_hashes'].items():assert sha(HERE/name)==digest,f'Inference source changed: {name}'
    for mapping in ('evaluation_hashes','evaluation_inputs','protected_entries'):
        for name,digest in fixed[mapping].items():assert sha(name)==digest,f'Frozen {mapping} changed: {name}'
    expected_evaluators={str(Path(__file__).resolve()),str((P/'batch_analysis_20260911/evaluate_batch.py').resolve())}
    assert expected_evaluators.issubset(fixed['evaluation_hashes'])
    assert freeze_evaluation_inputs()==fixed['evaluation_inputs'],'Evaluation input list or content changed'
    return fixed

def _verify_hash_map(mapping,required):
    assert isinstance(mapping,dict) and set(required).issubset(mapping),'Missing required bound files'
    for path,digest in mapping.items():assert sha(path)==digest,f'Bound file changed: {path}'

def validate_run_binding(case,variant,fixed):
    """Reject mixed-video, incomplete, changed-code or changed-output results."""
    d=ROOT/variant/case;front=ROOT/'frontend'/case;signals=ROOT/'signals'/case
    manifest=json.loads((d/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['reference_used'] is False
    assert manifest['case']==case and manifest['variant']==variant
    assert manifest['source_hashes']==fixed['source_hashes']
    assert manifest['patch_hr_config']==fixed['patch_hr_config']
    metadata=json.loads((front/'metadata.json').read_text())
    binding=json.loads((front/'batch_binding.json').read_text())
    original=json.loads((B/case/'inference/frame_trace.json').read_text())
    inputs=json.loads((B/'inputs_manifest.json').read_text())
    item=next(row for row in inputs if row['case']==case)
    expected_frames,expected_fps=original['n_frames'],original['fps']
    for record in (manifest,metadata,binding):
        assert record['frames']==expected_frames and record['fps']==expected_fps,'Frame/FPS identity changed'
    assert binding['case']==case and binding['fixed_protocol_sha256']==sha(ROOT/'protocol_before_inference.json')
    assert binding['source_hashes']==fixed['source_hashes']
    assert metadata['video_path']==str(Path(item['video']['path']).resolve())
    assert str(Path(binding['video']).resolve())==metadata['video_path']
    assert metadata['video_sha256']==binding['video_sha256']==item['video']['sha256']
    assert metadata['video_bytes']==item['video']['bytes'] and metadata['max_seconds'] is None
    assert metadata['reference_used'] is False and metadata['temporal_RGB_filling'] is False
    assert metadata['RGB_aggregation_across_patches'] is False and metadata['anchors_reinitialized_after_gap'] is False
    assert metadata['config']==fixed['frontend_config']
    for name,digest in metadata['source_hashes'].items():assert fixed['source_hashes'][name]==digest
    expected_front={'patch_trace.csv','frame_trace.csv','patch_anchors.json','landmark_trace.npz'}
    assert set(metadata['output_hashes'])==expected_front
    for name,digest in metadata['output_hashes'].items():assert sha(front/name)==digest
    if 'metadata_sha256' in binding:assert binding['metadata_sha256']==sha(front/'metadata.json')
    spatial=variant.split('_',1)[0] if variant in SPATIAL else 'late'
    signal_metadata_path=signals/f'{spatial}_signals.json'
    expected_signal_path=signals/f'{spatial}_patch_waveforms.csv'
    assert Path(manifest['patch_waveforms_path']).resolve()==expected_signal_path.resolve()
    assert manifest['patch_waveforms_sha256']==sha(expected_signal_path)
    assert manifest['patch_signal_metadata']==json.loads(signal_metadata_path.read_text())
    assert manifest['patch_signal_metadata']['spatial_mode']==spatial
    required={str((front/name).resolve()) for name in expected_front|{'metadata.json','batch_binding.json'}}
    required|={str((signals/'anchored_patch_trace.csv').resolve()),str(signal_metadata_path.resolve())}
    _verify_hash_map(manifest['source_files'],required)
    required_outputs={'waveform.csv','heart_rate.csv','fusion_heart_rate.csv'}
    if variant in SPATIAL:required_outputs.add('patch_diagnostics.csv')
    assert required_outputs.issubset(manifest['output_hashes'])
    for name,digest in manifest['output_hashes'].items():
        assert Path(name).name==name,'Output checksum keys must be local file names'
        assert sha(d/name)==digest,f'Inference output changed: {case}/{variant}/{name}'
    if variant in SPATIAL:
        assert manifest['primary_hr_source']=='patch_'+variant.split('_',1)[1]
        assert manifest['motion_on_primary'] is False and manifest['temporal_correction_on_primary'] is False
    else:
        temporal=variant.endswith('_dp');motion='_nomotion_' not in variant
        assert manifest['primary_hr_source']==('fused_wave_dp' if temporal else 'fused_wave_local')
        assert manifest['motion_on_primary']==motion and manifest['temporal_correction_on_primary']==temporal
        assert manifest['evidence_config']==fixed['evidence_config']
        parent=ROOT/'late_psd_cluster'/case
        assert Path(manifest['parent_spatial_manifest']).resolve()==(parent/'manifest.json').resolve()
        assert manifest['parent_spatial_manifest_sha256']==sha(parent/'manifest.json')
        assert sha(d/'waveform.csv')==sha(parent/'waveform.csv')
        assert sha(d/'fusion_heart_rate.csv')==sha(parent/'fusion_heart_rate.csv')
    return manifest

def assert_fused_support(finite,accepted,starts,window,label):
    """Only a fused-wave HR readout has this direct waveform-source contract."""
    finite=np.asarray(finite,bool);accepted=np.asarray(accepted,bool)
    assert len(accepted)==len(starts),f'{label}: incomplete HR clock'
    assert all(finite[a:a+window].all() and len(finite[a:a+window])==window
               for a in starts[accepted]),f'{label}: accepted HR lacks a full finite source waveform'
def aggregate(rows):
    n=sum(r['Nvalid'] for r in rows);nr=sum(r['Nref'] for r in rows);nc=sum(r['common_windows'] for r in rows)
    return dict(Nref=nr,Nvalid=n,Nwithin5=sum(r['Nwithin5'] for r in rows),
        MAE_bpm=sum(r['Nvalid']*r['MAE_bpm'] for r in rows if r['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(r['Nvalid']*r['RMSE_bpm']**2 for r in rows if r['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*sum(r['Nwithin5'] for r in rows)/n if n else np.nan,
        R5_all_reference_pct=100*sum(r['Nwithin5'] for r in rows)/nr,
        HR_coverage_pct=100*sum(r['Noutput'] for r in rows)/sum(r['Nplanned'] for r in rows),
        common_windows=nc,common_MAE_bpm=sum(r['common_windows']*r['common_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan,
        common_V25_MAE_bpm=sum(r['common_windows']*r['common_V25_MAE_bpm'] for r in rows if r['common_windows'])/nc if nc else np.nan)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True,choices=SPATIAL+READOUT);args=ap.parse_args();folder=ROOT/args.variant
    fixed=validate_protocol()
    assert not (folder/'evaluation_summary.json').exists(),'Preserve completed evaluations'
    rows=[];fusion_rows=[];sensitivity=[];hashes={str(Path(__file__)):sha(__file__),str(P/'batch_analysis_20260911/evaluate_batch.py'):sha(P/'batch_analysis_20260911/evaluate_batch.py')}
    for i in range(1,7):
        c=f'data{i}';d=folder/c;manifest=validate_run_binding(c,args.variant,fixed);out=d/'evaluation';out.mkdir()
        wave=pd.read_csv(d/'waveform.csv');hr=pd.read_csv(d/'heart_rate.csv');old=pd.read_csv(V25/c/'fusion_heart_rate.csv')
        ow=pd.read_csv(V25/c/'fusion_waveform.csv');ref=pd.read_csv(B/c/'evaluation/paired_windows.csv');metadata=json.loads((B/c/'inference/summary.json').read_text())
        fps=metadata['fps'];frames=metadata['frames'];assert len(wave)==frames
        np.testing.assert_allclose(wave.time_s,ow.time_s,atol=1e-8,rtol=0);np.testing.assert_allclose(hr.time_s,ref.time_s,atol=1e-8,rtol=0)
        np.testing.assert_allclose(old.time_s,ref.time_s,atol=1e-8,rtol=0)
        ok=core.bools(hr.accepted);ook=core.bools(old.accepted);y=hr.ridge_bpm.to_numpy(float);oy=old.ridge_bpm.to_numpy(float);r=ref.reference_bpm.to_numpy(float)
        assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()
        finite=np.isfinite(wave.base);starts=np.arange(0,frames-round(10*fps)+1,round(fps));assert len(starts)==len(hr)
        if args.variant in READOUT:assert_fused_support(finite,ok,starts,round(10*fps),'primary fused-wave HR')
        if 'covered' in wave:np.testing.assert_array_equal(finite,core.bools(wave.covered))
        assert not (core.bools(wave.observed)&~finite.to_numpy()).any()
        assert not (core.bools(wave.interpolated)&~finite.to_numpy()).any()
        expected_source=('spatial_aggregation_of_saved_patch_BVP' if args.variant in SPATIAL else
            'saved_fused_wave_'+('nomotion' if '_nomotion_' in args.variant else 'motion')+'_'+('dp' if args.variant.endswith('_dp') else 'local'))
        assert hr.hr_source.eq(expected_source).all(),'Primary HR provenance does not match variant'
        common=ok&ook&np.isfinite(r);new=ok&~ook&np.isfinite(r);lost=~ok&ook&np.isfinite(r)
        m=core.score(y,ok,r);prior=core.score(oy,ook,r)
        fh=pd.read_csv(d/'fusion_heart_rate.csv');np.testing.assert_allclose(fh.time_s,hr.time_s,atol=1e-8,rtol=0)
        fy=fh.ridge_bpm.to_numpy(float);fok=core.bools(fh.accepted)
        assert np.isnan(fy[~fok]).all() and np.isfinite(fy[fok]).all()
        assert fh.hr_source.eq('saved_fused_wave_motion_dp').all()
        assert_fused_support(finite,fok,starts,round(10*fps),'secondary fused-wave HR')
        fm=core.score(fy,fok,r);fc=fok&ook&np.isfinite(r)
        fusion_rows.append(dict(case=c,**fm,common_windows=int(fc.sum()),
            common_MAE_bpm=float(np.abs(fy[fc]-r[fc]).mean()) if fc.any() else np.nan,
            common_V25_MAE_bpm=float(np.abs(oy[fc]-r[fc]).mean()) if fc.any() else np.nan))
        row=dict(variant=args.variant,case=c,**m,V25_MAE_bpm=prior['MAE_bpm'],delta_MAE_bpm=m['MAE_bpm']-prior['MAE_bpm'],
            delta_P5_pp=m['P5_valid_pct']-prior['P5_valid_pct'],delta_R5_pp=m['R5_all_reference_pct']-prior['R5_all_reference_pct'],
            delta_hr_coverage_pp=m['hr_output_coverage_pct']-prior['hr_output_coverage_pct'],
            waveform_coverage_pct=100*finite.mean(),delta_waveform_coverage_pp=100*(finite.mean()-np.isfinite(ow.base).mean()),
            common_windows=int(common.sum()),new_windows=int(new.sum()),lost_windows=int(lost.sum()),
            common_MAE_bpm=np.abs(y[common]-r[common]).mean() if common.any() else np.nan,
            common_V25_MAE_bpm=np.abs(oy[common]-r[common]).mean() if common.any() else np.nan,
            new_MAE_bpm=np.abs(y[new]-r[new]).mean() if new.any() else np.nan,
            lost_V25_MAE_bpm=np.abs(oy[lost]-r[lost]).mean() if lost.any() else np.nan,
            mean_HR_bpm=float(np.mean(y[ok])) if ok.any() else np.nan,
            target_P5_95_met=bool(m['P5_valid_pct']>=95),status_counts=hr.status.value_counts().to_dict(),
            primary_hr_source=manifest['primary_hr_source'],primary_hr_interpretation=manifest['primary_hr_interpretation'],
            fusion_readout_metrics=fm,
            mean_patch_observed_pct=manifest['patch_availability']['mean_patch_observed_pct'],
            at_least_two_regions_observed_pct=manifest['patch_availability']['at_least_two_regions_observed_pct'],
            face_detection_pct=manifest['patch_availability']['face_detection_pct'])
        paired=ref[['window_index','time_s','window_start_s','window_end_s','reference_bpm','reference_valid']].copy()
        paired['estimated_bpm']=y;paired['accepted']=ok;paired['V25_bpm']=oy;paired['V25_accepted']=ook
        paired['error_bpm']=np.where(ok,y-r,np.nan);paired['abs_error_bpm']=np.abs(paired.error_bpm)
        paired['status']=hr.status;paired['common']=common;paired['new']=new;paired['lost']=lost
        paired['fusion_readout_bpm']=fy;paired['fusion_readout_accepted']=fok
        paired.to_csv(out/'paired_windows.csv',index=False);write(out/'metrics.json',row);rows.append(row)
        alignment=json.loads((B/c/'evaluation/alignment.json').read_text());raw=pd.read_csv(alignment['reference_path'])
        sr=[]
        for shift in [-5,-2,-1,0,1,2,5]:
            adjusted=core.reference_windows(raw,alignment['video_start_utc_ns'],ref.window_start_s,ref.window_end_s,shift)
            if shift==0:np.testing.assert_allclose(r,adjusted.reference_bpm,atol=1e-9,rtol=0,equal_nan=True)
            sr.append(dict(variant=args.variant,case=c,shift_s=shift,**core.score(y,ok,adjusted.reference_bpm.to_numpy(float))))
        pd.DataFrame(sr).to_csv(out/'alignment_sensitivity.csv',index=False);sensitivity+=sr
        for path in [d/'waveform.csv',d/'heart_rate.csv',d/'fusion_heart_rate.csv',d/'manifest.json',B/c/'evaluation/paired_windows.csv',B/c/'evaluation/alignment.json',Path(alignment['reference_path'])]:hashes[str(path)]=sha(path)
    agg=aggregate(rows);prior=json.loads((V25/'stage_evaluation.json').read_text())['pooled']
    checks=dict(pooled_MAE=agg['MAE_bpm']<prior['MAE_bpm'],pooled_R5_gain=agg['R5_all_reference_pct']>=prior['R5_all_reference_pct']+5,
        each_MAE=all(r['delta_MAE_bpm']<=3 for r in rows),each_P5=all(r['delta_P5_pp']>=-3 for r in rows),each_R5=all(r['delta_R5_pp']>=-3 for r in rows),
        each_HR_coverage=all(r['delta_hr_coverage_pp']>=-3 for r in rows),each_waveform_coverage=all(r['delta_waveform_coverage_pp']>=-3 for r in rows),
        common_MAE=agg['common_MAE_bpm']<agg['common_V25_MAE_bpm'])
    validate_protocol()
    for case in [f'data{i}' for i in range(1,7)]:validate_run_binding(case,args.variant,fixed)
    hashes.update(fixed['evaluation_inputs']);hashes.update(fixed['evaluation_hashes'])
    hashes[str(ROOT/'protocol_before_inference.json')]=sha(ROOT/'protocol_before_inference.json')
    report=dict(variant=args.variant,pooled=agg,cases=rows,fusion_readout_pooled=aggregate(fusion_rows),fusion_readout_cases=fusion_rows,
        promotion_checks=checks,promotion_pass=all(checks.values()),
        near_all_within_5bpm_target_met=all(r['target_P5_95_met'] for r in rows),scope='Previously seen development clips, not held-out validation; estimated original alignment. Spatial primary HR aggregates saved patch BVPs; the separate fused-wave readout is not substituted for it.',input_hashes=hashes)
    write(folder/'evaluation_summary.json',report);pd.DataFrame(rows).drop(columns=['status_counts','fusion_readout_metrics']).to_csv(folder/'metrics.csv',index=False)
    print(json.dumps(core.clean(dict(variant=args.variant,pooled=agg,promotion_checks=checks,promotion_pass=all(checks.values()),
        cases=[{k:r[k] for k in ['case','MAE_bpm','P5_valid_pct','R5_all_reference_pct','hr_output_coverage_pct','waveform_coverage_pct']} for r in rows])),ensure_ascii=False),flush=True)
if __name__=='__main__':main()


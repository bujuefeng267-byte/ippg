"""Replay saved V27 broadband waveforms through byte-identical V25 HR code.

This audit reads no reference or evaluation scores. It verifies source identity,
all 48 secondary CSVs and the 24 fused-wave primary CSVs. The four spatial
primary HR outputs intentionally have a different source and are not replaced.
The QA wrapper is supplemental post-implementation evidence, not inference.
"""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import argparse
import hashlib
import json
import importlib
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT_ROOT=Path('/home/fengbujue/项目/rppg识别/results/data1_6_v27_20260912')
CASES=tuple(f'data{i}' for i in range(1,7))
SPATIAL=('early_median','early_psd_cluster','late_median','late_psd_cluster')
READOUT=('fusion_nomotion_local','fusion_motion_local','fusion_nomotion_dp','fusion_motion_dp')
VARIANTS=SPATIAL+READOUT
SOURCES=('evidence_hr.py','motion_evidence.py','legacy_motion.py','analyze_rppg.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bools(values):
    a=np.asarray(values)
    assert not pd.isna(a).any() and np.isin(a,[True,False,0,1]).all()
    return a.astype(bool)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value


def compare_json(a,b,path,stats):
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for key in a:compare_json(a[key],b[key],path+'/'+str(key),stats)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare_json(x,y,path+'/'+str(i),stats)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert isinstance(b,(int,float)) and not isinstance(b,bool),path
        np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-10,equal_nan=True,err_msg=path)
        if np.isfinite(a) and np.isfinite(b):stats['max_numeric_difference']=max(stats['max_numeric_difference'],abs(float(a)-float(b)))
        stats['numeric_comparisons']+=1
    else:assert a==b,path


def compare_tables(actual,expected,label):
    assert list(actual)==list(expected),label+': column names/order changed'
    assert len(actual)==len(expected),label+': row count changed'
    stats=dict(rows=len(actual),columns=len(actual.columns),numeric_comparisons=0,max_numeric_difference=0.)
    for col in actual:
        a=actual[col];b=expected[col]
        np.testing.assert_array_equal(pd.isna(a),pd.isna(b),err_msg=label+'/'+col+'/missing')
        if col.endswith('_json'):
            for i,(x,y) in enumerate(zip(a,b)):compare_json(json.loads(x),json.loads(y),label+'/'+col+'/'+str(i),stats)
        elif pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            np.testing.assert_allclose(a.to_numpy(float),b.to_numpy(float),rtol=1e-10,atol=1e-9,equal_nan=True,err_msg=label+'/'+col)
            mask=np.isfinite(a.to_numpy(float))&np.isfinite(b.to_numpy(float))
            if mask.any():stats['max_numeric_difference']=max(stats['max_numeric_difference'],float(np.max(abs(a.to_numpy(float)[mask]-b.to_numpy(float)[mask]))))
            stats['numeric_comparisons']+=len(a)
        elif pd.api.types.is_bool_dtype(b):np.testing.assert_array_equal(bools(a),bools(b),err_msg=label+'/'+col)
        else:
            np.testing.assert_array_equal(a.fillna('').to_numpy(str),b.fillna('').to_numpy(str),err_msg=label+'/'+col)
    ok=bools(actual.accepted);y=actual.ridge_bpm.to_numpy(float)
    assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all(),label+'/output contract'
    stats.update(accepted=int(ok.sum()),label=label,passed=True)
    return stats


def reconstruct_readout(result,frames,fps,motion,temporal):
    result=result.copy(deep=True)
    if not temporal:result['ridge_bpm']=result.evidence_local_bpm
    result['raw_spectral_peak_bpm']=result.spectral_peak_bpm
    result.loc[~result.accepted,['spectral_peak_bpm','ridge_bpm']]=np.nan
    starts=np.arange(0,frames-round(10*fps)+1,round(fps))
    assert len(starts)==len(result)
    result['window_start_s']=starts/fps;result['window_end_s']=(starts+round(10*fps))/fps
    result['hr_source']='saved_fused_wave_'+('motion' if motion else 'nomotion')+'_'+('dp' if temporal else 'local')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--v25-source',type=Path,default=HERE.parent/'rppg_motion_v25')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--confirmed-complete',action='store_true')
    a=ap.parse_args()
    if not a.confirmed_complete:ap.error('Await complete six-video inference before reading results')
    root=a.root.resolve();output=(a.output or root/'qa_saved_waveform_readout.json').resolve()
    if output.exists():raise FileExistsError('Preserve prior audit; use a new output filename')
    required=[root/'protocol_before_inference.json']
    for c in CASES:
        required += [root/'frontend'/c/'frame_trace.csv',root/'frontend'/c/'metadata.json']
        for v in VARIANTS:required += [root/v/c/name for name in ('manifest.json','waveform.csv','heart_rate.csv','fusion_heart_rate.csv')]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise FileNotFoundError('Incomplete inference: '+json.dumps(missing,ensure_ascii=False))
    hashes={str(Path(__file__).resolve()):sha(__file__)}
    def readj(path):
        hashes[str(path.resolve())]=sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def readc(path):
        hashes[str(path.resolve())]=sha(path)
        return pd.read_csv(path)
    fixed=readj(root/'protocol_before_inference.json')
    assert fixed['reference_used_in_inference'] is False and fixed['variants']==list(VARIANTS)
    source_receipt={}
    for name in SOURCES:
        new=HERE/name;old=a.v25_source/name;nh=sha(new);oh=sha(old)
        assert nh==oh==fixed['source_hashes'][name],name+': not identical to frozen V25 readout source'
        hashes[str(new.resolve())]=nh;hashes[str(old.resolve())]=oh
        source_receipt[name]=dict(v27_sha256=nh,v25_sha256=oh,byte_identical=True)
    # The production estimator is re-executed, while wrapper selection,
    # source/flag contracts and table comparison are independently implemented.
    sys.path.insert(0,str(HERE))
    estimator=importlib.import_module('evidence_hr')
    assert asdict(estimator.DEFAULT_CONFIG)==fixed['evidence_config']
    for name in ('evidence_hr','legacy_motion','analyze_rppg'):
        assert Path(sys.modules[name].__file__).resolve()==(HERE/(name+'.py')).resolve()
    records=[];cache={};errors=[]
    for case in CASES:
        full_trace=readc(root/'frontend'/case/'frame_trace.csv')
        meta=readj(root/'frontend'/case/'metadata.json');fps=float(meta['fps']);n=int(meta['frames'])
        assert len(full_trace)==n
        np.testing.assert_allclose(full_trace.time_s,np.arange(n)/fps,rtol=0,atol=1e-8)
        # These are the ONLY columns consumed by motion_evidence. Explicitly
        # excluding everything else demonstrates no label/reference is passed.
        motion_columns=['motion_x','motion_y','face_y0','face_y1']
        motion_trace=full_trace.reindex(columns=motion_columns).copy()
        trace_hash=hashes[str((root/'frontend'/case/'frame_trace.csv').resolve())]
        for variant in VARIANTS:
            folder=root/variant/case;manifest=readj(folder/'manifest.json')
            assert manifest['case']==case and manifest['variant']==variant and manifest['status']=='complete'
            assert manifest['reference_used'] is False and manifest['source_hashes']==fixed['source_hashes']
            assert manifest['fps']==fps and manifest['frames']==n
            wave=readc(folder/'waveform.csv');secondary=readc(folder/'fusion_heart_rate.csv')
            assert len(wave)==n
            np.testing.assert_allclose(wave.time_s,full_trace.time_s,rtol=0,atol=1e-8)
            np.testing.assert_array_equal(bools(wave.covered),np.isfinite(wave.base))
            wave_hash=hashes[str((folder/'waveform.csv').resolve())]
            expected_hash=manifest['output_hashes']['waveform.csv']
            assert wave_hash==expected_hash
            def estimate(motion):
                key=(wave_hash,trace_hash,fps,motion)
                if key not in cache:
                    cache[key]=estimator.estimate_evidence(wave.base.to_numpy(float),
                        pd.DataFrame({'rgb_valid':bools(wave.observed)}),bools(wave.interpolated),fps,
                        10.,1.,42.,210.,motion_trace=motion_trace if motion else None)
                    assert cache[key].attrs['reference_used'] is False
                return cache[key]
            try:
                expected=reconstruct_readout(estimate(True),n,fps,True,True)
                records.append(compare_tables(secondary,expected,variant+'/'+case+'/secondary'))
                if variant in READOUT:
                    primary=readc(folder/'heart_rate.csv')
                    motion='_nomotion_' not in variant;temporal=variant.endswith('_dp')
                    expected=reconstruct_readout(estimate(motion),n,fps,motion,temporal)
                    records.append(compare_tables(primary,expected,variant+'/'+case+'/primary'))
                    assert wave_hash==hashes[str((root/'late_psd_cluster'/case/'waveform.csv').resolve())]
            except (AssertionError,ValueError,KeyError) as exc:
                errors.append(dict(case=case,variant=variant,error=str(exc)))
        print(json.dumps(dict(case=case,checked_tables=len(records),errors=len(errors)),ensure_ascii=False),flush=True)
    if 'motion_evidence' in sys.modules:assert Path(sys.modules['motion_evidence'].__file__).resolve()==(HERE/'motion_evidence.py').resolve()
    for path,digest in hashes.items():assert sha(path)==digest,'Input changed during replay: '+path
    passed=not errors and len(records)==72
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=passed,
        audit_kind='supplemental_saved_waveform_replay_not_new_inference_or_reference_validation',
        checked_secondary_tables=sum(r['label'].endswith('/secondary') for r in records),
        checked_fused_primary_tables=sum(r['label'].endswith('/primary') for r in records),
        expected_secondary_tables=48,expected_fused_primary_tables=24,
        distinct_estimator_calls=len(cache),source_identity=source_receipt,
        numeric_absolute_tolerance=1e-9,numeric_relative_tolerance=1e-10,
        discrete_flags_strings_and_NaN_masks_exact=not errors,
        motion_input_columns=['motion_x','motion_y','face_y0','face_y1'],
        reference_files_read=False,reference_arrays_passed=False,spatial_primary_intentionally_separate=True,
        records=records,errors=errors,input_hashes=hashes,
        limitation='Replays the frozen V25 estimator; verifies saved-wave readout consistency, not independent correctness of that estimator or physiological accuracy.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(clean(receipt),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(output=str(output),passed=passed,checked_tables=len(records),errors=errors),ensure_ascii=False),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()

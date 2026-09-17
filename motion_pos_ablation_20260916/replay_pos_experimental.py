"""Replay a tested POS branch from an existing video cache, without references.

The project default is unchanged. Raw3 and patch6 are fixed exploratory options;
no case identity, target HR, Polar data or per-video parameter choice is used.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
import numpy as np
import pandas as pd
import frontend_ablation as front

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--trace',type=Path,required=True,help='Original V28 frame_trace.csv with matching .json')
    ap.add_argument('--mode',choices=['raw3','patch6'],required=True)
    ap.add_argument('--regional-trace',type=Path,help='Optional previously extracted six-region CSV for patch6')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    p=args.project.resolve();runtime=front.load_runtime(p)
    trace,fps,meta,inputs=front.load_cache(args.trace)
    sources=dict(runtime.source_hashes);sources[str(Path(__file__).resolve())]=front.sha(__file__)
    sources.update({str(f):front.sha(f) for f in (p/'motion_upgrade_v28_20260912').glob('*.py')})
    expected_arm='front_baseline_region_median' if args.mode=='raw3' else 'pos6_select5'
    out.mkdir(parents=True)
    started={'created_utc':datetime.now(timezone.utc).isoformat(),'mode':args.mode,'equivalent_experiment_arm':expected_arm,'source_hashes':sources,'input_hashes':inputs,'reference_used':False,'default_replacement':False,'experimental':True,'window_s':10,'step_s':1,'min_bpm':42,'max_bpm':210}
    if args.mode=='patch6':
        sys.path.insert(0,str(p/'motion_dis_hybrid_v2_20260915'))
        import regional_motion
        for name in ['regional_motion.py','dis_core.py']:
            f=p/'motion_dis_hybrid_v2_20260915'/name;sources[str(f)]=front.sha(f)
        if args.regional_trace:
            region=args.regional_trace.resolve();rmeta=json.loads(region.with_suffix('.json').read_text())
            if front.sha(region)!=rmeta['trace_sha256']:raise ValueError('Regional cache checksum mismatch')
            if rmeta.get('old_cache_path')!=str(args.trace.resolve()):raise ValueError('Regional cache belongs to a different geometry source')
            if rmeta['geometry_cache']['trace_sha256']!=inputs[str(args.trace.resolve())]:raise ValueError('Regional geometry checksum mismatch')
            inputs[str(region)]=front.sha(region);inputs[str(region.with_suffix('.json'))]=front.sha(region.with_suffix('.json'))
    front.save_json(out/'inference_freeze.json',started)
    if args.mode=='raw3':
        products,region_sources,records,masks,native=front.extract_products(trace,fps,runtime)
        wave=products[expected_arm]
        front.save_csv(out/'region_sources.csv',region_sources[expected_arm])
        front.save_csv(out/'extraction_windows.csv',records[expected_arm])
        front.save_csv(out/'input_masks.csv',masks)
    else:
        if args.regional_trace:
            regional=pd.read_csv(region)
        else:
            regional,rmeta=regional_motion.extract_regional_trace(args.trace,out/'regional',threads=1)
        if len(regional)!=len(trace) or abs(float(rmeta['fps'])-fps)>1e-8:raise ValueError('Regional native clock differs')
        products,records=runtime.core.extract_all(regional,fps)
        wave=products['pos'];native=None
        front.save_csv(out/'extraction_windows.csv',records[records.method.eq('pos')])
    front.save_csv(out/'waveform.csv',wave)
    saved=pd.read_csv(out/'waveform.csv')
    hr=front.estimate_waveform(saved,fps,runtime)
    hr.insert(0,'window_index',np.arange(len(hr)))
    hr['hr_source']='saved_waveform_fixed_POS_experimental_no_motion_score'
    front.save_csv(out/'heart_rate.csv',hr)
    front._verify(inputs);front._verify(sources)
    front.save_json(out/'manifest.json',{**started,'status':'complete','fps':fps,'frames':len(trace),'native_input_validity':native,'Nplanned':len(hr),'Noutput':int(hr.accepted.sum()),'hr_output_coverage_pct':100*float(hr.accepted.mean()),'waveform_finite_pct':100*float(saved.covered.mean()),'physiological_accuracy_measured':False,'waveform_morphology_validated':False,'outputs':{f.name:front.sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps({'out':str(out),'mode':args.mode,'status':'complete','reference_used':False,'Noutput':int(hr.accepted.sum())},ensure_ascii=False))
if __name__=='__main__':main()

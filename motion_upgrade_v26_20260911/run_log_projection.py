"""Fixed log-color ablations, preserving V25 frontend and physical ROI gates."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from analyze_motion_v2 import validate_trace,source_hashes as v25_hashes
from motion_fusion import FusionConfig,fuse_windows
from guarded_fusion import GuardConfig,guarded_fuse
from waveform_hr import estimate_fused_waveform
from signal_candidates_v26 import make_candidate_signals,SignalCandidateConfig

HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别');BASE=P/'results/data1_6_20260911';ROOT=P/'results/data1_6_v26_20260911'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def frozen():return {**v25_hashes(),**{n:sha(HERE/n) for n in ['run_log_projection.py','signal_candidates_v26.py','development_protocol_v26.json','evaluate_v26.py']}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--modes',nargs='+',choices=['baseline','guarded'],default=['baseline','guarded']);args=ap.parse_args()
    config=FusionConfig(methods=('logpos','logchrom'),motion_evidence_mode='legacy');guard=GuardConfig()
    sources=frozen();inputs=json.loads((BASE/'inputs_manifest.json').read_text());ROOT.mkdir(exist_ok=True)
    for mode in args.modes:
        folder=ROOT/('log_projection_'+mode);folder.mkdir()
        plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=sources,
            projection_config=asdict(SignalCandidateConfig()),fusion_config=asdict(config),guard_config=asdict(guard),
            variant=mode,reference_used=False,hr_mode='evidence',window_s=10,step_s=1,min_bpm=42,max_bpm=210,max_gap_s=.10,
            hypothesis='Remove multiplicative common illumination before chromatic synthesis; baseline-only is a fixed tracking-reconstruction ablation, not per-video routing',
            clips=[dict(case=d['case'],trace_sha256=sha(BASE/d['case']/'inference/frame_trace.csv'),
                metadata_sha256=sha(BASE/d['case']/'inference/frame_trace.json')) for d in inputs])
        write(folder/'protocol_before_run.json',plan)
        results=[]
        for item in inputs:
            c=item['case'];start=time.perf_counter();out=folder/c;out.mkdir()
            cache=BASE/c/'inference/frame_trace.csv';meta=json.loads(cache.with_suffix('.json').read_text())
            assert sha(cache)==meta['trace_sha256'];identity=meta['identity'];stat=Path(item['video']['path']).stat()
            assert identity['video']==item['video']['path'] and identity['size']==stat.st_size and identity['mtime_ns']==stat.st_mtime_ns
            for name,v in identity['frontend_hashes'].items():assert sha(HERE/name)==v
            trace=pd.read_csv(cache);fps=meta['fps'];validate_trace(trace,fps)
            bs,bw,bd=make_candidate_signals(trace,fps,.10,42,210,rgb_source='baseline')
            ts,tw,td=make_candidate_signals(trace,fps,.10,42,210,rgb_source='active')
            for roi in config.roi_names:
                for tag in ['observed','interpolated']:np.testing.assert_array_equal(bw[f'{roi}_{tag}'],tw[f'{roi}_{tag}'])
            effective=trace.copy()
            for roi in config.roi_names:effective[f'{roi}_valid']=bw[f'{roi}_observed'].to_numpy(bool)
            if mode=='guarded':
                proposals,wave,diag,routing,branches=guarded_fuse(bs,ts,effective,fps,10,1,42,210,config=config,guard=guard)
                routing.to_csv(out/'branch_routing.csv',index=False)
            else:proposals,wave,diag=fuse_windows(bs,effective,fps,10,1,42,210,config=config)
            proposals['waveform_generated']=proposals.accepted & (proposals.waveform_roi_count>=config.min_rois)
            hr,prov=estimate_fused_waveform(wave,effective,bw,proposals,diag,fps,10,1,42,210,hr_mode='evidence')
            np.testing.assert_array_equal(np.isfinite(wave),prov.covered)
            prov.insert(1,'base',wave);prov.to_csv(out/'waveform.csv',index=False);hr.to_csv(out/'heart_rate.csv',index=False)
            proposals.to_csv(out/'proposals.csv',index=False);diag.to_csv(out/'fusion_diagnostics.csv',index=False)
            bw.to_csv(out/'baseline_roi_waveforms.csv',index=False);tw.to_csv(out/'tracked_roi_waveforms.csv',index=False)
            pd.concat([bd.assign(branch='baseline'),td.assign(branch='tracked')],ignore_index=True).to_csv(out/'projection_diagnostics.csv',index=False)
            summary=dict(case=c,variant='log_projection_'+mode,fps=fps,frames=len(trace),source_hashes=sources,
                trace_source=str(cache),trace_sha256=sha(cache),config=plan,methods_actual=['logpos','logchrom'],
                waveform_coverage_pct=100*float(np.isfinite(wave).mean()),hr_coverage_pct=100*float(hr.accepted.mean()),
                hr_from_saved_waveform=True,offline=True,reference_used=False,elapsed_s=time.perf_counter()-start)
            write(out/'summary.json',summary);results.append(dict(case=c,returncode=0,elapsed_s=summary['elapsed_s']))
            print(json.dumps(dict(variant=summary['variant'],case=c,waveform_coverage_pct=summary['waveform_coverage_pct'],hr_coverage_pct=summary['hr_coverage_pct']),ensure_ascii=False),flush=True)
        assert frozen()==sources;write(folder/'runs.json',results)

if __name__=='__main__':main()

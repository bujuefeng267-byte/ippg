"""Frozen sequential V25 stages; only unchanged frontend traces are reused."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time

HERE=Path(__file__).resolve().parent
PROJECT=Path('/home/fengbujue/项目/rppg识别')
BASE=PROJECT/'results/data1_6_20260911'
ROOT=PROJECT/'results/data1_6_v25_20260911'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def emit(**row):print(json.dumps(row,ensure_ascii=False),flush=True)

def run(stage,data):
    case=data['case'];folder=ROOT/stage['name'];out=folder/case
    cmd=[sys.executable,'-B',str(HERE/'analyze_motion_v2.py'),data['video']['path'],
        '--output',str(out),'--cache',str(BASE/case/'inference/frame_trace.csv'),
        '--pixel-mode','tracking_screened','--algorithm-mode','guarded_fusion',
        '--max-gap','.1','--window','10','--step','1','--min-bpm','42','--max-bpm','210',
        '--motion-evidence',stage['motion_evidence'],'--hr-mode',stage['hr_mode'],'--routing-mode',stage['routing_mode']]
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    start=time.perf_counter()
    with (folder/(case+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
    row=dict(stage=stage['name'],case=case,returncode=p.returncode,elapsed_s=time.perf_counter()-start,command=cmd)
    emit(**row);return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='*');args=ap.parse_args()
    from analyze_motion_v2 import source_hashes
    frozen=source_hashes();protocol=json.loads((HERE/'development_protocol.json').read_text())
    data=json.loads((BASE/'inputs_manifest.json').read_text())
    ROOT.mkdir(exist_ok=True)
    recorded=ROOT/'protocol_before_validation.json'
    if recorded.exists():assert json.loads(recorded.read_text())['source_hashes']==frozen,'Create separate revision after source changes'
    else:
        record=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,source_hashes=frozen,
            runner_sha256=sha(Path(__file__)),evaluator_sha256=sha(HERE/'evaluate_stages.py'),
            baseline_reference_hashes={d['case']:sha(BASE/d['case']/'evaluation/paired_windows.csv') for d in data},
            caches={d['case']:dict(csv=sha(BASE/d['case']/'inference/frame_trace.csv'),metadata=sha(BASE/d['case']/'inference/frame_trace.json')) for d in data},
            cache_policy='Exact identity and frontend source hashes validated by unchanged V24 loader; all waveforms/proposals/HR regenerated.',
            inference_reference_access='No reference file argument or read in inference modules')
        recorded.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    for stage in protocol['stages']:
        if args.stages and stage['name'] not in args.stages:continue
        folder=ROOT/stage['name']
        if folder.exists():
            if (folder/'stage_evaluation.json').exists():emit(stage=stage['name'],status='already_completed');continue
            raise FileExistsError('Incomplete stage must be inspected before retry: '+str(folder))
        folder.mkdir();emit(stage=stage['name'],event='stage_start')
        results=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(run,stage,d) for d in data]):results.append(future.result())
        (folder/'runs.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        assert source_hashes()==frozen,'Inference source changed'
        if any(r['returncode'] for r in results):raise RuntimeError('Stage failed; inspect per-case logs')
        subprocess.run([sys.executable,'-B',str(HERE/'evaluate_stages.py'),'--stage',stage['name']],check=True)
    emit(event='requested_stages_complete',source_unchanged=source_hashes()==frozen)

if __name__=='__main__':main()

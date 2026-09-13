"""One predeclared V24 development round: 2 structures x 2 gaps x 6 sources.

The expensive unmodified LK/screening is reused from verified V23 paired
increments. Reconstruction, quality routing, waveform and HR are recomputed.
No reference is supplied to inference. A fresh installed-video check is separate.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from dataclasses import asdict
import hashlib,io,json,subprocess,sys,time,unittest
from analyze_motion_v2 import source_hashes
from pixel_tracking import PixelConfig
from anchored_reconstruction import AnchorConfig
from guarded_fusion import GuardConfig

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROJECT=Path('/home/fengbujue/项目/rppg识别')
CASES={
 'user0904':Path('/mnt/c/Users/15011/Desktop/Video_20260904_145402918.avi'),
 'user0907':PROJECT/'videos/user_0907/Video_20260907_171742478.avi',
 'ubfc':PROJECT/'videos/ubfc_subject1/vid.avi',
 'kaggle_full':PROJECT/'videos/kaggle_motion_subject001/video.mov',
 'synthetic72':PROJECT/'videos/self_test_72bpm.mp4',
 'data1':ROOT/'rppg_data1_20260909/inputs/video/Video_20260909_YJC_2.avi'}
OUT=HERE/'validation'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def task(case):
    records=[]
    for algorithm,gapname,gap in [('bounded_tracking','gap10',.10),('bounded_tracking','gap15',.15),
                                  ('guarded_fusion','gap10',.10),('guarded_fusion','gap15',.15)]:
        label=f'{algorithm}_{gapname}'
        cmd=[sys.executable,'-B',str(HERE/'analyze_motion_v2.py'),str(CASES[case]),'--output',str(OUT/label/case),
             '--pixel-mode','tracking_screened','--algorithm-mode',algorithm,'--max-gap',str(gap)]
        if label=='bounded_tracking_gap10':
            cmd+=['--tracking-cache',str(ROOT/'rppg_motion_v23/validation/tracking_screened_gap10'/case/'frame_trace.csv')]
        else:
            cmd+=['--cache',str(OUT/'bounded_tracking_gap10'/case/'frame_trace.csv')]
        started=time.perf_counter()
        with (OUT/f'{case}_{label}.log').open('w',encoding='utf-8') as f:
            proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        row=dict(case=case,mode=label,exit_code=proc.returncode,elapsed_s=time.perf_counter()-started,command=cmd)
        records.append(row);print(json.dumps(row),flush=True)
        if proc.returncode:break
    return records

def main():
    if OUT.exists():raise FileExistsError('Validation output is immutable; diagnose before a separately named round')
    criteria=HERE/'evaluation_protocol.json'
    protocol=json.loads(criteria.read_text(encoding='utf-8'))
    if not protocol.get('parameters_frozen'):raise ValueError('Evaluation criteria and candidate parameters must be frozen first')
    for video in CASES.values():
        if not video.is_file():raise FileNotFoundError(video)
    OUT.mkdir()
    frozen=source_hashes()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    tests=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        passed=result.wasSuccessful(),source_hashes=frozen,test_hashes={p.name:sha(p) for p in sorted(HERE.glob('test_*.py'))})
    (OUT/'tests.log').write_text(log.getvalue(),encoding='utf-8')
    (OUT/'tests.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')
    print(json.dumps(tests),flush=True)
    if not result.wasSuccessful():print(log.getvalue(),flush=True);raise SystemExit(1)
    candidates={f'{a}_{g}':dict(algorithm_mode=a,pixel_mode='tracking_screened',max_gap_s=s,
        pixel_config=asdict(PixelConfig()),anchor_config=asdict(AnchorConfig()),
        guard_config=asdict(GuardConfig()) if a=='guarded_fusion' else None)
        for a in ['bounded_tracking','guarded_fusion'] for g,s in [('gap10',.1),('gap15',.15)]}
    tracking_caches={k:dict(path=str(ROOT/'rppg_motion_v23/validation/tracking_screened_gap10'/k/'frame_trace.csv'),
        sha256=sha(ROOT/'rppg_motion_v23/validation/tracking_screened_gap10'/k/'frame_trace.csv'),
        metadata_sha256=sha(ROOT/'rppg_motion_v23/validation/tracking_screened_gap10'/k/'frame_trace.json')) for k in CASES}
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=frozen,runner_sha256=sha(__file__),
        evaluation_protocol_sha256=sha(criteria),cases={k:dict(video=str(p),bytes=p.stat().st_size,sha256=sha(p)) for k,p in CASES.items()},
        candidates=candidates,default_candidate='guarded_fusion_gap10',
        data1_alignment_protocol_sha256=sha(ROOT/'rppg_data1_20260909/alignment_protocol.json'),
        tracking_caches=tracking_caches,reference_policy='No references in inference; fixed prior alignment and all seven sensitivity shifts.',
        population='Five previously observed human videos plus one synthetic control; development/regression, not held-out generalization.',
        frontend_reuse='Unchanged V23 LK/local screening/fallback paired increments with checked source and video identity. V24 reconstruction is recomputed.',
        unchanged='V22 geometry, observation masks, POS/CHROM, filter edges, HR grids, quality gates and 10s/1s window plan.',tests=tests)
    (OUT/'protocol_before_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(task,k) for k in CASES]
        for job in as_completed(jobs):
            records.extend(job.result())
            (OUT/'runs.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    if source_hashes()!=frozen:raise RuntimeError('Inference source changed')
    if len(records)!=24 or any(r['exit_code'] for r in records):raise SystemExit(1)
    print('All 24 V24 candidate runs completed.',flush=True)

if __name__=='__main__':main()

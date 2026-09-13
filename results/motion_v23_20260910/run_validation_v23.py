"""Frozen two-mode paired-pixel experiment; no references in inference."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import time
import unittest
from analyze_motion_v2 import source_hashes
from pixel_tracking import PixelConfig
from dataclasses import asdict

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROJECT=Path('/home/fengbujue/项目/rppg识别')
CASES={
 'user0904':Path('/mnt/c/Users/15011/Desktop/Video_20260904_145402918.avi'),
 'user0907':PROJECT/'videos/user_0907/Video_20260907_171742478.avi',
 'ubfc':PROJECT/'videos/ubfc_subject1/vid.avi',
 'kaggle_full':PROJECT/'videos/kaggle_motion_subject001/video.mov',
 'synthetic72':PROJECT/'videos/self_test_72bpm.mp4',
 'data1':ROOT/'rppg_data1_20260909/inputs/video/Video_20260909_YJC_2.avi',
}
OUT=HERE/'validation'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def task(case,mode):
    records=[]
    for suffix,gap in [('gap10',.10),('gap15',.15)]:
        label=f'{mode}_{suffix}'
        output=OUT/label/case
        cmd=[sys.executable,'-B',str(HERE/'analyze_motion_v2.py'),str(CASES[case]),
             '--output',str(output),'--pixel-mode',mode,'--max-gap',str(gap)]
        if suffix=='gap10':
            cmd+=['--geometry-cache',str(ROOT/'rppg_motion_v22/validation/trimmed_gap10'/case/'frame_trace.csv')]
        else:
            cmd+=['--cache',str(OUT/f'{mode}_gap10'/case/'frame_trace.csv')]
        started=time.perf_counter()
        with (OUT/f'{case}_{label}.log').open('w',encoding='utf-8') as f:
            proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        row=dict(case=case,mode=label,exit_code=proc.returncode,elapsed_s=time.perf_counter()-started,command=cmd)
        records.append(row)
        print(json.dumps(row),flush=True)
        if proc.returncode:break
    return records

def main():
    if OUT.exists():raise FileExistsError('Validation is immutable; choose a new run directory after a diagnosed code revision')
    for video in CASES.values():
        if not video.is_file():raise FileNotFoundError(video)
    criteria=HERE/'evaluation_upgrade_criteria.json'
    if not criteria.is_file():raise FileNotFoundError(criteria)
    OUT.mkdir()
    frozen=source_hashes()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    tests=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
               skipped=len(result.skipped),passed=result.wasSuccessful(),source_hashes=frozen,
               test_hashes={p.name:sha(p) for p in sorted(HERE.glob('test_*.py'))})
    (OUT/'tests.log').write_text(log.getvalue(),encoding='utf-8')
    (OUT/'tests.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')
    print(json.dumps(tests),flush=True)
    if not result.wasSuccessful():
        print(log.getvalue(),flush=True);raise SystemExit(1)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=frozen,
        runner_sha256=sha(__file__),criteria_sha256=sha(criteria),upgrade_criteria_sha256=sha(criteria),
        cases={k:dict(video=str(p),bytes=p.stat().st_size,sha256=sha(p)) for k,p in CASES.items()},
        candidates={f'{m}_{g}':dict(pixel_mode=m,max_gap_s=s,pixel_config=asdict(PixelConfig()))
                    for m in ['tracking_only','tracking_screened'] for g,s in [('gap10',.1),('gap15',.15)]},
        default_candidate='tracking_screened_gap10',baseline='V22 trimmed_gap10',
        data1_alignment_protocol_sha256=sha(ROOT/'rppg_data1_20260909/alignment_protocol.json'),
        geometry_caches={k:dict(path=str(ROOT/'rppg_motion_v22/validation/trimmed_gap10'/k/'frame_trace.csv'),
                               sha256=sha(ROOT/'rppg_motion_v22/validation/trimmed_gap10'/k/'frame_trace.csv')) for k in CASES},
        reference_policy='No references in inference; frozen existing alignment and all seven data1 sensitivity shifts.',
        unchanged='V22 face geometry/decisions/masks, 10s windows, 1s step, quality gates, POS/CHROM, fusion, actual-waveform HR readout.',
        candidate_signal='Paired patch log-color changes, with explicitly marked baseline ratio fallback and gap resets; relative-color reconstruction, not raw RGB.',
        population='Five human source videos plus one synthetic control; overlapping windows are not independent subjects; development/regression only.',
        tests=tests)
    (OUT/'protocol_before_validation.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(task,c,m) for c in CASES for m in ['tracking_only','tracking_screened']]
        for job in as_completed(jobs):
            records.extend(job.result())
            (OUT/'runs.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    if source_hashes()!=frozen:raise RuntimeError('Inference source changed during validation')
    if len(records)!=24 or any(r['exit_code'] for r in records):raise SystemExit(1)
    print('All six cases, two pixel modes and two frozen gap settings completed.',flush=True)

if __name__=='__main__':main()

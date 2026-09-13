"""Stage six user archives and run the installed, frozen V2.4 pipeline.

No reference values are supplied to inference. Original archives stay unchanged.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, stat, struct, subprocess, sys, time, zipfile

PROJECT = Path('/home/fengbujue/项目/rppg识别')
SOURCE = Path('/mnt/c/Users/15011/Desktop/视频')
INPUT = PROJECT/'videos/data1_6_20260911'
OUTPUT = PROJECT/'results/data1_6_20260911'
CODE = PROJECT/'motion_upgrade_v24_20260910'
MEDIA = {'.avi', '.mp4', '.mov', '.mkv'}

def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs), ensure_ascii=False), flush=True)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def archived_mtime(info):
    extra=info.extra
    while len(extra)>=4:
        tag,size=struct.unpack('<HH',extra[:4]); value=extra[4:4+size];extra=extra[4+size:]
        if tag==0x5455 and len(value)>=5 and value[0]&1:
            return struct.unpack('<I',value[1:5])[0]
    return None

def expand(archive, target, evidence, depth=0):
    if depth>3: raise ValueError('Unexpected nested ZIP depth')
    target.mkdir(parents=True, exist_ok=True)
    base=target.resolve()
    nested=[]
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            rel=Path(info.filename.replace('\\','/'))
            dest=(target/rel).resolve()
            if not dest.is_relative_to(base) or rel.is_absolute() or stat.S_ISLNK(info.external_attr>>16):
                raise ValueError('Unsafe archive member')
            if info.is_dir(): dest.mkdir(parents=True,exist_ok=True);continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists(): raise FileExistsError(dest)
            h=hashlib.sha256()
            with z.open(info) as src, dest.open('xb') as out:
                for block in iter(lambda:src.read(8*1024*1024),b''):
                    out.write(block);h.update(block)
            epoch=archived_mtime(info)
            if epoch is not None: os.utime(dest,(epoch,epoch))
            evidence.append(dict(archive=str(archive),member=info.filename,path=str(dest),
                bytes=info.file_size,crc32=info.CRC,sha256=h.hexdigest(),
                zip_dos_datetime=info.date_time,extended_unix_mtime_s=epoch))
            if dest.suffix.lower()=='.zip':nested.append(dest)
    for nested_zip in nested:
        expand(nested_zip,nested_zip.parent/(nested_zip.stem+'_unpacked'),evidence,depth+1)

def prepare(case):
    target=INPUT/case
    receipt=target/'staging_manifest.json'
    if receipt.exists(): return json.loads(receipt.read_text())
    if target.exists(): raise RuntimeError(f'Partial staging needs inspection: {target}')
    evidence=[];archive=SOURCE/(case+'.zip')
    emit('extract_start',case=case,archive=str(archive))
    expand(archive,target,evidence)
    videos=[r for r in evidence if Path(r['path']).suffix.lower() in MEDIA]
    if len(videos)!=1: raise ValueError(f'{case}: expected one video, found {len(videos)}')
    video=videos[0]
    probe=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',video['path']],
        check=True,capture_output=True,text=True)
    data=dict(case=case,archive=str(archive),archive_bytes=archive.stat().st_size,
        video=video,files=evidence,ffprobe=json.loads(probe.stdout))
    receipt.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    emit('extract_done',case=case,video=video['path'],bytes=video['bytes'])
    return data

def run_case(data):
    case=data['case']; out=OUTPUT/case/'inference';log=OUTPUT/case/'inference.log'
    out.parent.mkdir(parents=True,exist_ok=True)
    done=out.parent/'run_record.json'
    if done.exists():
        prior=json.loads(done.read_text())
        if prior['exit_code']==0 and (out/'summary.json').exists():return prior
    if out.exists():raise RuntimeError(f'Inspect incomplete inference before retry: {out}')
    command=[sys.executable,'-B',str(CODE/'analyze_motion_v2.py'),data['video']['path'],
        '--output',str(out),'--pixel-mode','tracking_screened','--algorithm-mode','guarded_fusion',
        '--max-gap','0.10','--window','10','--step','1','--min-bpm','42','--max-bpm','210']
    started=time.perf_counter();emit('inference_start',case=case)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with log.open('w') as f: p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
    record=dict(case=case,exit_code=p.returncode,elapsed_s=time.perf_counter()-started,
        command=command,video_sha256=data['video']['sha256'],reference_supplied_to_inference=False)
    done.write_text(json.dumps(record,ensure_ascii=False,indent=2));emit('inference_done',**record)
    return record

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(CODE))
    from analyze_motion_v2 import source_hashes
    frozen=source_hashes()
    freeze=OUTPUT/'protocol_before_inference.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())['source_hashes']==frozen
    else:
        freeze.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
            source_hashes=frozen,algorithm_mode='guarded_fusion',pixel_mode='tracking_screened',
            max_gap_s=.10,window_s=10,step_s=1,min_bpm=42,max_bpm=210,
            cases=[f'data{i}' for i in range(1,7)],
            primary_hr='ridge_bpm on accepted windows; offline estimator',
            reference_policy='After inference only; no HR-optimized time alignment.',
            analysis_scope='Six recordings; subject independence not established.'),ensure_ascii=False,indent=2))
    all_data=[];records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[]
        for i in range(1,7):
            data=prepare(f'data{i}');all_data.append(data)
            (OUTPUT/'inputs_manifest.json').write_text(json.dumps(all_data,ensure_ascii=False,indent=2))
            futures.append(pool.submit(run_case,data))
        for future in as_completed(futures):
            records.append(future.result())
            (OUTPUT/'runs.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    assert source_hashes()==frozen,'Inference source changed during batch'
    if any(r['exit_code'] for r in records):raise SystemExit(1)
    emit('batch_complete',cases=len(records),source_unchanged=True)

if __name__=='__main__':main()

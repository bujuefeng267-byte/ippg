"""Run retained V28 continuous PPG plus verified V32 window-based heart rate.

The two products have explicit sources: V32 HR refers to saved signal windows,
not to the unchanged V28 continuous context waveform. No reference input exists.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import subprocess
import sys
from common import sha, write_new, verify

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
if not (PROJECT/'motion_upgrade_v28_20260912').is_dir():
    PROJECT=Path('/home/fengbujue/项目/rppg识别')
PYTHON=PROJECT/'.venv/bin/python'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video',type=Path,help='New video: run the retained V28 frontend first')
    group.add_argument('--v28-output',type=Path,help='Existing verified V28 output: reuse its measured data')
    parser.add_argument('--trace-cache',type=Path,help='Original matching trace CSV or cache directory')
    parser.add_argument('--out',required=True,type=Path,help='New output folder; existing folders are rejected')
    args=parser.parse_args(argv)
    out=args.out.resolve()
    if out.exists():parser.error('Output already exists; use a new directory')
    old=args.v28_output.resolve() if args.v28_output else out/'continuous_v28'
    if args.v28_output and (not old.is_dir() or out.is_relative_to(old)):
        parser.error('Use a verified V28 directory and put output outside it')
    if args.video and not args.video.is_file():parser.error('Video does not exist')
    if args.trace_cache and not args.trace_cache.exists():parser.error('Trace cache does not exist')
    sources={str(p):sha(p) for p in (Path(__file__),HERE/'common.py')}
    for folder in ('motion_upgrade_v28_20260912','motion_upgrade_v31_20260912','motion_upgrade_v32_20260914'):
        sources.update({str(p):sha(p) for p in (PROJECT/folder).glob('*.py')})
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':sources,
        'reference_used':False,'case_specific_routing':False,'mode':'retained_V28_continuous_plus_V32_window_HR',
        'continuous_waveform_improvement_claimed':False,'original_v28':str(old)}
    out.mkdir(parents=True)
    write_new(out/'workflow_protocol.json',record)
    try:
        if args.video:
            command=[str(PYTHON),'-B',str(PROJECT/'motion_upgrade_v28_20260912/analyze_video_v28.py'),
                     '--video',str(args.video.resolve()),'--out',str(old)]
            if args.trace_cache:command+=['--trace-cache',str(args.trace_cache.resolve())]
            subprocess.run(command,check=True)
        command=[str(PYTHON),'-B',str(PROJECT/'motion_upgrade_v32_20260914/analyze_v28_result_v32.py'),
                 '--v28-output',str(old),'--out',str(out/'window_hr')]
        # A new V28 run writes its own bound trace metadata. For that path the
        # V32 adapter must use the new V28-local cache, not the external input.
        if args.v28_output and args.trace_cache:
            command+=['--trace-cache',str(args.trace_cache.resolve())]
        subprocess.run(command,check=True)
        verify(sources)
        child=json.loads((out/'window_hr/summary.json').read_text(encoding='utf-8'))
        if child['status']!='complete' or child['reference_used'] is not False:
            raise ValueError('Window inference did not produce completed reference-free output')
        verify({str(out/'window_hr'/name):value for name,value in child['output_hashes'].items()})
        artifacts=[out/'workflow_protocol.json',out/'window_hr/summary.json',
                   out/'window_hr/windows_manifest.csv',old/'waveform.csv',old/'heart_rate.csv']
        receipt={**record,'status':'complete','primary_heart_rate':str(out/'window_hr/heart_rate.csv'),
            'primary_heart_rate_sha256':sha(out/'window_hr/heart_rate.csv'),
            'heart_rate_waveform_manifest':str(out/'window_hr/windows_manifest.csv'),
            'heart_rate_waveform_kind':'individual_measured_10_second_windows',
            'continuous_context_waveform':str(old/'waveform.csv'),
            'continuous_context_heart_rate':str(old/'heart_rate.csv'),
            'continuous_context_is_source_of_replaced_hr':False,
            'new_candidate_scoring_enabled':False,'source_hashes_verified':True,
            'artifact_hashes':{str(p):sha(p) for p in artifacts}}
        write_new(out/'result_summary.json',receipt)
        print(json.dumps({k:receipt[k] for k in ('status','primary_heart_rate','heart_rate_waveform_manifest',
                                               'continuous_context_waveform')},ensure_ascii=False))
    except Exception as exc:
        write_new(out/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),
            'original_v28':str(old),'partial_outputs_are_not_success':True})
        raise


if __name__=='__main__':main()

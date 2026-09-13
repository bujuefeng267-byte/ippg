#!/usr/bin/env python3
"""Create an isolated copy of the frozen cPACE/RhythmMamba trials and replay it.

Run this entry with WSL Python. Each name creates a new direct child of this
Windows workspace. Existing paths are rejected, never overwritten or deleted.
--prepare-only copies and hashes sources/weights; it runs no tests or inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'rppg_paper_trials_20260909'
SCIENCE_PYTHON = Path('/home/fengbujue/项目/rppg识别/.venv/bin/python')
RHYTHM_PYTHON = Path('/home/fengbujue/项目/rppg识别/.venv-rhythm-trial/bin/python')
FORBIDDEN_PARTS = {'.git', '__pycache__', 'results', 'results_no_homodyne', 'evaluation'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require_inside(path, root):
    resolved, boundary = Path(path).resolve(), Path(root).resolve()
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise ValueError(f'Path is not strictly inside intended directory: {resolved}')
    return resolved


def output_path(name):
    if not re.fullmatch(r'[A-Za-z0-9_]+', name, flags=re.ASCII):
        raise ValueError('--name must contain only ASCII letters, digits, and underscore')
    candidate = ROOT/f'rppg_paper_replay_{name}'
    # Check the lexical entry as well as its resolution; dangling symlinks count.
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f'Replay directory already exists: {candidate}')
    resolved = require_inside(candidate, ROOT)
    if resolved.parent != ROOT:
        raise ValueError('Replay output must be a direct workspace child')
    return resolved


def source_plan(method):
    """Explicit source whitelist; no old result or passed/completed record copied."""
    wanted = {Path('cpace/project_estimator/legacy_motion.py'),
              Path('cpace/project_estimator/analyze_rppg.py')}
    if method in ('cpace', 'both'):
        wanted.update(Path('cpace')/name for name in (
            'run_cpace_trial.py', 'test_cpace_trial.py', 'verify_outputs.py', 'README.md'))
        wanted.update(Path('cpace/author_source')/p.name
                      for p in (SOURCE/'cpace/author_source').glob('*.py'))
        wanted.update(Path('cpace/author_source')/name for name in ('README.md','LICENSE'))
    if method in ('rhythm', 'both'):
        wanted.update(Path('rhythm')/name for name in (
            'run_rhythm_trial.py','backend.py','backend_test.py','test_preprocessing.py',
            'README.md','upstream_download.json'))
        wanted.update((Path('rhythm_source_audit.json'), Path('rhythm_source_audit.md')))
        wanted.update(Path('rhythm/third_party_licenses')/name for name in (
            'Mamba_LICENSE', 'Mamba_LICENSE_provenance.json'))
        # The download manifest lists exactly the frozen official files and one
        # selected checkpoint. Do not sweep up alternate weights or generated data.
        upstream = json_read(SOURCE/'rhythm/upstream_download.json')
        for record in upstream['files']:
            rel = Path(record['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Invalid relative path in upstream manifest')
            wanted.add(Path('rhythm/official_snapshot')/rel)

    plan = {}
    source_root = SOURCE.resolve(strict=True)
    if source_root.parent != ROOT or source_root != SOURCE:
        raise ValueError('Original trial source must be a direct workspace child')
    for rel in sorted(wanted, key=str):
        if rel.is_absolute() or '..' in rel.parts or any(p in FORBIDDEN_PARTS for p in rel.parts):
            raise ValueError(f'Forbidden snapshot path: {rel}')
        src = require_inside(source_root/rel, source_root)
        if not src.is_file():
            raise FileNotFoundError(src)
        plan[rel.as_posix()] = {'source':src, 'sha256':sha(src), 'bytes':src.stat().st_size}

    pins = {}
    if method in ('cpace', 'both'):
        frozen = json_read(SOURCE/'cpace/protocol_before_inference.json')
        pins.update({'cpace/'+key:value for key,value in frozen['code_hashes'].items()})
    if method in ('rhythm', 'both'):
        frozen = json_read(SOURCE/'rhythm/protocol_before_inference.json')
        pins.update(frozen['code_hashes'])
        for record in upstream['files']:
            pins['rhythm/official_snapshot/'+record['path']] = record['sha256']
    checked = []
    for rel, expected in pins.items():
        if rel in plan:
            if plan[rel]['sha256'] != expected:
                raise ValueError(f'Source changed since the frozen trial: {rel}')
            checked.append(rel)
    return plan, sorted(checked)


def commands(method, destination):
    result = []
    def add(label, python, relative, *args):
        result.append({'label':label,
                       'argv':[str(python), '-B', str(destination/relative), *args]})
    if method in ('cpace','both'):
        add('cpace_selfcheck', SCIENCE_PYTHON, 'cpace/test_cpace_trial.py')
        add('cpace_inference', SCIENCE_PYTHON, 'cpace/run_cpace_trial.py', '--run-all')
        add('cpace_saved_output_check', SCIENCE_PYTHON, 'cpace/verify_outputs.py')
    if method in ('rhythm','both'):
        add('rhythm_preprocessing_check', SCIENCE_PYTHON, 'rhythm/test_preprocessing.py')
        add('rhythm_backend_check', RHYTHM_PYTHON, 'rhythm/backend_test.py')
        add('rhythm_inference', RHYTHM_PYTHON, 'rhythm/run_rhythm_trial.py', '--run-all', '--device', 'cuda')
    return result


def verify_snapshot(destination, plan):
    for rel, record in plan.items():
        target = require_inside(destination/rel, destination)
        if not target.is_file() or sha(target) != record['sha256']:
            raise ValueError(f'Snapshot checksum mismatch: {rel}')


def prepare(destination, method, plan, pin_checked, scheduled, prepare_only):
    # Exclusive directory creation is the write boundary. Failure later leaves
    # partial files for inspection; this program never deletes or resumes them.
    destination.mkdir(exist_ok=False)
    for rel, record in plan.items():
        target = require_inside(destination/rel, destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        require_inside(target, destination)
        with record['source'].open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8*1024*1024)
    verify_snapshot(destination, plan)
    manifest = {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'method':method, 'workspace_root':str(ROOT), 'source_root':str(SOURCE),
        'destination':str(destination), 'prepare_only':prepare_only,
        'source_snapshot_verified':True,
        'files':{rel:record['sha256'] for rel,record in plan.items()},
        'file_bytes':{rel:record['bytes'] for rel,record in plan.items()},
        'verified_against_original_frozen_protocol_or_download':pin_checked,
        'wrapper_sha256':sha(__file__),
        'commands':scheduled,
        'original_results_copied':False, 'old_completion_or_pass_markers_copied':False,
        'videos_or_v22_traces_copied':False,
        'input_dependency':'Original workspace V2.2 trimmed_gap10 traces and original videos remain required; method runners retain their original ROOT lookup.',
        'runtime_policy':'Use existing separate WSL environments; no installs or environment modification; explicit CUDA for Rhythm; no silent fallback.',
        'scope':'Replay frozen inference only. Independent reference evaluation is not copied or run.'}
    write_new_json(destination/'snapshot_manifest.json', manifest)
    return manifest


def execute(destination, method, plan, scheduled):
    steps = []
    failure = None
    logdir = destination/'replay_logs'
    logdir.mkdir(exist_ok=False)
    try:
        for index, command in enumerate(scheduled, 1):
            verify_snapshot(destination, plan)
            print(f'[{index}/{len(scheduled)}] {command["label"]}', flush=True)
            begin = time.perf_counter()
            log_path = logdir/f'{index:02d}_{command["label"]}.log'
            # argv is a list; never invoke a command shell or installation script.
            with log_path.open('xb') as stream:
                result = subprocess.run(command['argv'], cwd=str(ROOT), shell=False,
                                        stdout=stream, stderr=subprocess.STDOUT, check=False)
            steps.append({'label':command['label'], 'argv':command['argv'],
                          'returncode':result.returncode, 'elapsed_s':time.perf_counter()-begin,
                          'log':str(log_path.relative_to(destination))})
            if result.returncode:
                raise RuntimeError(f'{command["label"]} failed ({result.returncode}); see {log_path}')
        verify_snapshot(destination, plan)
        for selected in ('cpace','rhythm') if method == 'both' else (method,):
            if not json_read(destination/selected/'run_summary.json').get('all_six_successful'):
                raise RuntimeError(f'{selected}: six-case completion marker missing')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_new_json(destination/'replay_execution.json', {
            'finished_utc':datetime.now(timezone.utc).isoformat(), 'method':method,
            'passed':failure is None, 'failure':failure, 'steps':steps,
            'source_snapshot_hashes_verified':failure is None,
            'reference_evaluation_run':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['cpace','rhythm','both'], required=True)
    parser.add_argument('--name', required=True, help='Unique ASCII letters/digits/underscore only')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Create and verify source/weight snapshot only; no tests or inference')
    args = parser.parse_args()
    if sys.platform != 'linux' or not str(ROOT).startswith('/mnt/c/'):
        parser.error('Run this workspace entry with WSL Python under /mnt/c; see replay_usage.md')
    destination = output_path(args.name)
    plan, pinned = source_plan(args.method)
    scheduled = commands(args.method, destination)
    if not args.prepare_only:
        for command in scheduled:
            python = Path(command['argv'][0])
            if not python.is_file() or not os.access(python, os.X_OK):
                raise FileNotFoundError(f'Required existing WSL interpreter unavailable: {python}')
    prepare(destination, args.method, plan, pinned, scheduled, args.prepare_only)
    print(json.dumps({'prepared':str(destination), 'method':args.method,
                      'files_verified':len(plan), 'inference_started':False}), flush=True)
    if args.prepare_only:
        print('Prepare-only completed. Use a NEW name for a full replay; existing names are never resumed.', flush=True)
        return
    execute(destination, args.method, plan, scheduled)
    print(json.dumps({'replay_complete':str(destination), 'method':args.method}), flush=True)


if __name__ == '__main__':
    main()

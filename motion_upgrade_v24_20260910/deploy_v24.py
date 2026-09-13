"""Install only the V24 experiment. No arguments: read-only plan; --verify: hashes.

This installer never promotes a default or overwrites a destination. It copies
code/tests/documentation and derived evidence, never source videos/references.
Run --install explicitly in the existing WSL project after validation is ready.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import uuid

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/fengbujue/项目/rppg识别')
CODE_NAME = 'motion_upgrade_v24_20260910'
RESULT_NAME = 'results/motion_v24_20260910'
LAUNCHER = 'run_motion_v24_experimental.sh'
CANDIDATE = 'guarded_fusion_gap10'
LAUNCHER_ARGS = ['--pixel-mode', 'tracking_screened', '--algorithm-mode',
                 'guarded_fusion', '--max-gap', '0.10']
CORE = {'analyze_motion_v2.py', 'analyze_rppg.py', 'legacy_motion.py',
        'motion_frontend.py', 'motion_fusion.py', 'stable_groups.py',
        'waveform_hr.py', 'baseline_frontend.py', 'pixel_tracking.py',
        'anchored_reconstruction.py', 'guarded_fusion.py'}
CASES = {'user0904', 'user0907', 'ubfc', 'kaggle_full', 'synthetic72', 'data1'}
MODES = {f'{a}_{g}' for a in ('bounded_tracking', 'guarded_fusion')
         for g in ('gap10', 'gap15')}
DOCS = {'README.md', 'research_v24.md', 'sources.json', 'installation_usage.md'}
TOOLS = {'deploy_v24.py', 'verify_install_v24.py', 'evaluate_v24.py',
         'run_validation_v24.py', 'qa_recompute_waveform_hr.py'}
EVIDENCE = {'evaluation_protocol.json', 'evaluation_plan.md',
            'evaluation_protocol_pre_freeze.json', 'evaluation_implementation_freeze.json',
            'evaluation_contract_precheck.json', 'V24_论文依据与实测报告.md',
            'independent_qa.json', 'independent_qa.md',
            'verify_results_v24.py', 'export_report_v24.py', 'qa_guard_routing.py',
            'validation/protocol_before_validation.json', 'validation/tests.json',
            'validation/runs.json', 'validation/waveform_replay_qa.json',
            'validation/guard_routing_qa.json',
            'evaluation/summary.json', 'evaluation/upgrade_gate.json'}
DERIVED_SUFFIXES = {'.csv', '.json', '.md', '.txt', '.log', '.png', '.svg', '.pdf', '.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(base, name, file=False):
    require(isinstance(name, str) and '\\' not in name, 'Expected a relative POSIX path')
    rel = PurePosixPath(name)
    require(name and not rel.is_absolute() and '..' not in rel.parts and
            rel.as_posix() == name and ':' not in name, f'Unsafe relative path: {name}')
    base = Path(base).resolve()
    path = base / name
    require(path.resolve().is_relative_to(base) and path.resolve() != base,
            f'Path escapes its declared root: {path}')
    probe = path
    while probe != base:
        require(not probe.is_symlink(), f'Symlink is not an input/destination: {probe}')
        probe = probe.parent
    if file:
        require(path.is_file(), f'Missing file: {path}')
    return path


def hash_map(mapping, root):
    require(isinstance(mapping, dict) and mapping, 'Expected a nonempty SHA256 map')
    for name, expected in mapping.items():
        require(isinstance(expected, str) and len(expected) == 64 and
                all(c in '0123456789abcdef' for c in expected), f'Invalid SHA256: {name}')
        require(digest(checked(root, name, file=True)) == expected, f'Hash mismatch: {name}')


def require_wsl():
    require(sys.platform == 'linux', 'Use the existing WSL project environment')
    require(PROJECT.is_dir() and PROJECT.resolve() == PROJECT, 'Project missing or aliased')
    # A virtualenv interpreter is normally an intentional symlink to system Python.
    require((PROJECT / '.venv/bin/python').is_file(), 'Existing project Python missing')


def preserved_snapshot():
    result = {}
    for path in sorted(PROJECT.iterdir()):
        if path.name == LAUNCHER:
            continue
        if path.is_file() and (path.suffix in {'.sh', '.py'} or path.name == 'README.md'):
            result[path.name] = digest(checked(PROJECT, path.name, file=True))
    return result


def preflight():
    for name in CORE | DOCS | TOOLS | EVIDENCE:
        checked(HERE, name, file=True)
    protocol = load(HERE / 'validation/protocol_before_validation.json')
    tests = load(HERE / 'validation/tests.json')
    runs = load(HERE / 'validation/runs.json')
    replay = load(HERE / 'validation/waveform_replay_qa.json')
    criteria = load(HERE / 'evaluation_protocol.json')
    evaluation = load(HERE / 'evaluation/summary.json')
    gate = load(HERE / 'evaluation/upgrade_gate.json')
    sources = protocol.get('source_hashes')
    require(set(sources or {}) == CORE, 'Exactly eleven frozen inference modules are required')
    hash_map(sources, HERE)
    require(criteria.get('parameters_frozen') is True, 'Candidate parameters are not frozen')
    require(protocol.get('evaluation_protocol_sha256') == digest(HERE / 'evaluation_protocol.json'),
            'Evaluation protocol changed after inference freeze')
    require(protocol.get('default_candidate') == CANDIDATE, 'Frozen primary candidate differs')
    require(set(protocol.get('cases', {})) == CASES and
            set(protocol.get('candidates', {})) == MODES, 'Incomplete six-source candidate plan')
    require(tests.get('passed') is True and tests.get('run', 0) > 0 and
            tests.get('failures') == tests.get('errors') == 0, 'Tests did not pass')
    require(tests.get('source_hashes') == sources, 'Test/source freeze differs')
    hash_map(tests.get('test_hashes'), HERE)
    require(set(tests['test_hashes']) == {p.name for p in HERE.glob('test_*.py')},
            'Current test set differs from tested set')
    require(isinstance(runs, list) and len(runs) == 24 and
            all(r.get('exit_code') == 0 for r in runs), '24 successful inference runs required')
    require({(r.get('case'), r.get('mode')) for r in runs} ==
            {(c, m) for c in CASES for m in MODES}, 'Missing/duplicate run identity')
    require(replay.get('passed') is True, 'Independent waveform replay did not pass')
    require(evaluation.get('source_protocol_v24') == protocol and
            evaluation.get('evaluation_protocol_sha256') == digest(HERE / 'evaluation_protocol.json') and
            evaluation.get('upgrade_gate') == gate, 'Evaluation provenance is inconsistent')
    required_qa = ('baseline_metrics_reproduced', 'baseline_SNR_reproduced',
                  'baseline_RGB_geometry_flags_and_sources_verified',
                  'four_candidates_share_frontend_with_1e_9_serialization_tolerance',
                  'provenance_from_positive_diagnostics_verified',
                  'reference_free_guard_routing_verified', 'input_hashes_unchanged',
                  'CSV_shape_missingness_and_integer_timestamp_roundtrip')
    require(all(evaluation.get('QA', {}).get(k) is True for k in required_qa),
            'Evaluation engineering QA did not pass')
    require(evaluation['QA'].get('waveform_replay', {}).get('passed') is True,
            'Evaluation does not confirm waveform replay')
    require(evaluation.get('output_manifest'), 'Evaluated output manifest missing')
    for name, item in evaluation['output_manifest'].items():
        require(digest(checked(HERE / 'evaluation', name, file=True)) == item['sha256'],
                f'Evaluated table changed: {name}')
    for case in CASES:
        for mode in MODES:
            summary = load(checked(HERE, f'validation/{mode}/{case}/summary.json', file=True))
            require(summary.get('source_hashes') == sources and
                    summary.get('reference_identity') is None, 'Unfrozen/reference-aware inference')
    return protocol, gate


def transfers():
    code = {n: checked(HERE, n, file=True) for n in CORE | DOCS | TOOLS}
    for path in sorted(HERE.glob('test_*.py')):
        code[path.name] = checked(HERE, path.name, file=True)
    results = {n: checked(HERE, n, file=True) for n in EVIDENCE}
    for dirname in ('validation', 'evaluation', 'audit'):
        top = checked(HERE, dirname)
        if not top.exists():
            continue
        for path in sorted(top.rglob('*')):
            require(not path.is_symlink(), f'Symlink in evidence: {path}')
            if path.is_file():
                require(path.suffix.lower() in DERIVED_SUFFIXES,
                        f'Unrecognized derived artifact; original inputs never copied: {path}')
                results[path.relative_to(HERE).as_posix()] = path
    return code, results


def launcher_bytes():
    args = ' '.join(LAUNCHER_ARGS)
    return ('#!/usr/bin/env bash\nset -euo pipefail\n'
            'project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'exec "$project_dir/.venv/bin/python" -B '
            f'"$project_dir/{CODE_NAME}/analyze_motion_v2.py" {args} "$@"\n').encode('utf-8')


def write_new(path, payload, executable=False):
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if executable:
        Path(path).chmod(0o755)


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def verify(record):
    require_wsl()
    require(record.get('schema_version') == 1 and record.get('candidate') == CANDIDATE,
            'Unexpected installation record')
    require(record.get('code') == str(PROJECT / CODE_NAME) and
            record.get('results') == str(PROJECT / RESULT_NAME) and
            record.get('launcher') == str(PROJECT / LAUNCHER), 'Unexpected installation targets')
    hash_map(record['files'], PROJECT)
    hash_map(record['preserved_entry_hashes'], PROJECT)
    require(digest(checked(PROJECT, LAUNCHER, file=True)) == record['launcher_sha256'],
            'Launcher changed')
    installed_record = checked(PROJECT, RESULT_NAME + '/installation_manifest.json', file=True)
    require(load(installed_record) == record, 'Installed manifest and local receipt differ')
    return {'verified': True, 'modified': False, 'files': len(record['files']),
            'default_changed': False, 'performance_gate': record['performance_gate']}


def install():
    require_wsl()
    protocol, gate = preflight()
    destinations = [checked(PROJECT, n) for n in (CODE_NAME, RESULT_NAME, LAUNCHER)]
    record_path = HERE / 'deployment.json'
    require(not os.path.lexists(record_path), 'Local installation receipt already exists; use --verify')
    for path in destinations:
        require(not os.path.lexists(path), f'Existing destination will not be overwritten: {path}')
    before = preserved_snapshot()
    code, results = transfers()
    hashes = {str(p): digest(p) for p in set(code.values()) | set(results.values())}
    token = uuid.uuid4().hex
    stages = [checked(PROJECT, '.' + CODE_NAME + '.stage-' + token),
              checked(PROJECT, 'results/.' + Path(RESULT_NAME).name + '.stage-' + token)]
    require(checked(PROJECT, 'results').is_dir(), 'Existing project results parent missing')
    for path in stages:
        path.mkdir(exist_ok=False)
    files = {}
    for mapping, stage, final in zip((code, results), stages, (CODE_NAME, RESULT_NAME)):
        for name, source in sorted(mapping.items()):
            target = checked(stage, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            require(digest(target) == hashes[str(source)], f'Copy mismatch: {source}')
            files[final + '/' + name] = hashes[str(source)]
    for path, expected in hashes.items():
        require(digest(path) == expected, f'Source changed during staging: {path}')
    require(preserved_snapshot() == before, 'An old project entry changed during staging')
    for path in destinations:
        require(not os.path.lexists(path), f'Destination appeared concurrently: {path}')
    record = dict(schema_version=1, installed_utc=datetime.now(timezone.utc).isoformat(),
        mode='experimental_only', candidate=CANDIDATE, launcher_args=LAUNCHER_ARGS,
        code=str(destinations[0]), results=str(destinations[1]), launcher=str(destinations[2]),
        default_changed=False, original_inputs_copied=False, source_hashes=protocol['source_hashes'],
        validation_protocol_sha256=digest(HERE / 'validation/protocol_before_validation.json'),
        evaluation_protocol_sha256=digest(HERE / 'evaluation_protocol.json'),
        performance_gate=gate, performance_gate_does_not_promote_defaults=True,
        files=files, preserved_entry_hashes=before,
        launcher_sha256=hashlib.sha256(launcher_bytes()).hexdigest(),
        source_input_hashes=hashes)
    write_new(stages[1] / 'installation_manifest.json', encode(record))
    # Explicit new-only paths. Failed staging is retained; no deletion or retry overwrite.
    stages[0].rename(destinations[0])
    stages[1].rename(destinations[1])
    write_new(destinations[2], launcher_bytes(), executable=True)
    require(preserved_snapshot() == before, 'An old project entry changed during installation')
    write_new(record_path, encode(record))
    return {'installed': True, **verify(record)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--install', action='store_true')
    actions.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.install:
        report = install()
    elif args.verify:
        report = verify(load(HERE / 'deployment.json'))
    else:
        report = dict(action='read_only_plan', modified=False, project=str(PROJECT),
            new_only=[CODE_NAME, RESULT_NAME, LAUNCHER], candidate=CANDIDATE,
            launcher_args=LAUNCHER_ARGS, core_files=sorted(CORE),
            original_inputs_copied=False, old_entries_changed=False,
            promotion_supported=False, installation_receipt_exists=(HERE / 'deployment.json').is_file(),
            missing_required_files=sorted(n for n in CORE | DOCS | TOOLS | EVIDENCE if not (HERE / n).is_file()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

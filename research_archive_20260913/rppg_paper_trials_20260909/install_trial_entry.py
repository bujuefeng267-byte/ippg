"""Install one explicitly named optional replay launcher in the user's project."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
PROJECT=Path('/home/fengbujue/项目/rppg识别').resolve(strict=True)
target=PROJECT/'run_paper_replay.sh'
if target.resolve().parent != PROJECT:
    raise ValueError('Launcher target escaped the named project')
body='''#!/usr/bin/env bash
set -euo pipefail
exec "/home/fengbujue/项目/rppg识别/.venv/bin/python" -B "/mnt/c/Users/15011/Documents/ChatGPT/New project/replay_paper_methods.py" "$@"
'''
if target.exists():
    if target.read_text(encoding='utf-8') != body:
        raise FileExistsError('Existing launcher differs; refusing replacement')
else:
    with target.open('x',encoding='utf-8',newline='\n') as f:
        f.write(body)
    target.chmod(0o755)
syntax=subprocess.run(['bash','-n',str(target)],capture_output=True,text=True)
if syntax.returncode:
    raise RuntimeError(syntax.stderr)
help_run=subprocess.run([str(target),'--help'],capture_output=True,text=True)
if help_run.returncode:
    raise RuntimeError(help_run.stderr)
check=subprocess.run([str(target),'--method','both','--name','entrycheck_20260909','--prepare-only'],capture_output=True,text=True)
if check.returncode:
    raise RuntimeError(check.stderr)
record=dict(passed=True,launcher=str(target),launcher_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bash_syntax_passed=True,help_passed=True,prepare_only_passed=True,
    prepare_output=check.stdout,models_rerun=False,existing_defaults_changed=False)
(HERE/'entrypoint_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))

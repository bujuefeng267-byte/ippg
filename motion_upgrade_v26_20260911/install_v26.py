"""Install the tested V26 experiment separately; preserve the V25 recommendation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil

HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别').resolve()
DEST=P/'motion_upgrade_v26_20260911'
RESULT=P/'results/data1_6_v26_20260911'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert P.is_dir() and DEST.parent.resolve()==P
    protected=[P/'run.sh', P/'run_motion_v25.sh']
    before={str(p):sha(p) for p in protected}
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and
        not any(part in {'__pycache__','.pytest_cache','.git','report_provisional','final_report'} for part in p.relative_to(HERE).parts)
        and p.suffix not in {'.pyc','.pyo'})
    expected={str(p.relative_to(HERE)):sha(p) for p in files}
    assert not DEST.exists(), 'Preserve existing installation; use a new version directory'
    launcher=P/'run_motion_v26_experimental.sh'
    assert not launcher.exists(), 'Preserve existing entry point'
    DEST.mkdir()
    for p in files:
        target=DEST/p.relative_to(HERE); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
        assert sha(target)==expected[str(p.relative_to(HERE))]
    script='#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(cd -- "$(dirname -- "$0")" && pwd)"\nexec "$ROOT/.venv/bin/python" -B "$ROOT/motion_upgrade_v26_20260911/analyze_components_v26.py" "$@"\n'
    with launcher.open('x',encoding='utf-8',newline='\n') as stream: stream.write(script)
    launcher.chmod(0o755)
    assert before=={str(p):sha(p) for p in protected}
    receipt=dict(installed_utc=datetime.now(timezone.utc).isoformat(),destination=str(DEST),
        files=expected,launcher_sha256=sha(launcher),preserved_recommended_entries=before,
        status='experimental; prior recommended V25 entry preserved',
        reference_used_in_inference=False)
    with (RESULT/'installation_receipt.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(destination=str(DEST),files=len(files),launcher=str(launcher),
                         preserved_recommendation=True),ensure_ascii=False))

if __name__=='__main__':main()

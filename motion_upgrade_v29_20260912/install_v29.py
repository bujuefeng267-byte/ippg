"""Install the independently selectable short-window modes and retain V28."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
R = P/'results/data1_6_v29_20260912'
D = P/'motion_upgrade_v29_20260912'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    protocol = json.loads((R/'protocol_before_run.json').read_text())
    for name, digest in protocol['source_hashes'].items():
        assert sha(HERE/name) == digest, name
    for path, digest in protocol['protected_entries'].items():
        assert sha(path) == digest, path
    assert (R/'evaluation_summary.json').is_file()
    for profile in ('short6', 'short6_relaxed'):
        for case in (f'data{i}' for i in range(1, 7)):
            result = json.loads((R/profile/case/'summary.json').read_text())
            assert result['status'] == 'complete' and result['reference_used'] is False
            for name, digest in result['output_hashes'].items():
                assert sha(R/profile/case/name) == digest, name
    launcher = P/'run_motion_v29.sh'
    if D.exists() or launcher.exists():
        raise FileExistsError('Preserve existing version or launcher')
    files = sorted(path for path in HERE.iterdir() if path.is_file()
                   and path.suffix in ('.py', '.md', '.sh', '.json'))
    D.mkdir()
    for path in files:
        shutil.copyfile(path, D/path.name)
        assert sha(path) == sha(D/path.name)
    shutil.copyfile(HERE/launcher.name, launcher)
    launcher.chmod(0o755)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), installed=str(D),
        files={path.name:sha(D/path.name) for path in files},
        launcher=str(launcher), launcher_sha256=sha(launcher),
        protocol_sha256=sha(R/'protocol_before_run.json'),
        previous_entries=protocol['protected_entries'],
        status='Optional 6s coverage modes; full six-video comparison disclosed; not an accuracy promotion or real-time claim.')
    (D/'installation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(installed=str(D), launcher=str(launcher), files=len(files)), ensure_ascii=False))

if __name__ == '__main__':
    main()

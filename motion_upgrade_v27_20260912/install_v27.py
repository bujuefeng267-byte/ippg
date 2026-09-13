"""Copy a verified version into the project without overwriting previous entries."""
from pathlib import Path
import json,hashlib,shutil

HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别')
ROOT=P/'results/data1_6_v27_20260912'
DEST=P/'motion_upgrade_v27_20260912'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    fixed=json.loads((ROOT/'protocol_before_inference.json').read_text())
    for name,digest in fixed['source_hashes'].items():assert sha(HERE/name)==digest
    for path,digest in fixed['protected_entries'].items():assert sha(path)==digest
    assert all((ROOT/name/'evaluation_summary.json').exists() for name in fixed['variants'])
    assert (HERE/'analyze_video_v27.py').is_file()
    assert not DEST.exists(),'Preserve an existing installed version'
    launcher=P/'run_motion_v27_experimental.sh'
    assert not launcher.exists(),'Preserve an existing launcher'
    files=sorted(p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.sh'))
    DEST.mkdir()
    for source in files:
        shutil.copyfile(source,DEST/source.name)
        assert sha(source)==sha(DEST/source.name)
    shutil.copyfile(HERE/launcher.name,launcher);launcher.chmod(0o755)
    release=dict(source=str(HERE),installed=str(DEST),reference_protocol=str(ROOT/'protocol_before_inference.json'),
        protocol_sha256=sha(ROOT/'protocol_before_inference.json'),
        files={p.name:sha(DEST/p.name) for p in files},launcher=str(launcher),launcher_sha256=sha(launcher),
        prior_entries_preserved=fixed['protected_entries'])
    (DEST/'installation_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2))
    for path,digest in fixed['protected_entries'].items():assert sha(path)==digest
    print(json.dumps(release,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

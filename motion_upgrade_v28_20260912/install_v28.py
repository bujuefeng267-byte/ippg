"""Install the evaluated V28 repair separately and preserve prior versions."""
from pathlib import Path
import hashlib,json,shutil

HERE=Path(__file__).resolve().parent
P=Path('/home/fengbujue/项目/rppg识别')
R=P/'results/data1_6_v28_20260912'
D=P/'motion_upgrade_v28_20260912'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    protocol=json.loads((R/'protocol_before_run.json').read_text())
    score=json.loads((R/'direct_guard/evaluation_summary.json').read_text())
    assert score['promotion_pass'] is True
    for name,h in protocol['source_hashes'].items():assert sha(HERE/name)==h
    for path,h in protocol['protected_entries'].items():assert sha(path)==h
    qa=json.loads((R/'qa_evaluation_v28.json').read_text());assert qa['passed'] is True
    assert (HERE/'analyze_video_v28.py').is_file()
    assert (R/'final_report/report.md').is_file()
    launcher=P/'run_motion_v28.sh'
    if D.exists() or launcher.exists():raise FileExistsError('Preserve existing version/launcher')
    files=sorted(p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.sh'))
    D.mkdir()
    for source in files:
        shutil.copyfile(source,D/source.name);assert sha(source)==sha(D/source.name)
    shutil.copyfile(HERE/launcher.name,launcher);launcher.chmod(0o755)
    release=dict(source=str(HERE),installed=str(D),
        frozen_protocol=str(R/'protocol_before_run.json'),protocol_sha256=sha(R/'protocol_before_run.json'),
        files={p.name:sha(D/p.name) for p in files},launcher=str(launcher),launcher_sha256=sha(launcher),
        previous_entries=protocol['protected_entries'],
        status='Passed original development-set promotion guards versus V25; requires held-out validation; near-all ±5 bpm target not achieved.')
    (D/'installation_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2))
    for path,h in protocol['protected_entries'].items():assert sha(path)==h
    print(json.dumps(dict(installed=str(D),launcher=str(launcher),files=len(files)),ensure_ascii=False))

if __name__=='__main__':main()

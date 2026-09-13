"""Install immutable validated V25 source and new entrypoints; preserve old ones."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, subprocess

HERE=Path(__file__).resolve().parent
PROJECT=Path('/home/fengbujue/项目/rppg识别').resolve()
ROOT=PROJECT/'results/data1_6_v25_20260911'
DEST=PROJECT/'motion_upgrade_v25_20260911'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

frozen=read(ROOT/'protocol_before_validation.json')['source_hashes']
assert read(ROOT/'stage4_preserve_waveform/stage_evaluation.json')['adoption_pass']
for name in ['qa_protocol_integrity_final.json','qa_v25_evidence.json']:
    assert read(ROOT/name)['passed'],name
for name,value in frozen.items():assert sha(HERE/name)==value,name
assert DEST.resolve().parent==PROJECT
if DEST.exists():
    assert not (DEST/'installation_manifest.json').exists(),'Do not overwrite completed installation'
    assert not (PROJECT/'run_motion_v25.sh').exists(),'Do not modify an active installation'
    for name,value in frozen.items():assert sha(DEST/name)==value,'Unexpected partial installation change: '+name
files=list(frozen)+[p.name for p in HERE.glob('test_*.py')]+[
    'development_protocol.json','run_stages.py','evaluate_stages.py','run_preserved_waveform.py',
    'qa_protocol_integrity.py','qa_v25_evidence.py','qa_recompute_waveform_hr_v24.py',
    'render_v25_comparison.py','build_v25_report.py','deploy_v25_release.py','README_V25.md']
assert all((HERE/f).is_file() for f in files)
DEST.mkdir(exist_ok=True)
for name in files:shutil.copy2(HERE/name,DEST/name)
shutil.copytree(HERE/'test_fixtures',DEST/'test_fixtures',dirs_exist_ok=True)
shutil.copy2(HERE/'README_V25.md',DEST/'README.md')
for name,value in frozen.items():assert sha(DEST/name)==value,name
tests=subprocess.run([str(PROJECT/'.venv/bin/python'),'-B','-m','unittest','discover','-s',str(DEST),'-p','test_*.py'],capture_output=True,text=True)
(ROOT/'installed_unit_tests.log').write_text(tests.stdout+tests.stderr)
assert tests.returncode==0,tests.stdout+tests.stderr
prefix='''#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
'''
launchers={
    'run_motion_v25.sh': '--motion-evidence legacy --hr-mode evidence --routing-mode legacy',
    'run_motion_v25_all_experimental.sh': '--motion-evidence reliable --hr-mode evidence --routing-mode evidence'}
for name,flags in launchers.items():
    target=PROJECT/name
    assert not target.exists(),name
    target.write_text(prefix+'exec "$project_dir/.venv/bin/python" -B "$project_dir/motion_upgrade_v25_20260911/analyze_motion_v2.py" --pixel-mode tracking_screened --algorithm-mode guarded_fusion --max-gap 0.10 '+flags+' "$@"\n')
    target.chmod(0o755)
readme=PROJECT/'README.md';before=readme.read_bytes()
backup=DEST/'project_README_before_v25.md';backup.write_bytes(before)
addition='''

## 2026-09-11：V2.5 保留覆盖的心率判断升级

运动研究的推荐入口为 `./run_motion_v25.sh VIDEO --output NEW_DIRECTORY`：保留 V2.4 像素跟踪、局部异常筛选、失败回退和实际波形，启用带运动证据的多候选心率与轨迹重获。

六段开发录像的合并 MAE 39.16 → 25.20 bpm，R5 24.27% → 30.10%；每段波形、HR 覆盖率均保持不变。data1 未改善，data4 的 ±5 bpm 窗数 46 → 45，data3/5/6 仍存在明显误差。参考沿用估计时间同步；本次不是独立泛化验证，完整链路仍为离线处理。

三步全部开启的 `run_motion_v25_all_experimental.sh` 在本批覆盖较差，仅作实验。原 `run.sh` 与所有历史版本均保留。

[V2.5 运行说明](motion_upgrade_v25_20260911/README.md) · [逐步修改与六视频指标变化](results/data1_6_v25_20260911/V25代码优化与六视频指标变化.md) · [六组 PPG/心率折线图](results/data1_6_v25_20260911/presentation/all_six_ppg_hr.png)
'''
assert b'run_motion_v25.sh' not in before
readme.write_bytes(before+addition.encode('utf-8'))
manifest=dict(installed_utc=datetime.now(timezone.utc).isoformat(),directory=str(DEST),
    inference_source_hashes=frozen,installed_files={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file()},
    launchers={name:sha(PROJECT/name) for name in launchers},
    project_readme_before_sha256=sha(backup),project_readme_after_sha256=sha(readme),
    selected_development_stage='stage4_preserve_waveform',
    recommendation_scope='Six-clip development regression; retained original default run.sh',
    unit_tests_returncode=tests.returncode,unit_test_log_sha256=sha(ROOT/'installed_unit_tests.log'))
(DEST/'installation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(ROOT/'installation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(dict(installed=str(DEST),new_launcher=str(PROJECT/'run_motion_v25.sh'),tests=tests.stderr.strip()),ensure_ascii=False))

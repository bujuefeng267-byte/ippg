"""Record the user's choice of the evaluated V28 accuracy baseline."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

P = Path('/home/fengbujue/项目/rppg识别')
D = P/'motion_upgrade_v28_20260912'
R = P/'results/data1_6_v28_20260912'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    manifest = json.loads((D/'installation_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(D/name) == digest, name
    assert sha(P/'run_motion_v28.sh') == manifest['launcher_sha256']
    assert json.loads((R/'final_verification.json').read_text())['passed'] is True
    launcher = P/'run_recommended.sh'
    content = '#!/usr/bin/env bash\nset -euo pipefail\nPROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nexec "$PROJECT_DIR/run_motion_v28.sh" "$@"\n'
    if launcher.exists() and launcher.read_text() != content:
        raise FileExistsError('Inspect existing recommended launcher before changing it')
    launcher.write_text(content)
    launcher.chmod(0o755)
    note = P/'当前推荐版本.md'
    markdown = '''# 当前推荐版本：V28

用户选择继续使用精度较高的V28，保留10秒窗口和原质量门槛。V29只作为短窗口连续性试验，后续精度改进均以V28为比较基线。

```bash
cd /home/fengbujue/项目/rppg识别
./run_recommended.sh --video /完整路径/视频.mp4 --out results/新视频_v28
```

该入口直接调用`run_motion_v28.sh`。原始`run.sh`仍是早期POS/CHROM示例入口，运行当前推荐版本请使用上面的命令。

六视频已有开发集结果：MAE 17.42 bpm，±5 bpm达标率43.46%，心率覆盖84.14%。达标率以有输出且有参考的窗口为分母。这些结果沿用估计时间对齐，不能视为新视频表现保证。

[V28完整说明](<//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/V28运动心率改进说明.md>)
'''
    if note.exists() and note.read_text() != markdown:
        raise FileExistsError('Inspect existing recommendation note before replacing it')
    note.write_text(markdown)
    receipt = dict(version='V28', selected_utc=datetime.now(timezone.utc).isoformat(),
        reason='User requested returning to the higher-accuracy baseline after V29 coverage experiment.',
        launcher=str(launcher), launcher_sha256=sha(launcher),
        target_launcher_sha256=sha(P/'run_motion_v28.sh'),
        inference_sha256=sha(D/'analyze_video_v28.py'),
        evaluation_sha256=sha(R/'direct_guard/evaluation_summary.json'))
    out = P/'recommended_version.json'
    if out.exists():
        raise FileExistsError('Preserve prior recommendation receipt')
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == '__main__':
    main()

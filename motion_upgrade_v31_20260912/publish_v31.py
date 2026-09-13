"""Install auditable experimental sources; keep the recommended V28 unchanged."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
R = P/'results/data1_6_v31_20260912'
DEST = P/'motion_upgrade_v31_20260912'
REPORT = P/'V31保护机制与六视频结果_20260912.md'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link(path, title): return f'[{title}](<//wsl.localhost/Ubuntu{path}>)'


def main():
    evaluation = json.loads((R/'evaluation_summary.json').read_text())
    qa = json.loads((R/'qa_evaluation_v31_completed.json').read_text())
    assert qa['passed'] is True
    for group in ('input_hashes', 'evaluation_output_hashes'):
        for path, expected in evaluation[group].items(): assert sha(path) == expected, path
    protocol = json.loads((R/'freeze_inference_v31.json').read_text())
    for path, expected in protocol['protected_entries'].items(): assert sha(path) == expected, path
    assert not any(v['promotion_pass'] for v in evaluation['variants'].values())
    assert json.loads((P/'recommended_version.json').read_text())['version'] == 'V28'
    assert not DEST.exists() and not REPORT.exists()
    DEST.mkdir()
    sources = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.suffix in ('.py','.md'):
            target = DEST/path.name
            shutil.copyfile(path,target)
            assert sha(path) == sha(target)
            sources[path.name] = sha(target)
    baseline = next(iter(evaluation['variants'].values()))['baseline_V28']
    labels = {'protected_cdf_motion':'V31：运动证据准入', 'protected_cdf_consensus':'V31：增加原区域主峰一致性准入'}
    lines = ['# V31 保护机制与六视频结果', '',
        '**保护机制已实现，但本轮没有精度增益。当前推荐版本继续使用 V28。**', '',
        '这次将完整原 V28 作为主路径，CDF 作为独立候选，修复“新预处理把保底分支也改掉”的结构问题。两个固定准入方案都跑完六视频；最终均撤销了局部替换，保留原 V28 波形与心率。因此不能把这些不变的结果称为准确率升级。', '',
        '## 指标是否上涨', '',
        '| 方案 | MAE ↓ | RMSE ↓ | 输出中 ±5 bpm 比例 ↑ | 全部参考窗中 ±5 比例 ↑ | 心率覆盖 ↑ | 波形时间覆盖 ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, item in [('原 V28', baseline['pooled'])]+[(labels[k],v['pooled']) for k,v in evaluation['variants'].items()]:
        lines.append(f"| {name} | {item['MAE_bpm']:.2f} bpm | {item['RMSE_bpm']:.2f} bpm | {item['P5_valid_pct']:.2f}% | {item['R5_all_reference_pct']:.2f}% | {item['HR_coverage_pct']:.2f}% | {item['waveform_time_coverage_pct']:.2f}% |")
    lines += ['', '- 同一组原 309 个十秒窗口，保持 260 个心率输出、113 个 ±5 bpm 达标窗口。',
        '- V28 原来正确的窗口被改错或丢失：**0**。新增 ≥20 bpm 大误差窗口：**0**。新增 >12 bpm 大跳次数：**0**。',
        '- 本轮两方案的最终波形与心率均与 V28 相同，所有精度和覆盖率增幅为 **0**；旧问题也仍然保留。', '',
        '| 视频 | V28 / V31 MAE（bpm） | V28 / V31 ±5 比例 | V28 / V31 心率覆盖率 |',
        '|---|---:|---:|---:|']
    for row in baseline['cases']:
        lines.append(f"| {row['case']} | {row['MAE_bpm']:.2f} | {row['P5_valid_pct']:.2f}% | {row['hr_output_coverage_pct']:.2f}% |")
    lines += ['', '以上是已看过的开发视频、原估计 Polar 对齐下的结果。波形覆盖率按时间加权；没有同步接触式 PPG 波形真值，不能据此声称形态或 SNR 提高。', '',
        '## 已完成的代码修改', '',
        '1. **完整原 V28 保底。** 新候选不会改写原 RGB 缓存或原 V28 分支；两路先分别计算。',
        '2. **波形保护。** 每个未准入的完整十秒窗口，其覆盖到的全部原 V28 样本、缺失及观测标记均受保护。新候选只能在剩余允许区域中进行 Hann 渐变混合，权重不能泄漏到保护区。',
        '3. **心率保护。** 保存最终波形后重新计算频谱证据；未准入窗口在确认原频率仍有实际谱峰支持后，约束动态规划保留原 V28 心率。不会直接复制不受实测信号支持的数字。',
        '4. **异常撤销。** 如果原本可输出的窗口失去质量门支持，或大跳次数增多、最大跳幅超过 max(12 bpm, 原最大跳幅)，撤销涉及的准入并重新计算，必要时完全回退。参考心率不参与这一过程。',
        '5. **无实际修改就继续保护。** 符合候选条件但没有可修改样本的窗口也会撤销准入，避免波形没改而心率跟踪状态被释放。', '',
        '## 为什么没有新增精度收益', '',
        '| 视频与方案 | 初步有证据的窗口 | 最终实际修改窗口 | 原因 |',
        '|---|---:|---:|---|',
        '| data5：两方案 | 8 | 0 | 连续候选不足以在重叠保护中留下可改样本 |',
        '| data6：区域一致性方案 | 2 | 0 | 孤立短段完全被相邻受保护窗口覆盖 |',
        '| 其余视频 | 0 | 0 | 新旧差异或原始区域/运动证据不足，保留 V28 |', '',
        '**十秒窗每秒滑动一次，相邻窗重叠九秒。** data5 的初步准入位于图上约 7–14 秒窗中心：原 V28 为 182 bpm，候选约 91–93 bpm，得到直接运动风险下降及至少两个原物理 ROI 支持。但若左右未准入窗口的整段波形必须精确保留，这八个窗口覆盖到的样本都会同时落入受保护窗口，没有可编辑核心。', '',
        '这揭示了当前结构的限制：**对一条全局波形做短片段修正，同时要求相邻所有重叠窗口逐样本完全不变，可能根本没有修改空间。** 本轮保护是严格的，保住旧结果的同时也挡住了短段改善，不能包装成已经解决精度问题。', '',
        '也不能直接放宽主峰一致性门槛。data6 的两个候选约为 190 bpm，虽被多个原区域主峰支持，但候选直接运动风险约 0.84–0.91，旧 93 bpm 约为 0.08–0.09；区域一致不等于脉搏正确。该情况没有进入最终输出。', '',
        '## 后续需要解决的具体问题', '',
        '- 先明确如何组织独立窗口信号，才能修正短段且不通过全局混合影响相邻结果。若采用独立窗口波形，必须单独保存并能从它重现心率，不能只换心率数字。',
        '- 区域一致性与强运动重合发生冲突时，需要更可靠的候选否决证据。',
        '- 继续把原 V28 正确窗口损伤、新增大误差、开头二十秒错误、大跳及全部覆盖率列入回归；本轮没有根据参考逐视频选方案或调低门槛。', '',
        '## 测试与保存位置', '',
        '本轮 44 项合成/集成测试通过，另有独立 CSV 复算及源文件哈希核验。两个方案都通过保护与不退步检查，但未通过预先规定的实质提升要求（整体 MAE 至少下降 1 bpm 或全部参考窗 ±5 比例增加 1 个百分点），所以不替换推荐版本。', '',
        '- '+link(P/'run_recommended.sh','当前推荐入口：原 V28'),
        '- '+link(DEST,'V31 实验源码与测试'),
        '- '+link(DEST/'protected_waveform_router.py','波形保护与心率约束'),
        '- '+link(DEST/'proposal_evidence_v31.py','两种固定候选准入规则'),
        '- '+link(R/'evaluation_summary.json','完整六视频比较'),
        '- '+link(R/'qa_evaluation_v31_completed.json','独立核验通过记录'),
        '- 每个 `protected_cdf_* / data1–data6` 目录均含实际 `waveform.csv`、`heart_rate.csv`、`proposal_evidence.csv`、`routing_decisions.csv`、`preservation_audit.json` 和逐窗参考比较。', '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    current_note = P/'当前推荐版本.md'
    if current_note.exists():
        text = current_note.read_text(encoding='utf-8')
        marker = 'V31保护机制与六视频结果_20260912.md'
        if marker not in text:
            current_note.write_text(text.rstrip()+'\n\nV31 两个保护实验与 V28 的最终波形、心率和指标完全相同，未获得新精度收益；推荐继续使用 V28。'+link(REPORT,'查看 V31 回归报告')+'\n',encoding='utf-8')
    readme = '# V31 实验代码\n\n未通过精度提升条件，推荐入口仍为原 V28。\n\n'+link(REPORT,'阅读六视频结果与结构限制')+'\n\n'
    if (DEST/'analyze_v28_result_v31.py').exists():
        readme += ('对已经完成的 V28 输出进行独立候选实验：\n\n```bash\n'
            './run_motion_v31_experimental.sh --v28-output results/原V28输出 --out results/新V31实验 --mode direct_motion\n'
            '```\n\nV28 输出应包含 frame_trace.csv 和同名 JSON；旧批量输出可额外传 --trace-cache 指向其原缓存。'
            '这是实验入口，不能据本次结果声称提高精度。\n')
        launcher = P/'run_motion_v31_experimental.sh'
        assert not launcher.exists()
        launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'exec "$PROJECT_DIR/.venv/bin/python" -B "$PROJECT_DIR/motion_upgrade_v31_20260912/analyze_v28_result_v31.py" "$@"\n',encoding='utf-8')
        launcher.chmod(0o755)
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    for path, expected in protocol['protected_entries'].items(): assert sha(path) == expected, path
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), installed_sources=sources,
        report_path=str(REPORT),report_sha256=sha(REPORT), recommended_version='V28', promoted=False,
        inference_protocol_sha256=sha(R/'freeze_inference_v31.json'),
        evaluation_sha256=sha(R/'evaluation_summary.json'), completed_QA_sha256=sha(R/'qa_evaluation_v31_completed.json'),
        protected_entries=protocol['protected_entries'])
    (DEST/'installation_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (R/'publication_receipt.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(REPORT),'code':str(DEST),'recommended':'V28','source_files':len(sources)},ensure_ascii=False))


if __name__ == '__main__': main()

"""Publish the completed, independently evaluated V28 accuracy experiments."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
ROOT = P/'results/data1_6_v30_20260912'
REPORT = P/'V28精度排查与改进实验_20260912.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link(path, title):
    return f'[{title}](<//wsl.localhost/Ubuntu{path}>)'


def publish():
    assert not REPORT.exists(), 'Preserve published report'
    summary_path = ROOT/'evaluation_summary.json'
    summary = json.loads(summary_path.read_text())
    for group in ('input_hashes', 'evaluation_output_hashes'):
        for path, expected in summary[group].items():
            assert sha(path) == expected, path
    variants = summary['variants']
    base = variants['motion_full_spectrum']['baseline_V28']
    assert not any(r['promotion_pass'] for r in variants.values()), 'Reconsider report decision if a variant passed'
    recommended = json.loads((P/'recommended_version.json').read_text())
    assert recommended['version'] == 'V28'
    archive = ROOT/'source_snapshot'
    archive.mkdir(exist_ok=False)
    archived = {}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.md', '.json', '.csv'):
            target = archive/path.name
            shutil.copyfile(path, target)
            assert sha(path) == sha(target)
            archived[str(target)] = sha(target)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), sharey=True)
    handles = None
    for axis, row in zip(axes.flat, variants['cdf_v28']['cases']):
        case = row['case']
        paired = pd.read_csv(ROOT/'cdf_v28'/case/'evaluation/paired_windows.csv')
        t = paired.time_s.to_numpy(float)
        reference = paired.reference_bpm.to_numpy(float)
        axis.fill_between(t, reference-5, reference+5, color='#adb5bd', alpha=.2)
        h0, = axis.plot(t, reference, '--', color='#20252b', lw=1.6,
                       label='Polar reference (estimated alignment)')
        h1, = axis.plot(t, paired.V28_bpm.where(paired.V28_accepted), color='#2864bd', lw=1.9,
                       label='Recommended V28')
        h2, = axis.plot(t, paired.estimated_bpm.where(paired.accepted), color='#cb642f', lw=1.6,
                       label='CDF experiment')
        old = next(r for r in base['cases'] if r['case'] == case)
        axis.set_title(f"{case}  |  MAE: V28 {old['MAE_bpm']:.1f}, CDF {row['MAE_bpm']:.1f} bpm", pad=9)
        axis.set_ylim(35, 215)
        axis.set_xlim(0, row['waveform_frames']/row['fps'])
        axis.grid(alpha=.2)
        axis.set_xlabel('Video time / window center (s)')
        axis.set_ylabel('Heart rate (bpm)')
        handles = [h0, h1, h2]
    fig.suptitle('Six-video accuracy check: V28 retained', fontsize=18, y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.935), ncol=3, frameon=False)
    fig.text(.5, .025, 'Grey band: reference +/-5 bpm. Missing outputs stay blank. 10 s offline windows; no reference-guided tuning.\n'
             'Full-spectrum motion normalization gives the same HR values as V28 and is omitted as an overlapping line.',
             ha='center', va='bottom', fontsize=10, color='#505962')
    fig.subplots_adjust(left=.065, right=.98, top=.845, bottom=.135, wspace=.18, hspace=.38)
    figure = ROOT/'V28_CDF_six_video_heart_rate.png'
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    labels = {'motion_full_spectrum':'运动评分归一化修正', 'cdf_v28':'CDF 颜色滤波＋V28'}
    lines = ['# V28 精度排查与改进实验', '',
        '继续使用 **V28**。本轮完成两个独立、固定配置的实验；它们均未通过“提高精度并保留原有优势”的全部检查，未替换推荐算法。', '',
        '## 当前使用入口', '',
        '推荐入口：'+link(P/'run_recommended.sh','run_recommended.sh')+'，实际调用原 V28。已核对 V28 安装文件哈希并运行入口 `--help`。', '',
        '```bash', './run_recommended.sh --video /完整路径/视频.mp4 --out results/新视频_v28', '```', '',
        '原 `run.sh` 是较早的 POS/CHROM 演示入口；建议使用上述明确绑定版本的入口。V29 短窗实验及所有历史输出保留。', '',
        '## 两个实验的实测变化', '',
        '| 方案 | 心率 MAE ↓ | RMSE ↓ | 已输出心率中 ±5 bpm 比例 ↑ | 全部参考窗口中 ±5 bpm 比例 ↑ | 心率输出覆盖率 ↑ | 波形时间覆盖率 ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, metrics in [('原 V28',base['pooled'])]+[(labels[k],v['pooled']) for k,v in variants.items()]:
        lines.append(f"| {name} | {metrics['MAE_bpm']:.2f} bpm | {metrics['RMSE_bpm']:.2f} bpm | {metrics['P5_valid_pct']:.2f}% | {metrics['R5_all_reference_pct']:.2f}% | {metrics['HR_coverage_pct']:.2f}% | {metrics['waveform_time_coverage_pct']:.2f}% |")
    cdf = variants['cdf_v28']
    lines += ['', '比较使用同一组 data1–data6、原 309 个十秒窗口和原参考对齐。心率准确比例只在有输出的窗口中计算，必须与覆盖率一起看；不把未输出的时间填成 0。波形覆盖率按各视频的真实秒数加权，避免 180 fps 视频被按帧数放大。', '',
        '运动评分修正在六段视频的心率数值、接受掩码、波形文件上均与原 V28 一致，本轮增益为 0。', '',
        '| 视频 | V28 → CDF MAE（bpm） | V28 → CDF ±5 比例 | V28 → CDF 心率覆盖率 |',
        '|---|---:|---:|---:|']
    for row in cdf['cases']:
        old = next(r for r in base['cases'] if r['case'] == row['case'])
        lines.append(f"| {row['case']} | {old['MAE_bpm']:.2f} → {row['MAE_bpm']:.2f} | {old['P5_valid_pct']:.2f}% → {row['P5_valid_pct']:.2f}% | {old['hr_output_coverage_pct']:.2f}% → {row['hr_output_coverage_pct']:.2f}% |")
    g = cdf['pooled_coverage_groups']
    lines += ['', 'CDF 的 data5 改善明显，但 data2、data6 的误差增大，data3 心率覆盖率下降 23.08 个百分点；因此不能作为全部视频统一使用的替代版本。原 V28 的优势需要保留。', '',
        f"CDF 与 V28 共同输出的 {g['common_new']['Nvalid']} 个窗口中，MAE 为 {g['common_V28']['MAE_bpm']:.2f} → {g['common_new']['MAE_bpm']:.2f} bpm。CDF 另外输出 {g['newly_covered']['Nvalid']} 个窗口，同时丢失 V28 原有的 {g['lost_V28']['Nvalid']} 个窗口。共同窗口也有改善，所以收益并非全部来自删去困难窗口；但新增四窗均超出 ±5 bpm，仍不能仅比较各自剩余窗口的平均误差。", '',
        link(figure,'查看六视频心率对照图')+'。灰色带为参考值 ±5 bpm；横轴为十秒窗中心，缺失位置断线，参考同步仍是估计。', '',
        '## 还有哪些值得改进的地方', '',
        '1. **优先保留局部区域里的有效候选，再决定如何融合。** 原 V28 的 147 个超出 ±5 bpm 的输出中，80 个在最终波形里没有参考附近的候选支持；这 80 个中有 74 个在融合前仍有至少两个物理 ROI 支持附近频率。说明分支选择和波形混合值得改进。这个判断使用参考值作事后诊断，不能作为可部署的准确率，也不证明该频率一定来自脉搏。', '',
        '2. **解决 data3 的半频误选，需要增加独立证据。** data3 的 59 个输出中，27 个接近参考心率的一半。周期相关性和二次谐波奖励会同时支持较低频率；但真实低心率也可能有强二次谐波，所以不能直接把心率乘二。后续可保留两条实际候选轨迹，以颜色变化方向、区域支持和直接运动证据区分。', '',
        '3. **单改平滑不是优先项。** data1、data3、data6 没有发现“局部最优候选已在 ±5 bpm 内，却被后续 DP 改错”的窗口。错误已在前面的候选强度或评分中出现。', '',
        '4. **运动评分归一化属于已验证的机制修正，本批数据没有增益。** 合成的 20 bpm 带外运动，其带内谱尾会被原归一化抬到约 0.80 风险；全谱归一化使其降到约 0.000028。真实带内 60/120 bpm 的运动风险基本不变。当前只在最终心率读出测试，未同时修改上游融合与路由。', '',
        '## 本轮具体改了什么', '',
        '- 运动实验：完整运动 PSD 峰值作为归一化分母，保持速度、可靠性、候选、周期/谐波评分、DP 和质量门槛。心率重新从保存的 V28 波形计算，波形逐字节不变。',
        '- CDF 实验：对原始和跟踪分支的三个 ROI RGB 做颜色频谱软滤波，再运行完整原 V28；固定十秒窗、一秒步长和 42–210 bpm。有效短段不足十秒、未覆盖尾段、异常滤波窗口保留原 RGB；不跨缺失插值。最终波形和覆盖率由程序重新得到。data1 约 55.48% 的已观测 ROI×分支样本实际经过 CDF，其余主要因短段回退；其他五段约为 98.71%–99.96%。这里是滤波应用比例，不是 PPG 覆盖率。',
        '- CDF 依据：[Wang 等，2017，Color-Distortion Filtering，作者原文](https://sstuijk.estue.nl/publications/fg17.pdf)。使用颜色方向降低亮度/反光扰动的影响；本项目的分段回退和时间窗是工程适配，论文结果不代表本地增益。', '',
        '## 验证与结论边界', '',
        '- 两套推理配置和评估规则均在新评分前冻结；全部十二次推理完成后统一读参考评分，未逐视频选方案、未根据参考改参数、未选择有利时间偏移。',
        '- 运动评分 5 项、CDF 9 项、评估器 7 项合成测试通过；另核验原 V28 指标可复现、保存波形支持每个输出心率、时间轴/NaN/输出掩码和文件哈希。',
        '- 替换规则：整体 MAE 下降、全部参考窗 ±5 比例至少增加 5 个百分点、共同输出 MAE 下降；各视频 MAE 退步不超过 3 bpm，准确比例与覆盖率下降不超过 3 个百分点。两实验均未全部通过。',
        '- 六段视频是反复检查过的开发数据，参考 Polar 对齐仍是估计；已固定报告 ±1/2/5 秒敏感性，未取最有利偏移。结果不能当作新拍视频上的泛化精度。',
        '- 当前参考是心率，不能证明 PPG 波形形态或信噪比提高；波形输出更多只表示覆盖增加。', '',
        '## 保存位置', '',
        '- '+link(ROOT/'evaluation_summary.json','完整比较、替换检查与文件哈希'),
        '- '+link(ROOT/'cdf_v28/metrics.csv','CDF 六视频指标表'),
        '- '+link(archive/'diagnosis_v28_candidates.md','V28 候选与评分详细诊断'),
        '- '+link(archive/'review_readout.md','运动评分与半频机制审查'),
        '- '+link(archive,'本轮实验源码与合成测试快照'),
        '- 每个方案的 `data1`–`data6` 文件夹含 `waveform.csv`、`heart_rate.csv` 和 `evaluation/paired_windows.csv`。CDF 另含明确标记的处理后 RGB 视图，原视频和原缓存未替换。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        selected_version='V28', promoted_variant=None,
        evaluation_sha256=sha(summary_path), source_snapshot=archived,
        report_path=str(REPORT), report_sha256=sha(REPORT),
        figure_path=str(figure), figure_sha256=sha(figure),
        recommendation_receipt_sha256=sha(P/'recommended_version.json'))
    (ROOT/'publication_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'report':str(REPORT),'figure':str(figure),'snapshot_files':len(archived)},ensure_ascii=False))


if __name__ == '__main__':
    publish()

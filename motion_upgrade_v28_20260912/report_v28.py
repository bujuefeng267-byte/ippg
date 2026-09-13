"""Export the completed V28 development evaluation without altering inference.

This report consumes saved evaluations only after explicit evaluation confirmation.
Matplotlib figures retain all six clips, failed-window NaNs and original clocks.
No model, threshold, alignment or per-video variant is selected by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


P = Path('/home/fengbujue/项目/rppg识别')
ROOT = P/'results/data1_6_v28_20260912'
R26 = P/'results/data1_6_v26_20260911'
CASES = tuple(f'data{i}' for i in range(1, 7))
ORDER = ('V25', 'V26_harmonic', 'V26_conservative', 'V28')
NAMES = {'V25': 'V25', 'V26_harmonic': 'V26 完整谐波',
         'V26_conservative': 'V26 保守融合', 'V28': 'V28 直接运动保护'}
COLORS = {'V25': '#7C838B', 'V26_harmonic': '#AD6830',
          'V26_conservative': '#AD6830', 'V28': '#176B9D', 'Reference': '#22272D'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def md_path(path):
    return '//wsl.localhost/Ubuntu' + str(Path(path))


def md_link(label, path):
    return f'[{label}](<{md_path(path)}>)'


def md_image(label, path):
    return f'![{label}](<{md_path(path)}>)'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def fmt(value):
    return '—' if value is None or not np.isfinite(float(value)) else f'{value:.2f}'


def chart_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'axes.edgecolor': '#BAC0C6',
        'axes.labelcolor': '#333940', 'xtick.color': '#555C64', 'ytick.color': '#555C64',
        'text.color': '#232930', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': '#E2E6E9', 'grid.linewidth': .7, 'axes.axisbelow': True,
        'savefig.facecolor': 'white', 'path.simplify': False})


def finish(fig, path, subtitle):
    fig.text(.06, .025, subtitle, fontsize=10, ha='left', va='bottom', color='#58626C')
    fig.savefig(path, dpi=170, bbox_inches=None)
    plt.close(fig)


def hr_values(paired, name):
    if name == 'V28':
        value, accepted = 'estimated_bpm', 'accepted'
    else:
        value, accepted = name+'_bpm', name+'_accepted'
    assert paired[accepted].notna().all() and paired[accepted].isin([True, False, 0, 1]).all()
    return np.where(paired[accepted].to_numpy(bool), paired[value].to_numpy(float), np.nan)


def plot_hr(ax, paired, versions, end, ylim):
    t, ref = paired.time_s.to_numpy(float), paired.reference_bpm.to_numpy(float)
    valid = paired.reference_valid.to_numpy(bool)
    ref = np.where(valid, ref, np.nan)
    ax.fill_between(t, ref-5, ref+5, color='#E1E3E5', alpha=.9, linewidth=0)
    styles = {'V25': '--', 'V26_conservative': '-.', 'V26_harmonic': '--', 'V28': '-'}
    for name in versions:
        y = hr_values(paired, name)
        ax.plot(t, y, color=COLORS[name], linewidth=1.45 if name != 'V28' else 1.8,
                linestyle=styles[name], zorder=3 if name == 'V28' else 2)
    ax.plot(t, ref, color=COLORS['Reference'], linewidth=1.65, zorder=4)
    ax.set(xlim=(0, end), ylim=ylim, xlabel='Time from video start (s)', ylabel='Heart rate (bpm)')
    ax.grid(axis='y')


def legend(versions):
    labels = {'V25': 'V25', 'V26_conservative': 'V26 conservative',
              'V26_harmonic': 'V26 full harmonic', 'V28': 'V28 direct guard'}
    styles = {'V25': '--', 'V26_conservative': '-.', 'V26_harmonic': '--', 'V28': '-'}
    return [Line2D([0], [0], color=COLORS['Reference'], lw=1.7, label='Polar reference'),
            *[Line2D([0], [0], color=COLORS[name], lw=1.7,
                     ls=styles[name], label=labels[name]) for name in versions]]


def generate(root=ROOT):
    out, folder = root/'final_report', root/'direct_guard'
    summary_path, qa_path = folder/'evaluation_summary.json', root/'qa_evaluation_v28.json'
    summary, qa = json.loads(summary_path.read_text()), json.loads(qa_path.read_text())
    assert qa['passed'] is True and not qa['errors'], 'Independent evaluation must pass first'
    assert summary['variant'] == 'direct_guard'
    assert tuple(r['case'] for r in summary['cases']) == CASES
    assert summary['pooled']['Nplanned'] == summary['pooled']['Nref'] == 309
    for name in ORDER[:-1]:
        assert tuple(r['case'] for r in summary['baselines'][name]['cases']) == CASES
    sources = {str(summary_path): sha(summary_path), str(qa_path): sha(qa_path),
               str(root/'protocol_before_run.json'): sha(root/'protocol_before_run.json')}
    # Verify the independent QA binds every plotted paired CSV and waveform.
    def read_csv(path):
        source_hash = sha(path)
        assert qa['input_hashes'][str(path)] == source_hash, 'Plot input changed after independent QA'
        sources[str(path)] = source_hash
        return pd.read_csv(path)
    paired, waves = {}, {}
    route_rows, shift_rows = [], []
    for case in CASES:
        paired[case] = read_csv(folder/case/'evaluation/paired_windows.csv')
        waves[case] = read_csv(folder/case/'waveform.csv')
        shift_rows.append(read_csv(folder/case/'evaluation/alignment_sensitivity.csv'))
        old_route_path = R26/'conservative_components'/case/'routing_decisions.csv'
        new_route_path = folder/case/'routing_decisions.csv'
        for path in (old_route_path, new_route_path):
            sources[str(path)] = sha(path)
        old_route, new_route = pd.read_csv(old_route_path), pd.read_csv(new_route_path)
        assert not (new_route.route_eligible & ~old_route.route_eligible).any()
        route_rows.append(dict(case=case,
            V26_eligible_windows=int(old_route.route_eligible.sum()),
            V28_eligible_windows=int(new_route.route_eligible.sum()),
            direct_motion_veto_windows=int((new_route.reason == 'direct_motion_evidence_insufficient').sum()),
            V28_candidate_positive_samples=int((waves[case].candidate_weight > 0).sum())))
    baselines = summary['baselines']
    pooled = {name: baselines[name]['pooled'] for name in ORDER[:-1]}
    pooled['V28'] = summary['pooled']
    cases = {name: {row['case']: row for row in baselines[name]['cases']} for name in ORDER[:-1]}
    cases['V28'] = {row['case']: row for row in summary['cases']}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(version=name, **pooled[name]) for name in ORDER]).to_csv(out/'four_version_pooled_metrics.csv', index=False)
    per_case = [dict(version=name, **cases[name][case]) for name in ORDER for case in CASES]
    pd.DataFrame(per_case).drop(columns=['status_counts', 'routing_reason_counts'], errors='ignore').to_csv(
        out/'four_version_case_metrics.csv', index=False)
    pd.DataFrame(route_rows).to_csv(out/'routing_comparison.csv', index=False)
    pd.concat([paired[case].assign(case=case) for case in CASES], ignore_index=True).to_csv(
        out/'six_video_hr_comparison.csv', index=False)
    sensitivity = pd.concat(shift_rows, ignore_index=True)
    sensitivity.to_csv(out/'alignment_sensitivity_all_cases.csv', index=False)
    shift_pooled = []
    for shift, group in sensitivity.groupby('shift_s', sort=True):
        n = int(group.Nvalid.sum())
        shift_pooled.append(dict(shift_s=int(shift), Nvalid=n, Nref=int(group.Nref.sum()),
            MAE_bpm=float((group.Nvalid*group.MAE_bpm).sum()/n),
            P5_valid_pct=float(100*group.Nwithin5.sum()/n),
            R5_all_reference_pct=float(100*group.Nwithin5.sum()/group.Nref.sum())))
    pd.DataFrame(shift_pooled).to_csv(out/'alignment_sensitivity_pooled.csv', index=False)
    chart_style()
    all_y = np.concatenate([hr_values(p, name) for p in paired.values() for name in ORDER] +
                           [p.reference_bpm.to_numpy(float) for p in paired.values()])
    ylim = (max(0, 20*np.floor(np.nanmin(all_y)/20)-20), 20*np.ceil(np.nanmax(all_y)/20)+20)
    for versions, title, filename in (
        (('V25', 'V26_conservative', 'V28'), 'Heart-rate comparison | V25, V26 conservative and V28',
         'six_video_hr_v25_v26conservative_v28.png'),
        (('V26_harmonic', 'V28'), 'Heart-rate comparison | Full V26 harmonic and V28',
         'six_video_hr_v26harmonic_v28.png')):
        fig, axes = plt.subplots(3, 2, figsize=(15, 11.6))
        fig.subplots_adjust(left=.07, right=.97, top=.865, bottom=.10, hspace=.43, wspace=.20)
        fig.suptitle(title, fontsize=17, y=.967, x=.06, ha='left', fontweight='semibold')
        fig.legend(handles=legend(versions), loc='upper left', bbox_to_anchor=(.055, .935),
                   frameon=False, ncol=len(versions)+1)
        for ax, case in zip(axes.flat, CASES):
            p, metric = paired[case], cases['V28'][case]
            end = metric['waveform_frames']/metric['fps']
            plot_hr(ax, p, versions, end, ylim)
            ax.set_title(f"{case}   |   V28 MAE {metric['MAE_bpm']:.2f} bpm; "
                         f"within 5 bpm {metric['Nwithin5']}/{metric['Nvalid']}", loc='left', pad=10)
        finish(fig, out/filename,
            '10 s windows / 1 s steps; gray band = reference +/-5 bpm, not uncertainty. '
            'Missing outputs remain gaps.\nFixed estimated alignment; six previously inspected development clips. '
            'Equal series overlap; all panels use the same heart-rate scale.')
    fig, axes = plt.subplots(3, 2, figsize=(15, 11.6))
    fig.subplots_adjust(left=.08, right=.97, top=.875, bottom=.105, hspace=.46, wspace=.22)
    fig.suptitle('V28 saved rPPG waveforms | All six videos', fontsize=17,
                 y=.965, x=.06, ha='left', fontweight='semibold')
    fig.text(.06, .915, 'Original saved samples; separate amplitude scales per video; no extra smoothing or filling.', fontsize=11)
    for ax, case in zip(axes.flat, CASES):
        wave, metric = waves[case], cases['V28'][case]
        ax.plot(wave.time_s, wave.base, color=COLORS['V28'], lw=.8)
        ax.set(xlim=(0, metric['waveform_frames']/metric['fps']), xlabel='Time from video start (s)',
               ylabel='Relative amplitude (a.u.)')
        ax.set_title(f"{case}   |   finite waveform coverage {metric['waveform_coverage_pct']:.2f}%", loc='left', pad=10)
        ax.grid(axis='y')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    finish(fig, out/'six_video_v28_ppg.png',
        'Finite coverage measures availability, not physiological correctness. '
        'Missing samples remain NaN and are not connected.\nThe saved convex mixture may include measured narrowband components; '
        'no waveform ground truth or morphology validation is available.')
    for case in CASES:
        metric, wave = cases['V28'][case], waves[case]
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.3), sharex=True)
        fig.subplots_adjust(left=.085, right=.97, top=.84, bottom=.14, hspace=.43)
        fig.suptitle(f'V28 rPPG and heart rate | {case}', x=.075, ha='left', y=.97,
                     fontsize=17, fontweight='semibold')
        fig.text(.075, .913, f"MAE {metric['MAE_bpm']:.2f} bpm   |   +/-5 bpm {metric['Nwithin5']}/{metric['Nvalid']} outputs   |   "
                 f"HR coverage {metric['hr_output_coverage_pct']:.2f}%", fontsize=11)
        axes[0].plot(wave.time_s, wave.base, color=COLORS['V28'], lw=.85)
        axes[0].set_ylabel('Relative amplitude (a.u.)')
        axes[0].set_title('Saved rPPG waveform', loc='left', pad=9)
        axes[0].grid(axis='y')
        axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        plot_hr(axes[1], paired[case], ('V28',), metric['waveform_frames']/metric['fps'], ylim)
        axes[1].set_title('Heart rate from 10 s windows / 1 s steps', loc='left', pad=9)
        axes[1].legend(handles=legend(('V28',)), loc='upper right', frameon=False, fontsize=9)
        finish(fig, out/f'{case}_v28_ppg_heart_rate.png',
            'Gray band = reference +/-5 bpm. Missing outputs remain gaps; fixed estimated alignment.\n'
            'Waveform availability does not prove morphology recovery; measured narrowband components may contribute.')
    new, old, conservative, harmonic = [pooled[k] for k in ('V28', 'V25', 'V26_conservative', 'V26_harmonic')]
    delta_mae = new['MAE_bpm']-old['MAE_bpm']
    intro = ('通过六视频开发集的既定升级条件。' if summary['promotion_pass'] else
             '未通过六视频开发集的既定升级条件。')
    text = [
        '# V28：保留 V26 保守融合收益，减少误替换', '',
        f"**{intro}** 相比 V25，总体 MAE 从 {old['MAE_bpm']:.2f} 降至 {new['MAE_bpm']:.2f} bpm，"
        f"下降 {-delta_mae:.2f} bpm（{-100*delta_mae/old['MAE_bpm']:.2f}%）；"
        f"±5 bpm 达标输出从 {old['Nwithin5']}/{old['Nvalid']} 增至 {new['Nwithin5']}/{new['Nvalid']}，"
        '六段视频的心率和波形覆盖率均保持 V25 水平。', '',
        '**本轮保留的是 V26 保守融合的收益，未保留完整谐波分支的全部优势。** '
        'data5 保留 19.44 bpm 的 MAE，完整谐波分支在该视频曾达到 3.66 bpm；'
        '完整谐波分支总体达标率、覆盖率和 RMSE 仍优于本轮。'
        '当前尚未达到“基本所有心率都在 ±5 bpm 内”，data1、data3、data6 仍需继续改进。', '',
        '## 四个固定版本的总体结果', '',
        table(['版本', 'MAE↓ bpm', 'RMSE↓ bpm', 'P5↑ %', 'R5↑ %', 'HR覆盖↑ %', '波形时间覆盖↑ %', '达标/输出'],
              [[NAMES[name], fmt(pooled[name]['MAE_bpm']), fmt(pooled[name]['RMSE_bpm']),
                fmt(pooled[name]['P5_valid_pct']), fmt(pooled[name]['R5_all_reference_pct']),
                fmt(pooled[name]['HR_coverage_pct']), fmt(pooled[name]['waveform_time_coverage_pct']),
                f"{pooled[name]['Nwithin5']}/{pooled[name]['Nvalid']}"] for name in ORDER]), '',
        'P5 = ±5 bpm 达标窗口数 / 有心率输出且有有效参考的窗口数。'
        'R5 = 达标窗口数 / 全部有参考的计划窗口数；缺失输出不算成功。'
        '心率覆盖率 = 输出窗口数 / 全部计划窗口数。本批四版本均有 309 个参考合格计划窗口。'
        '总体 MAE 按有效窗口汇总，未对六个视频 MAE 简单平均。', '',
        '波形时间覆盖率按每段有限样本数 / 原始 fps 累加，再除以总视频时长；'
        'data4 约 180 fps，其余约 30 fps，因此不能把总体帧覆盖率当成总体时间覆盖率。'
        f"V28 的总体帧覆盖率另为 {new['waveform_frame_coverage_pct']:.2f}%。"
        '波形覆盖表示有数值可输出，不等同于真实 PPG 形态恢复率。', '',
        '## 六段视频全部展示', '',
        table(['视频', 'V25 MAE', 'V26完整谐波 MAE', 'V26保守融合 MAE', 'V28 MAE'],
              [[case, *[fmt(cases[name][case]['MAE_bpm']) for name in ORDER]] for case in CASES]), '',
        '以上 MAE 单位均为 bpm。下表为最终 V28；所有指标均来自同一组保存输出，没有逐视频挑选最优版本。', '',
        table(['视频', 'P5 %', 'R5 %', 'HR覆盖 %', '波形覆盖 %', '达标/输出/计划'],
              [[case, fmt(cases['V28'][case]['P5_valid_pct']), fmt(cases['V28'][case]['R5_all_reference_pct']),
                fmt(cases['V28'][case]['hr_output_coverage_pct']), fmt(cases['V28'][case]['waveform_coverage_pct']),
                f"{cases['V28'][case]['Nwithin5']}/{cases['V28'][case]['Nvalid']}/{cases['V28'][case]['Nplanned']}"] for case in CASES]), '',
        'data1 恢复 V25 的 13.10 bpm，避免 V26 保守融合的 33.21 bpm 退步；'
        'data5 保留 V26 保守融合的 19.44 bpm，相比 V25 的 60.72 bpm 改善；'
        'data2、data3、data4、data6 的有效心率值和覆盖掩码与 V25 保持一致。'
        '因此本轮改善来自 data5 收益的保留和 data1 误替换的阻止，未解决 data3、data6 原有误差。', '',
        '## 代码只改变一个判断', '',
        'V26 保守融合原本使用旧心率处的综合运动风险（直接频率、双倍频、半频的加权最大值）'
        '决定是否替换波形。V28 在原判断全部通过后，还要求旧心率频率本身的直接运动风险达到 '
        '**同一个已有阈值 0.50**。未增加新阈值、未按视频调参，也未把参考心率输入推理。', '',
        '原规则继续保留：综合旧风险 ≥0.50、候选综合风险至少降低 0.30、至少两个不同物理面部区域、'
        '完整有限窗口、原 V25 缺失掩码、正 Hann 权重的凸混合，以及按实际贡献来源记录采样标记。'
        '新门只否决原有替换，不在原本拒绝的位置新增替换。最终心率从重新保存并读取的融合波形计算，'
        '未直接复制候选心率，也未用心率数值合成正弦波。', '',
        table(['视频', 'V26可替换窗口', 'V28可替换窗口', '直接运动证据不足而否决'],
              [[r['case'], r['V26_eligible_windows'], r['V28_eligible_windows'], r['direct_motion_veto_windows']] for r in route_rows]), '',
        '替换窗口使用完整 10 秒 Hann 权重叠加，因此“可替换窗口数”不等于改变后的心率窗口数或替换秒数。'
        '缺乏直接运动证据只说明本轮不应执行该替换，不能据此证明原心率正确。', '',
        '## 升级检查及仍未达到的目标', '',
        table(['事先约定的条件（相对 V25）', '结果'], [
            ['总体 MAE 下降', '通过' if summary['promotion_checks']['pooled_MAE'] else '未通过'],
            ['总体 R5 至少增加 5 个百分点', '通过' if summary['promotion_checks']['pooled_R5_gain'] else '未通过'],
            ['每段 MAE 增加不超过 3 bpm', '通过' if summary['promotion_checks']['each_MAE'] else '未通过'],
            ['每段 P5、R5 各下降不超过 3 个百分点', '通过' if summary['promotion_checks']['each_P5'] and summary['promotion_checks']['each_R5'] else '未通过'],
            ['每段 HR、波形覆盖率各下降不超过 3 个百分点', '通过' if summary['promotion_checks']['each_HR_coverage'] and summary['promotion_checks']['each_waveform_coverage'] else '未通过'],
            ['共同有效窗口 MAE 下降', '通过' if summary['promotion_checks']['common_MAE'] else '未通过']]), '',
        f"V28 与 V25 有相同的 {new['common_windows']} 个有效比较窗口，本次没有新增或丢失心率输出。"
        f"目前仅 {sum(r['target_P5_95_met'] for r in summary['cases'])}/6 段视频达到 P5≥95%。"
        '升级条件通过只支持继续开发使用；未证明已实现高准确率或对新视频同样有效。', '',
        '## 心率折线图', '',
        '主图比较 V25、V26 保守融合和 V28；黑线为 Polar 参考心率，灰带为参考 ±5 bpm，'
        '不是置信区间。各面板使用相同心率纵轴。多个版本重合时曲线会覆盖，完整数值保存在比较 CSV。', '',
        md_image('六视频心率：V25、V26保守融合、V28', out/'six_video_hr_v25_v26conservative_v28.png'), '',
        '另图完整保留 V26 谐波分支，便于同时查看本轮保住的结果和放弃的收益。', '',
        md_image('六视频心率：V26完整谐波、V28', out/'six_video_hr_v26harmonic_v28.png'), '',
        '## PPG 波形与逐视频输出', '',
        '下图直接绘制最终 waveform.csv 的 base 样本，不增加平滑、补零或跨缺失插值；NaN 显示为断线。'
        '各视频保留自己的幅度尺度，不用于跨视频比较脉搏振幅。'
        '保存波形为测得信号的凸混合，其中可以包含 V26 从实际光学信号中选出的窄带分量。'
        '窄带处理会影响形态，因此不能把外观更规则解释为生理 PPG 形态更准确。'
        '目前只有心率参考，没有同步 PPG 波形真值，未报告形态相关系数或 SNR 的改善。', '',
        md_image('六视频最终PPG波形', out/'six_video_v28_ppg.png'), '',
        table(['视频', '可直接分享的PPG＋心率图', '原始波形CSV', '原始心率CSV'],
              [[case, md_link('打开图', out/f'{case}_v28_ppg_heart_rate.png'),
                md_link('waveform.csv', folder/case/'waveform.csv'),
                md_link('heart_rate.csv', folder/case/'heart_rate.csv')] for case in CASES]), '',
        '## 参考对齐与适用范围', '',
        '参考为 Polar 设备约每秒推送的心率数值，不是本程序直接从 ECG 波形重算的心率。'
        '沿用原有 ZIP 内时间戳与实际视频时长得到的估计起点，没有硬件同步标记；'
        '采用同一套 10 秒窗口、约 1 秒步长与原始采样时钟。'
        'data4 以实际解码 10693 帧计算，而非容器声明的 10702 帧。', '',
        '这些视频此前已被多轮分析，属于开发集；重叠窗口不是独立样本。'
        '这轮只运行事前冻结的一个 V28 修改，未在看到新评分后调整阈值。'
        '仍应使用独立新拍、能确认同步的视频检验泛化效果。', '',
        '以下保留全部约定的参考平移敏感性结果，没有选择其中最优平移替代主结果。', '',
        table(['参考平移 s', '有效窗口', 'MAE bpm', 'P5 %', 'R5 %'],
              [[r['shift_s'], r['Nvalid'], fmt(r['MAE_bpm']), fmt(r['P5_valid_pct']), fmt(r['R5_all_reference_pct'])]
               for r in shift_pooled]), '',
        '## 文件与核验', '',
        f"独立评分核验通过，共检查 {qa['numeric_comparisons']} 个数值比较，"
        f"最大绝对差为 {qa['maximum_absolute_difference']:.3g}。"
        '核验包括固定参考、缺失分母、各视频指标和全部七个平移条件。'
        '推理输出已验证只采用真实保存波形、保留原 NaN 掩码，并从该波形独立重算心率。', '',
        '- '+md_link('正式评价 JSON', summary_path),
        '- '+md_link('独立核验 JSON', qa_path),
        '- '+md_link('四版本总体指标 CSV', out/'four_version_pooled_metrics.csv'),
        '- '+md_link('四版本逐视频指标 CSV', out/'four_version_case_metrics.csv'),
        '- '+md_link('六视频逐窗口心率比较 CSV', out/'six_video_hr_comparison.csv'),
        '- '+md_link('替换窗口统计 CSV', out/'routing_comparison.csv'),
        '- '+md_link('全部参考平移敏感性 CSV', out/'alignment_sensitivity_all_cases.csv'),
        '- '+md_link('事前冻结协议', root/'protocol_before_run.json'), '',
        '报告与图由 report_v28.py 根据已保存结果生成；报告不参与算法决策。', '',
    ]
    (out/'report.md').write_text('\n'.join(text), encoding='utf-8')
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        generator_sha256=sha(__file__), evaluation_confirmed=True,
        source_hashes=sources, figures_preserve_missingness=True,
        figures_show_all_six_cases=True, no_extra_waveform_filtering=True,
        scope='Previously inspected development clips; fixed estimated alignment',
        outputs={path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != 'report_manifest.json'})
    (out/'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(out/'report.md'), promotion_pass=summary['promotion_pass'],
                         pooled=new, figures=len(list(out.glob('*.png')))), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-confirmed', action='store_true')
    args = parser.parse_args()
    if not args.evaluation_confirmed:
        parser.error('Generate only after formal evaluation and independent QA are confirmed')
    generate()

"""Read-only V28 startup and missingness diagnosis; no human pulse reference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
R = P/'results/data1_6_v28_20260912/direct_guard'
V = P/'results/data1_6_v25_20260911/stage4_preserve_waveform'
ROIS = ('forehead', 'left_cheek', 'right_cheek')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(mask, fps):
    edge = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [dict(start_frame=int(a), stop_frame_exclusive=int(b), frames=int(b-a),
                 start_s=float(a/fps), stop_s_exclusive=float(b/fps), duration_s=float((b-a)/fps))
            for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def count_map(series):
    return {str(k): int(v) for k, v in series.value_counts().items()}


def finite_windows(wave, fps, frames):
    starts = np.arange(0, len(wave)-frames+1, round(fps), dtype=int)
    valid = np.array([np.isfinite(wave.base.iloc[a:a+frames]).all() for a in starts])
    first = int(starts[np.flatnonzero(valid)[0]]) if valid.any() else None
    return dict(window_frames=int(frames), effective_window_s=float(frames/fps),
        planned_windows=len(starts), fully_finite_windows=int(valid.sum()),
        fully_finite_fraction=float(valid.mean()) if len(valid) else None,
        first_window=None if first is None else dict(start_s=first/fps,
            center_s=(first+frames/2)/fps, end_s_exclusive=(first+frames)/fps),
        interpretation='Geometry of unchanged saved V28 samples only; no new HR, spectral quality, or reference scoring performed')


def main():
    inputs, cases = {}, []
    for name in (f'data{i}' for i in range(1, 7)):
        paths = [R/name/'waveform.csv', R/name/'heart_rate.csv', V/name/'frame_trace.json']
        for path in paths:
            inputs[str(path)] = sha(path)
        wave, hr = (pd.read_csv(path) for path in paths[:2])
        fps = float(json.loads(paths[2].read_text())['fps'])
        valid = np.isfinite(wave.base)
        first = hr.loc[hr.accepted].iloc[0]
        cases.append(dict(case=name, fps=fps, frames=len(wave), duration_s=len(wave)/fps,
            first_finite_waveform_sample_s=float(wave.time_s[valid].iloc[0]),
            first_accepted_hr_window=dict(start_s=float(first.window_start_s),
                center_s=float(first.time_s), end_s_exclusive=float(first.window_end_s)),
            waveform_coverage_fraction=float(valid.mean()), planned_hr_windows=len(hr),
            accepted_hr_windows=int(hr.accepted.sum()), hr_coverage_fraction=float(hr.accepted.mean()),
            waveform_missing_intervals=runs(~valid, fps), hr_status_counts=count_map(hr.status)))
    name = 'data6'
    extra_paths = [V/name/file for file in ('frame_trace.csv', 'roi_waveforms.csv',
        'fusion_proposals.csv', 'fusion_diagnostics.csv', 'branch_routing.csv',
        'baseline_branch_proposals.csv', 'tracked_branch_proposals.csv')]
    for path in extra_paths:
        inputs[str(path)] = sha(path)
    trace, roi, proposals, diag, routing, base, tracked = [pd.read_csv(p) for p in extra_paths]
    wave = pd.read_csv(R/name/'waveform.csv')
    fps = cases[-1]['fps']
    unique_diag = diag.drop_duplicates(['window_index', 'channel'])
    ambiguous = proposals.loc[proposals.proposal_status == 'ambiguous_consensus',
        ['time_s', 'window_start_s', 'window_end_s', 'quality_proxy', 'runner_up_ratio',
         'available_roi_count', 'roi_count']]
    detailed = dict(raw_rgb_missing=runs(~trace.rgb_valid, fps),
        fewer_than_two_raw_rois=runs(trace[[r+'_valid' for r in ROIS]].sum(axis=1) < 2, fps),
        source_counts=count_map(trace.source),
        per_roi={r: dict(raw_rgb_missing=runs(~trace[r+'_valid'], fps),
                         ppg_missing=runs(~np.isfinite(roi[r+'_pos']), fps)) for r in ROIS},
        fused_proposal_status_counts=count_map(proposals.proposal_status),
        branch_routing_reason_counts=count_map(routing.selection_reason),
        baseline_proposal_status_counts=count_map(base.status),
        tracked_proposal_status_counts=count_map(tracked.status),
        unique_window_channel_count=len(unique_diag),
        unique_window_channel_status_counts=count_map(unique_diag.channel_status),
        diagnostic_counts_are_first_failed_gate_not_exclusive_causal_ablation=True,
        ambiguous_fused_proposals=ambiguous.to_dict(orient='records'),
        usable_roi_quality_range=[float(unique_diag.loc[unique_diag.channel_status == 'usable', 'roi_quality'].min()),
                                 float(unique_diag.loc[unique_diag.channel_status == 'usable', 'roi_quality'].max())],
        unchanged_waveform_finite_window_counts=[finite_windows(wave, fps, frames) for frames in
            (round(10*fps), round(6*fps), int(np.ceil(6*fps)))])
    result = dict(reference_used=False, inference_changed=False,
        source_description='V28 final wave/HR and corresponding V25 preserved frontend/fusion diagnostics',
        intervals='Half-open frame-index intervals; exclusive end = stop frame / FPS',
        offline_timing='Stored waveform sample time and HR window center are retrospective video times, not measured real-time availability. Zero-phase filters, overlap-add and evidence DP use future samples/windows.',
        input_sha256=inputs, cases=cases, data6=detailed,
        selected_prospective_comparison=dict(standard_6s=dict(ambiguity_ratio=.8),
            relaxed_6s=dict(ambiguity_ratio=.9),
            unchanged=dict(min_rois=2, min_observed_fraction=.9, min_roi_quality=.2,
                max_gap_s=.1, filter_edge_s=1.6, hr_observed_fraction=.9,
                hr_max_interpolated_fraction=.1, hr_peak_concentration=.12),
            note='The two profiles were selected from failure mechanisms before new evaluation, not by reference accuracy. The unchanged-wave calculation is not a prediction of complete six-second pipeline accuracy or coverage.'))
    (HERE/'startup_missingness_v28.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# V28 起始时间和 data6 缺失原因诊断', '',
        '本诊断只读取已保存的程序输出、RGB/运动轨迹与融合诊断，没有读取参考心率，也没有修改模型或旧结果。', '',
        '## 六段视频从哪里开始有输出', '',
        '| 视频 | 第一波形点 | 第一心率窗口开始 | 心率图上的中心时间 | 窗口结束 |',
        '|---|---:|---:|---:|---:|']
    for row in cases:
        h = row['first_accepted_hr_window']
        lines.append(f"| {row['case']} | {row['first_finite_waveform_sample_s']:.2f} s | {h['start_s']:.2f} s | {h['center_s']:.2f} s | {h['end_s_exclusive']:.2f} s |")
    lines += ['', '只有 data6 的波形约到第 9 秒才出现；其他五段约从第 2 秒或第 4 秒开始。PPG 波形逐帧标时，心率值标在 10 秒窗口中心。例如 data6 图上第 14 秒的心率来自第 9–19 秒的样本。', '',
        '**这些是视频时间，不是实时等待时间。** 当前程序是离线处理：零相位滤波、重叠波形融合与心率路径搜索会使用未来数据。即使把窗口改成 6 秒，也不能据此宣称系统已经实现实时输出；窗口结束只能描述该局部估计所需样本的结束位置。', '',
        '## data6 的缺失来自哪里', '',
        '| 层次 | 已观察到的情况 |', '|---|---|',
        '| 原始 RGB | 三处区域共同缺失于约 0–1.37、1.60–6.47、51.07–51.60 秒 |',
        '| 每区域 PPG | 有效连续段需至少约 4 秒，两端各裁去 1.6 秒；缺失因此扩大到 0–8.07、49.47–53.20、70.00–71.60 秒 |',
        '| 融合波形 | 10 秒完整窗口、共识与时间网格后，缺失为 0–9、49–55、69–71.60 秒 |',
        '| 最终心率 | 62 个窗口中接受 36 个；拒绝的 26 个全部为 `gap_or_filter_edge` |', '',
        '早期 1.37–1.60 秒虽然短暂有 RGB，但不足持续信号处理所需长度。首次长段 RGB 从约 6.47 秒开始，再加 1.6 秒边缘裁剪，第一段区域 PPG 才能从约 8.07 秒开始。1 秒步长的完整融合窗口从第 9 秒开始。单独降低质量阈值不能补出这些缺失样本。', '',
        'data6 的融合结果有 34 个已接受提议、24 个区域不足、4 个共识模糊。底层两个分支共 744 个唯一“窗口×通道”诊断：453 个可用、216 个缺口/滤波边缘、72 个观察不足、3 个频谱过散；没有区域因 `poor_roi_quality` 被拒。可用区域的质量约为 0.996–1.000。诊断记录的是首先失败的门槛，这些计数不能当作各因素的独立因果贡献。', '',
        '因此，降低区域质量门槛 0.20，或最终心率的观察比例/频谱门槛，对当前 data6 的主要缺失没有直接证据。三处区域同时失效，允许单区域输出也无法补救这些时段。', '',
        '## 本轮保留的有限修改', '',
        '比较两个统一配置：6 秒窗口保持原标准，以及 6 秒窗口仅把共识模糊阈值从 0.80 放宽到 0.90。最少两个区域、观察比例 0.90、区域质量 0.20、短缺口上限 0.10 秒、滤波边缘以及最终心率门槛均保持不变。', '',
        '现有 data6 的四个模糊提议中心约为 59、62、63、65 秒；竞争候选/主候选分数比约为 0.984、0.838、0.819、0.859。0.90 允许一定竞争，但仍拒绝接近同分的 0.984。放宽后输出代表允许更模糊的解释，不能称为置信度提升。6 秒重新提取后的具体提议会改变，最终需要完整回归。', '',
        '只在现有 data6 波形上计算完整有限窗口数量，10 秒为 36/62；180 帧的约 6 秒窗口为 44/66。由于实际帧率是 30.00003 fps，180 帧略少于 6 秒；若沿用旧程序严格“至少 6 秒”的条件，需要 181 帧，实际为 6.0333 秒，此时为 42/66。这个计算只显示窗口缩短可减轻缺口对邻近窗口的影响，没有计算新心率，也不是新模型覆盖率或准确率结论。', '',
        '使用 181 帧时，现有波形首个完整窗口仍从约第 9 秒开始，中心约 12.02 秒，结束约 15.03 秒；相比旧窗口中心第 14 秒、结束第 19 秒，所需样本区间缩短。它不会让第 0–8.07 秒本来不存在的区域 PPG 提前出现。', '',
        '完整逐帧区间、未四舍五入的时间、各状态数量和输入校验和保存在同目录 `startup_missingness_v28.json`。']
    (HERE/'V28起始时间与data6缺失诊断.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(json=str(HERE/'startup_missingness_v28.json'),
        markdown=str(HERE/'V28起始时间与data6缺失诊断.md'), reference_used=False), ensure_ascii=False))


if __name__ == '__main__':
    main()

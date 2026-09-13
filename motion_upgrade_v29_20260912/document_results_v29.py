"""Write the code-change note from the completed, fixed V29 evaluation."""
from pathlib import Path
import hashlib
import json

P = Path('/home/fengbujue/项目/rppg识别')
R = P/'results/data1_6_v29_20260912'
HERE = Path(__file__).resolve().parent
PREFIX = '//wsl.localhost/Ubuntu/home/fengbujue/项目/rppg识别/'

def fmt(x, digits=2):
    return '无输出' if x is None else f'{x:.{digits}f}'

def link(label, relative):
    return f'[{label}](<{PREFIX}{relative}>)'

def main():
    score = json.loads((R/'evaluation_summary.json').read_text())
    old6 = next(x for x in score['baseline_V28']['cases'] if x['case'] == 'data6')
    profiles = [('V29：6秒原标准', 'short6'), ('V29：6秒放宽模糊筛选', 'short6_relaxed')]
    rows = [('V28：10秒', old6, old6['waveform_availability'], old6['hr_availability'])]
    for name, key in profiles:
        detail = score['profiles'][key]['data6_detail']
        rows.append((name, detail['native6s'], detail['availability']['waveform'], detail['availability']['native6s_hr']))
    lines = ['# V29：缩短分析窗口，检查输出连续性', '',
        '本轮将分析窗口从10秒缩为6秒，保留V28的像素跟踪、区域信号、谐波候选和直接运动证据保护。另提供轻度放宽模糊筛选的6秒模式，允许更接近的候选得分通过。两个模式在跑分前确定，六段视频使用相同规则。', '',
        '**使用建议：需要更短窗口时优先选`short6`。** 对data6，6秒原标准已把有波形时间从约54秒增加到56秒，心率从36/62个窗口增加到46/66个窗口。放宽模糊筛选没有再增加该视频的覆盖，却让原生6秒MAE从24.18变为25.19 bpm。两者的±5 bpm达标率均为0%，更多输出仍不能当作准确心率。', '',
        'V29作为可选短窗模式保存，不替换V28：六视频按共同10秒口径计算，原标准短窗MAE为19.28 bpm，V28为17.42 bpm，说明连续性收益没有带来整体准确率提升。', '',
        '## 最后一段 data6 的实际输出', '',
        '| 模式 | 波形覆盖 | 有波形时长（秒） | 心率输出/计划 | 心率覆盖 | 首个波形点（秒） | 首个心率中心（秒） | 首个心率窗口结束（秒） |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for name, metric, wave, hr in rows:
        lines.append(f'| {name} | {fmt(wave["waveform_coverage_pct"])}% | {fmt(wave["finite_duration_s"])} | '
            f'{metric["Noutput"]}/{metric["Nplanned"]} | {fmt(metric["hr_output_coverage_pct"])}% | '
            f'{fmt(wave["first_finite_time_s"])} | {fmt(hr["first_accepted_center_s"])} | {fmt(hr["first_accepted_window_end_s"])} |')
    lines += ['', '上表是各模式实际输出的位置和覆盖率：6秒模式的计划窗口比10秒模式更多，分母已分别列出。心率更新约每秒一次，窗口之间重叠；不能把有效窗口数乘以6秒或10秒当作覆盖时长。', '',
        '**这些仍是离线结果。** 心率点画在分析窗口中心；窗口结束时间只说明该窗口需要的输入数据范围。上游滤波和整段心率路径估计使用未来数据，因此不能据此宣称“实时6秒出结果”。', '',
        '## 新增输出是否准确', '',
        '| data6 模式 | 本模式参考窗口 | MAE（bpm） | ±5 bpm达标/有效输出 | 达标率 |',
        '| --- | --- | ---: | --- | ---: |']
    for name, metric, _, _ in rows:
        lines.append(f'| {name} | {10 if name.startswith("V28") else 6}秒 | {fmt(metric["MAE_bpm"])} | {metric["Nwithin5"]}/{metric["Nvalid"]} | {fmt(metric["P5_valid_pct"])}% |')
    lines += ['', '以上各模式按自身窗口内的Polar平均心率评分，不把不同窗口的达标率差直接解释为算法准确率增幅。下面把每个新波形重新按**同一组原10秒窗口、原严格心率门槛**读出，与V28统一比较：', '',
        '| data6 同一10秒口径 | MAE（bpm） | ±5 bpm达标率 | 心率覆盖 | 新增/丢失输出窗口 | 新增窗口 MAE |',
        '| --- | ---: | ---: | ---: | --- | ---: |',
        f'| V28 | {fmt(old6["MAE_bpm"])} | {fmt(old6["P5_valid_pct"])}% | {fmt(old6["hr_output_coverage_pct"])}% | 基准 | — |']
    for name, key in profiles:
        metric = score['profiles'][key]['data6_detail']['common10s']
        lines.append(f'| {name} | {fmt(metric["MAE_bpm"])} | {fmt(metric["P5_valid_pct"])}% | '
            f'{fmt(metric["hr_output_coverage_pct"])}% | {metric["new_windows"]}/{metric["lost_windows"]} | {fmt(metric["new_MAE_bpm"])} |')
    lines += ['', '更多输出只代表连续性变化。误差仍需看上述参考对比，波形形态本身没有同步PPG真值可验证。', '',
        '## 六段视频统一对照', '',
        '| 视频 | V28波形覆盖 | 6秒原标准波形覆盖 | 6秒放宽波形覆盖 | V28心率覆盖（10秒） | 6秒原标准心率覆盖 | 6秒放宽心率覆盖 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for old in score['baseline_V28']['cases']:
        case = old['case']
        strict = next(x for x in score['profiles']['short6']['native6s']['cases'] if x['case'] == case)
        relaxed = next(x for x in score['profiles']['short6_relaxed']['native6s']['cases'] if x['case'] == case)
        vals = [old['waveform_availability']['waveform_coverage_pct'], strict['waveform_coverage_pct'], relaxed['waveform_coverage_pct'],
                old['hr_output_coverage_pct'], strict['hr_output_coverage_pct'], relaxed['hr_output_coverage_pct']]
        lines.append('| '+case+' | '+' | '.join(fmt(x)+'%' for x in vals)+' |')
    lines += ['', '| 六视频汇总：共同10秒口径 | MAE（bpm） | RMSE（bpm） | ±5 bpm达标率 | 全部参考窗口达标率 | 心率覆盖 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    totals = [('V28', score['baseline_V28']['pooled'])]
    totals += [(name, score['profiles'][key]['common10s']['pooled']) for name, key in profiles]
    for name, metric in totals:
        values = [fmt(metric[k]) for k in ('MAE_bpm', 'RMSE_bpm', 'P5_valid_pct', 'R5_all_reference_pct', 'hr_output_coverage_pct')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '汇总按有效窗口数量加权。完整逐视频误差、共同/新增/丢失窗口、参考平移敏感性均保存在评分JSON和CSV中，没有只保留改善视频。', '',
        '## 为什么开头和中间仍会空白', '',
        'V28六段视频第一个波形点约在4、2、4、2、2、9秒，并非全部都要等10秒。data6约6.47秒之后才取得持续有效皮肤信号；原有每段1.6秒滤波边缘处理后，区域波形约8.07秒才可用，融合波形约9秒开始。缩短心率窗口不能补出此前没有稳定采到的皮肤信号。', '',
        'data6原先26个被拒的心率窗口全部含波形缺口或滤波边缘，没有因最终心率的观测比例或谱集中度门槛而拒绝。6秒窗口减少缺口影响的前后范围；轻度放宽区域候选模糊门槛则检验能否让更多实测波形参与融合。', '',
        '两个新模式的唯一质量门槛差异为`ambiguity_ratio`从0.80提高到0.90。仍要求至少两个物理区域，观测比例≥90%，插值比例≤10%，最终谱集中度≥0.12；不延长原0.1秒插值上限，不填零，不保持或插值缺失心率。', '',
        '窗口按`round(6*fps)`帧定义，实际起止时间写入输出；修复了约30.00003/180.0018 fps时恰好6秒被原浮点最小长度判断误拒的问题。', '',
        '## 图和使用方法', '',
        link('data6：三模式PPG波形与心率折线对比图', 'results/data1_6_v29_20260912/figures/data6_ppg_hr_short_window_comparison.png'), '',
        link('六视频：心率与Polar参考折线图', 'results/data1_6_v29_20260912/figures/six_video_hr_short_window_comparison.png'), '',
        '在WSL项目目录运行，输出目录必须尚不存在：', '', '```bash',
        'cd /home/fengbujue/项目/rppg识别',
        './run_motion_v29.sh --video /完整路径/视频.mp4 --out results/新视频_v29 --profile short6',
        '```', '', '将`--profile`改为`short6_relaxed`可试放宽模糊筛选的6秒模式。V28原入口继续保留。', '',
        '主输出为`waveform.csv`、`heart_rate.csv`、`heart_rate_10s.csv`、`routing_decisions.csv`及`ppg_and_hr.png`。`heart_rate.csv`是6秒模式实际心率，`heart_rate_10s.csv`用于原10秒口径对比。质量标记表示通过了哪些工程门槛，不是准确率保证。', '',
        link('完整评分JSON', 'results/data1_6_v29_20260912/evaluation_summary.json'), '',
        link('原生6秒逐视频指标CSV', 'results/data1_6_v29_20260912/metrics_native6s.csv'), '',
        link('共同10秒逐视频指标CSV', 'results/data1_6_v29_20260912/metrics_common10s.csv'), '',
        '仍沿用原Polar心率参考和估计时间对齐；这六段视频均已用于开发，不是独立新测试集。该模式用于检查较短窗口与更多输出的取舍，不自动作为V28的准确率升级。', '']
    content = '\n'.join(lines)
    local = HERE/'V29短窗口与连续性说明.md'
    local.write_text(content)
    (R/'V29短窗口与连续性说明.md').write_text(content)
    print(str(local))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Render frozen V24 / V25 comparisons, without estimating or altering data.

Saved waveforms and their NaN gaps are drawn directly. Metric values are read
from existing evaluations. References are the fixed paired windows, never
archive metadata interpreted as synchronization truth. No identity is shown.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path('/home/fengbujue/项目/rppg识别/results/data1_6_v25_20260911')
CASES = tuple(f'data{i}' for i in range(1, 7))
STAGE = 'stage4_preserve_waveform'
C = dict(old='#8d9aa9', new='#176b76', ref='#bd7c20', wave='#3f69a5',
         text='#202b39', muted='#687587', grid='#e6ebef')
NOTE = '离线估计；参考按文件时间估计同步，非硬件同步。六组均为开发回归数据。'
WAVE_NOTE = '波形为算法估计的归一化 rPPG，不代表传感器实测 PPG；直接绘制保存值，断线未补零。'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def flags(series):
    values = series.astype(str).str.lower()
    require(values.isin(['true', 'false', '1', '0']).all(), 'Invalid Boolean field')
    return values.isin(['true', '1']).to_numpy()


def setup_style():
    for filename in ('/mnt/c/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyh.ttc',
                     '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        if Path(filename).is_file():
            font_manager.fontManager.addfont(filename)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=filename).get_name()
            break
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
        'text.color': C['text'], 'axes.labelcolor': C['text'],
        'axes.edgecolor': '#aeb7c2', 'xtick.color': C['muted'],
        'ytick.color': C['muted'], 'axes.spines.top': False,
        'axes.spines.right': False, 'path.simplify': False,
        'agg.path.chunksize': 0, 'savefig.facecolor': 'white'})


def load_case(root, case):
    paths = {}
    data = {}
    for version, stage in [('old', 'v24_replay'), ('new', STAGE)]:
        directory = root/stage/case
        names = dict(wave='fusion_waveform.csv', hr='fusion_heart_rate.csv',
                     paired='evaluation/paired_windows.csv',
                     metrics='evaluation/metrics.json', summary='summary.json')
        for key, filename in names.items():
            path = directory/filename
            require(path.is_file(), f'Missing input: {path}')
            paths[f'{version}_{key}'] = path
            data[f'{version}_{key}'] = read_json(path) if path.suffix == '.json' else pd.read_csv(path)
        hr, paired = data[f'{version}_hr'], data[f'{version}_paired']
        require(len(hr) == len(paired) and np.allclose(hr.time_s, paired.time_s, rtol=0, atol=1e-9),
                'HR / reference timeline mismatch')
        mask = flags(hr.accepted)
        require(np.array_equal(mask, flags(paired.accepted)), 'HR / evaluation mask mismatch')
        require(np.allclose(hr.ridge_bpm[mask], paired.estimated_bpm[mask], rtol=0, atol=1e-9),
                'HR / evaluation estimates mismatch')
        require(hr.ridge_bpm[~mask].isna().all(), 'Rejected HR must remain NaN')
    wa, wb = data['old_wave'], data['new_wave']
    require(len(wa) == len(wb), 'Waveform lengths differ')
    for key in ('time_s', 'base', 'covered', 'observed', 'interpolated'):
        require(np.array_equal(wa[key].to_numpy(), wb[key].to_numpy(), equal_nan=True),
                f'This presentation requires preserved waveform field {key}')
    pa, pb = data['old_paired'], data['new_paired']
    for key in ('time_s', 'reference_bpm'):
        require(np.array_equal(pa[key].to_numpy(), pb[key].to_numpy(), equal_nan=True),
                f'Fixed reference field differs: {key}')
    require(np.array_equal(flags(pa.reference_valid), flags(pb.reference_valid)), 'Reference masks differ')
    require(np.array_equal(flags(data['old_hr'].accepted), flags(data['new_hr'].accepted)),
            'This comparison requires preserved HR coverage')
    fps, frames = float(data['new_summary']['fps']), int(data['new_summary']['frames'])
    require(len(wb) == frames and fps > 0, 'Waveform length / FPS mismatch')
    require(np.allclose(wb.time_s, np.arange(frames)/fps, rtol=0, atol=1e-7), 'Invalid frame timeline')
    data.update(case=case, duration=frames/fps, paths=paths,
                input_hashes={str(path): sha(path) for path in paths.values()})
    return data


def fmt(value, digits=2):
    return f'{float(value):.{digits}f}' if value is not None and np.isfinite(value) else 'NA'


def caption(data):
    old, new = data['old_metrics'], data['new_metrics']
    return (f'{data["case"]}  |  MAE {fmt(old["MAE_bpm"])} → {fmt(new["MAE_bpm"])} bpm'
            f'    HR覆盖 {fmt(new["hr_output_coverage_pct"], 1)}%'
            f'    波形覆盖 {fmt(new["waveform_coverage_pct"], 1)}%')


def lines(data, wave_ax, hr_ax):
    wave = data['new_wave']
    wave_ax.plot(wave.time_s, wave.base, color=C['wave'], lw=.72)
    for key, color, style, width, marker, zorder in (
            ('old', C['old'], (0, (5, 3)), 2.1, None, 2),
            ('new', C['new'], '-', 1.5, 'o', 3)):
        table = data[f'{key}_hr']
        hr_ax.plot(table.time_s, np.where(flags(table.accepted), table.ridge_bpm, np.nan),
                   color=color, ls=style, lw=width, marker=marker, markersize=2., zorder=zorder)
    pair = data['new_paired']
    hr_ax.plot(pair.time_s, np.where(flags(pair.reference_valid), pair.reference_bpm, np.nan),
               color=C['ref'], ls='--', lw=1.5, zorder=4)
    wave_ax.set_ylabel('归一化 rPPG\n相对幅度')
    hr_ax.set_ylabel('心率（bpm）')
    hr_ax.set_xlabel('视频经过时间（秒）')
    all_hr = np.r_[data['old_hr'].ridge_bpm, data['new_hr'].ridge_bpm, pair.reference_bpm]
    finite = all_hr[np.isfinite(all_hr)]
    require(not len(finite) or (finite.min() >= 40 and finite.max() <= 210),
            'Expand shared HR axes before displaying out-of-range values')
    hr_ax.set_ylim(40, 210)
    hr_ax.set_yticks([50, 100, 150, 200])
    for ax in (wave_ax, hr_ax):
        ax.set_xlim(0, data['duration'])
        ax.grid(axis='y', color=C['grid'], linewidth=.7)
        ax.tick_params(labelsize=9)
    wave_ax.tick_params(labelbottom=False)


def hr_legend(fig, y=.94):
    handles = [Line2D([], [], color=C['old'], lw=2.1, ls=(0, (5, 3)), label='V24 原程序心率'),
        Line2D([], [], color=C['new'], lw=1.7, marker='o', markersize=3, label='V25 新程序心率'),
        Line2D([], [], color=C['ref'], lw=1.6, ls='--', label='固定参考心率（估计同步）')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, y), ncol=3,
               frameon=False, fontsize=10)


def save(fig, base, overwrite):
    require(base.resolve().is_relative_to(ROOT_RESOLVED), 'Output escapes requested root')
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        output = base.with_suffix('.'+suffix)
        require(overwrite or not output.exists(), f'Output exists: {output}')
        fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


def per_case(root, data, overwrite):
    fig = plt.figure(figsize=(13, 7.5))
    fig.text(.075, .955, f'{data["case"]}｜V24 与 V25 波形、心率对照', fontsize=19, weight='bold')
    fig.text(.075, .915, caption(data).split('|', 1)[-1].strip(), fontsize=11)
    fig.text(.075, .875, NOTE, fontsize=9.5, color=C['muted'])
    hr_legend(fig, .85)
    gs = fig.add_gridspec(2, 1, left=.075, right=.97, top=.765, bottom=.16,
                         hspace=.17, height_ratios=[1, 1.35])
    lines(data, fig.add_subplot(gs[0]), fig.add_subplot(gs[1]))
    fig.text(.075, .075, WAVE_NOTE, fontsize=9.5, color=C['muted'])
    fig.text(.075, .04, '本次两版波形逐样本一致；两版线条重合处表示心率结果相同。',
             fontsize=9.5, color=C['muted'])
    save(fig, root/STAGE/data['case']/'presentation/ppg_hr', overwrite)


def overview(root, data, overwrite):
    fig = plt.figure(figsize=(18, 14))
    fig.text(.065, .965, '六组视频｜V24 与 V25 结果对照', fontsize=23, weight='bold')
    fig.text(.065, .937, NOTE, fontsize=11, color=C['muted'])
    hr_legend(fig, .922)
    outer = fig.add_gridspec(3, 2, left=.065, right=.98, top=.865, bottom=.087,
                            wspace=.20, hspace=.32)
    for index, case in enumerate(data):
        sub = outer[index//2, index%2].subgridspec(2, 1, hspace=.18, height_ratios=[1, 1.18])
        wave_ax, hr_ax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        lines(case, wave_ax, hr_ax)
        wave_ax.set_title(caption(case), fontsize=10.4, loc='left', pad=13)
    fig.text(.065, .043, WAVE_NOTE, fontsize=10.5, color=C['muted'])
    fig.text(.065, .021, 'HR 纵轴统一 40–210 bpm，PPG 幅度刻度独立；两版波形与输出覆盖完全相同。',
             fontsize=10.5, color=C['muted'])
    save(fig, root/'presentation/all_six_ppg_hr', overwrite)


def metric_comparison(root, data, pooled, overwrite):
    fig = plt.figure(figsize=(14, 9))
    fig.text(.065, .955, '心率误差与覆盖率｜V24 → V25', fontsize=21, weight='bold')
    summary = pooled['pooled']
    fig.text(.065, .915,
        f'合并 MAE {fmt(pooled["baseline_pooled_MAE_bpm"])} → {fmt(summary["MAE_bpm"])} bpm'
        f'    全参考 R5 {fmt(pooled["baseline_pooled_R5_pct"])}% → {fmt(summary["R5_all_reference_pct"])}%'
        f'    有效比较窗 {summary["Nvalid"]} / {summary["Nref"]}', fontsize=11)
    fig.legend(handles=[Patch(color=C['old'], label='V24'), Patch(color=C['new'], label='V25')],
               loc='upper right', bbox_to_anchor=(.97, .89), ncol=2, frameon=False)
    gs = fig.add_gridspec(2, 2, left=.065, right=.97, top=.82, bottom=.145, wspace=.17, hspace=.40)
    specs = [('MAE_bpm', '心率 MAE（越低越好）', 'bpm', 100.),
             ('R5_all_reference_pct', '全参考窗口中误差 ≤5 bpm 的比例', '%', 110.),
             ('hr_output_coverage_pct', '心率输出覆盖率（两版相同）', '%', 110.),
             ('waveform_coverage_pct', 'PPG 波形覆盖率（两版相同）', '%', 110.)]
    x = np.arange(len(data))
    for index, (metric, title, unit, ceiling) in enumerate(specs):
        ax = fig.add_subplot(gs[index//2, index%2])
        for offset, version, color in [(-.18, 'old', C['old']), (.18, 'new', C['new'])]:
            values = np.array([d[f'{version}_metrics'][metric] for d in data], float)
            require(np.isfinite(values).all() and (values >= 0).all(), 'Metric missing or outside plotted range')
            require(np.max(values) < ceiling-4, 'Metric exceeds plotted range')
            ax.bar(x+offset, values, width=.34, color=color, zorder=3)
            for xpos, value in zip(x+offset, values):
                ax.text(xpos, value+1.5, fmt(value, 1), ha='center', va='bottom', fontsize=8.5)
        ax.set_title(title, loc='left', fontsize=12, pad=10)
        ax.set_xticks(x, CASES)
        ax.set_ylabel(unit)
        ax.set_ylim(0, ceiling)
        if unit == '%':
            ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis='y', color=C['grid'], zorder=0)
    fig.text(.065, .075, NOTE, fontsize=10, color=C['muted'])
    fig.text(.065, .043, 'MAE 仅统计有效输出窗；R5 的分母包含全部有参考值的计划窗口。覆盖率保持不等于波形质量已提高。',
             fontsize=10, color=C['muted'])
    save(fig, root/'presentation/mae_coverage_comparison', overwrite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    global ROOT_RESOLVED
    ROOT_RESOLVED = args.root.resolve()
    setup_style()
    data = [load_case(args.root, case) for case in CASES]
    pooled_path = args.root/STAGE/'stage_evaluation.json'
    pooled = read_json(pooled_path)
    hashes = {str(pooled_path): sha(pooled_path)}
    for case in data:
        hashes.update(case['input_hashes'])
        per_case(args.root, case, args.overwrite)
    overview(args.root, data, args.overwrite)
    metric_comparison(args.root, data, pooled, args.overwrite)
    require(all(sha(path) == digest for path, digest in hashes.items()), 'An input changed during rendering')
    output_paths = list((args.root/'presentation').glob('*.png')) + list((args.root/'presentation').glob('*.svg'))
    for case in CASES:
        output_paths += list((args.root/STAGE/case/'presentation').glob('ppg_hr.*'))
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        renderer_sha256=sha(__file__), input_hashes=hashes, input_files_unchanged=True,
        cases=list(CASES), waveform_fields_exactly_preserved=True, hr_masks_exactly_preserved=True,
        reference_windows_exactly_preserved=True, waveform_interpolation=False,
        metric_source='Existing evaluation/metrics.json and stage_evaluation.json; no recomputed metric',
        reference_note=NOTE, waveform_note=WAVE_NOTE,
        output_sha256={str(path.relative_to(args.root)): sha(path) for path in sorted(output_paths)})
    (args.root/'presentation/render_manifest.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(cases=6, png_svg_files=len(output_paths),
                         inputs_unchanged=True, presentation=str(args.root/'presentation'))))


if __name__ == '__main__':
    main()

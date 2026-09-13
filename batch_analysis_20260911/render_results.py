#!/usr/bin/env python3
"""Render frozen six-video outputs; never estimate a signal or recompute metrics.

Per case: PNG/SVG, full-duration silent H.264 replay, preview frames and receipt.
Charts preserve every waveform sample and NaN gap. Video pixels are resized for
presentation only. The reference comes exclusively from paired_windows.csv;
archive metadata is never interpreted as verified physiological synchronization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ROOT = Path('/home/fengbujue/项目/rppg识别/results/data1_6_20260911')
CASES = tuple(f'data{i}' for i in range(1, 7))
W, H = 1280, 720
VIDEO_BOX = (24, 142, 504, 412)
CHART_BOX = (540, 142, 1264, 650)
COLORS = dict(wave='#3f69a5', estimate='#25576b', reference='#bd7c20',
              text='#202b39', muted='#687587', grid='#e6ebef', background='#f5f7fa')
FONT_PATH = next((p for p in (
    Path('/mnt/c/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/msyh.ttc'),
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
    Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')) if p.is_file()), None)
if FONT_PATH:
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
    'axes.edgecolor': '#aeb7c2', 'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'], 'xtick.color': COLORS['muted'],
    'ytick.color': COLORS['muted'], 'axes.spines.top': False,
    'axes.spines.right': False, 'savefig.facecolor': 'white',
    'path.simplify': False, 'agg.path.chunksize': 0})
SYNC_NOTE = '参考按文件时间估计同步；非硬件同步，误差受对齐假设影响'
OFFLINE_NOTE = '离线结果回放 · 非实时监测 · 断线表示未输出，未补零'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def safe_path(root, relative):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()) and path.resolve() != root.resolve(),
            f'Output escapes batch root: {path}')
    probe = path
    while probe != root:
        require(not probe.is_symlink(), f'Symlink not allowed in presentation path: {probe}')
        probe = probe.parent
    return path


def flags(column):
    values = column.astype(str).str.lower()
    require(values.isin(['true', 'false', '1', '0']).all(), f'Invalid Boolean field: {column.name}')
    return values.isin(['true', '1']).to_numpy()


def number(value, digits=1, suffix=''):
    if value is None:
        return 'NA'
    try:
        return f'{float(value):.{digits}f}{suffix}' if np.isfinite(float(value)) else 'NA'
    except (ValueError, TypeError):
        return 'NA'


def metric_caption(metrics):
    return (f'波形覆盖 {number(metrics.get("waveform_coverage_pct"),1,"%")}    '
            f'心率覆盖 {number(metrics.get("hr_output_coverage_pct"),1,"%")}    '
            f'MAE {number(metrics.get("MAE_bpm"),2)} bpm    '
            f'RMSE {number(metrics.get("RMSE_bpm"),2)} bpm')


def read_case(root, item):
    case = item['case']
    require(case in CASES, f'Unexpected case identity: {case}')
    directory = safe_path(root, case)
    paths = dict(wave=directory/'inference/fusion_waveform.csv',
                 hr=directory/'inference/fusion_heart_rate.csv',
                 summary=directory/'inference/summary.json',
                 paired=directory/'evaluation/paired_windows.csv',
                 metrics=directory/'evaluation/metrics.json')
    for path in paths.values():
        require(path.is_file(), f'Inference/evaluation is not ready: {path}')
    identity_hashes = {str(p): sha(p) for p in paths.values()}
    wave, hr, paired = (pd.read_csv(paths[k]) for k in ('wave', 'hr', 'paired'))
    summary, metrics = load_json(paths['summary']), load_json(paths['metrics'])
    require({'time_s', 'base'} <= set(wave), 'Waveform schema mismatch')
    require({'time_s', 'ridge_bpm', 'accepted', 'status'} <= set(hr), 'HR schema mismatch')
    require({'time_s', 'accepted', 'reference_valid', 'reference_bpm', 'estimated_bpm'} <= set(paired),
            'Paired-window schema mismatch')
    require(len(hr) == len(paired) and np.allclose(hr.time_s, paired.time_s, atol=1e-9, rtol=0),
            'Paired/reference windows do not match the inference time axis')
    accepted = flags(hr.accepted)
    require(np.array_equal(accepted, flags(paired.accepted)), 'Accepted masks differ')
    require(np.allclose(hr.ridge_bpm[accepted], paired.estimated_bpm[accepted],
                        atol=1e-9, rtol=0, equal_nan=True), 'Evaluation and inference HR differ')
    fps = float(summary['fps'])
    require(fps > 0 and len(wave) == int(summary['frames']), 'Frame count/FPS mismatch')
    times = wave.time_s.to_numpy(float)
    require(np.isfinite(times).all() and np.all(np.diff(times) > 0), 'Invalid waveform timeline')
    require(np.allclose(times, np.arange(len(wave))/fps, atol=1e-7, rtol=0),
            'Waveform timeline differs from original-frame sampling')
    video = Path(item['video']['path'])
    require(video.is_file(), f'Original video missing: {video}')
    estimate = hr.ridge_bpm.to_numpy(float).copy()
    estimate[~accepted] = np.nan
    reference = paired.reference_bpm.to_numpy(float).copy()
    reference[~flags(paired.reference_valid)] = np.nan
    require(np.isfinite(estimate[accepted]).all(), 'Accepted HR is not finite')
    stat = video.stat()
    return dict(case=case, directory=directory, paths=paths, input_hashes=identity_hashes,
                item=item, video=video, video_stat=(stat.st_size, stat.st_mtime_ns),
                video_sha256_from_manifest=item['video'].get('sha256'),
                wave=wave, hr=hr, paired=paired, summary=summary, metrics=metrics,
                estimate=estimate, reference=reference, accepted=accepted,
                fps=fps, duration=len(wave)/fps)


def style_axis(ax):
    ax.grid(axis='y', color=COLORS['grid'], linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def make_charts(data, figsize=(13, 6.7), dpi=140, compact=False):
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1]})
    if compact:
        fig.subplots_adjust(left=.12, right=.975, bottom=.12, top=.96, hspace=.4)
    else:
        fig.subplots_adjust(left=.075, right=.98, bottom=.12, top=.79, hspace=.35)
        fig.text(.075, .943, f'{data["case"]}  |  PPG 波形与心率结果', fontsize=19, weight='bold')
        fig.text(.075, .895, metric_caption(data['metrics']), fontsize=12)
        fig.text(.075, .853, OFFLINE_NOTE + '；' + SYNC_NOTE, fontsize=10, color=COLORS['muted'])
    plot_axes(data, axes)
    axes[1].set_xlabel('视频经过时间（秒）')
    if not compact:
        fig.text(.075, .035, '心率点对应10秒窗中心；参考线为同窗设备心率。波形幅度为相对单位，不是接触式PPG真值。',
                 fontsize=10, color=COLORS['muted'])
    fig.canvas.draw()
    return fig, axes


def plot_axes(data, axes):
    wave, times = data['wave'], data['hr'].time_s.to_numpy(float)
    axes[0].plot(wave.time_s, wave.base, color=COLORS['wave'], lw=.85, label='输出 rPPG')
    axes[0].set_ylabel('rPPG 相对幅值')
    if not np.isfinite(wave.base.to_numpy(float)).any():
        axes[0].text(.5, .5, '该视频未生成可用波形', transform=axes[0].transAxes, ha='center')
    axes[1].plot(times, data['estimate'], color=COLORS['estimate'], lw=1.7,
                 marker='o', ms=2.5, label='程序心率')
    axes[1].plot(times, data['reference'], color=COLORS['reference'], lw=1.35,
                 ls='--', label='参考心率（估计同步）')
    axes[1].set_ylabel('心率（bpm）')
    finite = data['reference'][np.isfinite(data['reference'])]
    low = min(40, math.floor(finite.min()/10)*10) if len(finite) else 40
    high = max(210, math.ceil(finite.max()/10)*10) if len(finite) else 210
    axes[1].set_ylim(low, high)
    for ax in axes:
        ax.set_xlim(0, data['duration'])
        style_axis(ax)
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    high_hr = np.r_[data['estimate'], data['reference']]
    if np.isfinite(high_hr).any() and np.nanmax(high_hr) >= 160:
        axes[1].legend(loc='lower right', bbox_to_anchor=(1, 1.01),
                       ncol=2, frameon=False, fontsize=9)
    else:
        axes[1].legend(loc='upper right', frameon=False, fontsize=9)


@lru_cache(maxsize=16)
def pil_font(size):
    require(FONT_PATH is not None, 'A CJK font is needed for Chinese presentation labels')
    return ImageFont.truetype(str(FONT_PATH), size=size)


def text(draw, xy, value, size=20, fill=None):
    draw.text(xy, str(value), font=pil_font(size), fill=fill or COLORS['text'])


def dashboard_background(data):
    image = Image.new('RGB', (W, H), COLORS['background'])
    draw = ImageDraw.Draw(image)
    text(draw, (24, 16), f'{data["case"]}  |  视频 PPG / 心率结果', 28)
    text(draw, (24, 62), OFFLINE_NOTE, 18, COLORS['muted'])
    text(draw, (24, 91), SYNC_NOTE, 17, COLORS['muted'])
    text(draw, (24, 119), '原视频画面（仅缩放展示）', 15, COLORS['muted'])
    draw.rectangle(VIDEO_BOX, fill='#10141c')
    fig, axes = make_charts(data, figsize=(7.24, 5.08), dpi=100, compact=True)
    chart = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    image.paste(Image.fromarray(chart), CHART_BOX[:2])
    coordinate_map = []
    for ax in axes:
        x0, y0 = ax.transData.transform((0, ax.get_ylim()[0]))
        x1, y1 = ax.transData.transform((data['duration'], ax.get_ylim()[1]))
        coordinate_map.append(dict(x0=CHART_BOX[0]+x0, x1=CHART_BOX[0]+x1,
            y0=CHART_BOX[1]+chart.shape[0]-y1, y1=CHART_BOX[1]+chart.shape[0]-y0))
    plt.close(fig)
    text(draw, (24, 430), '窗口心率（10秒窗中心）', 18, COLORS['muted'])
    text(draw, (24, 525), '整段视频评测', 17, COLORS['muted'])
    metrics = data['metrics']
    text(draw, (24, 557), f'波形覆盖  {number(metrics.get("waveform_coverage_pct"),1,"%")}', 20)
    text(draw, (270, 557), f'心率覆盖  {number(metrics.get("hr_output_coverage_pct"),1,"%")}', 20)
    text(draw, (24, 593), f'MAE  {number(metrics.get("MAE_bpm"),2)} bpm', 20)
    text(draw, (270, 593), f'RMSE  {number(metrics.get("RMSE_bpm"),2)} bpm', 20)
    text(draw, (24, 642), '曲线完整保留缺口；播放指针不改变原始信号。', 16, COLORS['muted'])
    draw.line((24, 687, 1256, 687), fill='#d4dce5', width=3)
    return image, coordinate_map


def nearest_window(data, t):
    centers = data['hr'].time_s.to_numpy(float)
    if not len(centers):
        return None
    index = int(np.argmin(np.abs(centers-t)))
    step = float(data['summary'].get('effective_step_s', 1))
    return index if abs(centers[index]-t) <= step/2+1e-8 else None


def compose_frame(background, coords, source, data, t):
    image = background.copy()
    image.paste(Image.fromarray(source), VIDEO_BOX[:2])
    draw = ImageDraw.Draw(image)
    progress = min(1., max(0., t/data['duration']))
    for ax in coords:
        x = round(ax['x0'] + progress*(ax['x1']-ax['x0']))
        draw.line((x, round(ax['y0']), x, round(ax['y1'])), fill='#922b50', width=2)
    i = nearest_window(data, t)
    predicted = None if i is None else data['estimate'][i]
    reference = None if i is None else data['reference'][i]
    text(draw, (24, 465), f'程序  {number(predicted,1)} bpm', 25, COLORS['estimate'])
    text(draw, (270, 465), f'参考  {number(reference,1)} bpm', 25, COLORS['reference'])
    draw.line((24, 687, 24+round(1232*progress), 687), fill='#922b50', width=3)
    text(draw, (24, 693), f'{t:05.1f} / {data["duration"]:.1f} 秒', 15)
    text(draw, (970, 693), 'NA = 当前没有有效输出', 15, COLORS['muted'])
    return image


def read_exact(stream, count):
    chunks, size = [], 0
    while size < count:
        chunk = stream.read(count-size)
        if not chunk:
            break
        chunks.append(chunk); size += len(chunk)
    if size == 0:
        return None
    require(size == count, 'FFmpeg returned an incomplete decoded frame')
    return b''.join(chunks)


def probe_video(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_type,codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration,size',
        '-of', 'json', str(path)], capture_output=True, text=True)
    require(result.returncode == 0, 'ffprobe failed: '+result.stderr[-1000:])
    return json.loads(result.stdout)


def render_video(data, presentation, output_fps):
    require(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg/ffprobe not found')
    background, coords = dashboard_background(data)
    background.save(presentation/'dashboard_background.png')
    vw, vh = VIDEO_BOX[2]-VIDEO_BOX[0], VIDEO_BOX[3]-VIDEO_BOX[1]
    movie = presentation/'result_video.mp4'
    decode_cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '2',
        '-i', str(data['video']), '-map', '0:v:0', '-an', '-sn', '-dn', '-vf',
        f'fps={output_fps}:round=near,scale={vw}:{vh}:force_original_aspect_ratio=decrease:flags=area,pad={vw}:{vh}:(ow-iw)/2:(oh-ih)/2:black',
        '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    encode_cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{W}x{H}', '-r', str(output_fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
        '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(movie)]
    sample_targets = {round(data['duration']*q*output_fps): f'preview_{label}.png'
                      for q, label in [(0.1,'early'),(.5,'middle'),(.9,'late')]}
    sample_targets[0] = 'preview_first.png'
    count = 0
    started = time.perf_counter()
    with (presentation/'decode.log').open('w') as dec_log, (presentation/'encode.log').open('w') as enc_log:
        decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=dec_log)
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=enc_log)
        try:
            while True:
                payload = read_exact(decoder.stdout, vw*vh*3)
                if payload is None:
                    break
                source = np.frombuffer(payload, np.uint8).reshape(vh, vw, 3)
                t = count/output_fps
                require(t < data['duration']+1/output_fps+.001, 'Decoded presentation exceeds source duration')
                frame = compose_frame(background, coords, source, data, t)
                if count in sample_targets:
                    frame.save(presentation/sample_targets[count])
                encoder.stdin.write(frame.tobytes())
                count += 1
            encoder.stdin.close()
            decode_exit, encode_exit = decoder.wait(), encoder.wait()
            require(decode_exit == encode_exit == 0, 'FFmpeg failed; inspect preserved encode/decode logs')
        except BaseException:
            decoder.kill(); encoder.kill(); decoder.wait(); encoder.wait()
            raise
        finally:
            decoder.stdout.close()
    require(count > 0, 'No source frames were rendered')
    frame.save(presentation/'preview_last.png')
    contact = Image.new('RGB', (W, H*3), 'white')
    for row, name in enumerate(('preview_first.png', 'preview_middle.png', 'preview_last.png')):
        with Image.open(presentation/name) as preview:
            contact.paste(preview, (0, H*row))
    contact.save(presentation/'contact_first_middle_last.png')
    duration_error = abs(count/output_fps-data['duration'])
    require(duration_error <= 1/output_fps+.002, 'Rendered duration differs by more than one output frame')
    probe = probe_video(movie)
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    require(len(video_streams) == 1 and len(probe['streams']) == 1, 'Output must have one video stream and no audio')
    stream = video_streams[0]
    require(stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p' and
            (stream['width'], stream['height']) == (W, H) and
            float(Fraction(stream['r_frame_rate'])) == output_fps and
            int(stream['nb_read_frames']) == count, 'Encoded format/frame count mismatch')
    with (presentation/'decode_verification.log').open('w') as log:
        result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2',
            '-i', str(movie), '-f', 'null', '-'], stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Full output decode verification failed')
    return dict(frames=count, output_fps=output_fps, original_duration_s=data['duration'],
        rendered_duration_s=count/output_fps, absolute_duration_difference_s=duration_error,
        full_source_decode_exit_code=decode_exit, encode_exit_code=encode_exit,
        output_full_decode_exit_code=result.returncode, ffprobe=probe,
        elapsed_s=time.perf_counter()-started, decode_command=decode_cmd, encode_command=encode_cmd,
        audio=False, waveform_resampled=False,
        source_video_display_resampling='15 fps presentation only; signal/time arrays are unchanged')


def verify_input_unchanged(data):
    for path, expected in data['input_hashes'].items():
        require(sha(path) == expected, f'Frozen input changed during presentation: {path}')
    stat = data['video'].stat()
    require((stat.st_size, stat.st_mtime_ns) == data['video_stat'], 'Original video stat changed')


def render_case(data, args):
    presentation = safe_path(args.root, data['case']+'/presentation')
    receipt = presentation/'render_manifest.json'
    if presentation.exists() and not args.overwrite:
        if receipt.is_file():
            old = load_json(receipt)
            require(old.get('source_input_hashes') == data['input_hashes'], 'Existing presentation uses different input')
            require(args.static_only or old.get('video') is not None,
                    'Existing presentation is static only; use --overwrite to add the requested full video')
            require(all((presentation/n).is_file() and sha(presentation/n) == h
                        for n, h in old.get('outputs', {}).items()), 'Existing presentation output checksum mismatch')
            print(json.dumps({'case':data['case'], 'already_rendered_and_verified':True,
                'renderer_revision_matches':old.get('renderer_sha256') == sha(__file__),
                'retained_renderer_sha256':old.get('renderer_sha256')}), flush=True)
            return old
        raise FileExistsError(f'Incomplete presentation exists: {presentation}; explicit --overwrite only replaces presentation artifacts')
    presentation.mkdir(parents=True, exist_ok=True)
    fig, _ = make_charts(data)
    fig.savefig(presentation/'ppg_hr.png', dpi=160)
    fig.savefig(presentation/'ppg_hr.svg')
    plt.close(fig)
    movie = None if args.static_only else render_video(data, presentation, args.fps)
    verify_input_unchanged(data)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), case=data['case'],
        renderer_sha256=sha(__file__), source_input_hashes=data['input_hashes'],
        original_video=str(data['video']), original_video_sha256_from_manifest=data['video_sha256_from_manifest'],
        original_video_stat_unchanged=True, original_video_rehashed_by_renderer=False,
        manifest_role='Video identity and duration only; archived timestamps are not interpreted as synchronization truth.',
        metrics_source=str(data['paths']['metrics']), displayed_metrics=data['metrics'],
        sync_status=data['metrics'].get('sync_status'), reference_display_note=SYNC_NOTE,
        amplitude='Actual saved fusion_waveform.csv base; relative units, no new filtering or interpolation',
        HR='Actual ridge_bpm masked by accepted; paired reference masked by reference_valid',
        current_readout='Nearest window center only within half of the existing step; never carry HR through missing output',
        offline=True, input_files_unchanged=True, video=movie,
        outputs={p.name:sha(p) for p in sorted(presentation.iterdir()) if p.is_file() and p != receipt})
    json_write(receipt, report)
    print(json.dumps({'case':data['case'], 'presentation':str(presentation),
        'video_frames':None if movie is None else movie['frames'], 'input_files_unchanged':True}), flush=True)
    return report


def overview(root, items, overwrite):
    missing = [item['case'] for item in items
               if not (root/item['case']/'evaluation/metrics.json').is_file()]
    if missing:
        print(json.dumps({'overview':'waiting_for_evaluation', 'missing':missing}), flush=True)
        return None
    data = [read_case(root, item) for item in items]
    target = safe_path(root, 'presentation')
    target.mkdir(exist_ok=True)
    outputs = [target/'all_six_ppg_hr.png', target/'all_six_ppg_hr.svg', target/'overview_manifest.json']
    if any(p.exists() for p in outputs) and not overwrite:
        print(json.dumps({'overview':'already_exists', 'path':str(target)}), flush=True)
        return None
    fig = plt.figure(figsize=(18, 13), dpi=140)
    outer = fig.add_gridspec(3, 2, left=.065, right=.98, top=.905, bottom=.065, wspace=.19, hspace=.36)
    for i, d in enumerate(data):
        cell = outer[i//2, i%2].subgridspec(2, 1, hspace=.3)
        axes = [fig.add_subplot(cell[j]) for j in range(2)]
        plot_axes(d, axes)
        axes[0].set_title(d['case']+'  |  '+metric_caption(d['metrics']), fontsize=10.5, loc='left', pad=10)
        axes[0].tick_params(labelbottom=False)
        axes[1].set_xlabel('视频经过时间（秒）', fontsize=9)
        for ax in axes:
            ax.tick_params(labelsize=8)
            ax.yaxis.label.set_size(9)
            for label in ax.get_legend().get_texts():
                label.set_fontsize(7.5)
    fig.text(.065, .963, '六组视频 PPG 与心率结果总览', fontsize=23, weight='bold')
    fig.text(.065, .933, OFFLINE_NOTE+'；'+SYNC_NOTE, fontsize=11, color=COLORS['muted'])
    fig.text(.065, .018, '完整原视频时间轴；本次六组HR纵轴统一40–210 bpm，PPG幅度刻度独立。指标来自已核验 metrics.json，未重新计算。',
             fontsize=11, color=COLORS['muted'])
    fig.savefig(outputs[0], dpi=160); fig.savefig(outputs[1]); plt.close(fig)
    for d in data:
        verify_input_unchanged(d)
    report = dict(cases=[d['case'] for d in data], renderer_sha256=sha(__file__),
        source_hashes={d['case']:d['input_hashes'] for d in data}, metrics_recomputed=False,
        source_files_unchanged=True, outputs={p.name:sha(p) for p in outputs[:2]})
    json_write(outputs[2], report)
    print(json.dumps({'overview':'complete','path':str(outputs[0])}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--fps', type=int, default=15, choices=[15])
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--overview-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Replace only generated presentation artifacts')
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    require(FONT_PATH is not None, 'No Chinese font found')
    manifest = load_json(args.root/'inputs_manifest.json')
    require(isinstance(manifest, list) and {i['case'] for i in manifest} == set(CASES),
            'Expected the fixed six-case input manifest')
    items = {i['case']:i for i in manifest}
    if not args.overview_only:
        for case in args.cases:
            print(json.dumps({'event':'render_start','case':case}), flush=True)
            render_case(read_case(args.root, items[case]), args)
    overview(args.root, [items[c] for c in CASES], args.overwrite)


if __name__ == '__main__':
    main()

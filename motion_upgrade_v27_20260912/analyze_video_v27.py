"""Experimental V27 analysis of any single local video; no reference input.

Default: late_psd_cluster. This is an explicit experimental preset, not a claim
that it is the best-performing or promoted model. Primary spatial HR is a patch
statistic; the separate secondary HR is read from the actual saved fused wave.
Use --variant all for the eight frozen comparisons, without selecting a winner.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import cv2
import numpy as np
import pandas as pd

from stable_patch_frontend_v27 import extract_patches, source_hashes as frontend_sources, PatchConfig, PATCH_IDS
from prepare_patch_signals_v27 import (anchor_patches, prepare_signals, read_saved_signals,
                                        validate_inputs, mask)
from patch_hr_v27 import infer_patch_hr, PatchHRConfig, protocol_description
from evidence_hr import DEFAULT_CONFIG as EVIDENCE_CONFIG
from run_v27 import readout


SPATIAL_VARIANTS = ('early_median', 'early_psd_cluster', 'late_median', 'late_psd_cluster')
READOUT_VARIANTS = ('fusion_nomotion_local', 'fusion_motion_local',
                    'fusion_nomotion_dp', 'fusion_motion_dp')
VARIANTS = SPATIAL_VARIANTS + READOUT_VARIANTS
FRONTEND_FILES = ('patch_trace.csv', 'frame_trace.csv', 'patch_anchors.json', 'landmark_trace.npz')
DEPENDENCIES = ('analyze_video_v27.py', 'stable_patch_frontend_v27.py',
                'prepare_patch_signals_v27.py', 'patch_hr_v27.py', 'run_v27.py',
                'legacy_motion.py', 'analyze_rppg.py', 'baseline_frontend.py',
                'pixel_tracking.py', 'anchored_reconstruction.py', 'evidence_hr.py', 'motion_evidence.py')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_hashes():
    here = Path(__file__).resolve().parent
    return {name: sha(here/name) for name in DEPENDENCIES}


def probe_video(video, count_frames):
    """Read actual FPS; cache reuse additionally verifies decoded frame count."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot open input video')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 7:
            raise ValueError('Video FPS cannot support the frozen 42-210 BPM band')
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = None
        if count_frames:
            decoded = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if image is None or image.ndim != 3:
                    raise ValueError('Decoder returned an invalid frame')
                decoded += 1
        return dict(fps=fps, declared_frames=declared, independently_decoded_frames=decoded)
    finally:
        cap.release()


def validate_frontend(front, video, *, count_video_frames=True):
    """Verify full-video cache binding, sample clocks, masks and local identities."""
    front, video = Path(front).resolve(), Path(video).resolve()
    meta_path = front/'metadata.json'
    metadata_sha = sha(meta_path)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    if metadata.get('reference_used') is not False or metadata.get('max_seconds') is not None:
        raise ValueError('Require a complete reference-free frontend extraction')
    if metadata.get('temporal_RGB_filling') is not False or metadata.get('RGB_aggregation_across_patches') is not False:
        raise ValueError('Cache must retain independent patch observations without temporal filling')
    if metadata.get('source_hashes') != frontend_sources():
        raise ValueError('Frontend cache source hashes do not match installed sources')
    if metadata.get('config') != asdict(PatchConfig()):
        raise ValueError('Cache configuration differs from the fixed frontend')
    video_sha = sha(video)
    if metadata.get('video_sha256') != video_sha or metadata.get('video_bytes') != video.stat().st_size:
        raise ValueError('Frontend cache belongs to different video bytes')
    if set(metadata.get('output_hashes', {})) != set(FRONTEND_FILES):
        raise ValueError('Cache must bind all four complete frontend outputs')
    for name in FRONTEND_FILES:
        if sha(front/name) != metadata['output_hashes'][name]:
            raise ValueError('Frontend output hash mismatch: '+name)
    probe = probe_video(video, count_video_frames)
    fps = float(metadata['fps'])
    n = metadata['frames']
    if not isinstance(n, int) or n < round(10*fps):
        raise ValueError('At least ten seconds are required for the frozen window')
    if not np.isclose(probe['fps'], fps, atol=1e-6, rtol=0):
        raise ValueError('Video FPS differs from cached FPS')
    if probe['declared_frames'] != metadata.get('declared_frames'):
        raise ValueError('Video container frame metadata changed')
    if count_video_frames and probe['independently_decoded_frames'] != n:
        raise ValueError('Actual decoded video frame count differs from cache')
    frames = pd.read_csv(front/'frame_trace.csv')
    patches = pd.read_csv(front/'patch_trace.csv')
    if len(frames) != n or metadata.get('patch_ids') != list(PATCH_IDS):
        raise ValueError('Cache frame count or fixed patch identities disagree')
    blocks = validate_inputs(patches, frames, fps)
    for ident, block in blocks.items():
        valid = mask(block.valid)
        tracked = block[[f'tracked_{c}' for c in 'rgb']].to_numpy(float)
        if not np.array_equal(np.isfinite(tracked).all(1), valid):
            raise ValueError('Tracked RGB validity differs from raw support: '+ident)
        if not np.array_equal(mask(block.tracked_valid), valid):
            raise ValueError('Tracked-valid mask contradicts saved RGB: '+ident)
        if not np.isfinite(block.pixel_tracks).all() or (block.pixel_tracks < 0).any():
            raise ValueError('Invalid tracking-support counts')
        if not block.pixel_source.isin(['missing', 'baseline_reset', 'numerical_reset',
                                       'tracked_ratio', 'baseline_ratio_fallback']).all():
            raise ValueError('Unknown patch source')
        if not np.array_equal(block.pixel_source.to_numpy() == 'missing', ~valid):
            raise ValueError('Missing source does not match the actual sampling mask')
        reset = block.pixel_source.isin(['baseline_reset', 'numerical_reset']).to_numpy()
        if not np.array_equal(mask(block.pixel_reset), reset):
            raise ValueError('Reset flag contradicts source')
    with np.load(front/'landmark_trace.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['frame'], np.arange(n))
        np.testing.assert_allclose(archive['time_s'], frames.time_s, atol=1e-8, rtol=0)
        points = archive['points_px']
        if points.ndim != 3 or points.shape[0] != n or points.shape[2] != 2:
            raise ValueError('Incomplete per-frame landmark record')
        missing = frames.source.to_numpy() == 'missing'
        if not np.isnan(points[missing]).all():
            raise ValueError('Missing landmark frames must retain NaN')
        if (~missing).any() and (points.shape[1] == 0 or not np.isfinite(points[~missing]).all()):
            raise ValueError('Available landmark frames must contain actual finite points')
    anchors = json.loads((front/'patch_anchors.json').read_text(encoding='utf-8'))
    if anchors['patches'] is None:
        if patches.valid.any():
            raise ValueError('Valid patches cannot precede any stable identity definition')
    else:
        if set(anchors['patches']) != set(PATCH_IDS):
            raise ValueError('Stable anchor identities are incomplete')
        initial = anchors['initial_frame']
        if not isinstance(initial, int) or not 0 <= initial < n or not frames.source.iloc[initial] in ('mesh', 'redetected'):
            raise ValueError('Anchors require their first genuine detection')
        for ident, anchor in anchors['patches'].items():
            ids = anchor['landmark_indices']
            if len(ids) != PatchConfig().local_landmarks or len(set(ids)) != len(ids):
                raise ValueError('Invalid fixed local landmark indices')
            if anchor['region'] != ident.rsplit('_', 1)[0]:
                raise ValueError('Anchor parent region changed')
    if sha(meta_path) != metadata_sha:
        raise ValueError('Cache metadata changed while being validated')
    receipt = dict(frontend_path=str(front), metadata_sha256=metadata_sha, video_sha256=video_sha,
        output_hashes=metadata['output_hashes'], source_hashes=metadata['source_hashes'],
        frames=n, fps=fps, probe=probe, complete_clock_verified=True,
        twelve_patch_identity_and_support_verified=True, reference_used=False)
    return patches, frames, metadata, receipt


def plot_result(wave, primary, secondary, variant, output):
    """Static source-backed figure. Missing values are never connected through."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelcolor': '#29323D', 'text.color': '#29323D',
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1.4, 1, 1]})
    figure.subplots_adjust(top=.85, bottom=.16, hspace=.36, left=.1, right=.97)
    figure.suptitle('Experimental V27 | '+variant, x=.1, ha='left', y=.97, fontsize=17, fontweight='bold')
    figure.text(.1, .92, 'Offline rPPG | 10 s windows / 1 s step | No physiological reference', fontsize=10)
    axes[0].plot(wave.time_s, wave.base, color='#2C628B', lw=.9)
    axes[0].set_title('Saved fused optical waveform', loc='left', fontsize=11)
    axes[0].set_ylabel('Relative amplitude\n(a.u.)')
    spatial = variant in SPATIAL_VARIANTS
    primary_label = 'Primary: spatial patch HR statistic' if spatial else 'Primary: saved-wave HR / '+variant.removeprefix('fusion_')
    labels = (primary_label, 'Secondary: saved-wave HR / motion + DP')
    for axis, table, label, color, style in zip(axes[1:], [primary, secondary], labels,
                                               ['#2C628B', '#B96A2B'], ['-', '--']):
        accepted = table.accepted.to_numpy(bool)
        values = np.where(accepted, table.ridge_bpm.to_numpy(float), np.nan)
        axis.plot(table.time_s, values, color=color, linestyle=style, marker='o', ms=2.8, lw=1.2)
        axis.set_title(f'{label}  |  {int(accepted.sum())}/{len(table)} windows', loc='left', fontsize=11)
        axis.set_ylim(40, 212)
        axis.set_ylabel('HR (bpm)')
    for axis in axes:
        axis.grid(axis='y', color='#DCE1E6', linewidth=.6)
        axis.set_xlim(0, float(wave.time_s.iloc[-1]) if len(wave) else 1.)
    axes[-1].set_xlabel('Video time (s)')
    note = ('The primary spatial statistic can differ from the displayed fused-wave HR.' if spatial else
            'Both HR outputs read the saved waveform; motion evidence and temporal selection are explicit settings.')
    figure.text(.1, .07, note+'\nGaps remain blank. Relative waveform amplitude is not contact-PPG morphology. Coverage is not accuracy.',
                fontsize=9, linespacing=1.6)
    figure.savefig(output, dpi=160, facecolor='white')
    plt.close(figure)


def _variant_manifest(name, directory, primary, secondary, waves_path, meta, fixed, requested):
    spatial = name in SPATIAL_VARIANTS
    return dict(variant=name, experimental=True, requested=name in requested, reference_used=False,
        source_hashes=fixed, primary_hr_source=('spatial_aggregation_of_saved_patch_BVP' if spatial else 'saved_fused_waveform'),
        primary_is_displayed_fused_waveform_HR=not spatial,
        motion_on_primary=False if spatial else '_nomotion_' not in name,
        temporal_on_primary=False if spatial else name.endswith('_dp'),
        primary_interpretation=('Hierarchical patch/region HR statistic; not the HR of the single displayed fused waveform.' if spatial else
                                'Read from the saved late-PSD-cluster fused waveform with the named motion/local/DP settings.'),
        secondary_hr_source='saved_fused_waveform_motion_DP', secondary_is_saved_waveform_readout=True,
        primary_accepted_windows=int(primary.accepted.sum()), secondary_accepted_windows=int(secondary.accepted.sum()),
        planned_windows=len(primary), patch_waveforms_path=str(waves_path), patch_waveforms_sha256=sha(waves_path),
        signal_metadata=meta, patch_hr_protocol=protocol_description(),
        output_hashes={path.name: sha(path) for path in directory.iterdir() if path.is_file()}, status='complete')


def analyze_video(video, output, *, variant='late_psd_cluster', frontend_cache=None):
    if variant not in VARIANTS+('all',):
        raise ValueError('Choose one of the eight fixed variants or all')
    video, output = Path(video).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; preserve it and choose a new directory')
    fixed = source_hashes()
    input_sha = sha(video)
    requested = list(VARIANTS) if variant == 'all' else [variant]
    if frontend_cache is not None:
        patches, frames, metadata, binding = validate_frontend(frontend_cache, video, count_video_frames=True)
        output.mkdir(parents=True)
        front = output/'frontend'
        front.mkdir()
        for name in FRONTEND_FILES+('metadata.json',):
            shutil.copyfile(Path(frontend_cache)/name, front/name)
        for name, expected in binding['output_hashes'].items():
            if sha(front/name) != expected:
                raise ValueError('Copied cache output hash changed')
        if sha(front/'metadata.json') != binding['metadata_sha256']:
            raise ValueError('Copied cache metadata hash changed')
        extraction_mode = 'verified_frontend_cache'
    else:
        output.mkdir(parents=True)
        front = output/'frontend'
        extract_patches(video, front)
        patches, frames, metadata, binding = validate_frontend(front, video, count_video_frames=False)
        extraction_mode = 'fresh_full_video_extraction'
    save(output/'frontend_binding.json', dict(extraction_mode=extraction_mode, **binding))
    fps = metadata['fps']
    signals_dir = output/'signals'
    signals_dir.mkdir()
    anchored_path = signals_dir/'anchored_patch_trace.csv'
    anchor_patches(patches, frames, fps).to_csv(anchored_path, index=False)
    anchored = pd.read_csv(anchored_path)
    needed = set(name for name in requested if name in SPATIAL_VARIANTS)
    if any(name in READOUT_VARIANTS for name in requested):
        needed.add('late_psd_cluster')
    variant_dirs, products = {}, {}
    for spatial in ('early', 'late'):
        modes = [mode for mode in ('median', 'psd_cluster') if f'{spatial}_{mode}' in needed]
        if not modes:
            continue
        _, waveforms, meta = prepare_signals(anchored, frames, fps, spatial)
        waves_path = signals_dir/f'{spatial}_patch_waveforms.csv'
        waveforms.to_csv(waves_path, index=False)
        save(signals_dir/f'{spatial}_signals.json', meta)
        inputs = read_saved_signals(waves_path, meta)
        for mode in modes:
            name = f'{spatial}_{mode}'
            directory = output/'variants'/name
            directory.mkdir(parents=True)
            fused, primary, diagnostics = infer_patch_hr(*inputs, frames, fps, mode=mode)
            fused.to_csv(directory/'waveform.csv', index=False)
            primary.to_csv(directory/'primary_hr.csv', index=False)
            diagnostics.to_csv(directory/'patch_diagnostics.csv', index=False)
            saved = pd.read_csv(directory/'waveform.csv')
            secondary = readout(saved, frames, fps)
            secondary.to_csv(directory/'secondary_hr.csv', index=False)
            primary = pd.read_csv(directory/'primary_hr.csv')
            secondary = pd.read_csv(directory/'secondary_hr.csv')
            plot_result(saved, primary, secondary, name, directory/'ppg_hr.png')
            manifest = _variant_manifest(name, directory, primary, secondary, waves_path, meta, fixed, requested)
            save(directory/'manifest.json', manifest)
            variant_dirs[name], products[name] = directory, (saved, secondary, waves_path, meta)
    for name in READOUT_VARIANTS:
        if name not in requested:
            continue
        parent = variant_dirs['late_psd_cluster']
        saved, secondary, waves_path, meta = products['late_psd_cluster']
        directory = output/'variants'/name
        directory.mkdir(parents=True)
        shutil.copyfile(parent/'waveform.csv', directory/'waveform.csv')
        shutil.copyfile(parent/'secondary_hr.csv', directory/'secondary_hr.csv')
        primary = readout(pd.read_csv(directory/'waveform.csv'), frames, fps,
                          motion='_nomotion_' not in name, temporal=name.endswith('_dp'))
        primary.to_csv(directory/'primary_hr.csv', index=False)
        plot_result(saved, pd.read_csv(directory/'primary_hr.csv'), secondary, name, directory/'ppg_hr.png')
        manifest = _variant_manifest(name, directory, primary, secondary, waves_path, meta, fixed, requested)
        manifest.update(parent_spatial_variant='late_psd_cluster',
                        parent_manifest_sha256=sha(parent/'manifest.json'),
                        motion_on_primary='_nomotion_' not in name, temporal_on_primary=name.endswith('_dp'),
                        evidence_config=asdict(EVIDENCE_CONFIG))
        save(directory/'manifest.json', manifest)
        if sha(directory/'waveform.csv') != sha(parent/'waveform.csv'):
            raise AssertionError('Readout comparisons must use identical saved waveform bytes')
        variant_dirs[name] = directory
    if fixed != source_hashes() or input_sha != sha(video):
        raise ValueError('Input video or executable sources changed during analysis')
    for name, expected in binding['output_hashes'].items():
        if sha(front/name) != expected:
            raise ValueError('Frontend output changed during inference')
    if sha(front/'metadata.json') != binding['metadata_sha256']:
        raise ValueError('Frontend metadata changed during inference')
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='complete', experimental=True,
        video_path=str(video), video_sha256=input_sha, video_bytes=video.stat().st_size,
        reference_used=False, estimated_synchronization_used=False, physiological_accuracy_evaluated=False,
        requested_variants=requested, computed_variants=list(variant_dirs),
        support_variants=[name for name in variant_dirs if name not in requested],
        source_hashes=fixed, fps=fps, frames=len(frames), frontend_binding=binding,
        primary_secondary_distinction='Spatial HR statistics and fused-wave HR are separate named outputs; no truth-based selection.',
        outputs={str(path.relative_to(output)): sha(path) for path in output.rglob('*') if path.is_file()})
    save(output/'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS+('all',), default='late_psd_cluster',
                        help='Explicit experimental preset; default late_psd_cluster, not a promoted model')
    parser.add_argument('--frontend-cache', type=Path,
                        help='Reuse only a complete, source/video/output-hash-bound V27 frontend')
    args = parser.parse_args()
    result = analyze_video(args.video, args.output, variant=args.variant, frontend_cache=args.frontend_cache)
    print(json.dumps({key: result[key] for key in ('status', 'experimental', 'requested_variants', 'frames', 'fps')})+
          '\n'+str(args.output.resolve()/'manifest.json'), flush=True)


if __name__ == '__main__':
    main()

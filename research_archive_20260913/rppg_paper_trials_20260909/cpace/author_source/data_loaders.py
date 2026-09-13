"""
Data loaders for ROI CSVs (from extract_rois.py) and ground-truth PPG.

Supports two ground-truth formats:
    - UBFC-rPPG: ground_truth.txt with 3 rows (PPG waveform, HR BPM, timestamps)
    - UBFC-Phys: bvp_*.csv with single column of BVP samples at 64 Hz
"""

import os
import csv
import numpy as np
from scipy.interpolate import interp1d


# =========================================================================
# ROI CSV LOADER (output of extract_rois.py)
# =========================================================================

def load_roi_csv(csv_path):
    """Load a ROI CSV produced by extract_rois.py.

    The CSV format is:
        Line 1: comment with metadata, e.g.
            #fps=30.0,n_frames=1800,face_detect_rate=0.98
        Line 2: header: frame, ROI1_R, ROI1_G, ROI1_B, ROI2_R, ...
        Lines 3+: data, one row per video frame

    Returns:
        fs: float, frame rate
        roi_rgb: dict of ROI name -> (r, g, b) tuples of 1D arrays
        meta: dict of metadata
    """
    # Metadata
    with open(csv_path, 'r') as f:
        first = f.readline().strip()
    meta = {}
    if first.startswith('#'):
        for kv in first[1:].split(','):
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    meta[k.strip()] = float(v)
                except ValueError:
                    meta[k.strip()] = v
    fs = float(meta.get('fps', 30.0))

    # Data
    rows = []
    with open(csv_path, 'r') as f:
        f.readline()  # skip metadata
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Identify ROIs
    roi_names = set()
    for col in fieldnames:
        if col == 'frame':
            continue
        parts = col.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('R', 'G', 'B'):
            roi_names.add(parts[0])

    n_frames = len(rows)
    roi_rgb = {}
    for roi in sorted(roi_names):
        r = np.array([float(row[f'{roi}_R']) for row in rows])
        g = np.array([float(row[f'{roi}_G']) for row in rows])
        b = np.array([float(row[f'{roi}_B']) for row in rows])
        roi_rgb[roi] = (r, g, b)

    return fs, roi_rgb, meta


# =========================================================================
# GROUND TRUTH LOADERS
# =========================================================================

def load_ubfc_rppg_ground_truth(gt_path, fs_video, n_frames):
    """Load UBFC-rPPG ground_truth.txt.

    The file has 3 whitespace-separated rows:
        line 1: PPG waveform (irregular sampling)
        line 2: HR in BPM (matched timestamps)
        line 3: timestamps in seconds

    Returns:
        Dict with 'ppg', 'hr_bpm_per_frame', 'mean_hr_bpm'.
    """
    with open(gt_path, 'r') as f:
        lines = f.readlines()

    ppg = np.array([float(x) for x in lines[0].split()])
    hr_bpm = np.array([float(x) for x in lines[1].split()])
    timestamps = np.array([float(x) for x in lines[2].split()])

    # Resample HR to video frame rate (nearest interpolation)
    video_times = np.arange(n_frames) / fs_video
    video_times = np.clip(video_times, timestamps[0], timestamps[-1])
    f_interp = interp1d(timestamps, hr_bpm, kind='nearest',
                        fill_value='extrapolate')
    hr_per_frame = f_interp(video_times)

    return {
        'ppg': ppg,
        'hr_bpm_per_frame': hr_per_frame,
        'mean_hr_bpm': float(np.mean(hr_per_frame)),
    }


def load_ubfc_phys_ground_truth(bvp_path, fs_bvp=64.0):
    """Load UBFC-Phys bvp_*.csv (single column at 64 Hz).

    The BVP-derived ground-truth heart rate uses the same zero-padded
    PSD with parabolic interpolation as the rPPG HR estimator, applied
    to the full BVP recording, giving sub-BPM resolution.

    Returns:
        Dict with 'bvp', 'fs_bvp', 'mean_hr_bpm'.
    """
    bvp = []
    with open(bvp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                bvp.append(float(line))
            except ValueError:
                continue
    bvp = np.array(bvp)

    # HR via zero-padded PSD with parabolic interpolation
    # (matches the rPPG HR estimator in metrics.py)
    from metrics import _psd_hr_zeropad
    mean_hr_bpm, _ = _psd_hr_zeropad(bvp - bvp.mean(), fs_bvp,
                                     lo=0.7, hi=3.0, pad_factor=8)

    return {
        'bvp': bvp,
        'fs_bvp': fs_bvp,
        'mean_hr_bpm': float(mean_hr_bpm) if not np.isnan(mean_hr_bpm) else np.nan,
    }

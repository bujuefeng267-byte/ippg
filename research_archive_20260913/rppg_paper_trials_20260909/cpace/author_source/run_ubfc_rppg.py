"""
Batch processing: run all rPPG methods on the full UBFC-rPPG dataset.

Usage:
    # 1. Extract ROIs for the whole dataset
    python extract_rois.py --batch-rppg /path/to/UBFC-rPPG/ --output roi_csvs/

    # 2. Run all methods and compute per-subject metrics
    python run_ubfc_rppg.py --rois roi_csvs/ --gt /path/to/UBFC-rPPG/ \
        --output results.json
"""

import argparse
import json
import os

import numpy as np

from data_loaders import load_roi_csv, load_ubfc_rppg_ground_truth
from pipeline import run_all_methods


def process_subject(roi_csv, gt_path, methods, apply_homodyne):
    fs, roi_rgb, meta = load_roi_csv(roi_csv)
    roi_rgb = {k: v for k, v in roi_rgb.items() if k != 'Full_Face'}

    n_frames = len(next(iter(roi_rgb.values()))[0])
    gt = load_ubfc_rppg_ground_truth(gt_path, fs, n_frames)
    gt_hr = gt['mean_hr_bpm']

    results = run_all_methods(
        roi_rgb, fs, methods=methods, apply_homodyne=apply_homodyne
    )

    summary = {
        'gt_hr_bpm': gt_hr,
        'fs': fs,
        'n_frames': n_frames,
        'methods': {},
    }
    for method, r in results.items():
        if 'error' in r:
            summary['methods'][method] = {'error': r['error']}
            continue
        hr = r['hr_consensus_bpm']
        summary['methods'][method] = {
            'hr_bpm': hr,
            'mae_vs_gt': float(abs(hr - gt_hr)) if hr == hr else None,
            'plv_lr_cheek': r['plv_lr_cheek'],
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch UBFC-rPPG evaluation.")
    parser.add_argument('--rois', required=True,
                        help='Directory of ROI CSVs (from extract_rois.py)')
    parser.add_argument('--gt', required=True,
                        help='UBFC-rPPG root directory (with subject*/ subdirs)')
    parser.add_argument('--output', default='results.json',
                        help='Output JSON path')
    parser.add_argument(
        '--methods', nargs='+',
        default=['GREEN', 'CHROM', 'POS', 'OMIT', 'LGI', 'PBV', 'ICA', 'cPACE'],
    )
    parser.add_argument('--no-homodyne', action='store_true')
    args = parser.parse_args()

    # Discover subjects
    subjects = []
    for entry in sorted(os.listdir(args.gt)):
        sub_dir = os.path.join(args.gt, entry)
        gt_path = os.path.join(sub_dir, 'ground_truth.txt')
        csv_path = os.path.join(args.rois, f'{entry}.csv')
        if os.path.isdir(sub_dir) and os.path.exists(gt_path) \
                and os.path.exists(csv_path):
            subjects.append((entry, csv_path, gt_path))

    print(f"Found {len(subjects)} subjects with ROI CSV and ground truth.\n")

    all_results = {}
    for sid, csv_path, gt_path in subjects:
        print(f"  Processing {sid} ...", end=' ', flush=True)
        try:
            all_results[sid] = process_subject(
                csv_path, gt_path, args.methods, not args.no_homodyne
            )
            print('ok')
        except Exception as e:
            print(f'ERROR: {e}')
            all_results[sid] = {'error': str(e)}

    # Summary
    print(f"\n{'Method':<8} {'mean MAE':>12} {'median PLV':>12} {'n':>6}")
    print('-' * 45)
    for method in args.methods:
        maes, plvs = [], []
        for r in all_results.values():
            if 'error' in r:
                continue
            m = r['methods'].get(method, {})
            if 'mae_vs_gt' in m and m['mae_vs_gt'] is not None:
                maes.append(m['mae_vs_gt'])
            if 'plv_lr_cheek' in m and m['plv_lr_cheek'] is not None:
                plvs.append(m['plv_lr_cheek'])
        mae_str = f"{np.mean(maes):.2f}" if maes else "N/A"
        plv_str = f"{np.median(plvs):.3f}" if plvs else "N/A"
        print(f"{method:<8} {mae_str:>12} {plv_str:>12} {len(maes):>6}")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == '__main__':
    main()

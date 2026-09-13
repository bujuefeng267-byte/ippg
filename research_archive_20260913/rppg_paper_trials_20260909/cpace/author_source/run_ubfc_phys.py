"""
Batch processing: run all rPPG methods on the full UBFC-Phys dataset.

UBFC-Phys is the headline dataset in the paper, with two tasks:
    T1 - rest condition (subject relaxed)
    T2 - speech task (induces moderate motion, lower HR fidelity)

Per-task evaluation is the standard convention in the paper. T3 is
present in the dataset but not used in the paper (interview task with
more head motion).

Usage:
    # 1. Extract ROIs for all subjects and tasks
    python extract_rois.py --batch /path/to/UBFC-Phys/ --tasks T1 T2 \
        --output roi_csvs/

    # 2. Run all methods and compute per-task per-subject metrics
    python run_ubfc_phys.py --rois roi_csvs/ --gt /path/to/UBFC-Phys/ \
        --output results_phys.json

Expected directory structure:
    UBFC-Phys/
        s1/
            vid_s1_T1.avi
            bvp_s1_T1.csv
            vid_s1_T2.avi
            bvp_s1_T2.csv
        s2/
            ...

ROI CSV directory (output of extract_rois.py):
    roi_csvs/
        s1_T1.csv
        s1_T2.csv
        s2_T1.csv
        ...
"""

import argparse
import json
import os

import numpy as np

from data_loaders import load_roi_csv, load_ubfc_phys_ground_truth
from pipeline import run_all_methods


def process_subject_task(roi_csv, bvp_path, methods, apply_homodyne):
    """Process one subject-task pair."""
    fs, roi_rgb, meta = load_roi_csv(roi_csv)
    roi_rgb = {k: v for k, v in roi_rgb.items() if k != 'Full_Face'}

    gt = load_ubfc_phys_ground_truth(bvp_path)
    gt_hr = gt['mean_hr_bpm']

    results = run_all_methods(
        roi_rgb, fs, methods=methods, apply_homodyne=apply_homodyne
    )

    n_frames = len(next(iter(roi_rgb.values()))[0])
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


def find_subject_files(ubfc_phys_dir, subj_id, task):
    """Locate the bvp file for a UBFC-Phys subject/task. Returns None if missing."""
    subj_num = subj_id.replace('s', '')
    subj_dir = os.path.join(ubfc_phys_dir, subj_id)
    if not os.path.isdir(subj_dir):
        return None
    for bname in [f'bvp_{subj_id}_{task}.csv', f'bvp_s{subj_num}_{task}.csv']:
        bp = os.path.join(subj_dir, bname)
        if os.path.exists(bp):
            return bp
    return None


def main():
    parser = argparse.ArgumentParser(description="Batch UBFC-Phys evaluation.")
    parser.add_argument('--rois', required=True,
                        help='Directory of ROI CSVs (from extract_rois.py). '
                             'Files named <subject>_<task>.csv e.g. s1_T1.csv.')
    parser.add_argument('--gt', required=True,
                        help='UBFC-Phys root directory with subject subdirs '
                             '(s1/, s2/, ...) containing bvp_*.csv files.')
    parser.add_argument('--output', default='results_phys.json',
                        help='Output JSON path')
    parser.add_argument('--tasks', nargs='+', default=['T1', 'T2'],
                        help='Tasks to evaluate (default: T1 T2). '
                             'T3 is present in the dataset but not used in the paper.')
    parser.add_argument(
        '--methods', nargs='+',
        default=['GREEN', 'CHROM', 'POS', 'OMIT', 'LGI', 'PBV', 'ICA', 'cPACE'],
    )
    parser.add_argument('--no-homodyne', action='store_true')
    args = parser.parse_args()

    # Discover subject-task pairs from the ROI CSV directory
    pairs = []
    for fname in sorted(os.listdir(args.rois)):
        if not fname.endswith('.csv'):
            continue
        base = os.path.splitext(fname)[0]
        # Expected format: s<N>_T<M>
        parts = base.split('_')
        if len(parts) != 2 or not parts[0].startswith('s') or not parts[1].startswith('T'):
            continue
        subj_id, task = parts
        if task not in args.tasks:
            continue
        csv_path = os.path.join(args.rois, fname)
        bvp_path = find_subject_files(args.gt, subj_id, task)
        if bvp_path is None:
            continue
        pairs.append((subj_id, task, csv_path, bvp_path))

    if not pairs:
        print(f"No subject-task pairs found in {args.rois} with ground truth in {args.gt}.")
        print(f"Looking for tasks: {args.tasks}")
        return

    print(f"Found {len(pairs)} subject-task pairs.")
    by_task = {}
    for subj_id, task, _, _ in pairs:
        by_task.setdefault(task, []).append(subj_id)
    for task in args.tasks:
        n = len(by_task.get(task, []))
        print(f"  {task}: {n} subjects")
    print()

    # Process each pair
    all_results = {}
    for i, (subj_id, task, csv_path, bvp_path) in enumerate(pairs, 1):
        key = f'{subj_id}_{task}'
        print(f"  [{i}/{len(pairs)}] {key} ...", end=' ', flush=True)
        try:
            res = process_subject_task(
                csv_path, bvp_path, args.methods, not args.no_homodyne
            )
            res['subject'] = subj_id
            res['task'] = task
            all_results[key] = res
            print('ok')
        except Exception as e:
            print(f'ERROR: {e}')
            all_results[key] = {'error': str(e), 'subject': subj_id, 'task': task}

    # Per-task summary table
    print('\n' + '=' * 64)
    print('SUMMARY (per task)')
    print('=' * 64)
    for task in args.tasks:
        task_results = {k: v for k, v in all_results.items()
                        if v.get('task') == task and 'error' not in v}
        if not task_results:
            continue
        print(f"\n{task} (n={len(task_results)}):")
        print(f"  {'Method':<8} {'mean MAE':>12} {'median PLV':>12} {'n':>6}")
        print(f"  {'-' * 42}")
        for method in args.methods:
            maes, plvs = [], []
            for r in task_results.values():
                m = r['methods'].get(method, {})
                if m.get('mae_vs_gt') is not None:
                    maes.append(m['mae_vs_gt'])
                if m.get('plv_lr_cheek') is not None:
                    plvs.append(m['plv_lr_cheek'])
            mae_str = f"{np.mean(maes):.2f}" if maes else "N/A"
            plv_str = f"{np.median(plvs):.3f}" if plvs else "N/A"
            print(f"  {method:<8} {mae_str:>12} {plv_str:>12} {len(maes):>6}")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == '__main__':
    main()

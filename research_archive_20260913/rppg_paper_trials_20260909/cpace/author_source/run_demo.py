"""
Demo: run all rPPG methods on a single recording and print results.

Usage:
    # 1. Extract ROIs from a video
    python extract_rois.py /path/to/video.avi --output roi_csvs/my_rec.csv

    # 2. Run all methods on the ROI CSV
    python run_demo.py roi_csvs/my_rec.csv

    # 3. Optional: provide a UBFC-rPPG reference signal for HR error
    python run_demo.py roi_csvs/my_rec.csv --reference ground_truth.txt
"""

import argparse

from data_loaders import load_roi_csv, load_ubfc_rppg_ground_truth
from pipeline import run_all_methods


def main():
    parser = argparse.ArgumentParser(
        description="Run all rPPG methods on one recording."
    )
    parser.add_argument('roi_csv', help='ROI CSV produced by extract_rois.py')
    parser.add_argument('--reference', '-r', default=None,
                        help='Optional UBFC-rPPG ground_truth.txt reference '
                             'signal (for HR comparison)')
    parser.add_argument(
        '--methods', nargs='+',
        default=['GREEN', 'CHROM', 'POS', 'OMIT', 'LGI', 'PBV', 'ICA', 'cPACE'],
        help='Methods to evaluate'
    )
    parser.add_argument('--no-homodyne', action='store_true',
                        help='Disable homodyne envelope correction')
    args = parser.parse_args()

    # Load ROIs
    fs, roi_rgb, meta = load_roi_csv(args.roi_csv)
    print(f"\nLoaded {args.roi_csv}")
    print(f"  fps = {fs}, ROIs = {list(roi_rgb.keys())}")
    print(f"  n_frames = {int(meta.get('n_frames', 0))}, "
          f"face_detect_rate = {meta.get('face_detect_rate', 'N/A')}")

    # Drop Full_Face (cPACE wants per-region ROIs)
    roi_rgb = {k: v for k, v in roi_rgb.items() if k != 'Full_Face'}

    # Run all methods
    results = run_all_methods(
        roi_rgb, fs,
        methods=args.methods,
        apply_homodyne=not args.no_homodyne,
    )

    # Optional reference signal for HR error comparison
    ref_hr = None
    if args.reference:
        n_frames = len(next(iter(roi_rgb.values()))[0])
        ref = load_ubfc_rppg_ground_truth(args.reference, fs, n_frames)
        ref_hr = ref['mean_hr_bpm']
        print(f"  Reference mean HR = {ref_hr:.2f} BPM")

    # Print table
    print()
    print(f"{'Method':<8} {'HR (BPM)':>10}", end='')
    if ref_hr is not None:
        print(f" {'|err|':>8}", end='')
    print(f" {'L-R PLV':>10}")
    print('-' * 50)
    for method in args.methods:
        r = results.get(method, {})
        if 'error' in r:
            print(f"{method:<8} ERROR: {r['error']}")
            continue
        hr = r['hr_consensus_bpm']
        plv = r['plv_lr_cheek']
        print(f"{method:<8} {hr:>10.2f}", end='')
        if ref_hr is not None:
            err = abs(hr - ref_hr) if hr == hr else float('nan')
            print(f" {err:>8.2f}", end='')
        plv_str = f"{plv:.3f}" if plv is not None else "N/A"
        print(f" {plv_str:>10}")
    print()


if __name__ == '__main__':
    main()

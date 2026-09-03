#!/usr/bin/env python3
"""Compare an rPPG summary with UBFC's three-line ground-truth file."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    rows = [
        np.fromstring(line, sep=" ")
        for line in args.ground_truth.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 3:
        raise ValueError("Expected UBFC ground truth with PPG, HR, and timestamp rows")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    gt_hr = rows[1]
    estimated = float(summary["median_bpm"])
    reference = float(np.median(gt_hr))

    output = {
        "ground_truth_samples": [int(len(row)) for row in rows[:3]],
        "reference_hr_mean_bpm": float(np.mean(gt_hr)),
        "reference_hr_median_bpm": reference,
        "reference_hr_std_bpm": float(np.std(gt_hr)),
        "reference_hr_min_bpm": float(np.min(gt_hr)),
        "reference_hr_max_bpm": float(np.max(gt_hr)),
        "estimated_median_bpm": estimated,
        "absolute_error_bpm": abs(estimated - reference),
        "signed_error_bpm": estimated - reference,
        "face_detection_rate": float(summary["face_detection_rate"]),
        "median_snr_db": float(summary["median_snr_db"]),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

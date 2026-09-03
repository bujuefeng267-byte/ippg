#!/usr/bin/env python3
"""Relate motion measurements to POS/CHROM rPPG stability for one candidate clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stats(frame: pd.DataFrame, mask: pd.Series, prefix: str) -> dict:
    part = frame.loc[mask]
    return {
        f"{prefix}_windows": int(len(part)),
        f"{prefix}_median_bpm": float(part.bpm.median()),
        f"{prefix}_bpm_std": float(part.bpm.std(ddof=0)),
        f"{prefix}_median_snr_db": float(part.snr_db.median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", type=Path, required=True)
    ap.add_argument("--pos", type=Path, required=True)
    ap.add_argument("--chrom", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    motion = pd.read_csv(args.motion)
    motion = motion[motion.time_s <= 60].copy()
    pos = pd.read_csv(args.pos).rename(columns={"bpm": "pos_bpm", "snr_db": "pos_snr"})
    chrom = pd.read_csv(args.chrom).rename(columns={"bpm": "chrom_bpm", "snr_db": "chrom_snr"})
    hr = pd.merge(pos, chrom, on="time_s", how="inner")
    hr["motion_score"] = np.interp(hr.time_s, motion.time_s, motion.motion_score_window)
    hr["method_disagreement_bpm"] = (hr.pos_bpm - hr.chrom_bpm).abs()
    q25, q75 = hr.motion_score.quantile([0.25, 0.75])

    result = {
        "motion_low_threshold": float(q25),
        "motion_high_threshold": float(q75),
        "inter_method_disagreement_all_median_bpm": float(hr.method_disagreement_bpm.median()),
        "inter_method_disagreement_low_motion_median_bpm": float(hr.loc[hr.motion_score <= q25, "method_disagreement_bpm"].median()),
        "inter_method_disagreement_high_motion_median_bpm": float(hr.loc[hr.motion_score >= q75, "method_disagreement_bpm"].median()),
    }
    for name, bpm_col, snr_col in (("pos", "pos_bpm", "pos_snr"), ("chrom", "chrom_bpm", "chrom_snr")):
        temp = hr[["time_s", "motion_score", bpm_col, snr_col]].rename(columns={bpm_col: "bpm", snr_col: "snr_db"})
        result[name] = {}
        result[name].update(stats(temp, temp.motion_score <= q25, "low_motion"))
        result[name].update(stats(temp, temp.motion_score >= q75, "high_motion"))
        result[name].update(stats(temp, temp.time_s < 30, "first_half"))
        result[name].update(stats(temp, temp.time_s >= 30, "second_half"))

    hr.to_csv(args.output / "motion_hr_joined.csv", index=False)
    (args.output / "motion_effect_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(motion.time_s, motion.motion_score_window, color="tab:red")
    axes[0].set_ylabel("Motion score")
    axes[1].plot(hr.time_s, hr.pos_bpm, label="POS")
    axes[1].plot(hr.time_s, hr.chrom_bpm, label="CHROM")
    axes[1].set_ylabel("Estimated HR (bpm)")
    axes[1].legend()
    axes[2].plot(hr.time_s, hr.pos_snr, label="POS")
    axes[2].plot(hr.time_s, hr.chrom_snr, label="CHROM")
    axes[2].axhline(0, color="black", lw=0.7)
    axes[2].set_ylabel("Spectral SNR (dB)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.output / "motion_vs_rppg.png", dpi=170)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

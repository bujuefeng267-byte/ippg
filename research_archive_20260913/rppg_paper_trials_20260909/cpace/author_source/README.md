# cPACE

A reference implementation of cPACE (chromatic Pulsation with Absorption-direction Cancellation and Eigenvector selection) and seven baseline unsupervised remote photoplethysmography (rPPG) methods.

cPACE recovers the cardiac pulse waveform from RGB facial video. Its central idea is that the temporally-averaged skin reflectance direction `q_hat` in RGB-space carries a non-cardiac (isochromatic) signal that mixes into the bandpass-filtered output of methods that operate in a chrominance plane. cPACE removes this direction by projecting RGB onto the `q_hat`-perpendicular subspace before estimating the cardiac direction via eigenvector decomposition.

This repository provides cPACE along with reference implementations of GREEN, CHROM, POS, OMIT, LGI, PBV, and ICA, all sharing the same preprocessing pipeline.

## Installation

```bash
pip install numpy scipy mediapipe opencv-python scikit-learn
```

Python 3.8+ recommended. `scikit-learn` is only needed for the ICA baseline.

## Methods

| Method | Reference |
|---|---|
| GREEN | Verkruysse et al. 2008, raw green channel baseline |
| CHROM | de Haan & Jeanne 2013 |
| POS | Wang et al. 2017 |
| OMIT | Casado & Lopez 2023, Face2PPG |
| LGI | Pilz et al. 2018 |
| PBV | de Haan & van Leest 2014 |
| ICA | Poh et al. 2010 |
| cPACE | This implementation |

## Pipeline

Every method uses the same pipeline. The only difference is the projection rule.

1. ROI extraction from video using MediaPipe FaceMesh (forehead, left cheek, right cheek, nose) - `extract_rois.py`.
2. Per-channel temporal normalization (mean division) inside each projection method.
3. Cardiac-band bandpass filter (0.7-3.0 Hz, 4th-order Butterworth, zero-phase).
4. Method-specific projection to a 1D cardiac signal per ROI.
5. Optional homodyne envelope correction at the seed heart-rate frequency.
6. Heart-rate estimation via windowed PSD (10 s windows, 2 s step) with 8x zero-padding and parabolic interpolation for sub-BPM resolution; power-weighted median across windows.
7. Cross-ROI phase-locking value (PLV) on the L_Cheek-R_Cheek pair.

The heart-rate seed used by cPACE (for its narrowband filter and the eigenvector window) defaults to the green-channel PSD peak from the forehead ROI. Users with access to a synchronized reference signal can override this through the `hr_seed_hz` argument.

## Usage

### Single recording

```bash
# Step 1: extract ROIs from a video
python extract_rois.py /path/to/video.avi --output roi_csvs/my_recording.csv

# Step 2: run all methods on the ROI CSV
python run_demo.py roi_csvs/my_recording.csv
```

### Batch on UBFC-Phys (T1 and T2 conditions)

```bash
# Step 1: extract ROIs for T1 and T2 across all subjects
python extract_rois.py --batch /path/to/UBFC-Phys/ --tasks T1 T2 \
    --output roi_csvs/

# Step 2: per-task evaluation against the reference BVP signal
python run_ubfc_phys.py --rois roi_csvs/ --gt /path/to/UBFC-Phys/ \
    --output results_phys.json
```

### Batch on UBFC-rPPG

```bash
# Step 1: extract ROIs for all subjects
python extract_rois.py --batch-rppg /path/to/UBFC-rPPG/ --output roi_csvs/

# Step 2: per-subject evaluation against the reference PPG
python run_ubfc_rppg.py --rois roi_csvs/ --gt /path/to/UBFC-rPPG/ \
    --output results_rppg.json
```

### Python API

```python
from data_loaders import load_roi_csv
from pipeline import run_method, run_all_methods

# Load extracted ROIs
fs, roi_rgb, _ = load_roi_csv("roi_csvs/my_recording.csv")
roi_rgb = {k: v for k, v in roi_rgb.items() if k != "Full_Face"}

# Run cPACE
result = run_method(roi_rgb, fs, method="cPACE")
print(f"HR = {result['hr_consensus_bpm']:.1f} BPM")
print(f"L-R cheek PLV = {result['plv_lr_cheek']:.3f}")

# Run all 8 methods on the same recording (shared pipeline + shared seed)
all_results = run_all_methods(roi_rgb, fs)

# Override the seed from an external reference (e.g., synchronized BVP)
result = run_method(roi_rgb, fs, method="cPACE", hr_seed_hz=1.25)  # 75 BPM

# Enable the optional cross-ROI coherence gate inside cPACE
from projections import cpace_signal
signals = cpace_signal(roi_rgb, fs, hr_seed_hz=1.25, apply_coherence_gate=True)
```

## Module layout

```
extract_rois.py    MediaPipe FaceMesh ROI extraction; writes ROI CSVs.
utils.py           Shared utilities (bandpass filter).
projections.py     All 8 projection methods. Same signature for all:
                   (roi_rgb dict, fs, **kwargs) -> dict of ROI -> 1D signal.
homodyne.py        Envelope correction applied after projection (optional).
metrics.py         HR seed (green PSD), windowed HR, L-R cheek PLV.
pipeline.py        run_method and run_all_methods orchestrators.
data_loaders.py    ROI CSV and reference-signal loaders (UBFC-rPPG, UBFC-Phys).
run_demo.py        Single-recording demo.
run_ubfc_phys.py   Batch evaluation on UBFC-Phys T1 and T2.
run_ubfc_rppg.py   Batch evaluation on UBFC-rPPG.
```

## cPACE in more detail

cPACE operates per ROI:

1. **Per-channel temporal normalization and detrend.** The R, G, B channels are each divided by their temporal mean and centred.

2. **q-perpendicular projection.** The skin reflectance direction is `q_hat = (R_mean, G_mean, B_mean) / ||...||`. The projection matrix `P = I - q_hat * q_hat^T` projects RGB onto the 2-D plane perpendicular to `q_hat`. This removes the isochromatic component.

3. **Narrowband filter around the heart-rate seed.** The projected signal is bandpassed to `(hr_seed_hz - bw, hr_seed_hz + bw)` for the eigenvector covariance estimation.

4. **Eigenvector decomposition.** The 3x3 covariance of the bandpassed signal yields top eigenvectors v1 and v2 (after q-perpendicular projection, one of these eigenvalues is effectively zero; v1 and v2 span the chrominance plane).

5. **v1 or v2 selection by cross-ROI PLV.** Both candidates are computed, then the one whose per-ROI signals show higher cross-ROI phase coherence is chosen. This is defensive against windows where motion variance exceeds cardiac variance, where v1 would track motion instead.

6. **Optional cross-ROI coherence gate.** An additional amplitude-only gate, controlled by the instantaneous cross-ROI Kuramoto order parameter, can be applied to attenuate samples where per-ROI phases disagree (e.g., during transient motion). Off by default; enable via `apply_coherence_gate=True`.

7. **Optional homodyne envelope correction.** Applied at the pipeline level (not inside `cpace_signal`); flattens amplitude modulation at the seed frequency. On by default.

## Parameters

Default values:

| Parameter | Value | Purpose |
|---|---|---|
| Cardiac frequency band | 0.7-3.0 Hz | Bandpass + HR estimation |
| Bandpass filter | 4th-order Butterworth, zero-phase | Shared across methods |
| HR seed | Green PSD peak from Forehead ROI | cPACE seed; homodyne center |
| Eigen-window half-width | 0.30 Hz | Narrowband filter around seed |
| Envelope smoothing | 0.30 Hz | Homodyne low-pass cutoff |
| HR window | 10 s, 2 s step | Per-ROI HR estimation |
| Zero-pad factor | 8x | Sub-BPM PSD resolution |
| Aggregation across windows | Power-weighted median | Robust to outlier windows |
| Aggregation across ROIs | Median | Per-recording HR |
| PLV pair | L_Cheek - R_Cheek | Symmetric, cohort-comparable |
| Coherence gate | Off by default | Optional motion robustness |

## Datasets

Two datasets are supported by the batch scripts:

- **UBFC-rPPG**: Bobbia et al., *Pattern Recognit. Lett.* 124, 82-90 (2019). https://sites.google.com/view/ybenezeth/ubfcrppg
- **UBFC-Phys**: Sabour et al., *IEEE Trans. Affect. Comput.* (2021).

The `data_loaders.py` module includes loaders for both datasets' reference PPG/BVP formats. ROI extraction (`extract_rois.py`) is dataset-agnostic and works on any video file.

## License

[MIT or Apache 2.0]

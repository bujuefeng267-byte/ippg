"""V30 readout-only full-spectrum normalization of V28 motion evidence.

The frozen frontend stores both motion axes as displacement / IMAGE HEIGHT per
frame. Divide by the contemporaneous normalized face height, then multiply by
FPS. These are geometric motion proxies, not measured inertial velocities.
Only contiguous observed stretches are used for spectra; no gap is interpolated.

Fixed engineering scales, selected before real-video V25 evaluation:
* >=50% observed/spectrally supported samples and >=3 s contiguous stretches;
* speed .25 face heights/s gives half of the bounded interference strength.
No physiological reference or estimated heart rate enters this function.

The sole arithmetic change is the normalization denominator: use the peak of
the length-weighted, summed-axis full motion PSD, including DC and Nyquist.
Each observed block retains its original Welch/nfft/detrend calculation. Full
spectra share the largest block FFT grid before averaging; powers of two make
this grid include every original PSD knot. Slow-motion leakage is not promoted
to full-scale risk by restricting the normalization peak to the HR query band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

MIN_OBSERVED_FRACTION = .50
MIN_SEGMENT_SECONDS = 3.
HALF_STRENGTH_SPEED_FACE_PER_S = .25

NORMALIZATION_CONFIG = dict(
    method='length_weighted_full_motion_psd_peak',
    full_spectrum='one_sided_0_to_nyquist_inclusive',
    include_dc=True, detrend='constant', window='hann',
    axes='sum_both_axes_before_block_average',
    block_weight='observed_contiguous_block_frame_count',
    common_frequency_axis='largest_original_block_nfft',
    preserve_candidate_band_neutral_threshold=1e-20,
    min_observed_fraction=MIN_OBSERVED_FRACTION,
    min_segment_seconds=MIN_SEGMENT_SECONDS,
    half_strength_speed_face_per_s=HALF_STRENGTH_SPEED_FACE_PER_S,
    intended_scope='saved_v28_waveform_final_hr_readout_only')


def _column(trace, name, start, stop):
    if name not in trace:
        return np.full(stop-start, np.nan)
    return pd.to_numeric(trace[name].iloc[start:stop], errors='coerce').to_numpy(float)


def motion_evidence(trace, start, stop, fps, grid):
    """Return bounded interference profile plus honest availability metadata.

``profile`` is full-spectrum-normalized motion PSD * strength * reliability, not a
probability of artifact. An unavailable profile is neutral zero for arithmetic;
``available=False`` and missing speed/strength prohibit interpreting it as proof
of low motion. Flat observed motion remains available with a zero spectrum.
"""
    grid=np.asarray(grid,float)
    if (not np.isfinite(fps) or fps<=0 or not isinstance(start,(int,np.integer)) or
            not isinstance(stop,(int,np.integer)) or not 0<=start<stop<=len(trace) or
            grid.ndim!=1 or not len(grid) or not np.isfinite(grid).all() or
            np.any(grid<=0) or np.any(np.diff(grid)<=0) or grid[-1]>=fps*30):
        raise ValueError('Invalid motion window, FPS or ordered sub-Nyquist BPM grid')
    n=stop-start
    result=dict(profile=np.zeros(len(grid)),available=False,observed_fraction=0.,
                speed_face_per_s=np.nan,strength=np.nan,reliability=0.,
                spectral_coverage=0.,status='insufficient_observed_motion',
                normalization_method=NORMALIZATION_CONFIG['method'],
                normalization_includes_dc=True, full_spectrum_peak_power=np.nan,
                query_band_peak_power=np.nan, query_peak_over_full_peak=np.nan)
    dx=_column(trace,'motion_x',start,stop);dy=_column(trace,'motion_y',start,stop)
    height=_column(trace,'face_y1',start,stop)-_column(trace,'face_y0',start,stop)
    valid=np.isfinite(dx)&np.isfinite(dy)&np.isfinite(height)&(height>1e-6)
    result['observed_fraction']=float(valid.mean())
    if valid.mean()<MIN_OBSERVED_FRACTION:
        return result
    velocity=np.full((n,2),np.nan)
    velocity[valid]=np.column_stack([dx[valid],dy[valid]])*fps/height[valid,None]
    edges=np.diff(np.r_[False,valid,False].astype(int))
    starts,stops=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
    spectra=[];counts=[];full_spectra=[];block_frequencies=[]
    for a,b in zip(starts,stops):
        if b-a<max(4,int(np.ceil(MIN_SEGMENT_SECONDS*fps-1e-9))):
            continue
        power=np.zeros(len(grid))
        block=velocity[a:b]
        nfft=max(8192,2**int(np.ceil(np.log2(max(4*len(block),256*fps)))))
        full_power=np.zeros(nfft//2+1)
        frequencies=np.fft.rfftfreq(nfft,1/fps)
        for axis in range(2):
            if np.std(block[:,axis])<1e-10:
                continue
            # Bound FFT frequency spacing in Hz, rather than fixing a sample
            # count that becomes six times coarser at 180 versus 30 FPS. Zero
            # padding improves interpolation only, not physical resolution.
            frequencies,psd=welch(block[:,axis],fs=fps,window='hann',nperseg=len(block),
                                  noverlap=0,nfft=nfft,detrend='constant')
            power+=np.interp(grid/60.,frequencies,psd)
            full_power+=psd
        spectra.append(power);counts.append(b-a)
        full_spectra.append(full_power);block_frequencies.append(frequencies)
    supported=sum(counts)
    result['spectral_coverage']=float(supported/n)
    if supported/n<MIN_OBSERVED_FRACTION:
        result['status']='insufficient_contiguous_motion_support'
        return result
    # RMS vector speed preserves genuine steady translation, but a detrended
    # constant translation has no periodic interference and hence profile zero.
    speed=float(np.sqrt(np.mean(np.sum(velocity[valid]**2,axis=1))))
    strength=float(speed**2/(speed**2+HALF_STRENGTH_SPEED_FACE_PER_S**2))
    reliability=float(supported/n)
    power=np.average(np.stack(spectra),axis=0,weights=counts)
    common=block_frequencies[int(np.argmax([len(f) for f in block_frequencies]))]
    full_power=np.average(np.stack([np.interp(common,f,p)
        for f,p in zip(block_frequencies,full_spectra)]),axis=0,weights=counts)
    full_peak=float(full_power.max());query_peak=float(power.max())
    # Keep the original query-band neutral/flat condition. Only the positive
    # denominator changes, so no formerly neutral query becomes a new risk.
    normalized=power/full_peak if query_peak>1e-20 and full_peak>1e-20 else np.zeros(len(grid))
    result.update(profile=np.clip(normalized*strength*reliability,0.,1.),available=True,
                  speed_face_per_s=speed,strength=strength,reliability=reliability,
                  status='observed_motion' if power.max()>1e-20 else 'observed_flat_motion')
    result.update(full_spectrum_peak_power=full_peak, query_band_peak_power=query_peak,
                  query_peak_over_full_peak=query_peak/full_peak if full_peak>1e-20 else 0.)
    return result

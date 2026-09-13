"""Bound paired-patch integration drift using a slow raw-colour anchor.

The LK correspondence, spatial screening and failure fallback are unchanged.
This is an independently implemented complementary reconstruction, not a
reproduction of a neural-network paper. No HR label or reference enters it.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from baseline_frontend import REGION_NAMES


@dataclass(frozen=True)
class AnchorConfig:
    anchor_hz: float = 0.15
    anchor_order: int = 4

    def __post_init__(self):
        if not np.isfinite(self.anchor_hz) or self.anchor_hz <= 0:
            raise ValueError('anchor_hz must be positive and finite')
        if self.anchor_order != 4:
            raise ValueError('Only the predeclared fourth-order anchor is supported')


def anchored_reconstruction(trace, fps, config=AnchorConfig()):
    """Preserve timestamps/missingness; correct accumulated log-level error.

    l[t] = integral(paired_delta)[t] + LP4(log(baseline)-integral)[t].
    A perfect common pulse present in both inputs is preserved exactly. The
    The fourth-order low-pass reduces original artefact leakage compared to
    a first-order anchor, but is not a hard frequency separation or quality
    guarantee. Its state is causal; the later rPPG pipeline remains offline.
    Missing observations reset state and never become held/zero RGB.
    """
    if not np.isfinite(fps) or fps < 5:
        raise ValueError('Invalid FPS')
    out=trace.copy()
    sos=butter(config.anchor_order,config.anchor_hz,fs=fps,btype='lowpass',output='sos')
    levels={r:None for r in REGION_NAMES}
    filter_states={r:np.zeros((len(sos),2,3)) for r in REGION_NAMES}
    previous={r:None for r in REGION_NAMES}
    previous_frame=None
    result={r:[] for r in REGION_NAMES}
    sources={r:[] for r in REGION_NAMES}
    corrections={r:[] for r in REGION_NAMES}
    for row in trace.to_dict('records'):
        frame=int(row['frame'])
        adjacent=previous_frame is not None and frame==previous_frame+1
        for roi in REGION_NAMES:
            raw=np.asarray([row[f'baseline_{roi}_{c}'] for c in 'rgb'],float)
            valid=bool(row[f'{roi}_valid']) and np.isfinite(raw).all() and (raw>0).all()
            if not valid:
                levels[roi]=previous[roi]=None
                filter_states[roi].fill(0)
                value=np.full(3,np.nan);src='missing';correction=np.nan
            else:
                lograw=np.log(raw)
                if not adjacent or levels[roi] is None or previous[roi] is None:
                    levels[roi]=lograw.copy();src='baseline_reset';correction=0.
                    filter_states[roi].fill(0)
                    value=raw.copy()
                else:
                    delta=np.asarray([row[f'{roi}_pixel_log_delta_{c}'] for c in 'rgb'],float)
                    pixel_source=str(row[f'{roi}_pixel_source'])
                    # Numerical reset concerned the old unbounded integrator;
                    # the paired increment is usable if correspondence passed.
                    tracked=(pixel_source=='tracked_ratio' or
                             (pixel_source=='numerical_reset' and int(row[f'{roi}_pixel_tracks'])>=12))
                    if tracked and np.isfinite(delta).all():
                        src='tracked_ratio'
                    else:
                        delta=lograw-np.log(previous[roi]);src='baseline_ratio_fallback'
                    levels[roi]=levels[roi]+delta
                    drift,filter_states[roi]=sosfilt(sos,(lograw-levels[roi])[None,:],
                                                    axis=0,zi=filter_states[roi])
                    corrected=levels[roi]+drift[0]
                    correction=float(np.linalg.norm(drift[0]))
                    if not np.isfinite(corrected).all() or np.max(np.abs(corrected))>50:
                        levels[roi]=lograw.copy();corrected=lograw.copy();src='numerical_reset'
                        filter_states[roi].fill(0)
                    value=np.exp(corrected)
                previous[roi]=raw.copy()
            result[roi].append(value);sources[roi].append(src);corrections[roi].append(correction)
        previous_frame=frame
    values=[]
    for roi in REGION_NAMES:
        rgb=np.asarray(result[roi],float).reshape(-1,3)
        for j,c in enumerate('rgb'):
            out[f'unanchored_{roi}_{c}']=out[f'{roi}_{c}']
            out[f'{roi}_{c}']=rgb[:,j]
        out[f'{roi}_unanchored_pixel_source']=out[f'{roi}_pixel_source']
        out[f'{roi}_pixel_source']=sources[roi]
        out[f'{roi}_anchor_correction_log']=corrections[roi]
        values.append(rgb)
    stack=np.asarray(values)
    finite=np.isfinite(stack).all(2)
    count=finite.sum(0)
    merged=np.full((len(trace),3),np.nan)
    np.divide(np.where(finite[:,:,None],stack,0).sum(0),count[:,None],
              out=merged,where=count[:,None]>0)
    for j,c in enumerate('rgb'):
        out[f'unanchored_{c}']=out[c];out[c]=merged[:,j]
    out['pixel_rgb_kind']='anchored_paired_relative_colour'
    out['anchor_hz']=float(config.anchor_hz)
    out['anchor_order']=int(config.anchor_order)
    out.attrs.update(trace.attrs,anchor_config=asdict(config))
    return out


def baseline_view(trace):
    """Read the preserved original sampler, without inventing observations."""
    out=trace.copy()
    for roi in REGION_NAMES:
        for c in 'rgb':out[f'{roi}_{c}']=trace[f'baseline_{roi}_{c}']
    for c in 'rgb':out[c]=trace[f'baseline_{c}']
    return out

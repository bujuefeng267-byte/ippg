"""Vectorized measured-colour projections; no HR or reference inputs."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pos_projection(rgb, fps, normalize_overlap=False):
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Finite Nx3 colour measurements required')
    width = max(2, round(1.6*fps))
    if len(rgb) < width:
        return np.zeros(len(rgb),dtype=float)
    windows = sliding_window_view(rgb, width, axis=0)
    normalized = windows/(windows.mean(axis=2,keepdims=True)+1e-9)
    projection = np.array([[0.,1.,-1.],[-2.,1.,1.]])
    projected = np.matmul(projection, normalized)
    first, second = projected[:,0],projected[:,1]
    pulse = first + first.std(axis=1,keepdims=True)/(second.std(axis=1,keepdims=True)+1e-9)*second
    pulse -= pulse.mean(axis=1,keepdims=True)
    output = np.zeros(len(rgb),dtype=float)
    weights = np.zeros(len(rgb),dtype=float)
    # Same ascending-window accumulation order as retained POS.
    indices = np.arange(len(windows))[:,None]+np.arange(width)[None,:]
    np.add.at(output,indices.ravel(),pulse.ravel())
    if normalize_overlap:
        np.add.at(weights,indices.ravel(),1.)
        output /= np.maximum(weights,1.)
    return output


def overlap_count(n, fps):
    width=max(2,round(1.6*fps))
    out=np.zeros(n)
    for a in range(max(0,n-width+1)):
        out[a:a+width]+=1
    return out

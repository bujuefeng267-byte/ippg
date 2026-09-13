"""V2.3 optional paired-pixel front end; V2.2 detector/geometry stays frozen."""
from baseline_frontend import *
from baseline_frontend import extract as baseline_extract
from pixel_tracking import PixelConfig, apply_pixel_tracking


def extract(video,max_seconds=None,qa_dir=None,rgb_aggregation='trimmed_mean',
            pixel_mode='disabled',geometry_trace=None,geometry_fps=None):
    if geometry_trace is None:
        trace,fps=baseline_extract(video,max_seconds,qa_dir,rgb_aggregation)
    else:
        trace,fps=geometry_trace,geometry_fps
    return apply_pixel_tracking(video,trace,fps,pixel_mode,PixelConfig()),fps

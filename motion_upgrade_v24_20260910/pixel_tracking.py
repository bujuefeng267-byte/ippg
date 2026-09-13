"""Paired skin-patch optical flow and common-mode preserving screening.

Inspired by hschn58/rPPG's pixel tracking experiments, independently implemented.
Only paired samples from the same two frames enter a change estimate. Seeds are
renewed each frame, so losing a point never deletes its whole-video history.
The reconstructed RGB is a motion-compensated relative colour signal, not raw
camera RGB. Original baseline RGB, fallback decisions and all timestamps remain.
No physiological reference, HR candidate or target frequency enters this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np
import pandas as pd

from baseline_frontend import REGION_BOXES, REGION_NAMES

MODES = ('disabled', 'tracking_only', 'tracking_screened')


@dataclass(frozen=True)
class PixelConfig:
    max_points_per_roi: int = 100
    patch_radius: int = 3
    min_tracks: int = 12
    fb_max_px: float = 0.5
    screen_z: float = 3.0
    screen_floor: float = 0.003
    tracking_width: int = 960


def paired_log_change(previous_rgb, current_rgb, screened=True, config=PixelConfig()):
    """Robust mean log change of corresponding Nx3 patch samples.

    Screening removes deviations from the *shared* RGB change, not the shared
    change itself. In particular a coherent pulse or lighting change survives.
    This is an engineering outlier score, not a physiological quality metric.
    """
    old, new = np.asarray(previous_rgb, float), np.asarray(current_rgb, float)
    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != 3:
        raise ValueError('Expected two corresponding N x 3 arrays')
    keep = (np.isfinite(old).all(1) & np.isfinite(new).all(1) &
            (old > 20).all(1) & (old < 245).all(1) &
            (new > 20).all(1) & (new < 245).all(1))
    n_finite = int(keep.sum())
    delta = np.full(old.shape, np.nan)
    delta[keep] = np.log(new[keep] / old[keep])
    if screened and n_finite >= config.min_tracks:
        center = np.median(delta[keep], axis=0)
        distance = np.linalg.norm(delta[keep] - center, axis=1)
        middle = np.median(distance)
        scale = 1.4826 * np.median(np.abs(distance - middle))
        cutoff = middle + config.screen_z * max(float(scale), config.screen_floor)
        idx = np.flatnonzero(keep)
        keep[idx[distance > cutoff]] = False
    info = dict(n_input=len(old), n_finite=n_finite, n_kept=int(keep.sum()),
                screen_rejected=n_finite-int(keep.sum()))
    if keep.sum() < config.min_tracks:
        return None, keep, info
    values = np.sort(delta[keep], axis=0)
    cut = int(np.floor(.10*len(values)))
    answer = np.mean(values[cut:len(values)-cut], axis=0, dtype=np.float64)
    info['delta'] = answer.tolist()
    return answer, keep, info


def roi_bounds(row, name, shape):
    h, w = shape[:2]
    box = np.asarray([row[f'face_{s}'] for s in ('x0','y0','x1','y1')], float)
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    x0,y0,x1,y1 = box * [w,h,w,h]
    a,b,c,d = REGION_BOXES[name]
    return np.array([x0+a*(x1-x0), y0+b*(y1-y0),
                     x0+c*(x1-x0), y0+d*(y1-y0)])


def seed_grid(bounds, shape, config):
    if bounds is None:
        return np.empty((0,2), np.float32)
    h,w=shape[:2]
    a,b,c,d=bounds
    margin=config.patch_radius+1
    a,b=max(a+margin,margin),max(b+margin,margin)
    c,d=min(c-margin,w-margin-1),min(d-margin,h-margin-1)
    if c<=a or d<=b:
        return np.empty((0,2),np.float32)
    side=int(np.sqrt(config.max_points_per_roi))
    nx,ny=min(side,int((c-a)/3)+1),min(side,int((d-b)/3)+1)
    x,y=np.meshgrid(np.linspace(a,c,nx),np.linspace(b,d,ny))
    return np.column_stack([x.ravel(),y.ravel()]).astype(np.float32)


def sample_patches(rgb, points, radius):
    """Bilinearly sample 7x7 (default) patches, with no clipping/wraparound."""
    points=np.asarray(points,np.float32).reshape(-1,2)
    out=np.full((len(points),3),np.nan)
    if not len(points):return out
    h,w=rgb.shape[:2]
    valid=(np.isfinite(points).all(1) & (points[:,0]>=radius) &
           (points[:,1]>=radius) & (points[:,0]<w-radius-1) & (points[:,1]<h-radius-1))
    if not valid.any():return out
    dx,dy=np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1))
    idx=np.flatnonzero(valid)
    mx=(points[idx,0,None]+dx.ravel()).astype(np.float32)
    my=(points[idx,1,None]+dy.ravel()).astype(np.float32)
    # Float input is essential: uint8 remap would quantize the subpixel signal.
    values=cv2.remap(np.asarray(rgb,np.float32),mx,my,cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(np.float64)
    ok=np.isfinite(values).all(2) & (values>20).all(2) & (values<245).all(2)
    count=ok.sum(1)
    good=count>=max(25,int(.8*values.shape[1]))
    out[idx[good]]=np.where(ok[...,None],values,0).sum(1)[good]/count[good,None]
    return out


class PixelTracker:
    def __init__(self, config=PixelConfig(), mode='tracking_screened'):
        if mode not in MODES:raise ValueError('Unknown pixel mode')
        self.config,self.mode=config,mode
        self.previous_gray=None
        self.previous_rgb=None
        self.previous_row=None
        self.previous_index=None
        self.level={name:None for name in REGION_NAMES}
        self.lk=dict(winSize=(21,21),maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))

    def _flow(self, gray, points):
        n=len(points)
        moved=np.full((n,2),np.nan,np.float32)
        good=np.zeros(n,bool)
        error=np.full(n,np.nan)
        if not n or self.previous_gray is None:return moved,good,error
        try:
            p1,s1,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,points,None,**self.lk)
            if p1 is None or s1 is None:return moved,good,error
            moved=np.asarray(p1,np.float32).reshape(-1,2)
            h,w=gray.shape
            eligible=(s1.ravel().astype(bool) & np.isfinite(moved).all(1) &
                      (moved[:,0]>=0)&(moved[:,0]<w)&(moved[:,1]>=0)&(moved[:,1]<h))
            ii=np.flatnonzero(eligible)
            if not len(ii):return moved,good,error
            p0,s0,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,moved[ii],None,**self.lk)
            if p0 is None or s0 is None:return moved,good,error
            back=np.asarray(p0).reshape(-1,2)
            error[ii]=np.linalg.norm(back-points[ii],axis=1)
            good[ii]=(s0.ravel().astype(bool) & np.isfinite(error[ii]) &
                       (error[ii]<=self.config.fb_max_px))
        except cv2.error:
            pass
        return moved,good,error

    def update(self, frame_rgb, baseline_row, frame_index):
        row=dict(baseline_row)
        if self.mode=='disabled':return row
        image=np.asarray(frame_rgb)
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected RGB image')
        sampling_image=np.asarray(image,np.float32)
        h,w=image.shape[:2]
        ratio=min(1.,self.config.tracking_width/w)
        small=cv2.resize(image,(round(w*ratio),round(h*ratio))) if ratio<1 else image
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        if gray.dtype!=np.uint8:gray=np.clip(gray,0,255).astype(np.uint8)
        sh,sw=gray.shape
        scale=np.array([w/sw,h/sh])
        consecutive=(self.previous_index is not None and frame_index==self.previous_index+1 and
                     self.previous_gray.shape==gray.shape and self.previous_rgb.shape==image.shape)
        if not consecutive:
            self.level={name:None for name in REGION_NAMES}
        points,parts=[],{}
        for name in REGION_NAMES:
            old_ok=consecutive and bool(self.previous_row[f'{name}_valid'])
            new_ok=bool(row[f'{name}_valid'])
            pts=(seed_grid(roi_bounds(self.previous_row,name,small.shape),small.shape,self.config)
                 if old_ok and new_ok else np.empty((0,2),np.float32))
            parts[name]=(sum(len(p) for p in points),len(pts))
            points.append(pts)
        all_points=np.vstack(points)
        moved,flow_ok,errors=self._flow(gray,all_points) if consecutive else (
            np.full_like(all_points,np.nan),np.zeros(len(all_points),bool),np.full(len(all_points),np.nan))
        estimates=[]
        for name in REGION_NAMES:
            for c in 'rgb':row[f'baseline_{name}_{c}']=row[f'{name}_{c}']
            start,count=parts[name]
            sl=slice(start,start+count)
            tracked=flow_ok[sl].copy()
            q=moved[sl]
            bound=roi_bounds(row,name,small.shape)
            if bound is not None and count:
                a,b,c,d=bound
                tracked &= (q[:,0]>=a)&(q[:,0]<=c)&(q[:,1]>=b)&(q[:,1]<=d)
            src='missing'; delta=None; reset=False
            info=dict(n_input=0,n_finite=0,n_kept=0,screen_rejected=0)
            if tracked.any():
                prev_samples=sample_patches(self.previous_rgb,all_points[sl][tracked]*scale,self.config.patch_radius)
                curr_samples=sample_patches(sampling_image,q[tracked]*scale,self.config.patch_radius)
                delta,keep,info=paired_log_change(prev_samples,curr_samples,
                    self.mode=='tracking_screened',self.config)
            valid=bool(row[f'{name}_valid'])
            current=np.array([row[f'{name}_{c}'] for c in 'rgb'],float)
            previous=(np.array([self.previous_row[f'{name}_{c}'] for c in 'rgb'],float)
                      if consecutive else np.full(3,np.nan))
            if valid and np.isfinite(current).all() and (current>0).all():
                if self.level[name] is None or not consecutive or not np.isfinite(previous).all() or (previous<=0).any():
                    self.level[name]=current.copy();src='baseline_reset';reset=True
                else:
                    if delta is not None:src='tracked_ratio'
                    else:
                        delta=np.log(current/previous);src='baseline_ratio_fallback'
                    updated=self.level[name]*np.exp(delta)
                    if not np.isfinite(updated).all() or (updated<=1e-6).any() or (updated>1e6).any():
                        updated=current.copy();src='numerical_reset';reset=True
                    self.level[name]=updated
                estimates.append(self.level[name])
                for c,value in zip('rgb',self.level[name]):row[f'{name}_{c}']=float(value)
            else:
                self.level[name]=None
                for c in 'rgb':row[f'{name}_{c}']=np.nan
            finite_errors=errors[sl][np.isfinite(errors[sl])]
            row.update({f'{name}_pixel_source':src,f'{name}_pixel_tracks':info['n_kept'],
                f'{name}_pixel_seeds':count,f'{name}_pixel_screen_rejected':info['screen_rejected'],
                f'{name}_pixel_color_rejected':info['n_input']-info['n_finite'],
                f'{name}_pixel_fb_rejected':int(count-flow_ok[sl].sum()),
                f'{name}_pixel_geometry_rejected':int(flow_ok[sl].sum()-tracked.sum()),
                f'{name}_pixel_fb_error_median':float(np.median(finite_errors)) if len(finite_errors) else np.nan,
                f'{name}_pixel_reset':reset,
                f'{name}_pixel_log_delta_r':float(delta[0]) if delta is not None else np.nan,
                f'{name}_pixel_log_delta_g':float(delta[1]) if delta is not None else np.nan,
                f'{name}_pixel_log_delta_b':float(delta[2]) if delta is not None else np.nan})
        for c in 'rgb':row[f'baseline_{c}']=row[c]
        value=np.mean(estimates,axis=0) if estimates else np.full(3,np.nan)
        for c,v in zip('rgb',value):row[c]=float(v)
        row['pixel_mode']=self.mode
        row['pixel_rgb_kind']='paired_relative_color_reconstruction'
        self.previous_rgb=sampling_image.copy(); self.previous_gray=gray.copy()
        self.previous_row=dict(baseline_row);self.previous_index=frame_index
        return row


def apply_pixel_tracking(video, baseline_trace, fps, mode, config=PixelConfig()):
    if mode not in MODES:raise ValueError('Unknown pixel mode')
    if mode=='disabled':return baseline_trace.copy()
    tracker=PixelTracker(config,mode)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError(f'Cannot reopen video: {video}')
    actual_fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isclose(actual_fps,fps,rtol=0,atol=1e-6):
        cap.release();raise ValueError('Geometry cache FPS does not match video')
    rows=[]
    try:
        for i,record in enumerate(baseline_trace.to_dict('records')):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Video ended before geometry trace')
            rows.append(tracker.update(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),record,i))
            if (i+1)%300==0:print(f'{mode}: {i+1} frames',flush=True)
    finally:cap.release()
    result=pd.DataFrame(rows)
    result.attrs.update(baseline_trace.attrs)
    result.attrs.update(pixel_config=asdict(config),pixel_mode=mode)
    return result

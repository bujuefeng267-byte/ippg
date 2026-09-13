"""Render measured patch geometry at fixed timestamps; no HR or reference inputs."""
from pathlib import Path
import json,argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(front,out):
    front=Path(front);out=Path(out)
    meta=json.loads((front/'metadata.json').read_text())
    patches=pd.read_csv(front/'patch_trace.csv')
    cap=cv2.VideoCapture(meta['video_path'])
    indices=[round((meta['frames']-1)*f) for f in (.1,.5,.9)]
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for index,ax in zip(indices,axes):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read()
        if not ok:raise ValueError(f'Cannot decode frame {index}')
        ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        rows=patches[patches.frame==index]
        for row in rows.itertuples():
            q=np.array([[getattr(row,f'corner{i}_x'),getattr(row,f'corner{i}_y')] for i in range(4)])
            if not np.isfinite(q).all():continue
            q=np.vstack([q,q[0]])
            color=dict(forehead='cyan',left_cheek='orange',right_cheek='lime')[row.region]
            ax.plot(q[:,0],q[:,1],color=color,linewidth=1)
            ax.text(q[0,0],q[0,1],row.patch_id[-1],color=color,fontsize=7)
        ax.set_title(f'{index/meta["fps"]:.2f} s | valid patches {rows.valid.sum()}/12')
        ax.axis('off')
    cap.release()
    fig.suptitle(front.name+' | fixed local skin patches, original image coordinates')
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=140);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('frontend');p.add_argument('output')
    a=p.parse_args();render(a.frontend,a.output)

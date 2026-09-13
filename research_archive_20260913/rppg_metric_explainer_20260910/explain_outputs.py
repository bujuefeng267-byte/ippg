from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'rppg_motion_v22/validation/trimmed_gap15/user0907'

def runs(mask):
    d=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def main():
    meta=json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    tr=pd.read_csv(SOURCE/'frame_trace.csv')
    wave=pd.read_csv(SOURCE/'fusion_waveform.csv')
    hr=pd.read_csv(SOURCE/'fusion_heart_rate.csv')
    fps=meta['fps']; duration=len(tr)/fps
    valid=np.isfinite(wave.base.to_numpy(float))
    rgb=tr.rgb_valid.to_numpy(bool)
    accepted=hr.accepted.to_numpy(bool)
    assert len(tr)==len(wave)==1557 and len(hr)==42
    assert np.array_equal(valid,wave.covered.to_numpy(bool))
    assert valid.sum()==1260 and rgb.sum()==1415 and accepted.sum()==25
    assert wave.loc[~valid,'base'].isna().all()
    assert hr.loc[~accepted,'ridge_bpm'].isna().all()
    font=Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(3,1,figsize=(13.5,8.4),sharex=True,layout='constrained',
                        gridspec_kw={'height_ratios':[.8,2.2,2.0]})
    for a,b in runs(np.ones(len(rgb),bool)):
        ax[0].broken_barh([(a/fps,(b-a)/fps)],(0,1),facecolors='#e4e7eb')
    for a,b in runs(rgb):
        ax[0].broken_barh([(a/fps,(b-a)/fps)],(0,1),facecolors='#248769')
    ax[0].set(yticks=[],ylim=(-.1,1.1),title='人脸区域可采样：1415 / 1557 帧 = 90.88%（包含光流跟踪）')
    ax[0].legend(handles=[Patch(color='#248769',label='有真实 RGB 采样'),Patch(color='#e4e7eb',label='前端无可用人脸区域')],
                 loc='lower right',ncols=2,fontsize=9)
    for a,b in runs(~valid):
        ax[1].axvspan(a/fps,b/fps,color='#e4e7eb',alpha=.8)
    ax[1].plot(wave.time_s,wave.base,color='#216cb1',lw=.85)
    ax[1].axhline(0,color='#90949a',lw=.5,alpha=.5)
    ax[1].set(ylabel='归一化幅值（非绝对光强）',title='最终 rPPG 波形：1260 / 1557 点 = 80.92%，约 2–44 秒有输出')
    ax[1].text(.02,.95,'灰区 = 缺失值 NaN，曲线断开；不是填 0',transform=ax[1].transAxes,va='top',fontsize=10,
               bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax[2].plot(hr.time_s,hr.ridge_bpm.where(hr.accepted),marker='o',ms=4,lw=1.2,color='#216cb1',label='接受窗口的离线心率')
    for i in np.flatnonzero(~accepted):
        t=hr.time_s.iloc[i]
        ax[2].axvspan(t-.35,t+.35,color='#e4e7eb',alpha=.7)
    ax[2].set(ylabel='心率（bpm）',xlabel='原视频时间 / 心率窗口中心（秒）',
              title='最终心率：25 / 42 个计划窗口 = 59.52%（每窗 10 秒，每秒评估一次）')
    ax[2].text(.02,.95,'无参考心率：点表示程序输出，不表示已经测准',transform=ax[2].transAxes,va='top',fontsize=10,
               bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax[2].set_ylim(42,210)
    for a in ax:
        a.set_xlim(0,duration);a.spines[['top','right']].set_visible(False)
        a.grid(axis='x',alpha=.15)
    fig.suptitle('09-07 运动视频 · V2.2（0.15 秒短缺口设置）\n同一时间轴：可采样人脸 → 保留波形 → 通过条件的心率窗口',fontsize=15)
    fig.savefig(HERE/'最新版本_输出与缺失时间轴.png',dpi=160)
    plt.close(fig)
    examples=[]
    for t in [0,1,2,10,44,50]:
        i=round(t*fps)
        r=wave.iloc[i]
        examples.append({'frame':i,'time_s':float(r.time_s),'waveform_base':None if pd.isna(r.base) else float(r.base),
                         'rgb_valid':bool(rgb[i]),'covered':bool(r.covered),'observed':bool(r.observed),'interpolated':bool(r.interpolated)})
    evidence={'source_directory':str(SOURCE),'version':'V2.2 trimmed_gap15','fps':fps,'frames':len(tr),
        'duration_s':duration,'rgb_available_frames':int(rgb.sum()),'waveform_finite_samples':int(valid.sum()),
        'waveform_observed_samples':int(wave.observed.sum()),'waveform_interpolated_samples':int(wave.interpolated.sum()),
        'waveform_absent_samples':int((~valid).sum()),'wave_absent_but_rgb_present':int((~valid&rgb).sum()),
        'wave_and_rgb_absent':int((~valid&~rgb).sum()),'waveform_spans_s':[[a/fps,b/fps] for a,b in runs(valid)],
        'hr_accepted_windows':int(accepted.sum()),'hr_planned_windows':len(hr),'hr_status_counts':hr.status.value_counts().to_dict(),
        'hr_accepted_center_spans_s':[[float(hr.time_s.iloc[a]),float(hr.time_s.iloc[b-1])] for a,b in runs(accepted)],
        'examples':examples,'reference_available':False,'accuracy_claim':False,
        'wave_missing_encoded_as':'empty CSV cell / NaN, never forced to 0',
        'waveform_coverage_is_morphological_quality':False,
        'checks_passed':True,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [SOURCE/'summary.json',SOURCE/'frame_trace.csv',SOURCE/'fusion_waveform.csv',SOURCE/'fusion_heart_rate.csv']}}
    (HERE/'核对数据.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('source_hashes','examples')},ensure_ascii=True))

if __name__=='__main__':main()

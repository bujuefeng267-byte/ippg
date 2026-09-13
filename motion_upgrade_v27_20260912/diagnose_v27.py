"""Post-evaluation descriptive diagnostics; never changes predictions or parameters."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from motion_evidence import motion_evidence

ROOT=Path('/home/fengbujue/项目/rppg识别/results/data1_6_v27_20260912')

def main():
    rows=[];summaries=[];hashes={}
    for i in range(1,7):
        case=f'data{i}';front=ROOT/'frontend'/case;out=ROOT/'late_psd_cluster'/case
        paths=[front/'frame_trace.csv',front/'metadata.json',out/'evaluation/paired_windows.csv',out/'patch_diagnostics.csv']
        for p in paths:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        trace=pd.read_csv(paths[0]);fps=json.loads(paths[1].read_text())['fps']
        pair=pd.read_csv(paths[2]);diag=pd.read_csv(paths[3]);grid=np.arange(42.,211.)
        for r in pair.itertuples():
            start=round(r.window_start_s*fps);stop=start+round(10*fps)
            motion=motion_evidence(trace,start,stop,fps,grid)
            profile=motion['profile'];peak=grid[np.argmax(profile)] if profile.max()>0 else np.nan
            hr=r.estimated_bpm
            at_hr=np.interp(hr,grid,profile)/profile.max() if np.isfinite(hr) and profile.max()>0 else np.nan
            selected=diag[(diag.record_type=='contributor')&(diag.window_index==r.window_index)]
            tracked=selected.channel.str.startswith('tracked/').mean() if len(selected) else np.nan
            rows.append(dict(case=case,window_index=r.window_index,time_s=r.time_s,
                accepted=r.accepted,estimated_bpm=hr,reference_bpm=r.reference_bpm,
                error_bpm=hr-r.reference_bpm,main_motion_peak_bpm=peak,
                motion_strength=motion['strength'],motion_available=motion['available'],
                relative_motion_power_at_estimate=at_hr,contributor_tracked_fraction=tracked))
        part=pd.DataFrame(rows);part=part[(part.case==case)&part.accepted]
        summaries.append(dict(case=case,accepted_windows=len(part),
            HR_median_bpm=part.estimated_bpm.median(),reference_median_on_same_windows=part.reference_bpm.median(),
            above_reference_by_more_than_20_bpm=int((part.error_bpm>20).sum()),
            below_reference_by_more_than_20_bpm=int((part.error_bpm < -20).sum()),
            median_motion_strength=part.motion_strength.median(),
            median_relative_motion_power_at_estimate=part.relative_motion_power_at_estimate.median(),
            median_tracked_contributor_fraction=part.contributor_tracked_fraction.median()))
    out=ROOT/'post_evaluation_diagnostics';out.mkdir()
    pd.DataFrame(rows).to_csv(out/'late_cluster_motion_and_branch.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    (out/'provenance.json').write_text(json.dumps(dict(input_hashes=hashes,
        post_evaluation_only=True,parameters_changed=False,
        interpretation='Descriptive associations, not causal isolation. Motion profile is a geometric proxy; relative PSD is not artifact probability. No prediction or threshold is altered.'),indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()

"""Read-only reconstruction of the two rejection stages for the frozen video."""
from pathlib import Path
import csv
import json
import hashlib
from collections import Counter, defaultdict

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'rppg_motion_v22/validation/trimmed_gap15/user0907'

def read(name):
    with (SOURCE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    diag=read('fusion_diagnostics.csv')
    proposal=read('fusion_proposals.csv')
    hr=read('fusion_heart_rate.csv')
    unique={}
    for row in diag:
        key=(int(row['window_index']),row['channel'])
        if key in unique:
            old=unique[key]
            assert old['channel_status']==row['channel_status']
            assert old['observed_fraction']==row['observed_fraction']
        unique[key]=row
    groups=defaultdict(list)
    for (wi,ch),row in unique.items():groups[wi].append(row)
    assert len(proposal)==len(hr)==42 and len(unique)==252
    output=[]
    for i,(p,h) in enumerate(zip(proposal,hr)):
        rows=groups[i]
        assert len(rows)==6 and p['time_s']==h['time_s']
        states=Counter(r['channel_status'] for r in rows)
        output.append(dict(window_index=i,time_s=p['time_s'],window_start_s=p['window_start_s'],
            window_end_s=p['window_end_s'],channel_status_counts=json.dumps(dict(states)),
            original_roi_observed_fraction_min=min(float(r['observed_fraction']) for r in rows),
            original_roi_observed_fraction_max=max(float(r['observed_fraction']) for r in rows),
            waveform_proposal_status=p['proposal_status'],waveform_generated=p['waveform_generated'],
            final_hr_status=h['status'],final_hr_accepted=h['accepted'],
            final_wave_observed_fraction=h['observed_fraction'],
            final_wave_interpolated_fraction=h['interpolated_fraction']))
    with (HERE/'拒绝原因逐窗核对.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=output[0].keys());writer.writeheader();writer.writerows(output)
    insufficient=[i for i,p in enumerate(proposal) if p['proposal_status']=='insufficient_rois']
    breakdown=Counter()
    for i in insufficient:
        states={r['channel_status'] for r in groups[i]}
        assert len(states)==1
        breakdown[next(iter(states))]+=1
    observed=Counter(r['channel_status'] for r in unique.values())
    record=dict(passed=True,version='V2.2 trimmed_gap15/user0907',
        raw_diagnostic_rows=len(diag),unique_window_channels=len(unique),
        channel_counts=dict(observed),waveform_proposal_counts=dict(Counter(p['proposal_status'] for p in proposal)),
        insufficient_rois_window_breakdown=dict(breakdown),
        final_hr_counts=dict(Counter(h['status'] for h in hr)),
        stages_are_not_additive=True,physical_cause_not_proven=True,
        source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in
            ['fusion_diagnostics.csv','fusion_proposals.csv','fusion_heart_rate.csv']})
    (HERE/'拒绝原因核对.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=True))

if __name__=='__main__':main()

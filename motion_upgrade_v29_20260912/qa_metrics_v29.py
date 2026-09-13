"""Small independent numeric audit of saved V29 predictions and references."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

P = Path('/home/fengbujue/项目/rppg识别')
B = P/'results/data1_6_20260911'
OLD = P/'results/data1_6_v28_20260912/direct_guard'
ROOT = P/'results/data1_6_v29_20260912'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, (bool,np.bool_)): return bool(x)
    if isinstance(x, (int,np.integer)): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return x


def flags(s):
    assert s.notna().all() and s.isin([True,False,0,1]).all()
    return s.to_numpy(bool)


def score(y, ok, ref):
    eligible = np.isfinite(ref) & (ref > 0)
    valid = ok & np.isfinite(y) & eligible
    ae = abs(y[valid]-ref[valid])
    return dict(Nplanned=len(y),Noutput=int(ok.sum()),Nref=int(eligible.sum()),Nvalid=int(valid.sum()),
        Nwithin5=int((ae <= 5).sum()), MAE_bpm=float(ae.mean()) if len(ae) else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(ae**2))) if len(ae) else np.nan,
        P5_valid_pct=100*float((ae <= 5).mean()) if len(ae) else np.nan,
        R5_all_reference_pct=100*int((ae <= 5).sum())/int(eligible.sum()) if eligible.any() else np.nan,
        hr_output_coverage_pct=100*float(ok.mean()) if len(ok) else np.nan)


def references(raw, origin, starts, ends):
    t = (raw.host_utc_ns.to_numpy(np.int64)-np.int64(origin)).astype(float)/1e9
    y = raw.hr_bpm.to_numpy(float)
    keep = np.isfinite(y) & (y > 0)
    t,y = t[keep],y[keep]
    assert len(t) > 1 and np.all(np.diff(t) > 0)
    tail = np.median(np.diff(t))
    values = []
    for start,end in zip(starts,ends):
        chosen = (t >= start) & (t < end)
        points = t[chosen]
        valid = start >= t[0] and end <= t[-1]+tail and len(points) >= 2
        valid = valid and np.max(np.diff(np.r_[start,points,end])) <= 2
        values.append(float(y[chosen].mean()) if valid else np.nan)
    return np.array(values)


def pool(rows):
    n = sum(x['Nvalid'] for x in rows)
    nr = sum(x['Nref'] for x in rows)
    n5 = sum(x['Nwithin5'] for x in rows)
    return dict(Nplanned=sum(x['Nplanned'] for x in rows),Noutput=sum(x['Noutput'] for x in rows),
        Nref=nr,Nvalid=n,Nwithin5=n5,
        MAE_bpm=sum(x['Nvalid']*x['MAE_bpm'] for x in rows if x['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(x['Nvalid']*x['RMSE_bpm']**2 for x in rows if x['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*n5/n if n else np.nan,R5_all_reference_pct=100*n5/nr if nr else np.nan,
        hr_output_coverage_pct=100*sum(x['Noutput'] for x in rows)/sum(x['Nplanned'] for x in rows))


def main():
    receipt = ROOT/'qa_metrics_v29.json'
    assert not receipt.exists(), 'Preserve existing QA receipt'
    report_path = ROOT/'evaluation_summary.json'
    report = json.loads(report_path.read_text())
    hashes = {str(report_path):sha(report_path), str(Path(__file__)):sha(__file__)}
    errors, detail, comparison_count, maximum_difference = [], {}, 0, 0.

    def bind(path):
        hashes[str(path)] = sha(path)

    def check(actual, expected, label):
        nonlocal comparison_count,maximum_difference
        comparison_count += 1
        a = np.nan if actual is None else float(actual)
        e = np.nan if expected is None else float(expected)
        if np.isnan(a) and np.isnan(e): return
        if np.isfinite(a) and np.isfinite(e):
            maximum_difference = max(maximum_difference,abs(a-e))
            if np.isclose(a,e,atol=1e-9,rtol=1e-11): return
        errors.append(dict(label=label,saved=actual,recomputed=expected))

    def compare_dict(saved, expected, label):
        for key,value in expected.items(): check(saved.get(key),value,label+'/'+key)

    for profile in ('short6','short6_relaxed'):
        rows, data6 = [], {}
        for i in range(1,7):
            case = f'data{i}'
            folder = ROOT/profile/case
            hp,op = folder/'heart_rate_10s.csv',OLD/case/'heart_rate.csv'
            ap = B/case/'evaluation/alignment.json'
            alignment = json.loads(ap.read_text())
            rawp = Path(alignment['reference_path'])
            h,oh,raw = pd.read_csv(hp),pd.read_csv(op),pd.read_csv(rawp)
            for path in (hp,op,ap,rawp): bind(path)
            np.testing.assert_allclose(h.time_s,oh.time_s,atol=1e-8,rtol=0)
            np.testing.assert_allclose(h.window_start_s,oh.window_start_s,atol=1e-8,rtol=0)
            np.testing.assert_allclose(h.window_end_s,oh.window_end_s,atol=1e-8,rtol=0)
            r = references(raw,alignment['video_start_utc_ns'],h.window_start_s,h.window_end_s)
            y,ok,oldok = h.ridge_bpm.to_numpy(float),flags(h.accepted),flags(oh.accepted)
            assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()
            m = score(y,ok,r); rows.append(m)
            saved = next(x for x in report['profiles'][profile]['common10s']['cases'] if x['case'] == case)
            compare_dict(saved,m,profile+'/'+case+'/common10s')
            if case == 'data6':
                new = ok & ~oldok
                new_metric = score(y[new],ok[new],r[new])
                common = ok & oldok
                compare_dict(saved,dict(new_windows=int(new.sum()),new_MAE_bpm=new_metric['MAE_bpm'],
                    new_Nwithin5=new_metric['Nwithin5'],new_P5_valid_pct=new_metric['P5_valid_pct'],
                    common_windows=int(common.sum()), common_MAE_bpm=score(y[common],ok[common],r[common])['MAE_bpm']),profile+'/data6/common10s_added')
                data6.update(common10s=m,newly_covered_common10s=new_metric)
                nativep,wavep,oldwavep = folder/'heart_rate.csv',folder/'waveform.csv',OLD/case/'waveform.csv'
                meta_path = B/case/'inference/summary.json'
                native,wave,oldwave = pd.read_csv(nativep),pd.read_csv(wavep),pd.read_csv(oldwavep)
                fps = float(json.loads(meta_path.read_text())['fps'])
                for path in (nativep,wavep,oldwavep,meta_path): bind(path)
                native_r = references(raw,alignment['video_start_utc_ns'],native.window_start_s,native.window_end_s)
                nm = score(native.ridge_bpm.to_numpy(float),flags(native.accepted),native_r)
                compare_dict(report['profiles'][profile]['data6_detail']['native6s'],nm,profile+'/data6/native6s')
                data6['native6s'] = nm
                for key,table in [('waveform',wave),('V28_waveform',oldwave)]:
                    finite = np.isfinite(table.base)
                    np.testing.assert_array_equal(finite,flags(table.covered))
                    np.testing.assert_allclose(table.time_s,np.arange(len(table))/fps,atol=1e-8,rtol=0)
                    indices = np.flatnonzero(finite)
                    missing_edges = np.diff(np.r_[False,~finite,False].astype(int))
                    gap_lengths = np.flatnonzero(missing_edges == -1)-np.flatnonzero(missing_edges == 1)
                    wm = dict(total_duration_s=len(table)/fps,finite_frames=int(finite.sum()),
                        finite_duration_s=int(finite.sum())/fps,waveform_coverage_pct=100*float(finite.mean()),
                        first_finite_time_s=float(table.time_s.iloc[indices[0]]) if len(indices) else None,
                        missing_duration_s=int((~finite).sum())/fps,longest_missing_s=float(gap_lengths.max())/fps if len(gap_lengths) else 0.)
                    compare_dict(report['profiles'][profile]['data6_detail']['availability'][key],wm,profile+'/data6/'+key)
                    data6[key] = wm
                for key,table in [('native6s_hr',native),('common10s_hr',h),('V28_hr',oh)]:
                    first = table.loc[flags(table.accepted)].iloc[0]
                    first_values = dict(first_accepted_center_s=float(first.time_s),
                        first_accepted_window_start_s=float(first.window_start_s),first_accepted_window_end_s=float(first.window_end_s))
                    compare_dict(report['profiles'][profile]['data6_detail']['availability'][key],first_values,profile+'/data6/'+key)
                    data6[key] = first_values
        pooled = pool(rows)
        compare_dict(report['profiles'][profile]['common10s']['pooled'],pooled,profile+'/all_six_common10s_pooled')
        detail[profile] = dict(data6=data6,all_six_common10s_pooled=pooled)
    for path,value in hashes.items():
        assert sha(path) == value, 'QA input mutated during audit: '+path
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not errors,errors=errors,
        comparisons=comparison_count,maximum_absolute_difference=maximum_difference,
        comparison_scope=['Both profiles: data6 native 6s and strict common 10s MAE/P5/R5/coverage.',
            'Both profiles: data6 newly covered common 10s accuracy and common-output accuracy.',
            'Both profiles: data6/V28 waveform durations, coverage, first finite point and longest missing gap.',
            'Both profiles: data6 native/common10/V28 first accepted HR center/start/end.',
            'Both profiles: all six common10 case metrics and pooled metrics (309 original planned windows).',
            'References recomputed directly from original Polar int64 timestamps and unchanged alignment; no evaluator/model imports, no offset search.'],
        results=detail,source_hashes=hashes)
    receipt.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean(dict(passed=not errors,comparisons=comparison_count,maximum_absolute_difference=maximum_difference,
        errors=errors,results=detail,receipt=str(receipt))),ensure_ascii=False),flush=True)
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()

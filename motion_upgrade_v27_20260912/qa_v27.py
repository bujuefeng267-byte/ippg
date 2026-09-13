"""Independent V27 input, reference, metric and result QA.

Evaluation only: this module must never be imported by video/patch inference.
It does not import the production evaluator or choose a reference offset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/fengbujue/项目/rppg识别')
BASE = PROJECT/'results/data1_6_20260911'
V25 = PROJECT/'results/data1_6_v25_20260911/stage4_preserve_waveform'
EXPECTED_FRAMES = dict(zip([f'data{i}' for i in range(1, 7)], [1555, 1393, 2221, 10693, 1866, 2148]))
EXPECTED_WINDOWS = dict(zip(EXPECTED_FRAMES, [42, 37, 65, 50, 53, 62]))
SPATIAL=('early_median','early_psd_cluster','late_median','late_psd_cluster')
READOUT=('fusion_nomotion_local','fusion_motion_local','fusion_nomotion_dp','fusion_motion_dp')
REGIONS=('forehead','left_cheek','right_cheek')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save(path, data):
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def bools(series):
    if series.isna().any() or not series.isin([True, False, 0, 1]).all():
        raise AssertionError(f'Invalid boolean values in {series.name}')
    return series.to_numpy(bool)


def score(pred, accepted, reference):
    """Independent denominator arithmetic; missing output never counts correct."""
    pred, accepted, reference = np.asarray(pred, float), np.asarray(accepted, bool), np.asarray(reference, float)
    if pred.shape != accepted.shape or pred.shape != reference.shape or pred.ndim != 1:
        raise AssertionError('Metric arrays must share one complete window axis')
    valid_reference = np.isfinite(reference) & (reference > 0)
    output = accepted & np.isfinite(pred)
    paired = output & valid_reference
    error = pred[paired]-reference[paired]
    n, no, nr, nv = len(pred), int(output.sum()), int(valid_reference.sum()), int(paired.sum())
    within5 = int((np.abs(error) <= 5).sum())
    return dict(Nplanned=n, Noutput=no, Nref=nr, Nvalid=nv, Nwithin5=within5,
        MAE_bpm=float(np.abs(error).mean()) if nv else np.nan,
        RMSE_bpm=float(np.sqrt(np.mean(error**2))) if nv else np.nan,
        Bias_bpm=float(error.mean()) if nv else np.nan,
        P5_valid_pct=100*within5/nv if nv else np.nan,
        R5_all_reference_pct=100*within5/nr if nr else np.nan,
        hr_output_coverage_pct=100*no/n if n else np.nan,
        paired_reference_coverage_pct=100*nv/nr if nr else np.nan)


def pool(rows):
    result = {k: sum(row[k] for row in rows) for k in
              ('Nplanned', 'Noutput', 'Nref', 'Nvalid', 'Nwithin5')}
    n, nr = result['Nvalid'], result['Nref']
    result.update(MAE_bpm=sum(r['MAE_bpm']*r['Nvalid'] for r in rows if r['Nvalid'])/n if n else np.nan,
        RMSE_bpm=np.sqrt(sum(r['RMSE_bpm']**2*r['Nvalid'] for r in rows if r['Nvalid'])/n) if n else np.nan,
        P5_valid_pct=100*result['Nwithin5']/n if n else np.nan,
        R5_all_reference_pct=100*result['Nwithin5']/nr if nr else np.nan,
        HR_coverage_pct=100*result['Noutput']/result['Nplanned'] if result['Nplanned'] else np.nan)
    return result


def promotion(rows, pooled, baseline):
    """The same eight V26 guards; no per-case variant selection permitted."""
    common_n = sum(r['common_windows'] for r in rows)
    new_common = sum(r['common_windows']*r['common_MAE_bpm'] for r in rows if r['common_windows'])/common_n if common_n else np.nan
    old_common = sum(r['common_windows']*r['common_V25_MAE_bpm'] for r in rows if r['common_windows'])/common_n if common_n else np.nan
    return dict(pooled_MAE=pooled['MAE_bpm'] < baseline['MAE_bpm'],
        pooled_R5_gain=pooled['R5_all_reference_pct'] >= baseline['R5_all_reference_pct']+5,
        each_MAE=all(r['delta_MAE_bpm'] <= 3 for r in rows),
        each_P5=all(r['delta_P5_pp'] >= -3 for r in rows),
        each_R5=all(r['delta_R5_pp'] >= -3 for r in rows),
        each_HR_coverage=all(r['delta_hr_coverage_pp'] >= -3 for r in rows),
        each_waveform_coverage=all(r['delta_waveform_coverage_pp'] >= -3 for r in rows),
        common_MAE=new_common < old_common)


def independent_reference(raw, origin_ns, starts, ends):
    """Original half-open windows and reference-quality rule, no offset search."""
    absolute = raw.host_utc_ns.to_numpy(np.int64)
    if not np.all(np.diff(absolute) > 0):
        raise AssertionError('Reference timestamps duplicated or unordered')
    times = (absolute-np.int64(origin_ns)).astype(float)/1e9
    hr = raw.hr_bpm.to_numpy(float)
    good = np.isfinite(hr) & (hr > 0)
    times, hr = times[good], hr[good]
    if len(times) < 2:
        raise AssertionError('Reference requires at least two valid notifications')
    tail = np.median(np.diff(times))
    estimates, accepted = [], []
    for start, end in zip(starts, ends):
        used = (times >= start) & (times < end)
        points = times[used]
        maxgap = np.max(np.diff(np.r_[start, points, end]))
        valid = start >= times[0] and end <= times[-1]+tail and len(points) >= 2 and maxgap <= 2
        accepted.append(valid)
        estimates.append(float(hr[used].mean()) if valid else np.nan)
    return np.asarray(estimates), np.asarray(accepted, bool)


def assert_hr(table, starts, fps, frames):
    if len(table) != len(starts):
        raise AssertionError('HR table does not retain all planned windows')
    np.testing.assert_allclose(table.time_s, (starts+round(10*fps)/2)/fps, atol=1e-8, rtol=0)
    accepted = bools(table.accepted)
    value = table.ridge_bpm.to_numpy(float)
    if not np.isnan(value[~accepted]).all() or not np.isfinite(value[accepted]).all():
        raise AssertionError('Main HR missingness disagrees with accepted')
    if np.any((value[accepted] < 42) | (value[accepted] > 210)):
        raise AssertionError('Main HR exceeds fixed range')
    return value, accepted


def _runs(mask):
    edge=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)))


def qa_spectral_summary(grid, power, cfg):
    k=int(np.argmax(power));peak=float(grid[k]);half=power[k]/2
    left=right=k
    while left>0 and power[left-1]>=half:left-=1
    while right<len(power)-1 and power[right+1]>=half:right+=1
    lo=float(grid[left]);hi=float(grid[right])
    if left>0 and power[left]!=power[left-1]:
        lo=float(np.interp(half,power[left-1:left+1],grid[left-1:left+1]))
    if right<len(power)-1 and power[right]!=power[right+1]:
        hi=float(np.interp(half,power[right:right+2][::-1],grid[right:right+2][::-1]))
    width=max(float(grid[1]-grid[0]),hi-lo)
    concentration=float(power[np.abs(grid-peak)<=cfg['concentration_radius_bpm']].sum()/power.sum())
    return dict(peak_bpm=peak,peak_concentration=concentration,peak_fwhm_bpm=width,
                spectral_score=concentration/(1+width/cfg['width_scale_bpm']))


def qa_spectrum(samples, fps, cfg):
    x=np.asarray(samples,float)
    if x.std()<cfg['flat_std']:return None
    grid=np.arange(cfg['min_bpm'],cfg['max_bpm']+.001,cfg['grid_step_bpm'])
    nfft=max(cfg['welch_nfft'],2**int(np.ceil(np.log2(max(4*len(x),256*fps)))))
    frequencies,power=welch(x,fs=fps,window='hann',nperseg=len(x),noverlap=0,
                            nfft=nfft,detrend='constant')
    band=np.interp(grid/60,frequencies,power)
    if not np.isfinite(band).all() or np.linalg.norm(band)<=0 or band.sum()<=0:return None
    return dict(grid=grid,power_fraction=band/band.sum(),shape=band/np.linalg.norm(band),
                **qa_spectral_summary(grid,band,cfg))


def qa_complete_clusters(items, cutoff):
    """Independent complete-link construction using full spectral vectors."""
    shapes=np.asarray([item['shape'] for item in items])
    shapes=shapes/np.linalg.norm(shapes,axis=1)[:,None]
    distances=np.clip(1-shapes@shapes.T,0,2)
    groups=[(i,) for i in range(len(items))]
    while True:
        eligible=[]
        for i,a in enumerate(groups):
            for j in range(i+1,len(groups)):
                b=groups[j];distance=max(float(distances[x,y]) for x in a for y in b)
                if distance<=cutoff+1e-12:eligible.append((distance,a,b,i,j))
        if not eligible:break
        _,a,b,i,j=min(eligible)
        groups=sorted([g for k,g in enumerate(groups) if k not in (i,j)]+[tuple(sorted(a+b))])
    return groups


def audit_patch_result(directory, signal_table, metadata, trace, fps, cfg):
    """Recompute patch spectra/aggregation and replay measured broadband OLA.

    No production patch inference or evaluation function is imported. The
    saved contributor signs identify the actual chosen polarity; all signal
    samples and spectral evidence are independently read from the BVP CSV.
    """
    hr=pd.read_csv(directory/'heart_rate.csv');wave=pd.read_csv(directory/'waveform.csv')
    diagnostics=pd.read_csv(directory/'patch_diagnostics.csv')
    n=len(trace);window=round(cfg['window_s']*fps);hop=round(cfg['step_s']*fps)
    starts=np.arange(0,n-window+1,hop);pred,accepted=assert_hr(hr,starts,fps,n)
    regions=metadata['patch_regions'];keys=sorted(metadata['channel_order'])
    np.testing.assert_allclose(signal_table.time_s,trace.time_s,atol=1e-8,rtol=0)
    if len(signal_table)!=n:raise AssertionError('Patch BVP lost original frames')
    masks={};quality={};sources={};safe={}
    for patch in regions:
        obs=bools(signal_table[f'{patch}/observed']);fill=bools(signal_table[f'{patch}/interpolated'])
        if (obs&fill).any():raise AssertionError('Observed/interpolated patch samples overlap')
        q=signal_table[f'{patch}/quality'].to_numpy(float)
        if not np.isfinite(q).all() or ((q<0)|(q>1)).any():raise AssertionError('Invalid patch quality')
        allowed=np.zeros(n,bool)
        for a,b in _runs(~obs):
            if a>0 and b<n and b-a<=np.floor(cfg['max_gap_s']*fps+1e-9):allowed[a:b]=True
        masks[patch]=(obs,fill);safe[patch]=allowed&fill;quality[patch]=q
        sources[patch]=signal_table[f'{patch}/tracking_source'].to_numpy(str)
    grouped={int(k):v for k,v in diagnostics.groupby('window_index')}
    total=np.zeros(n);weights=np.zeros(n);observed=np.ones(n,bool);filled=np.zeros(n,bool)
    taper=np.hanning(window+2)[1:-1];tie_records=[];channels_verified=0;clusters_verified=0
    for wi,a in enumerate(starts):
        b=a+window;rows=grouped.get(wi,pd.DataFrame(columns=diagnostics.columns))
        dchannels=rows.loc[rows.record_type=='channel']
        if len(dchannels)!=len(keys) or dchannels.channel.duplicated().any():raise AssertionError('Missing/duplicate channel decisions')
        dchannels=dchannels.set_index('channel');eligible={};by_patch={}
        for key in keys:
            branch,patch,_=key.split('/');obs,fill=masks[patch];x=signal_table[key].to_numpy(float)[a:b]
            qmean=float(quality[patch][a:b].mean());src=sources[patch][a:b]
            tracked=float((src=='tracked_ratio').mean());reset=float(np.isin(src,['baseline_reset','numerical_reset']).mean())
            status='eligible'
            if not np.isfinite(x).all():status='nonfinite_bvp'
            elif obs[a:b].mean()<cfg['min_observed']:status='insufficient_observed'
            elif not (obs[a:b]|safe[patch][a:b]).all():status='unverified_or_long_interpolation'
            elif qmean<cfg['min_quality']:status='low_quality'
            elif branch=='tracked' and (tracked<cfg['min_tracked_fraction'] or reset>cfg['max_reset_fraction']):status='tracking_ineligible'
            spectrum=qa_spectrum(x,fps,cfg) if status=='eligible' else None
            if status=='eligible' and spectrum is None:status='flat_or_zero_band_signal'
            row=dchannels.loc[key]
            if row.status!=status:raise AssertionError(f'Channel gate mismatch {wi}/{key}: {row.status} vs {status}')
            for field,value in [('observed_fraction',obs[a:b].mean()),('mean_quality',qmean),('tracked_fraction',tracked),('reset_fraction',reset)]:
                np.testing.assert_allclose(row[field],value,atol=1e-10,rtol=0)
            if spectrum is not None:
                for field in ('peak_bpm','peak_concentration','peak_fwhm_bpm','spectral_score'):
                    np.testing.assert_allclose(row[field],spectrum[field],atol=1e-9,rtol=0)
                item=dict(channel=key,patch_id=patch,region=regions[patch],**spectrum)
                eligible[key]=item;by_patch.setdefault(patch,[]).append(item)
            channels_verified+=1
        marked=dchannels.index[dchannels.representative.eq(True)].tolist()
        if len(marked)!=len(by_patch):raise AssertionError('Not exactly one representative per eligible patch')
        reps=[]
        for patch,items in sorted(by_patch.items()):
            actual=[eligible[key] for key in marked if eligible[key]['patch_id']==patch]
            if len(actual)!=1:raise AssertionError('Duplicate physical-patch representative')
            actual=actual[0];best=max(row['spectral_score'] for row in items)
            if best-actual['spectral_score']>1e-9:raise AssertionError('Representative lacks maximal spectral score')
            canonical=sorted(items,key=lambda row:(-row['spectral_score'],row['channel']))[0]
            if canonical['channel']!=actual['channel']:
                tie_records.append(dict(window=wi,kind='representative',actual=actual['channel'],alternate=canonical['channel'],score_difference=best-actual['spectral_score']))
            reps.append(actual)
        physical=set(item['region'] for item in reps)
        if hr.loc[wi,'eligible_patches']!=len(reps) or hr.loc[wi,'eligible_regions']!=len(physical):raise AssertionError('Eligibility counts changed')
        chosen=[]
        if len(physical)>=cfg['min_regions']:
            if hr.loc[wi,'aggregation_mode']=='median':chosen=reps
            else:
                cluster_rows=rows.loc[rows.record_type=='cluster']
                groups=qa_complete_clusters(reps,cfg['complete_link_cosine_distance'])
                if len(cluster_rows)!=len(groups):raise AssertionError('Complete-link cluster count differs')
                options=[]
                for indices in groups:
                    members=[reps[i] for i in indices];patch_ids=[item['patch_id'] for item in members]
                    actual=cluster_rows.loc[cluster_rows.patch_ids_json.map(json.loads).map(lambda ids:ids==patch_ids)]
                    if len(actual)!=1:raise AssertionError('Full PSD cluster membership differs')
                    actual=actual.iloc[0]
                    region_psds=[np.mean([item['power_fraction'] for item in members if item['region']==region],axis=0)
                                 for region in REGIONS if any(item['region']==region for item in members)]
                    summary=qa_spectral_summary(reps[0]['grid'],np.mean(region_psds,axis=0),cfg)
                    score_value=summary['spectral_score']*len(region_psds)/3
                    np.testing.assert_allclose(actual.cluster_score,score_value,atol=1e-9,rtol=0)
                    qualified=len(region_psds)>=cfg['min_regions']
                    if bool(actual.qualified)!=qualified:raise AssertionError('Cluster physical-region qualification differs')
                    if qualified:options.append((score_value,tuple(patch_ids),members,bool(actual.selected)))
                    clusters_verified+=1
                if options:
                    selected=[option for option in options if option[3]]
                    if len(selected)!=1:raise AssertionError('Expected one selected qualified cluster')
                    selected=selected[0];best=max(option[0] for option in options)
                    if best-selected[0]>1e-9:raise AssertionError('Selected cluster does not maximize declared score')
                    chosen=selected[2]
                    np.testing.assert_allclose(hr.loc[wi,'selected_cluster_score'],selected[0],atol=1e-9,rtol=0)
        if bool(chosen)!=accepted[wi]:raise AssertionError('Primary acceptance differs from independent patch support')
        contributors=rows.loc[rows.record_type=='contributor']
        if not chosen:
            if len(contributors):raise AssertionError('Rejected window contributes signal')
            continue
        selected_keys=[item['channel'] for item in chosen]
        if json.loads(hr.loc[wi,'selected_channels_json'])!=selected_keys:raise AssertionError('Selected channel provenance differs')
        if len(contributors)!=len(chosen) or set(contributors.channel)!=set(selected_keys):raise AssertionError('Contributor list differs')
        medians={region:float(np.median([item['peak_bpm'] for item in chosen if item['region']==region]))
                 for region in REGIONS if any(item['region']==region for item in chosen)}
        if json.loads(hr.loc[wi,'region_medians_json'])!=medians:raise AssertionError('Region medians differ')
        if pred[wi]!=float(np.median(list(medians.values()))):raise AssertionError('Main HR is not the hierarchical spatial median')
        if hr.loc[wi,'contributing_patches']!=len(chosen) or hr.loc[wi,'contributing_regions']!=len(medians):raise AssertionError('Contribution counts differ')
        mixed=np.zeros(window)
        for item in chosen:
            actual=contributors.loc[contributors.channel==item['channel']].iloc[0]
            count=sum(row['region']==item['region'] for row in chosen)
            expected_weight=1/(len(medians)*count)
            np.testing.assert_allclose(actual.coefficient,expected_weight,atol=1e-12,rtol=0)
            if actual.polarity not in (-1.,1.):raise AssertionError('Invalid contributor polarity')
            x=signal_table[item['channel']].to_numpy(float)[a:b]
            mixed+=(x-x.mean())/x.std()*float(actual.coefficient)*float(actual.polarity)
            patch=item['patch_id'];observed[a:b]&=masks[patch][0][a:b];filled[a:b]|=masks[patch][1][a:b]
        total[a:b]+=mixed*taper;weights[a:b]+=taper
    covered=weights>0;replay=np.full(n,np.nan);np.divide(total,weights,out=replay,where=covered)
    np.testing.assert_array_equal(bools(wave.covered),covered)
    np.testing.assert_array_equal(bools(wave.observed),observed&covered)
    np.testing.assert_array_equal(bools(wave.interpolated),filled&covered)
    np.testing.assert_allclose(wave.base,replay,atol=1e-8,rtol=0,equal_nan=True)
    return dict(passed=True,planned_windows=len(hr),accepted_windows=int(accepted.sum()),
        channel_decisions_verified=channels_verified,clusters_verified=clusters_verified,
        waveform_max_absolute_difference=float(np.nanmax(np.abs(wave.base-replay))) if covered.any() else 0.,
        main_HR_recomputed_from='saved patch BVP Welch spectra; physical-region median',
        waveform_replayed_from='saved actual broadband patch samples and contributor coefficients/polarities',ties=tie_records)


def preflight(output):
    """One byte-hash pass, no video decoding; original files remain read-only."""
    if output.exists():
        raise FileExistsError('Preserve existing preflight receipt; choose a new output')
    manifest_path = BASE/'inputs_manifest.json'
    manifest = read_json(manifest_path)
    if [r['case'] for r in manifest] != list(EXPECTED_FRAMES):
        raise AssertionError('The six original input identities changed')
    hashes = {str(manifest_path): sha(manifest_path)}
    result = dict(passed=False, generated_utc=datetime.now(timezone.utc).isoformat(),
        mode='preflight', video_decoding_performed=False, reference_offset_optimized=False,
        reference_used_only_in_this_evaluation=True, cases=[], hashes=hashes)
    ancestors = [Path('/'), Path('/home'), Path('/home/fengbujue'), PROJECT.parent, PROJECT]
    result['agents_paths'] = {str(p/'AGENTS.md'): (sha(p/'AGENTS.md') if (p/'AGENTS.md').is_file() else None) for p in ancestors}
    operations = PROJECT/'PROJECT_OPERATIONS.md'
    hashes[str(operations)] = sha(operations)
    v25_metrics = []
    for entry in manifest:
        case = entry['case']
        root = BASE/case
        video = Path(entry['video']['path'])
        stat = video.stat()
        if stat.st_size != entry['video']['bytes'] or sha(video) != entry['video']['sha256']:
            raise AssertionError(f'Original video bytes changed: {case}')
        hashes[str(video)] = entry['video']['sha256']
        trace_path = root/'inference/frame_trace.csv'
        trace_meta = read_json(root/'inference/frame_trace.json')
        inference = read_json(root/'inference/summary.json')
        frames, fps = int(trace_meta['n_frames']), float(trace_meta['fps'])
        if frames != EXPECTED_FRAMES[case] or inference['frames'] != frames or inference['fps'] != fps:
            raise AssertionError(f'Original decoded frame/FPS identity mismatch: {case}')
        if sha(trace_path) != trace_meta['trace_sha256']:
            raise AssertionError(f'Original trace checksum mismatch: {case}')
        identity = trace_meta['identity']
        if identity['video'] != str(video) or identity['size'] != stat.st_size or identity['mtime_ns'] != stat.st_mtime_ns:
            raise AssertionError(f'Trace video stat mismatch: {case}')
        trace = pd.read_csv(trace_path)
        if len(trace) != frames:
            raise AssertionError('Original trace row count mismatch')
        np.testing.assert_array_equal(trace.frame, np.arange(frames))
        np.testing.assert_allclose(trace.time_s, np.arange(frames)/fps, atol=1e-8, rtol=0)
        starts = np.arange(0, frames-round(10*fps)+1, round(fps))
        if len(starts) != EXPECTED_WINDOWS[case]:
            raise AssertionError('Original planned window count mismatch')
        stream = next(s for s in entry['ffprobe']['streams'] if s['codec_type'] == 'video')
        if abs(float(Fraction(stream['avg_frame_rate']))-fps) > 1e-8:
            raise AssertionError('Original frame rate metadata changed')
        alignment_path = root/'evaluation/alignment.json'
        alignment = read_json(alignment_path)
        origin = int(entry['video']['extended_unix_mtime_s'])*1_000_000_000-int(round(frames/fps*1_000_000_000))
        if alignment['video_start_utc_ns'] != origin or alignment['primary_shift_s'] != 0:
            raise AssertionError('Fixed original reference offset changed')
        if alignment['decoded_frames'] != frames or alignment['original_video_sha256'] != entry['video']['sha256']:
            raise AssertionError('Alignment belongs to different video identity')
        reference_path = Path(alignment['reference_path'])
        candidates = [r for r in entry['files'] if Path(r['path']) == reference_path]
        if len(candidates) != 1 or candidates[0]['sha256'] != alignment['reference_sha256'] or sha(reference_path) != alignment['reference_sha256']:
            raise AssertionError('Original Polar reference bytes changed')
        paired_path = root/'evaluation/paired_windows.csv'
        paired = pd.read_csv(paired_path)
        if len(paired) != len(starts):
            raise AssertionError('Reference planned windows changed')
        np.testing.assert_array_equal(paired.window_index, np.arange(len(starts)))
        np.testing.assert_allclose(paired.window_start_s, starts/fps, atol=1e-8, rtol=0)
        np.testing.assert_allclose(paired.window_end_s, (starts+round(10*fps))/fps, atol=1e-8, rtol=0)
        references, valid = independent_reference(pd.read_csv(reference_path), origin,
                                                   paired.window_start_s, paired.window_end_s)
        np.testing.assert_array_equal(bools(paired.reference_valid), valid)
        np.testing.assert_allclose(paired.reference_bpm, references, atol=1e-9, rtol=0, equal_nan=True)
        old_hr_path, old_wave_path = V25/case/'fusion_heart_rate.csv', V25/case/'fusion_waveform.csv'
        old_hr, old_wave = pd.read_csv(old_hr_path), pd.read_csv(old_wave_path)
        pred, accepted = assert_hr(old_hr, starts, fps, frames)
        if len(old_wave) != frames:
            raise AssertionError('V25 waveform frame count changed')
        np.testing.assert_allclose(old_wave.time_s, trace.time_s, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(np.isfinite(old_wave.base), bools(old_wave.covered))
        metric = score(pred, accepted, references)
        v25_metrics.append(metric)
        paths = [trace_path, root/'inference/frame_trace.json', root/'inference/summary.json',
                 alignment_path, paired_path, reference_path, old_hr_path, old_wave_path]
        for path in paths:
            hashes[str(path)] = sha(path)
        # Confirm original archived eval still binds the same manifest and raw reference.
        historical = read_json(root/'evaluation/evaluation_provenance.json')['inputs']
        for path in [manifest_path, reference_path, trace_path]:
            if historical[str(path)] != hashes[str(path)]:
                raise AssertionError(f'Original evaluation binding changed: {path}')
        result['cases'].append(dict(case=case, frames=frames, fps=fps, planned_windows=len(starts),
            reference_windows=int(valid.sum()), video=dict(path=str(video), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, sha256=entry['video']['sha256']),
            container_header_frames=int(stream['nb_frames']), width=stream['width'], height=stream['height'],
            video_start_utc_ns=origin, sync_status=alignment['sync_status'],
            reference_sha256=alignment['reference_sha256'], V25_metrics=metric,
            V25_waveform_coverage_pct=100*np.isfinite(old_wave.base).mean()))
        print(json.dumps(dict(case=case, identity_passed=True, frames=frames, fps=fps,
                              planned_windows=len(starts), reference_windows=int(valid.sum()))), flush=True)
    result['V25_pooled'] = pool(v25_metrics)
    summary_path = V25/'stage_evaluation.json'
    prior = read_json(summary_path)['pooled']
    for name in ('MAE_bpm', 'RMSE_bpm', 'R5_all_reference_pct'):
        np.testing.assert_allclose(result['V25_pooled'][name], prior[name], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result['V25_pooled']['P5_valid_pct'],
                               100*prior['Nwithin5']/prior['Nvalid'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(result['V25_pooled']['HR_coverage_pct'],
                               prior['hr_output_coverage_pct'], atol=1e-10, rtol=0)
    if sum(r['planned_windows'] for r in result['cases']) != 309 or sum(r['reference_windows'] for r in result['cases']) != 309:
        raise AssertionError('The complete original 309-window comparison changed')
    hashes[str(summary_path)] = sha(summary_path)
    result['passed'] = True
    save(output, result)
    return result


def check_metric_mapping(actual, reported, names=None):
    for name in (actual if names is None else names):
        expected=actual[name];saved=reported[name]
        if expected is None or (isinstance(expected,(float,np.floating)) and not np.isfinite(expected)):
            if saved is not None and np.isfinite(saved):raise AssertionError(f'{name}: missing metric became finite')
        else:
            np.testing.assert_allclose(saved,expected,atol=1e-8,rtol=0,err_msg=name)


def audit_results(root, output):
    """Eight global variants, each compared on the original six full clocks."""
    if output.exists():raise FileExistsError('Preserve completed independent QA')
    fixed=read_json(root/'protocol_before_inference.json')
    if fixed['variants']!=list(SPATIAL+READOUT) or fixed['reference_used_in_inference'] is not False:
        raise AssertionError('Unexpected frozen design/reference use')
    for name,digest in fixed['source_hashes'].items():
        if sha(HERE/name)!=digest:raise AssertionError(f'Frozen production source changed: {name}')
    for section in ('evaluation_hashes','evaluation_inputs','protected_entries'):
        for name,digest in fixed[section].items():
            if sha(name)!=digest:raise AssertionError(f'Frozen {section} changed: {name}')
    pre=read_json(HERE/'qa_inputs_preflight.json')
    if not pre['passed']:raise AssertionError('Original inputs have not passed independent preflight')
    original={row['case']:row for row in pre['cases']}
    baseline_rows=[];shared={}
    for case in EXPECTED_FRAMES:
        trace=pd.read_csv(root/'frontend'/case/'frame_trace.csv')
        metadata=read_json(root/'frontend'/case/'metadata.json')
        binding=read_json(root/'frontend'/case/'batch_binding.json')
        fps,n=original[case]['fps'],original[case]['frames']
        if len(trace)!=n or metadata['frames']!=n or metadata['fps']!=fps:raise AssertionError('Frontend frame/FPS mismatch')
        np.testing.assert_array_equal(trace.frame,np.arange(n))
        np.testing.assert_allclose(trace.time_s,np.arange(n)/fps,atol=1e-8,rtol=0)
        if metadata['video_sha256']!=original[case]['video']['sha256'] or metadata['reference_used'] is not False:
            raise AssertionError('Frontend video/reference identity differs')
        if binding['fixed_protocol_sha256']!=sha(root/'protocol_before_inference.json') or binding['source_hashes']!=fixed['source_hashes']:
            raise AssertionError('Frontend belongs to another frozen run')
        if binding['metadata_sha256']!=sha(root/'frontend'/case/'metadata.json'):raise AssertionError('Frontend metadata not bound')
        for name,digest in metadata['output_hashes'].items():
            if sha(root/'frontend'/case/name)!=digest:raise AssertionError('Frontend output changed')
        patches=pd.read_csv(root/'frontend'/case/'patch_trace.csv',usecols=['frame','patch_id','region','valid'])
        if len(patches)!=n*12 or patches[['frame','patch_id']].duplicated().any():raise AssertionError('Raw patch availability grain differs')
        observed=patches.pivot(index='frame',columns='patch_id',values='valid')
        if list(observed.index)!=list(range(n)) or len(observed.columns)!=12:raise AssertionError('Raw patch frame clock missing')
        for column in observed:bools(observed[column])
        per_region=np.asarray([observed[[p for p in observed if p.startswith(region+'_')]].any(axis=1) for region in REGIONS])
        availability=dict(mean_patch_observed_pct=100*observed.to_numpy(bool).mean(),
            at_least_two_regions_observed_pct=100*(per_region.sum(axis=0)>=2).mean(),
            face_detection_pct=100*bools(trace.face_detected).mean())
        ref=pd.read_csv(BASE/case/'evaluation/paired_windows.csv')
        starts=np.arange(0,n-round(10*fps)+1,round(fps));references=ref.reference_bpm.to_numpy(float)
        if len(starts)!=EXPECTED_WINDOWS[case] or len(ref)!=len(starts):raise AssertionError('Original 309-window plan changed')
        np.testing.assert_allclose(ref.time_s,(starts+round(10*fps)/2)/fps,atol=1e-8,rtol=0)
        old=pd.read_csv(V25/case/'fusion_heart_rate.csv');old_wave=pd.read_csv(V25/case/'fusion_waveform.csv')
        old_y,old_ok=assert_hr(old,starts,fps,n);metric=score(old_y,old_ok,references);baseline_rows.append(metric)
        shared[case]=dict(trace=trace,reference=references,starts=starts,fps=fps,n=n,old_y=old_y,old_ok=old_ok,
                          old_metric=metric,old_wave_coverage=np.isfinite(old_wave.base).mean(),availability=availability,
                          per_patch_availability={name:100*bools(observed[name]).mean() for name in observed})
    prior=pool(baseline_rows);result=dict(passed=False,reference_offset_optimized=False,
        scope='Independent metric arithmetic, patch aggregation and actual broadband-mixture replay; no video/model inference',
        cases_per_variant=6,variants={},V25_pooled=prior,qa_source_sha256=sha(__file__),
        readout_limit='Local/DP source fields, masks and shared-wave identity checked here; numerical V25 readout replay is a separate receipt.',
        original_preflight_sha256=sha(HERE/'qa_inputs_preflight.json'))
    signal_cache={}
    for variant in SPATIAL+READOUT:
        rows=[];fusion_rows=[];case_receipts=[]
        reported=read_json(root/variant/'evaluation_summary.json')
        for case,common in shared.items():
            directory=root/variant/case;manifest=read_json(directory/'manifest.json')
            if manifest['status']!='complete' or manifest['reference_used'] is not False or manifest['source_hashes']!=fixed['source_hashes']:
                raise AssertionError('Incomplete, reference-using or changed-source inference')
            if manifest['variant']!=variant or manifest['case']!=case:raise AssertionError('Variant/case routing changed')
            for name,digest in manifest['output_hashes'].items():
                if sha(directory/name)!=digest:raise AssertionError('Saved inference output changed')
            for name,digest in manifest['source_files'].items():
                if sha(name)!=digest:raise AssertionError('Saved inference source input changed')
            check_metric_mapping(common['availability'],manifest['patch_availability'])
            check_metric_mapping(common['per_patch_availability'],manifest['patch_availability']['per_patch_observed_pct'])
            wave=pd.read_csv(directory/'waveform.csv');hr=pd.read_csv(directory/'heart_rate.csv')
            fh=pd.read_csv(directory/'fusion_heart_rate.csv');fps=common['fps'];n=common['n'];starts=common['starts']
            np.testing.assert_allclose(wave.time_s,common['trace'].time_s,atol=1e-8,rtol=0)
            finite=np.isfinite(wave.base.to_numpy(float));np.testing.assert_array_equal(finite,bools(wave.covered))
            y,ok=assert_hr(hr,starts,fps,n);fy,fok=assert_hr(fh,starts,fps,n)
            for a in starts[fok]:
                if not finite[a:a+round(10*fps)].all():raise AssertionError('Fusion diagnostic accepted a nonfinite source window')
            primary_expected=('patch_'+variant.split('_',1)[1] if variant in SPATIAL else
                              'fused_wave_dp' if variant.endswith('_dp') else 'fused_wave_local')
            if manifest['primary_hr_source']!=primary_expected:raise AssertionError('Main HR source changed')
            patch_receipt=None
            if variant in SPATIAL:
                if not hr.hr_source.eq('spatial_aggregation_of_saved_patch_BVP').all():raise AssertionError('Patch HR mislabeled as fused-wave HR')
                key=str(manifest['patch_waveforms_path'])
                if sha(key)!=manifest['patch_waveforms_sha256']:raise AssertionError('Saved BVP input changed')
                if key not in signal_cache:signal_cache[key]=pd.read_csv(key)
                patch_receipt=audit_patch_result(directory,signal_cache[key],manifest['patch_signal_metadata'],
                                                 common['trace'],fps,fixed['patch_hr_config'])
            else:
                parent=root/'late_psd_cluster'/case
                if sha(directory/'waveform.csv')!=sha(parent/'waveform.csv'):raise AssertionError('Readout factor changed waveform')
                if sha(directory/'fusion_heart_rate.csv')!=sha(parent/'fusion_heart_rate.csv'):raise AssertionError('Readout diagnostic changed')
                for a in starts[ok]:
                    if not finite[a:a+round(10*fps)].all():raise AssertionError('Readout primary accepted a nonfinite source window')
                if variant.endswith('_local'):
                    np.testing.assert_allclose(y[ok],hr.evidence_local_bpm.to_numpy(float)[ok],atol=0,rtol=0)
                elif variant=='fusion_motion_dp':
                    np.testing.assert_array_equal(ok,fok);np.testing.assert_allclose(y,fy,atol=0,rtol=0,equal_nan=True)
            r=common['reference'];old_y=common['old_y'];old_ok=common['old_ok'];old_metric=common['old_metric']
            metric=score(y,ok,r);secondary=score(fy,fok,r)
            overlap=ok&old_ok&np.isfinite(r);extra=ok&~old_ok&np.isfinite(r);lost=~ok&old_ok&np.isfinite(r)
            row=dict(case=case,**metric,common_windows=int(overlap.sum()),
                common_MAE_bpm=float(np.abs(y[overlap]-r[overlap]).mean()) if overlap.any() else np.nan,
                common_V25_MAE_bpm=float(np.abs(old_y[overlap]-r[overlap]).mean()) if overlap.any() else np.nan,
                new_windows=int(extra.sum()),lost_windows=int(lost.sum()),
                delta_MAE_bpm=metric['MAE_bpm']-old_metric['MAE_bpm'],
                delta_P5_pp=metric['P5_valid_pct']-old_metric['P5_valid_pct'],
                delta_R5_pp=metric['R5_all_reference_pct']-old_metric['R5_all_reference_pct'],
                delta_hr_coverage_pp=metric['hr_output_coverage_pct']-old_metric['hr_output_coverage_pct'],
                waveform_coverage_pct=100*finite.mean(),delta_waveform_coverage_pp=100*(finite.mean()-common['old_wave_coverage']))
            actual=next(item for item in reported['cases'] if item['case']==case)
            check_metric_mapping(row,actual,[name for name in row if name!='case'])
            check_metric_mapping(secondary,actual['fusion_readout_metrics'])
            rows.append(row);fusion_rows.append(secondary)
            case_receipts.append(dict(case=case,metrics_passed=True,patch_replay=patch_receipt,
                                      primary_hr_source=primary_expected,main=metric,fusion=secondary))
        pooled=pool(rows);fusion_pooled=pool(fusion_rows)
        check_metric_mapping(pooled,reported['pooled'],[name for name in pooled if name not in ('Nplanned','Noutput')])
        check_metric_mapping(fusion_pooled,reported['fusion_readout_pooled'],[name for name in fusion_pooled if name not in ('Nplanned','Noutput')])
        checks=promotion(rows,pooled,prior)
        if checks!=reported['promotion_checks'] or all(checks.values())!=reported['promotion_pass']:
            raise AssertionError('Promotion decision differs from independent eight guards')
        result['variants'][variant]=dict(passed=True,pooled=pooled,fusion_pooled=fusion_pooled,
            promotion_checks=checks,promotion_pass=all(checks.values()),cases=case_receipts,
            evaluation_summary_sha256=sha(root/variant/'evaluation_summary.json'))
        print(json.dumps(dict(variant=variant,qa_passed=True,pooled=pooled)),flush=True)
    if sum(len(v['cases']) for v in result['variants'].values())!=48:raise AssertionError('Incomplete eight-by-six QA')
    for name,digest in fixed['source_hashes'].items():
        if sha(HERE/name)!=digest:raise AssertionError('Production source changed during QA')
    result['passed']=True;save(output,result);return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--results-root', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.preflight and args.results_root:ap.error('Choose preflight or completed-result QA')
    if args.preflight:result=preflight(args.out)
    elif args.results_root:result=audit_results(args.results_root,args.out)
    else:ap.error('Use --preflight or --results-root')
    print(json.dumps(clean(dict(passed=result['passed'], V25_pooled=result['V25_pooled'])), ensure_ascii=False))


if __name__ == '__main__':
    main()

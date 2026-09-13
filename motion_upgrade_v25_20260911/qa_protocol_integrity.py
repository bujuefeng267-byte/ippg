"""Post-run independent integrity/arithmetic audit; imports no production evaluator.

This script cannot make supplemental hashes retroactively pre-run freezes.
It does not run models, read hidden truth, choose offsets, or change thresholds.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/fengbujue/项目/rppg识别')
BASE = PROJECT / 'results/data1_6_20260911'
ROOT = PROJECT / 'results/data1_6_v25_20260911'
SHIFTS = (-5, -2, -1, 0, 1, 2, 5)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if math.isfinite(float(x)) else None
    return x


class Audit:
    def __init__(self):
        self.errors = []
        self.checks = 0
        self.max_numeric_difference = 0.

    def check(self, condition, context):
        self.checks += 1
        if not bool(condition): self.errors.append(context)

    def equal(self, got, expected, context, tolerance=1e-8):
        self.checks += 1
        if isinstance(expected, (bool, np.bool_)) or isinstance(expected, str):
            if got != expected: self.errors.append(f'{context}: {got!r} != {expected!r}')
            return
        empty_a = got is None or (isinstance(got, (float, np.floating)) and not np.isfinite(got))
        empty_b = expected is None or (isinstance(expected, (float, np.floating)) and not np.isfinite(expected))
        if empty_a or empty_b:
            if empty_a != empty_b: self.errors.append(f'{context}: missingness differs')
            return
        d = abs(float(got) - float(expected))
        self.max_numeric_difference = max(d, self.max_numeric_difference)
        if d > tolerance: self.errors.append(f'{context}: difference {d:.12g}')

    def metrics(self, reported, expected, context):
        for k, v in expected.items():
            self.check(k in reported, context+'.missing_key.'+k)
            if k in reported: self.equal(reported[k], v, context+'.'+k)


def bools(s):
    if not s.notna().all() or not s.isin([True, False, 0, 1]).all():
        raise ValueError('Invalid boolean column')
    return s.to_numpy(bool)


def compare_arrays(qa, a, b, label, tolerance=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    qa.check(a.shape == b.shape, label+'.shape')
    if a.shape != b.shape: return
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        qa.check(np.array_equal(np.isfinite(a), np.isfinite(b)), label+'.finite_mask')
        qa.check(np.allclose(a, b, atol=tolerance, rtol=0, equal_nan=True), label+'.values')
    else:
        qa.check(np.array_equal(a, b), label+'.values')


def score(y, ok, ref):
    y, ref = np.asarray(y, float), np.asarray(ref, float)
    output = ok & np.isfinite(y)
    ref_ok = np.isfinite(ref) & (ref > 0)
    valid = output & ref_ok
    errors = y[valid] - ref[valid]
    absolute = np.abs(errors)
    n, no, nr, nv = len(y), int(output.sum()), int(ref_ok.sum()), int(valid.sum())
    n5, n10 = int((absolute <= 5).sum()), int((absolute <= 10).sum())
    ratio = lambda a, b: 100. * a / b if b else np.nan
    mae = float(np.mean(absolute)) if nv else np.nan
    rms = float(np.sqrt(np.mean(np.square(errors)))) if nv else np.nan
    bias = float(np.mean(errors)) if nv else np.nan
    sd = float(np.std(errors, ddof=1)) if nv > 1 else np.nan
    return dict(Nplanned=n, Noutput=no, Nref=nr, Nvalid=nv, Nwithin5=n5, Nwithin10=n10,
        hr_output_coverage_pct=ratio(no,n), paired_reference_coverage_pct=ratio(nv,nr),
        MAE_bpm=mae, RMSE_bpm=rms, Bias_bpm=bias,
        MAPE_pct=float(np.mean(absolute/ref[valid])*100) if nv else np.nan,
        P5_valid_pct=ratio(n5,nv), P10_valid_pct=ratio(n10,nv),
        R5_all_reference_pct=ratio(n5,nr), R10_all_reference_pct=ratio(n10,nr),
        P95_abs_error_bpm=float(np.quantile(absolute,.95)) if nv else np.nan,
        Max_abs_error_bpm=float(absolute.max()) if nv else np.nan,
        LoA_lower_descriptive_bpm=bias-1.96*sd, LoA_upper_descriptive_bpm=bias+1.96*sd)


def reference_from_raw(raw, start_ns, starts, ends, shift):
    # Difference is performed as int64 BEFORE converting short elapsed time to float.
    absolute_time = raw.host_utc_ns.to_numpy(np.int64)
    t = (absolute_time - np.int64(start_ns)).astype(float)/1e9 - shift
    bpm = raw.hr_bpm.to_numpy(float)
    valid = np.isfinite(bpm) & (bpm > 0)
    t, bpm = t[valid], bpm[valid]
    assert len(t)>1 and np.all(np.diff(t)>0)
    tail = float(np.median(np.diff(t)))
    values, flags = [], []
    for left, right in zip(starts, ends):
        take = (t >= left) & (t < right)
        edges = np.r_[left, t[take], right]
        good = left >= t[0] and right <= t[-1]+tail and take.sum() >= 2 and np.diff(edges).max() <= 2
        flags.append(bool(good))
        values.append(float(bpm[take].mean()) if good else np.nan)
    return np.asarray(values), np.asarray(flags)


def average_absolute(y, ref, mask):
    return float(np.abs(y[mask]-ref[mask]).mean()) if mask.any() else np.nan


def pooled(rows):
    n = sum(x['Nvalid'] for x in rows)
    nr = sum(x['Nref'] for x in rows)
    nc = sum(x['common_windows'] for x in rows)
    return dict(Nvalid=n, Nref=nr, Nwithin5=sum(x['Nwithin5'] for x in rows),
        MAE_bpm=sum(x['MAE_bpm']*x['Nvalid'] for x in rows if x['Nvalid'])/n if n else np.nan,
        RMSE_bpm=math.sqrt(sum(x['RMSE_bpm']**2*x['Nvalid'] for x in rows if x['Nvalid'])/n) if n else np.nan,
        R5_all_reference_pct=100*sum(x['Nwithin5'] for x in rows)/nr if nr else np.nan,
        hr_output_coverage_pct=100*sum(x['Noutput'] for x in rows)/sum(x['Nplanned'] for x in rows),
        common_MAE_bpm=sum(x['common_MAE_bpm']*x['common_windows'] for x in rows if x['common_windows'])/nc if nc else np.nan,
        common_baseline_MAE_bpm=sum(x['common_baseline_MAE_bpm']*x['common_windows'] for x in rows if x['common_windows'])/nc if nc else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT/'qa_protocol_integrity.json')
    ap.add_argument('--stage4-protocol', type=Path)
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output name to preserve prior audit: '+str(args.output))
    qa = Audit()
    frozen_path = ROOT/'protocol_before_validation.json'
    frozen = read(frozen_path)
    qa.check(len(frozen['source_hashes'])==13, '13 frozen inference modules')
    for name, expected in frozen['source_hashes'].items():
        qa.check(sha(HERE/name)==expected, 'current frozen source '+name)
    qa.check(sha(HERE/'run_stages.py')==frozen['runner_sha256'], 'frozen runner unchanged')
    qa.check(sha(HERE/'evaluate_stages.py')==frozen['evaluator_sha256'], 'frozen evaluator unchanged')
    qa.check(read(HERE/'development_protocol.json')==frozen['protocol'], 'initial protocol unchanged')
    core_path = PROJECT/'batch_analysis_20260911/evaluate_batch.py'
    supplemental = {str(p):sha(p) for p in [core_path,BASE/'comparison_metrics.json',BASE/'inputs_manifest.json']}
    historical_entries = {}
    stages = list(frozen['protocol']['stages'])
    stage4 = 'stage4_preserve_waveform'
    if (ROOT/stage4/'stage_evaluation.json').exists():
        stages.append(dict(name=stage4,motion_evidence='legacy',hr_mode='evidence',routing_mode='legacy'))
    if args.stage4_protocol:
        supplemental[str(args.stage4_protocol)] = sha(args.stage4_protocol)
        extra_protocol = read(args.stage4_protocol)
        expected_extra = dict(name=stage4,motion_evidence='legacy',hr_mode='evidence',routing_mode='legacy')
        qa.check(extra_protocol['stage']==expected_extra, 'stage4 supplemental declared modes')
        qa.check(extra_protocol['source_hashes']==frozen['source_hashes'], 'stage4 no inference parameter/source revision')
        qa.check(extra_protocol['evaluator_sha256']==frozen['evaluator_sha256'], 'stage4 unchanged evaluator')
        qa.check(extra_protocol['original_adoption_gate_unchanged'] is True, 'stage4 declares unchanged gate')
        qa.check(extra_protocol['parameters_changed'] is False, 'stage4 declares unchanged numerical parameters')
        qa.check('not preregistered' in extra_protocol['selection_context'], 'stage4 explicitly post-development')
        qa.check(extra_protocol['original_protocol_sha256'] in [sha(frozen_path),sha(HERE/'development_protocol.json')], 'stage4 binds original protocol')
        for prior_stage, digest in extra_protocol['observed_stage_reports'].items():
            qa.check(sha(ROOT/prior_stage/'stage_evaluation.json')==digest, 'stage4 binds observed development result '+prior_stage)
    manifest = {x['case']:x for x in read(BASE/'inputs_manifest.json')}
    baseline_reported = {x['case']:x for x in read(BASE/'comparison_metrics.json')}
    qa.check(set(manifest)=={f'data{i}' for i in range(1,7)}, 'exact six input cases')
    qa.check(set(baseline_reported)==set(manifest), 'baseline aggregate exact six cases')
    baseline = {}
    summaries = []
    all_rows = []
    input_hashes = {}
    replay_notes = []
    paired_qa_rows = []
    local_diagnostics = []
    for case in sorted(manifest):
        prior = BASE/case/'inference'
        refdir = BASE/case/'evaluation'
        refpath = refdir/'paired_windows.csv'
        alignpath = refdir/'alignment.json'
        qa.check(sha(refpath)==frozen['baseline_reference_hashes'][case], case+' frozen reference windows')
        for name, file in [('csv','frame_trace.csv'),('metadata','frame_trace.json')]:
            qa.check(sha(prior/file)==frozen['caches'][case][name], case+' frozen cache '+name)
        ref = pd.read_csv(refpath)
        align = read(alignpath)
        rawpath = Path(align['reference_path'])
        raw = pd.read_csv(rawpath, dtype={'host_utc_ns':'int64'})
        qa.check(sha(rawpath)==align['reference_sha256'], case+' raw reference matches original alignment receipt')
        for p in [alignpath,rawpath,prior/'summary.json',prior/'fusion_heart_rate.csv',prior/'fusion_waveform.csv']:
            supplemental[str(p)] = sha(p)
        original_hr = pd.read_csv(prior/'fusion_heart_rate.csv')
        original_wave = pd.read_csv(prior/'fusion_waveform.csv')
        old_ok = bools(original_hr.accepted)
        old_y = original_hr.ridge_bpm.to_numpy(float)
        values = ref.reference_bpm.to_numpy(float)
        valid_ref = bools(ref.reference_valid)
        qa.check(np.array_equal(valid_ref,np.isfinite(values)&(values>0)), case+' eligible reference mask')
        compare_arrays(qa,original_hr.time_s,ref.time_s,case+' prior HR time axis')
        compare_arrays(qa,ref.window_index,np.arange(len(ref)),case+' reference index',tolerance=0)
        base_score = score(old_y,old_ok,values)
        qa.metrics(baseline_reported[case],base_score,case+' frozen baseline main metrics')
        baseline[case] = base_score
        shifts = {}
        for shift in SHIFTS:
            yref, eligibility = reference_from_raw(raw,align['video_start_utc_ns'],ref.window_start_s,ref.window_end_s,shift)
            shifts[shift] = yref
            if shift == 0:
                compare_arrays(qa,yref,values,case+' raw/reference zero-shift values')
                compare_arrays(qa,eligibility,valid_ref,case+' raw/reference zero-shift validity')
        first_summary = None
        for stage in stages:
            name = stage['name']; folder = ROOT/name/case
            context = name+'/'+case
            s = read(folder/'summary.json');cfg=s['config']
            qa.check(s['source_hashes']==frozen['source_hashes'],context+' source hashes')
            expect_cfg = dict(algorithm_mode='guarded_fusion',pixel_mode='tracking_screened',
                rgb_aggregation='trimmed_mean',window=10,step=1,max_gap=.1,min_bpm=42,max_bpm=210,
                max_seconds=None,reference_ubfc=None,geometry_cache=None,tracking_cache=None,
                cache=str(prior/'frame_trace.csv'),video=manifest[case]['video']['path'],output=str(folder),
                motion_evidence=stage['motion_evidence'],hr_mode=stage['hr_mode'],routing_mode=stage['routing_mode'])
            qa.check(all(cfg.get(k)==v for k,v in expect_cfg.items()),context+' actual summary flags')
            qa.check(s.get('reference_identity') is None,context+' no inference reference identity')
            qa.check(s['fusion_config']['motion_evidence_mode']==stage['motion_evidence'],context+' fusion mode')
            qa.check(s['guard_config']['routing_mode']==stage['routing_mode'],context+' guard mode')
            qa.check(s['identity']==read(prior/'frame_trace.json')['identity'],context+' frozen frontend identity')
            if first_summary is None: first_summary=s
            for k in ['pixel_config','anchor_config','identity','fps','frames']:
                qa.check(s[k]==first_summary[k],context+' unchanged '+k)
            for block,ignore in [('fusion_config','motion_evidence_mode'),('guard_config','routing_mode')]:
                qa.check({k:v for k,v in s[block].items() if k!=ignore}=={k:v for k,v in first_summary[block].items() if k!=ignore},context+' fixed '+block+' numerical parameters')
            hr = pd.read_csv(folder/'fusion_heart_rate.csv'); wave=pd.read_csv(folder/'fusion_waveform.csv')
            compare_arrays(qa,hr.time_s,original_hr.time_s,context+' HR axis')
            compare_arrays(qa,wave.time_s,original_wave.time_s,context+' wave axis')
            fps,frames=float(s['fps']),int(s['frames']);w=round(10*fps);hop=round(fps)
            starts=np.arange(0,frames-w+1,hop)
            qa.check(len(starts)==len(hr)==len(ref) and len(wave)==frames,context+' expected lengths')
            compare_arrays(qa,hr.time_s,(starts+w/2)/fps,context+' nominal centers')
            ok=bools(hr.accepted);y=hr.ridge_bpm.to_numpy(float);finite=np.isfinite(wave.base.to_numpy(float))
            qa.check(np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all(),context+' rejected HR missing')
            qa.check(np.array_equal(finite,bools(wave.covered)),context+' waveform finite/covered')
            qa.check(all(finite[a:a+w].all() for a in starts[ok]),context+' HR accepted only on fully finite windows')
            if name == 'v24_replay':
                for column in ['accepted','ridge_bpm','spectral_peak_bpm','status']:
                    compare_arrays(qa,hr[column],original_hr[column],context+' replay '+column)
                for column in ['base','covered','observed','interpolated']:
                    compare_arrays(qa,wave[column],original_wave[column],context+' replay wave '+column)
            m=score(y,ok,values);common=ok&old_ok&valid_ref;new=ok&~old_ok&valid_ref;lost=~ok&old_ok&valid_ref
            bfinite=np.isfinite(original_wave.base.to_numpy(float))
            m.update(baseline_MAE_bpm=base_score['MAE_bpm'],delta_MAE_bpm=m['MAE_bpm']-base_score['MAE_bpm'],
                delta_R5_pp=m['R5_all_reference_pct']-base_score['R5_all_reference_pct'],
                delta_hr_coverage_pp=m['hr_output_coverage_pct']-base_score['hr_output_coverage_pct'],
                waveform_coverage_pct=100*finite.mean(),baseline_waveform_coverage_pct=100*bfinite.mean(),
                delta_waveform_coverage_pp=100*(finite.mean()-bfinite.mean()),
                common_windows=int(common.sum()),new_windows=int(new.sum()),lost_windows=int(lost.sum()),
                common_MAE_bpm=average_absolute(y,values,common),common_baseline_MAE_bpm=average_absolute(old_y,values,common),
                new_MAE_bpm=average_absolute(y,values,new),lost_baseline_MAE_bpm=average_absolute(old_y,values,lost),
                correct_new_windows=int((new&(np.abs(y-values)<=5)).sum()),
                lost_correct_windows=int((lost&(np.abs(old_y-values)<=5)).sum()),
                mean_hr_bpm=float(y[ok].mean()) if ok.any() else np.nan)
            reported=read(folder/'evaluation/metrics.json')
            qa.metrics(reported,m,context+' main metrics')
            qa.check(reported['status_counts']==hr.status.value_counts().to_dict(),context+' status counts')
            qa.check(reported['branch_counts']==s['branch_selection_counts'],context+' branch counts')
            pair=pd.read_csv(folder/'evaluation/paired_windows.csv')
            for col,val in [('reference_bpm',values),('accepted',ok),('estimated_bpm',y),('baseline_accepted',old_ok),('baseline_bpm',old_y),('common',common),('new',new),('lost',lost),('error_bpm',np.where(ok&valid_ref,y-values,np.nan))]:
                compare_arrays(qa,pair[col],val,context+' paired '+col)
            sensitive=pd.read_csv(folder/'evaluation/alignment_sensitivity.csv')
            qa.check(sorted(sensitive.shift_s.tolist())==list(SHIFTS),context+' all fixed shifts')
            for shift in SHIFTS:
                got=sensitive.loc[sensitive.shift_s==shift].iloc[0].to_dict()
                qa.metrics(got,score(y,ok,shifts[shift]),context+f' shift {shift}')
            provenance=read(folder/'evaluation/provenance.json')
            qa.check(provenance['evaluator_sha256']==frozen['evaluator_sha256'],context+' evaluation uses frozen evaluator')
            qa.check(provenance['reference_core_sha256']==sha(core_path),context+' post-run core hash matches each evaluation receipt')
            for p,digest in {**provenance['frozen_reference_files'],**provenance['inputs']}.items():
                qa.check(sha(p)==digest,context+' reported provenance '+Path(p).name)
            for p in [folder/'summary.json',folder/'fusion_heart_rate.csv',folder/'fusion_waveform.csv',folder/'evaluation/metrics.json']:
                input_hashes[str(p)]=sha(p)
            local_diagnostics.append(dict(stage=name,case=case,estimator='spectral_peak_bpm',**score(hr.spectral_peak_bpm.to_numpy(float),ok,values)))
            paired_qa_rows.append(dict(stage=name,case=case,common_RMSE_bpm=float(np.sqrt(np.mean((y[common]-values[common])**2))) if common.any() else np.nan,
                common_baseline_RMSE_bpm=float(np.sqrt(np.mean((old_y[common]-values[common])**2))) if common.any() else np.nan))
            all_rows.append(dict(stage=name,case=case,**m))
        a=ROOT/'stage1_motion'/case/'fusion_waveform.csv';b=ROOT/'stage2_hr'/case/'fusion_waveform.csv'
        a_df,b_df=pd.read_csv(a),pd.read_csv(b)
        qa.check(list(a_df)==list(b_df),case+' stage1/2 waveform schema')
        for col in a_df:compare_arrays(qa,a_df[col],b_df[col],case+' stage1/2 waveform '+col,tolerance=0)
        replay_notes.append(dict(case=case,stage1_stage2_waveform_file_sha_equal=sha(a)==sha(b)))
        if any(x['name']==stage4 for x in stages):
            target=pd.read_csv(ROOT/stage4/case/'fusion_waveform.csv')
            replay=pd.read_csv(ROOT/'v24_replay'/case/'fusion_waveform.csv')
            for col in replay:compare_arrays(qa,replay[col],target[col],case+' stage4/replay preserved waveform '+col,tolerance=0)
    bmae=sum(x['MAE_bpm']*x['Nvalid'] for x in baseline.values())/sum(x['Nvalid'] for x in baseline.values())
    br5=100*sum(x['Nwithin5'] for x in baseline.values())/sum(x['Nref'] for x in baseline.values())
    for stage in stages:
        name=stage['name'];folder=ROOT/name
        runs=read(folder/'runs.json')
        qa.check({r['case'] for r in runs}==set(manifest) and len(runs)==6,name+' exact six run records')
        for run in runs:
            c=run['case'];cmd=run['command']
            # The historical command must not be rewritten to this QA script's
            # current installation directory. Index 2 is the entry script;
            # index 3 is the video. Its content is bound to the frozen source.
            historical_entry = Path(cmd[2])
            qa.check(historical_entry.name=='analyze_motion_v2.py',name+'/'+c+' recorded entry basename')
            historical_entries[str(historical_entry)] = {'currently_exists':historical_entry.exists(),
                'expected_sha256':frozen['source_hashes']['analyze_motion_v2.py'],
                'actual_sha256':sha(historical_entry) if historical_entry.exists() else None}
            if historical_entry.exists():
                qa.check(sha(historical_entry)==frozen['source_hashes']['analyze_motion_v2.py'],name+'/'+c+' recorded entry frozen content')
                supplemental[str(historical_entry)] = sha(historical_entry)
            expected=[cmd[0],'-B',str(historical_entry),manifest[c]['video']['path'],
                '--output',str(folder/c),'--cache',str(BASE/c/'inference/frame_trace.csv'),
                '--pixel-mode','tracking_screened','--algorithm-mode','guarded_fusion','--max-gap','.1',
                '--window','10','--step','1','--min-bpm','42','--max-bpm','210',
                '--motion-evidence',stage['motion_evidence'],'--hr-mode',stage['hr_mode'],'--routing-mode',stage['routing_mode']]
            qa.check(cmd==expected,name+'/'+c+' exact recorded command')
            qa.check(run['returncode']==0 and run['stage']==name,name+'/'+c+' successful recorded run')
        rows=[x for x in all_rows if x['stage']==name];agg=pooled(rows)
        qa.check(sum(x['Nplanned'] for x in rows)==309,name+' frozen planned 309 windows')
        report=read(folder/'stage_evaluation.json')
        qa.metrics(report['pooled'],agg,name+' pooled')
        qa.equal(report['baseline_pooled_MAE_bpm'],bmae,name+' recomputed baseline pooled MAE')
        qa.equal(report['baseline_pooled_R5_pct'],br5,name+' recomputed baseline pooled R5')
        for item in report['cases']:
            qa.metrics(item,{k:v for k,v in next(x for x in rows if x['case']==item['case']).items() if k not in ('stage','case')},name+'/'+item['case']+' aggregate copy')
        gates=dict(pooled_MAE_reduction=agg['MAE_bpm']<=.9*bmae,pooled_R5_gain=agg['R5_all_reference_pct']>=br5+5,
            all_case_MAE=all(x['delta_MAE_bpm']<=5 for x in rows),
            protected_MAE=all(x['delta_MAE_bpm']<=1 for x in rows if x['case'] in ('data2','data4')),
            protected_R5=all(x['delta_R5_pp']>=-5 for x in rows if x['case'] in ('data2','data4')),
            HR_coverage=all(x['delta_hr_coverage_pp']>=-5 for x in rows),waveform_coverage=all(x['delta_waveform_coverage_pp']>=-5 for x in rows),
            common_MAE=agg['common_MAE_bpm']<agg['common_baseline_MAE_bpm'])
        qa.check(report['adoption_checks']==gates,name+' unchanged adoption gate calculation')
        qa.check(report['adoption_pass']==all(gates.values()),name+' adoption result')
        summaries.append(dict(stage=name,post_initial_results_ablation=name==stage4,pooled=agg,adoption_checks=gates,adoption_pass=all(gates.values())))
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=not qa.errors,
        qa_script_sha256=sha(__file__),checks=qa.checks,max_scalar_difference=qa.max_numeric_difference,
        stages_checked=[x['name'] for x in stages],cases_checked=len(all_rows),sensitivity_rows_checked=len(all_rows)*7,
        original_freeze_sha256=sha(frozen_path),original_freeze_created_utc=frozen['created_utc'],
        original_pre_run_bound_items=['13 inference source hashes','runner hash','evaluator hash','paired-reference hashes','cache CSV and metadata hashes','initial protocol content'],
        supplemental_hash_scope='POST-RUN supplemental evidence, not retroactively pre-registered or frozen before inference',
        supplemental_hashes=supplemental,stage4_protocol_path=str(args.stage4_protocol) if args.stage4_protocol else None,
        historical_entry_content_checks=historical_entries,
        inputs_recomputed_hashes=input_hashes,stage1_stage2_waveform_comparison=replay_notes,
        summaries=summaries,local_HR_post_run_diagnostics=local_diagnostics,common_RMSE_post_run_diagnostics=paired_qa_rows,
        errors=qa.errors,limitations=[
            'Reference core, raw reference, alignment, baseline aggregates and input manifest were not separately bound by the V25 original pre-run manifest. Their current hashes and original/evaluation receipts are checked post-run; this cannot prove every earlier instant of immutability.',
            'Observed run records and saved outputs support configured execution; this is not an operating-system file-access trace and cannot prove absence of every possible reference read.',
            'Six previously inspected recordings, with repeated participants and overlapping windows, form a development regression set, not independent held-out validation.',
            'Pooled metrics weight available windows; they are not equally weighted participant estimates or confidence intervals.',
            'Gate allows up to 5 bpm per-case MAE increase and 5 percentage-point coverage decrease (protected cases have separate limits); passing does not mean every case/metric improves.',
            'RMSE, local HR, waveform SNR and longest gaps are not adoption criteria in the frozen protocol. Supplemental local/common-RMSE diagnostics do not alter that gate.',
            'Alignment remains original file-mtime-based, not hardware synchronization; all seven shifts are verified and none selected.',
            'stage4_preserve_waveform, if included, is an explicitly post-results causal ablation and must not be described as initially pre-registered.',
            'Finite waveform coverage is computability, not proof of recovered PPG morphology.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean(dict(passed=result['passed'],checks=qa.checks,cases_checked=len(all_rows),sensitivity_rows_checked=len(all_rows)*7,max_scalar_difference=qa.max_numeric_difference,errors=qa.errors,output=str(args.output))),ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()

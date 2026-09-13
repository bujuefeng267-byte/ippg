"""Fixed cPACE/RhythmMamba comparison against frozen V2.1/V2.2 outputs.

The 597 overlapping evaluation windows are not independent training samples.
No models are loaded, trained, selected or tuned here. The frozen shared legacy
readout is re-run only as an audit of the supplied saved waveform/HR contract.
Use the existing WSL project .venv Python (NumPy, pandas, SciPy) for that audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V22 = ROOT/'rppg_motion_v22'
spec = importlib.util.spec_from_file_location('_fixed_v22_evaluation', V22/'evaluate_v22.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
CASES = prior.CASES
SYSTEMS = ('v21/fusion','trimmed_gap10/fusion','trimmed_gap15/fusion',
           'cpace','rhythm','cpace_no_homodyne')
ROLES = {s:('diagnostic' if s=='cpace_no_homodyne' else 'baseline' if '/fusion' in s else 'primary')
         for s in SYSTEMS}
CANDIDATES = {'cpace':HERE/'cpace/results', 'rhythm':HERE/'rhythm/results',
              'cpace_no_homodyne':HERE/'cpace/results_no_homodyne'}
PAIRS = list(itertools.combinations(SYSTEMS,2))


def load_readout(evidence):
    """Load the explicitly frozen decoder, never either candidate model."""
    path=V22/'legacy_motion.py'
    evidence.register(path); evidence.register(V22/'analyze_rppg.py')
    previous_path=list(sys.path)
    sys.path.insert(0,str(V22))
    try:
        sp=importlib.util.spec_from_file_location('_paper_shared_readout',path)
        module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
    finally: sys.path[:]=previous_path
    return module.estimate


def reproduced_readout(data, estimate, case, system):
    wave, table = data['wave'],data['hr']
    observed=prior.bools(wave.observed,'waveform.observed')
    interpolated=prior.bools(wave.interpolated,'waveform.interpolated')
    audit_trace=pd.DataFrame({'rgb_valid':observed})
    predicted=estimate(wave.base.to_numpy(float),audit_trace,interpolated,
                       data['fps'],10,1,42,210)
    accepted=prior.bools(predicted.accepted,'recomputed.accepted')
    np.testing.assert_array_equal(accepted,data['accepted'])
    np.testing.assert_array_equal(predicted.status.astype(str),table.status.astype(str))
    for col in prior.ESTIMATORS.values(): predicted.loc[~accepted,col]=np.nan
    maximum=0.
    for col in ('time_s','spectral_peak_bpm','ridge_bpm'):
        np.testing.assert_allclose(predicted[col],table[col],rtol=0,atol=1e-9,equal_nan=True,
                                   err_msg=f'{case}/{system}: saved-waveform readout mismatch {col}')
        both=np.isfinite(predicted[col])&np.isfinite(table[col])
        if both.any():maximum=max(maximum,float(np.abs(predicted.loc[both,col]-table.loc[both,col]).max()))
    return dict(case=case,system=system,passed=True,Nplanned=len(table),maximum_absolute_difference=maximum,
        definition='Frozen V22 legacy estimate on actual saved base, observed and interpolated; rejected final columns masked NA.')


def load_candidate(case,system,baseline,inventory,evidence):
    directory=CANDIDATES[system]/case
    summary=evidence.json(directory/'summary.json')
    wave=evidence.csv(directory/'waveform.csv'); hr=evidence.csv(directory/'heart_rate.csv')
    for col in ('time_s','base','observed','interpolated','covered'):
        if col not in wave:raise ValueError(f'{system}/{case}: missing waveform column {col}')
    frames,fps=int(summary['frames']),float(summary['fps'])
    assert frames==baseline['frames']==len(wave)
    np.testing.assert_allclose(fps,baseline['fps'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(wave.time_s,np.arange(frames)/fps,rtol=0,atol=1e-8)
    np.testing.assert_allclose(hr.time_s,baseline['centers'],rtol=0,atol=1e-8)
    expected=next(x['video']['sha256'] for x in inventory['sources'] if x['source_id']==case)
    identity_hash=summary.get('source_video_sha256',summary.get('video_sha256'))
    assert identity_hash==expected,f'{system}/{case}: missing or incorrect original-video identity'
    trace_path=V22/'validation/trimmed_gap10'/case/'frame_trace.csv'
    assert summary.get('trace_sha256')==prior.sha(evidence.register(trace_path)),f'{system}/{case}: frozen trace identity'
    reference_used=summary.get('reference_used',summary.get('parameters',{}).get('reference_used'))
    if reference_used is None and 'reference_identity' in summary:reference_used=summary['reference_identity'] is not None
    assert reference_used is False,f'{system}/{case}: reference-free inference must be explicit'
    accepted=prior.bools(hr.accepted,'accepted')
    for col in prior.ESTIMATORS.values():
        assert np.isfinite(hr.loc[accepted,col]).all()
        assert hr.loc[~accepted,col].isna().all()
    finite=np.isfinite(wave.base.to_numpy(float))
    np.testing.assert_array_equal(prior.bools(wave.covered,'covered'),finite)
    assert not np.any(prior.bools(wave.observed,'observed')&~finite)
    assert not np.any(prior.bools(wave.interpolated,'interpolated')&~finite)
    for name,expected_hash in summary.get('output_hashes',{}).items():
        assert prior.sha(evidence.register(directory/name))==expected_hash,f'Candidate output checksum: {system}/{name}'
    for filename in ('segment_diagnostics.json','native_diagnostics.json','run_manifest.json'):
        if (directory/filename).exists():evidence.register(directory/filename)
    if system=='rhythm':
        model_meta=summary.get('model',{})
        for name,record in model_meta.get('source_files',{}).items():
            if record.get('storage')=='official_snapshot':
                assert prior.sha(evidence.register(HERE/'rhythm/official_snapshot'/name))==record['sha256']
        for name in ('backend_verification.json','preprocessing_verification.json','protocol_before_inference.json'):
            audit=evidence.json(HERE/'rhythm'/name)
            if 'verification' in name:assert audit.get('passed') is True
        evidence.register(HERE/'rhythm_source_audit.json')
    result=dict(directory=directory,summary=summary,wave=wave,hr=hr,fps=fps,frames=frames,
        starts=baseline['starts'],centers=baseline['centers'],window=baseline['window'],
        step=baseline['step'],accepted=accepted,finite_samples=finite,
        finite_windows=np.array([finite[a:a+baseline['window']].all() for a in baseline['starts']]))
    return result


def fairness_record(case,system,data):
    s=data['summary']
    if system.startswith('cpace'):
        return dict(case=case,system=system,role=ROLES[system],method=s.get('method'),
            learned_model=False,training_or_finetuning_on_current_videos=False,
            pretraining_dataset='not applicable: nonlearned author signal-processing method',
            pretraining_record_overlap='not applicable',
            input_policy='V22 fixed 10% trimmed-mean per-ROI video RGB; fixed Forehead output; constant ROI-set segments',
            native_band_hz=[.7,3.0],common_readout_search_bpm=[42,210],
            native_hr_scope=s.get('native_hr_scope','per-segment author diagnostic only'),
            source_commit=s.get('author_commit'),source_hashes=s.get('code_hashes'),
            parameters=s.get('parameters'),
            whole_segment_future_information=True,
            homodyne_changes_morphology=s.get('homodyne_changes_morphology'),
            primary_or_control='default homodyne' if system=='cpace' else 'prespecified no-homodyne diagnostic',
            original_paper_protocol_reproduction=False,
            limitations=s.get('limitations',[])+[
                'Shared 10s decoder does not reproduce native author multi-ROI HR aggregation.',
                'The narrower .7-3.0Hz core can affect SNR denominators relative to methods retaining .7-3.5Hz.',
                'No morphology recovery claim follows from homodyne output or a higher spectral ratio.'])
    if system=='rhythm':
        model_meta=s.get('model') if isinstance(s.get('model'),dict) else {}
        checkpoint=model_meta.get('checkpoint')
        documented_pretraining=s.get('pretraining_dataset',s.get('training_dataset'))
        if documented_pretraining is None and checkpoint=='PreTrainedModels/PURE_cross_RhythmMamba.pth':
            documented_pretraining='PURE, per fixed official PURE_cross checkpoint/config identity; training itself not rerun or audited'
        return dict(case=case,system=system,role=ROLES[system],learned_model=True,
            model=s.get('model_name','RhythmMamba'),model_metadata=model_meta,
            training_or_finetuning_on_current_videos=False,
            pretraining_dataset=documented_pretraining or 'not explicitly documented in wrapper metadata',
            pretraining_record_overlap=s.get('pretraining_overlap','Unknown at individual-record/subject level; do not label this a held-out benchmark without provenance.'),
            weights=s.get('weights',s.get('weight_identity',dict(path=checkpoint,sha256=model_meta.get('checkpoint_sha256')))),
            source_commit=s.get('author_commit',s.get('source_commit',model_meta.get('official_commit'))),
            execution_backend=model_meta.get('backend'),
            official_cuda_kernel_parity_verified=model_meta.get('official_cuda_kernel_parity_verified'),
            source_hashes=s.get('code_hashes',s.get('source_hashes')),
            parameters=s.get('parameters',s.get('preprocessing',s.get('config'))),
            original_paper_protocol_reproduction=False,
            limitations=s.get('limitations',[])+[
                'Pretrained dataset name alone does not prove absence of subject or source-record overlap.',
                'Weight conversion/runtime, face crop, resizing, resampling, context and edge masks may differ from the paper.',
                'Reference scan execution audits do not establish parity or performance equivalence with the original fused CUDA kernels.',
                'Uniform legacy HR decoding evaluates the saved waveform under a shared readout, not the original benchmark pipeline.'],
            complete_summary=s)
    return dict(case=case,system=system,role='baseline',learned_model=False,
        training_or_finetuning_on_current_videos=False,
        input_policy='Frozen previously inspected V2.1/V2.2 signal-processing output; no new inference.',
        source_hashes=s.get('source_hashes'),parameters=s.get('config'),
        original_paper_protocol_reproduction=False)


def wave_row(case,system,data,reference,scores,mask,scope,shift):
    finite=data['finite_samples']; runs=prior.spans(finite)
    interpolated=prior.bools(data['wave'].interpolated,'interpolated')
    observed=prior.bools(data['wave'].observed,'observed')
    values=scores[mask]
    row=dict(case=case,wave=system,system=system,role=ROLES[system],scope=scope,shift_s=shift,
        reference_kind=prior.REFERENCE_KIND.get(case,'none'),Nplanned=len(reference),
        Nref=int(np.isfinite(reference).sum()),Naccepted=int(data['accepted'].sum()),
        Nfinite_waveform_windows=int(data['finite_windows'].sum()),Nscored=int(mask.sum()),
        finite_sample_count=int(finite.sum()),finite_sample_pct=100*finite.mean(),
        finite_sample_duration_s=float(finite.sum()/data['fps']),
        longest_finite_run_samples=max((b-a for a,b in runs),default=0),
        longest_finite_run_s=max((b-a for a,b in runs),default=0)/data['fps'],
        observed_samples=int(observed.sum()),interpolated_samples=int(interpolated.sum()),
        interpolated_all_samples_pct=100*interpolated.mean(),
        snr_mean_db=float(values.mean()) if len(values) else None,
        snr_median_db=float(np.median(values)) if len(values) else None,r_wave=None)
    for field,column in [('waveform_generated_count','waveform_generated'),('neighbor_covered_count','neighbor_covered'),
                         ('proposal_accepted_count','proposal_accepted')]:
        row[field]=int(prior.bools(data['hr'][column],column).sum()) if column in data['hr'] else None
    row['generation_applicability']='fusion proposal provenance' if '/fusion' in system else 'not a comparable multi-ROI overlap-fusion generation metric'
    return row


def completed_candidates(evidence,rhythm_marker):
    cpace=evidence.json(HERE/'cpace/run_summary.json')
    assert cpace.get('all_six_successful') is True
    assert set(cpace['cases'])==set(CASES)
    for case in CASES:assert {'default','no_homodyne'} <= set(cpace['cases'][case])
    rhythm=evidence.json(rhythm_marker)
    # Support either a declared successful summary or the root runner's list of
    # per-case exit records. In both cases all six named sources are required.
    if isinstance(rhythm,list):
        assert len(rhythm)==6 and {x['case'] for x in rhythm}==set(CASES)
        assert all(x.get('exit_code')==0 for x in rhythm)
    else:
        assert rhythm.get('all_six_successful') is True or rhythm.get('all_successful') is True
        cases=rhythm.get('cases',rhythm.get('runs'))
        names=set(cases) if isinstance(cases,dict) else {x['case'] for x in cases}
        assert names==set(CASES)
    return dict(cpace=cpace,rhythm=rhythm)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'evaluation')
    parser.add_argument('--rhythm-marker',type=Path,default=HERE/'rhythm/run_summary.json')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    self_check=prior.self_check()
    if args.self_check:
        print(json.dumps(self_check));return
    if args.output.exists():raise FileExistsError('Existing evaluation evidence is never overwritten')
    evidence=prior.Evidence();evidence.register(Path(__file__))
    protocol=evidence.json(HERE/'evaluation_protocol.json')
    for relative,expected in protocol['frozen_input_hashes'].items():
        assert prior.sha(evidence.register(ROOT/relative))==expected,f'Frozen scoring input changed: {relative}'
    assert protocol['planned_evaluation_windows_total']==597 and protocol['source_groups']==6
    markers=completed_candidates(evidence,args.rhythm_marker)
    frozen=evidence.json(V22/'validation/protocol_before_validation.json')
    alignment=evidence.json(ROOT/'rppg_data1_20260909/alignment_protocol.json')
    inventory=evidence.json(V22/'dataset_inventory.json')
    baseline_metrics=evidence.csv(V22/'evaluation/metrics.csv')
    baseline_snr=evidence.csv(V22/'evaluation/waveform_metrics.csv')
    estimate=load_readout(evidence)
    for filename,expected in frozen['source_hashes'].items():
        assert prior.sha(evidence.register(V22/filename))==expected
    names=['metrics','common_metrics','continuity','paired_metrics','deltas','waveform_metrics',
           'waveform_deltas','waveform_windows','aligned_windows','reference_windows',
           'sensitivity','sensitivity_common','waveform_sensitivity']
    tables={key:[] for key in names};fairness=[];readout_audit=[];baseline_audit=[];metadata={}
    for case in CASES:
        previous=prior.load_case(case,V22/'validation',frozen,evidence)
        data={version+'/fusion':d for version,d in previous.items()}
        base=previous['v21']
        for system in CANDIDATES:
            data[system]=load_candidate(case,system,base,inventory,evidence)
            readout_audit.append(reproduced_readout(data[system],estimate,case,system))
        for system in SYSTEMS:fairness.append(fairness_record(case,system,data[system]))
        branches={}
        for system in SYSTEMS:
            d=data[system]
            for estimator,column in prior.ESTIMATORS.items():
                label=system+'/'+estimator
                branches[label]=dict(system=system,estimator=estimator,prediction=d['hr'][column].to_numpy(float),output=d['accepted'])
                tables['continuity'].append(dict(case=case,branch=label,system=system,role=ROLES[system],estimator=estimator,
                    **prior.continuity(d['accepted'],d['starts'],d['window'],d['fps'])))
        common=np.logical_and.reduce([b['output'] for b in branches.values()])
        primary_common=np.logical_and.reduce([b['output'] for b in branches.values() if ROLES[b['system']]!='diagnostic'])
        shifts=alignment['sensitivity_shifts_s'] if case=='data1' else [0]
        for shift in shifts:
            reference,ref_details=prior.reference_for(case,base['centers'],shift,evidence,alignment)
            for i,c in enumerate(base['centers']):
                tables['reference_windows'].append(dict(case=case,shift_s=shift,window_index=i,time_s=c,
                    reference_kind=prior.REFERENCE_KIND.get(case,'none'),reference_bpm=reference[i],
                    reference_eligible=np.isfinite(reference[i]),**ref_details[i],
                    video_window_center_utc_ns=alignment['video_start_utc_ns']+round((float(c)+shift)*1e9) if case=='data1' else None))
            own_rows={}
            for label,b in branches.items():
                keys=dict(case=case,branch=label,system=b['system'],role=ROLES[b['system']],estimator=b['estimator'],
                          reference_kind=prior.REFERENCE_KIND.get(case,'none'),shift_s=shift)
                own=dict(**keys,scope='own',**prior.metrics(b['prediction'],b['output'],reference));own_rows[label]=own
                shared=dict(**keys,scope='strict_common_all12',Ncommon_output=int(common.sum()),
                            **prior.metrics(b['prediction'],common,reference))
                if case=='data1':tables['sensitivity'].append(own);tables['sensitivity_common'].append(shared)
                if shift!=0:continue
                tables['metrics'].append(own);tables['common_metrics'].append(shared)
                # A diagnostic control must not silently determine the primary
                # comparison subset: also retain the five-system intersection.
                if ROLES[b['system']]!='diagnostic':
                    tables['common_metrics'].append(dict(**keys,scope='strict_common_primary10',Ncommon_output=int(primary_common.sum()),
                        **prior.metrics(b['prediction'],primary_common,reference)))
                for i,c in enumerate(base['centers']):
                    valid=b['output'][i] and np.isfinite(reference[i])
                    tables['aligned_windows'].append(dict(case=case,branch=label,system=b['system'],role=ROLES[b['system']],
                        window_index=i,time_s=c,actual_window_start_s=base['starts'][i]/base['fps'],
                        actual_window_end_s=(base['starts'][i]+base['window'])/base['fps'],prediction_bpm=b['prediction'][i],
                        reference_bpm=reference[i],reference_eligible=np.isfinite(reference[i]),valid_output=b['output'][i],
                        paired_valid=valid,all12_common_output=common[i],primary10_common_output=primary_common[i],
                        error_bpm=b['prediction'][i]-reference[i] if valid else np.nan,status=data[b['system']]['hr'].status.iloc[i]))
                if '/fusion' in b['system']:
                    old=baseline_metrics[baseline_metrics.case.eq(case)&baseline_metrics.branch.eq(label)].iloc[0]
                    for col in ['Nplanned','Noutput','Nref','Nvalid','MAE_bpm','RMSE_bpm','Bias_bpm','P5_valid_pct','R5_all_reference_pct']:
                        np.testing.assert_allclose(own[col] if own[col] is not None else np.nan,old[col],rtol=0,atol=1e-9,equal_nan=True)
                    baseline_audit.append(dict(case=case,branch=label,matched=True))
            if shift==0:
                for before,after in PAIRS:
                    for estimator in prior.ESTIMATORS:
                        a,b=before+'/'+estimator,after+'/'+estimator;aa,bb=branches[a],branches[b]
                        both=aa['output']&bb['output'];added=~aa['output']&bb['output'];lost=aa['output']&~bb['output']
                        keys=dict(case=case,comparison=a+'->'+b,before=a,after=b,estimator=estimator,
                            comparison_role='diagnostic' if 'diagnostic' in (ROLES[before],ROLES[after]) else 'primary',
                            reference_kind=prior.REFERENCE_KIND.get(case,'none'),shift_s=0,
                            Ncommon_output=int(both.sum()),Nadded_output=int(added.sum()),Nlost_output=int(lost.sum()))
                        subset={}
                        for scope,label,mask in [('common_before',a,both),('common_after',b,both),('added_after',b,added),('lost_before',a,lost)]:
                            row=dict(**keys,scope=scope,branch=label,**prior.metrics(branches[label]['prediction'],mask,reference))
                            tables['paired_metrics'].append(row);subset[scope]=row
                        cdelta=subset['common_after']['Nwithin5']-subset['common_before']['Nwithin5']
                        assert own_rows[b]['Nwithin5']-own_rows[a]['Nwithin5']==cdelta+subset['added_after']['Nwithin5']-subset['lost_before']['Nwithin5']
                        tables['deltas'].append(dict(**keys,
                            common_delta_MAE_bpm=prior.difference(subset['common_after']['MAE_bpm'],subset['common_before']['MAE_bpm']),
                            own_delta_MAE_bpm=prior.difference(own_rows[b]['MAE_bpm'],own_rows[a]['MAE_bpm']),
                            delta_C_out_pct=prior.difference(own_rows[b]['C_out_pct'],own_rows[a]['C_out_pct']),
                            delta_R5_all_reference_pct=prior.difference(own_rows[b]['R5_all_reference_pct'],own_rows[a]['R5_all_reference_pct']),
                            added_Nwithin5=subset['added_after']['Nwithin5'],lost_Nwithin5=subset['lost_before']['Nwithin5'],common_Nwithin5_delta=cdelta))
            scores={system:prior.wave_scores(d,reference) for system,d in data.items()}
            wave_common=np.logical_and.reduce([np.isfinite(v) for v in scores.values()])
            for system in SYSTEMS:
                d,values=data[system],scores[system]
                for scope,mask in [('own',np.isfinite(values)),('strict_common_all6',wave_common)]:
                    row=wave_row(case,system,d,reference,values,mask,scope,shift)
                    if case=='data1':tables['waveform_sensitivity'].append(row)
                    if shift==0:tables['waveform_metrics'].append(row)
                    if '/fusion' in system and scope=='own' and shift==0:
                        old=baseline_snr[baseline_snr.case.eq(case)&baseline_snr.wave.eq(system)&baseline_snr.scope.eq('own')].iloc[0]
                        for col in ['snr_mean_db','snr_median_db']:
                            np.testing.assert_allclose(row[col] if row[col] is not None else np.nan,old[col],rtol=0,atol=1e-8,equal_nan=True)
                if shift!=0:continue
                for i,c in enumerate(base['centers']):
                    tables['waveform_windows'].append(dict(case=case,wave=system,system=system,role=ROLES[system],window_index=i,time_s=c,
                        reference_bpm=reference[i],accepted=d['accepted'][i],finite_segment=d['finite_windows'][i],
                        snr_ref_h1_db=values[i],all6_common_snr=wave_common[i]))
            if shift==0:
                for a,b in PAIRS:
                    mask=np.isfinite(scores[a])&np.isfinite(scores[b]);delta=scores[b][mask]-scores[a][mask]
                    tables['waveform_deltas'].append(dict(case=case,before=a,after=b,Npaired_snr=len(delta),
                        comparison_role='diagnostic' if 'diagnostic' in (ROLES[a],ROLES[b]) else 'primary',
                        mean_paired_delta_snr_db=float(delta.mean()) if len(delta) else None,
                        median_paired_delta_snr_db=float(np.median(delta)) if len(delta) else None,
                        positive_delta_pct=100*np.mean(delta>1e-9) if len(delta) else None,
                        numerical_tie_pct=100*np.mean(np.abs(delta)<=1e-9) if len(delta) else None))
        metadata[case]=dict(frames=base['frames'],fps=base['fps'],Nplanned=len(base['centers']),
            actual_window_s=base['window']/base['fps'],actual_step_s=base['step']/base['fps'],
            reference_kind=prior.REFERENCE_KIND.get(case,'none'),
            directories={system:str(d['directory']) for system,d in data.items()})
        print(f'Completed comparison: {case}',flush=True)
    expected={'metrics':72,'common_metrics':132,'continuity':72,'paired_metrics':720,'deltas':180,
        'waveform_metrics':72,'waveform_deltas':90,'waveform_windows':3582,'aligned_windows':7164,
        'reference_windows':849,'sensitivity':84,'sensitivity_common':84,'waveform_sensitivity':84}
    for name,n in expected.items():assert len(tables[name])==n,(name,len(tables[name]),n)
    assert len(readout_audit)==18 and len(baseline_audit)==36
    evidence.verify();args.output.mkdir(parents=True,exist_ok=False)
    manifest={}
    for name,rows in tables.items():
        frame=pd.DataFrame(rows);utc_columns=[col for col in frame if col.endswith('_utc_ns')]
        for col in utc_columns:frame[col]=pd.array([r.get(col) for r in rows],dtype='Int64')
        path=args.output/(name+'.csv');frame.to_csv(path,index=False)
        loaded=pd.read_csv(path,dtype={col:'Int64' for col in utc_columns})
        assert frame.shape==loaded.shape and list(frame)==list(loaded)
        np.testing.assert_array_equal(frame.isna(),loaded.isna())
        for col in utc_columns:pd.testing.assert_series_equal(frame[col],loaded[col],check_names=False)
        manifest[path.name]=dict(rows=len(frame),sha256=prior.sha(path),columns=list(frame))
    for name,records in [('readout_reproduction',readout_audit),('baseline_reproduction',baseline_audit)]:
        path=args.output/(name+'.csv');pd.DataFrame(records).to_csv(path,index=False)
        manifest[path.name]=dict(rows=len(records),sha256=prior.sha(path))
    (args.output/'fairness.json').write_text(json.dumps(prior.clean(fairness),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),evaluator_sha256=prior.sha(Path(__file__)),
        runtime={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        primary_shift_s=0,systems=list(SYSTEMS),roles=ROLES,cases=list(CASES),case_metadata=metadata,
        sample_count_correction=protocol['sample_count_correction'],
        evaluation_protocol=protocol,candidate_completion=markers,
        primary_metrics=tables['metrics'],waveform_metrics=tables['waveform_metrics'],deltas=tables['deltas'],
        fairness=fairness,readout_reproduction=readout_audit,baseline_reproduction=baseline_audit,
        definitions=dict(P5='Nwithin5/Nvalid; R5=Nwithin5/original Nref; zero output and no reference are not zero error.',
            common='All12 strict common and separate primary10 common; all pairs retain original denominators and report added/lost output subsets.',
            continuity='Consecutive accepted-window runs and union of actual raw-frame support; overlap is not independent duration or validated HR time.',
            SNR='Exactly inherited V22 reference-H1 ratio on saved waveforms; cPACE native narrower band is a disclosed fairness limitation.',
            timing='Both readouts are offline; cPACE may use entire-segment future data; data1 timing is a fixed approximate archive-mtime assumption.',
            native_HR='Author native segment/multi-ROI HR is diagnostic, not silently substituted into shared 10s-window results.'),
        QA=dict(self_check=self_check,all_candidate_saved_waveform_readouts_reproduced=True,
            all36_baseline_HR_branches_reproduced=True,all_baseline_waveform_SNR_reproduced=True,
            source_input_hashes_unchanged=True,CSV_rows_missingness_and_exact_UTC_roundtrip=True),
        limitations=['Previously inspected six source groups; not an untouched or independent-subject benchmark.',
            '597 overlapping windows are not 597 independent training examples; this evaluator performs no training.',
            'Approximate data1 synchronization and unknown device notification lag limit absolute accuracy claims.',
            'Pretraining overlap/weight conversion/preprocessing and native method differences are recorded, not inferred away.',
            'No waveform morphology recovery or real-time implementation claim is established by these comparisons.'],
        input_manifest=evidence.manifest,output_manifest=manifest)
    (args.output/'summary.json').write_text(json.dumps(prior.clean(report),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(tables['metrics'])[['case','branch','Noutput','Nref','MAE_bpm','P5_valid_pct','R5_all_reference_pct']].to_string(index=False))
    print(f'Completed: {args.output}')


if __name__=='__main__':main()

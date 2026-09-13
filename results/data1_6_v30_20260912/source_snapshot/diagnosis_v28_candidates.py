"""Post-hoc V28 candidate failure diagnosis; references never enter inference.

Reads already saved wave-derived candidates and already paired references.
Oracle selections below are explanatory upper bounds, not a new estimator.
No inference, threshold search, temporal realignment, or source modification.
"""
from pathlib import Path
from dataclasses import asdict
import collections
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = Path('/home/fengbujue/项目/rppg识别')
V28 = P/'results/data1_6_v28_20260912/direct_guard'
V26 = P/'results/data1_6_v26_20260911/component_harmonics'
B = P/'results/data1_6_20260911'
sys.path.insert(0, str(P/'motion_upgrade_v28_20260912'))
from evidence_hr import DEFAULT_CONFIG


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, (np.integer, int)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    return value


def error_runs(mask, centers, fps):
    edge = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [dict(first_window_index=int(a), stop_window_index_exclusive=int(b),
                 first_center_s=float(centers[a]), last_center_s=float(centers[b-1]),
                 updates=int(b-a), update_duration_s=float((b-a)*round(fps)/fps))
            for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def terms(candidate, state):
    c = DEFAULT_CONFIG
    result = dict(spectral=c.spectral_weight*np.log(max(candidate['relative_power'], 1e-30)),
        periodic=c.periodic_weight*candidate['periodic_support'],
        harmonic=c.harmonic_weight*candidate['harmonic_support'],
        motion=-c.motion_weight*candidate['motion_overlap'],
        grid_offset=-c.candidate_offset_penalty*(state-candidate['bpm'])**2)
    base = sum(v for k, v in result.items() if k != 'grid_offset')
    assert abs(base-candidate['evidence_score']) < 1e-10
    return {**result, 'total_emission': sum(result.values())}


def main():
    inputs = {str(P/'motion_upgrade_v28_20260912/evidence_hr.py'):
              sha(P/'motion_upgrade_v28_20260912/evidence_hr.py')}
    cases, all_windows = [], []
    for case in (f'data{i}' for i in range(1, 7)):
        paths = [V28/case/'heart_rate.csv', V28/case/'evaluation/paired_windows.csv',
                 V26/case/'all_roi_candidates.json', B/case/'inference/frame_trace.json']
        for path in paths: inputs[str(path)] = sha(path)
        hr, paired = [pd.read_csv(path) for path in paths[:2]]
        roi_candidates = json.loads(paths[2].read_text())
        fps = float(json.loads(paths[3].read_text())['fps'])
        np.testing.assert_allclose(hr.time_s, paired.time_s, rtol=0, atol=1e-8)
        np.testing.assert_allclose(hr.ridge_bpm, paired.estimated_bpm, rtol=0, atol=0, equal_nan=True)
        assert np.array_equal(hr.accepted, paired.accepted)
        ref = paired.reference_bpm.to_numpy(float)
        reference_valid = paired.reference_valid.to_numpy(bool)
        accepted = hr.accepted.to_numpy(bool) & reference_valid
        errors = hr.ridge_bpm.to_numpy(float)-ref
        wrong = accepted & (abs(errors) > 5)
        by_window = collections.defaultdict(list)
        for row in roi_candidates: by_window[int(row['window_index'])].append(row)
        windows = []
        for i, row in hr.iterrows():
            candidates = json.loads(row.evidence_candidates_json)
            peaks = [c['bpm'] for c in candidates]
            states = [bpm for c in candidates for bpm in c['supported_bpm']]
            min_peak = min((abs(bpm-ref[i]) for bpm in peaks), default=np.inf)
            min_state = min((abs(bpm-ref[i]) for bpm in states), default=np.inf)
            old_at_state = collections.defaultdict(set)
            for c in by_window[i]:
                physical = c['channel'].split('/')[1]
                assert physical in ('forehead', 'left_cheek', 'right_cheek')
                for bpm in c['supported_bpm']:
                    if abs(bpm-ref[i]) <= 5:
                        old_at_state[float(bpm)].add(physical)
            max_rois = max((len(rois) for rois in old_at_state.values()), default=0)
            info = dict(case=case, window_index=int(i), time_s=float(row.time_s),
                window_start_s=float(row.window_start_s), window_end_s=float(row.window_end_s),
                reference_bpm=float(ref[i]), reference_valid=bool(reference_valid[i]),
                accepted=bool(row.accepted), status=str(row.status), output_bpm=float(row.ridge_bpm),
                error_bpm=float(errors[i]), local_peak_bpm=float(row.evidence_local_bpm),
                local_error_bpm=float(row.evidence_local_bpm-ref[i]),
                candidate_count=len(candidates), min_candidate_peak_error_bpm=min_peak,
                min_candidate_supported_state_error_bpm=min_state,
                oracle_peak_within5=bool(accepted[i] and min_peak <= 5),
                oracle_peak_within10=bool(accepted[i] and min_peak <= 10),
                oracle_supported_within5=bool(accepted[i] and min_state <= 5),
                oracle_supported_within10=bool(accepted[i] and min_state <= 10),
                roi_same_near_grid_max_physical_rois=max_rois,
                roi_near_grid_support={str(k): sorted(v) for k, v in sorted(old_at_state.items())})
            if not accepted[i]:
                info['failure_class'] = 'no_valid_output_or_reference'
            elif not wrong[i]:
                info['failure_class'] = 'within5_output'
            elif min_state > 5:
                info['failure_class'] = 'wrong_no_supported_state_within5'
            else:
                chosen = [c for c in candidates if c['bpm'] == row.evidence_selected_peak_bpm]
                assert len(chosen) == 1
                chosen = chosen[0]
                candidate_options = []
                for c in candidates:
                    near = [bpm for bpm in c['supported_bpm'] if abs(bpm-ref[i]) <= 5]
                    for bpm in near:
                        candidate_options.append((terms(c, bpm)['total_emission'], c, bpm))
                assert candidate_options
                _, near, near_state = max(candidate_options, key=lambda option: option[0])
                selected_terms = terms(chosen, float(row.ridge_bpm))
                assert abs(selected_terms['total_emission']-row.evidence_score) < 1e-10
                near_terms = terms(near, near_state)
                same = chosen['bpm'] == near['bpm']
                info['failure_class'] = ('wrong_supported_near_state_on_same_selected_peak' if same
                    else 'wrong_other_near_candidate_local_peak_already_correct' if abs(info['local_error_bpm']) <= 5
                    else 'wrong_other_near_candidate_local_scoring_also_wrong')
                info['near_candidate_diagnostic'] = dict(selected_peak_bpm=chosen['bpm'],
                    near_peak_bpm=near['bpm'], near_supported_state_bpm=near_state,
                    same_selected_peak=bool(same), selected_relative_power=chosen['relative_power'],
                    near_relative_power=near['relative_power'], selected_terms=selected_terms,
                    near_terms=near_terms,
                    selected_minus_near={k: selected_terms[k]-near_terms[k] for k in selected_terms},
                    selection='For diagnosis only, highest existing emission among supported states within reference ±5. Reference is not a model input.')
            windows.append(info)
        count = lambda field: sum(bool(w[field]) for w in windows)
        peak5, supported5 = count('oracle_peak_within5'), count('oracle_supported_within5')
        relative_half = wrong & (abs(hr.ridge_bpm.to_numpy(float)-.5*ref) <= 5)
        relative_double = wrong & (abs(hr.ridge_bpm.to_numpy(float)-2*ref) <= 5)
        near_wrong = [w for w in windows if w.get('near_candidate_diagnostic')]
        different = [w for w in near_wrong if not w['near_candidate_diagnostic']['same_selected_peak']]
        means = lambda rows: {k: float(np.mean([w['near_candidate_diagnostic']['selected_minus_near'][k] for w in rows]))
                              for k in ('spectral', 'periodic', 'harmonic', 'motion', 'grid_offset', 'total_emission')} if rows else {}
        summary = dict(case=case, fps=fps, Nplanned=len(hr), Nref=int(reference_valid.sum()),
            Noutput=int(accepted.sum()), Nwithin5=int((accepted & ~wrong).sum()), Nwrong=int(wrong.sum()),
            MAE_bpm=float(np.mean(abs(errors[accepted]))),
            reference_bpm_range=[float(np.min(ref[reference_valid])), float(np.max(ref[reference_valid]))],
            output_bpm_range=[float(np.min(hr.ridge_bpm[accepted])), float(np.max(hr.ridge_bpm[accepted]))],
            peak_oracle_within5=peak5, peak_oracle_within10=count('oracle_peak_within10'),
            supported_oracle_within5=supported5, supported_oracle_within10=count('oracle_supported_within10'),
            wrong_with_supported_near_candidate=len(near_wrong),
            wrong_without_supported_near_candidate=sum(w['failure_class'] == 'wrong_no_supported_state_within5' for w in windows),
            wrong_with_local_peak_within5=int((wrong & (abs(hr.evidence_local_bpm.to_numpy(float)-ref) <= 5)).sum()),
            failure_class_counts=dict(collections.Counter(w['failure_class'] for w in windows)),
            wrong_half_reference_count=int(relative_half.sum()),
            wrong_half_reference_pct_of_output=100*float(relative_half.sum())/int(accepted.sum()),
            wrong_double_reference_count=int(relative_double.sum()),
            wrong_double_reference_pct_of_output=100*float(relative_double.sum())/int(accepted.sum()),
            error_over5_runs=error_runs(wrong, hr.time_s.to_numpy(float), fps),
            error_over10_runs=error_runs(accepted & (abs(errors)>10), hr.time_s.to_numpy(float), fps),
            half_reference_runs=error_runs(relative_half, hr.time_s.to_numpy(float), fps),
            roi_two_region_oracle_on_valid_outputs=sum(w['accepted'] and w['reference_valid'] and w['roi_same_near_grid_max_physical_rois'] >= 2 for w in windows),
            final_no_near_candidate_but_two_roi_near=sum(w['failure_class'] == 'wrong_no_supported_state_within5'
                and w['roi_same_near_grid_max_physical_rois'] >= 2 for w in windows),
            score_mean_difference_all_wrong_near=means(near_wrong),
            score_mean_difference_distinct_wrong_near=means(different),
            score_difference_convention='Positive term favors the selected wrong state; negative favors the diagnostic reference-near state. These are emission differences, not an ablation of DP transitions.',
            windows=windows)
        cases.append(summary)
        all_windows.extend(windows)
    sums = {name: sum(c[name] for c in cases) for name in ('Nplanned', 'Nref', 'Noutput', 'Nwithin5', 'Nwrong',
        'peak_oracle_within5', 'peak_oracle_within10', 'supported_oracle_within5', 'supported_oracle_within10',
        'wrong_with_supported_near_candidate', 'wrong_without_supported_near_candidate',
        'wrong_with_local_peak_within5', 'roi_two_region_oracle_on_valid_outputs',
        'final_no_near_candidate_but_two_roi_near')}
    sums['actual_P5_valid_pct'] = 100*sums['Nwithin5']/sums['Noutput']
    sums['supported_oracle_P5_valid_pct'] = 100*sums['supported_oracle_within5']/sums['Noutput']
    sums['supported_oracle_R5_all_reference_pct'] = 100*sums['supported_oracle_within5']/sums['Nref']
    result = dict(reference_used_for_posthoc_diagnosis=True, reference_used_for_inference=False,
        inference_run=False, thresholds_changed=False, source_modified=False,
        source_hashes=inputs, evidence_config=asdict(DEFAULT_CONFIG), pooled=sums, cases=cases,
        limitations=['Oracle metrics use the reference to identify candidates retrospectively. They are not deployable accuracy estimates.',
            'A ±5-supported-state oracle is more permissive than a ±5-peak-center oracle because candidates retain nearby supported BPM grid states.',
            'Missing output windows are excluded from output-based candidate oracles and remain failures in all-reference success.',
            'Two physical ROI support is counted on a common existing BPM grid; correlated POS/CHROM and baseline/tracked branches do not add ROI votes.',
            'Reference-near spectral support does not establish pulse morphology or prove that the spectral component is physiological.',
            'The six videos and estimated Polar alignment are already inspected development data; no offset is optimized here.',
            'Error-run duration is wrong-update count times hop, not the union of overlapping 10-second windows.'])
    for path, expected in inputs.items(): assert sha(Path(path)) == expected
    (HERE/'diagnosis_v28_candidates.json').write_text(json.dumps(clean(result), ensure_ascii=False, indent=2)+'\n')
    pd.DataFrame([{k: v for k, v in w.items() if not isinstance(v, dict)} for w in all_windows]).to_csv(
        HERE/'diagnosis_v28_windows.csv', index=False)
    lines = ['# V28 精度瓶颈：候选存在性与评分诊断', '',
        '只读分析已保存的 V28 输出和融合前区域候选。参考心率仅用于事后判断哪些窗口估错，没有进入推理、候选生成或参数选择。', '',
        '## 最终波形里是否已有接近正确值的候选', '',
        '| 视频 | 有效输出 | 实际±5正确 | 峰中心±5理想上限 | 支持频点±5理想上限 | 支持频点±10理想上限 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f"| {c['case']} | {c['Noutput']} | {c['Nwithin5']} | {c['peak_oracle_within5']} | {c['supported_oracle_within5']} | {c['supported_oracle_within10']} |")
    lines += ['', f"全部 {sums['Noutput']} 个有效输出中，实际±5正确 {sums['Nwithin5']} 个（{sums['actual_P5_valid_pct']:.2f}%）。假如事后借助参考值在保存的候选支持频点中挑选，最多可得到 {sums['supported_oracle_within5']} 个±5结果（{sums['supported_oracle_P5_valid_pct']:.2f}%）。**这个上限使用了参考答案，不能作为改进后的准确率。**", '',
        f"{sums['Nwrong']} 个错误输出中，{sums['wrong_with_supported_near_candidate']} 个已有±5支持频点，{sums['wrong_without_supported_near_candidate']} 个没有。支持频点包含峰旁约±3 bpm，因此比要求峰中心本身接近参考更宽松。", '',
        '## 三段主要困难视频', '',
        '- data1：37个输出全部超出±5；只有5个窗口有±5内的峰中心。13个支持频点可接近参考的错误中，4个只是原选中峰的边缘，并非另一个正确峰。错误峰往往更强，接近参考的独立峰常较弱。',
        '- data3：59个输出中48个超出±5，27个错误输出接近参考心率的一半。某些窗口接近参考的峰比低频峰更强，但周期与谐波奖励仍使低频峰获胜。',
        '- data6：36个输出全部超出±5；16个已有±5支持频点，其中10个有±5峰中心。接近参考的候选常只有最强峰约5%–15%的功率，现有周期和谐波项仍偏向错误解释。', '',
        '这三段中，“局部最佳峰已在±5内、随后被DP改错”的计数均为0。因此不能把主要问题归结为心率曲线过度平滑。DP仍可能延长错误段，但仅调整DP没有充分证据能解决主要误差。', '',
        '## 融合前还剩多少信息', '',
        '| 视频 | 最终缺少±5支持频点的窗口 | 其中原区域层仍有至少2个ROI支持同一邻近频点 |',
        '|---|---:|---:|']
    for c in cases:
        lines.append(f"| {c['case']} | {c['wrong_without_supported_near_candidate']} | {c['final_no_near_candidate_but_two_roi_near']} |")
    lines += ['', '这里检索的是 V26 保存的每区域原始候选，按额头、左脸、右脸三个物理区域计票，同一区域的多个算法/分支不重复计票。多数最终缺失候选的困难窗口在区域层仍有参考附近支持，提示需要检查分支选择和波形混合是否削弱了有效频率。频率接近参考并不证明它一定是生理脉搏，也不能据此用参考挑区域。', '',
        '## 得分项的方向', '',
        '已按现有公式复算：0.35×对数相对功率＋1.10×周期支持＋1.00×谐波支持－1.25×运动重合－频点偏移代价，并核对保存的候选/选中状态得分。下表是存在邻近候选的错误窗口中，“实际选中状态减事后邻近状态”的平均项差；正数意味着该项支持了错误选择。这是解释现有得分，不是关闭某项后的实验结果。', '',
        '| 视频 | 频谱项差 | 周期项差 | 谐波项差 | 运动项差 |', '|---|---:|---:|---:|---:|']
    for c in cases:
        d = c['score_mean_difference_all_wrong_near']
        lines.append(f"| {c['case']} | {d.get('spectral', np.nan):+.3f} | {d.get('periodic', np.nan):+.3f} | {d.get('harmonic', np.nan):+.3f} | {d.get('motion', np.nan):+.3f} |")
    lines += ['', '后续可检验的方向是：检查周期/谐波评分对错误子谐波的奖励，以及保留区域候选后再选择实测分量。应继续保留 V28 的直接运动证据保护，固定统一规则后再回归；本诊断没有给出按视频调参或使用参考心率切换方案的规则。', '',
        'JSON保存每个窗口的候选存在性、错误类别、半频/倍频比例、持续错误段、得分项差及输入SHA；CSV保存便于筛查的窗口摘要。旧视频对齐仍是估计同步，所有结果属于已看过的开发数据。']
    (HERE/'diagnosis_v28_candidates.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(clean(sums), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()

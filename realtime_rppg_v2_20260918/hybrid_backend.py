"""Fixed-delay R1 fallback with an explicitly guarded V28 component upgrade.

Both producers receive exactly the same captured rows. R1 owns the accepted
mask and fallback values. A retained component may replace a same-window R1
result only when the *existing* V28 router admitted it, R1 agrees within the
existing 6 bpm peak-distance tolerance with the old HR that router examined,
and the retained final HR is a measured supported state within the existing
3 bpm radius of the admitted component proposal. No reference data is read.
"""
from __future__ import annotations

import copy
import math
import time

from fast_backend import StreamingBackend as FastR1
from retained_fast_backend import StreamingBackend as RetainedFast
from retained_backend import StreamingConfig, _number


VARIANT = "realtime_r1_guarded_retained_component_2s_delay_v3"
OLD_AGREEMENT_BPM = 6.
CANDIDATE_RADIUS_BPM = 3.


class RetainedTelemetry(RetainedFast):
    """Expose actual router evidence without changing its signal or decisions."""
    def _pipeline(self, trace):
        result = super()._pipeline(trace)
        self._current_decisions = result[2]
        return result

    def _infer(self, received_end):
        self._current_decisions = None
        result = super()._infer(received_end)
        result.update(retained_guard_old_bpm=None, retained_guard_candidate_bpm=None,
                      retained_guard_old_direct_motion_risk=None,
                      retained_guard_candidate_motion_risk=None)
        if self._current_decisions is not None:
            index = round((result["window_start_s"]-result["buffer_start_s"])/self.config.step_s)
            row = self._current_decisions.iloc[index]
            result.update(retained_guard_old_bpm=_number(row.old_bpm,None),
                          retained_guard_candidate_bpm=_number(row.candidate_bpm,None),
                          retained_guard_old_direct_motion_risk=_number(row.old_motion_risk_direct,None),
                          retained_guard_candidate_motion_risk=_number(row.candidate_motion_risk,None))
        return result


def choose_aligned(fast, retained):
    """Return a copied observation; rejected/missing fast windows stay missing."""
    result = copy.deepcopy(fast)
    reason = "retained_window_unavailable"
    replace = False
    if retained is not None:
        for key in ("window_start_s", "window_end_s"):
            if not math.isclose(float(fast[key]),float(retained[key]),rel_tol=0.,abs_tol=1e-6):
                raise ValueError("Hybrid producers must describe exactly the same measurement window")
        old = _number(retained.get("retained_guard_old_bpm"))
        proposed = _number(retained.get("retained_guard_candidate_bpm"))
        measured = _number(retained.get("hr_bpm"))
        fast_bpm = _number(fast.get("hr_bpm"))
        if not fast["accepted"]:
            reason = "r1_rejected_preserved"
        elif not retained["accepted"]:
            reason = "retained_rejected_keep_r1"
        elif not retained.get("component_route_eligible",False):
            reason = "v28_component_not_admitted"
        elif not math.isfinite(old) or not math.isfinite(fast_bpm) or abs(fast_bpm-old)>OLD_AGREEMENT_BPM:
            reason = "r1_disagrees_with_guarded_old_hr"
        elif not math.isfinite(proposed) or not math.isfinite(measured) or abs(measured-proposed)>CANDIDATE_RADIUS_BPM:
            reason = "retained_final_hr_not_admitted_component"
        elif not any(measured in candidate.get("supported_bpm",[]) for candidate in retained.get("evidence_candidates",[])):
            reason = "retained_final_hr_has_no_measured_spectral_support"
        else:
            reason = "existing_motion_guard_and_matching_r1_old_hr"
            replace = True
            result = copy.deepcopy(retained)
    result.update(hybrid_replaced=replace,hybrid_decision_reason=reason,
                  hybrid_fallback_hr_bpm=fast.get("hr_bpm"),
                  hybrid_fallback_accepted=bool(fast["accepted"]),
                  hybrid_waveform_source="retained_component" if replace else "fast_r1",
                  hybrid_guard_old_bpm=retained.get("retained_guard_old_bpm") if retained else None,
                  hybrid_guard_candidate_bpm=retained.get("retained_guard_candidate_bpm") if retained else None)
    return result


class StreamingBackend:
    def __init__(self, project_root=None, config=None):
        self.retained = RetainedTelemetry(project_root,config)
        self.fast = FastR1(project_root)
        self.config = self.retained.config
        self.pending = {}
        self.last_emitted_end = None
        self.events_emitted = 0

    @property
    def configuration(self):
        return dict(variant=VARIANT,reference_used=False,window_s=10.,step_s=1.,
                    fixed_delay_s=2.,history_s=22.,sample_hz=30.,max_gap_s=.1,
                    earliest_event_s=12.,earliest_possible_accepted_s=12.,
                    waveform_kind="independent_fixed_delay_measured_window_snapshots",
                    full_offline_v28_v32_equivalent=False,
                    acceptance_policy="preserve_same_window_R1_accepted_mask",
                    old_hr_agreement_bpm=OLD_AGREEMENT_BPM,
                    final_component_radius_bpm=CANDIDATE_RADIUS_BPM,
                    mode_selection="one_fixed_policy_all_inputs")

    def append(self,row):
        started=time.perf_counter()
        fast=self.fast.append(row)
        if fast is not None:
            self.pending[round(fast["window_end_s"],7)]=fast
        retained=self.retained.append(row)
        current=float(row["time_s"])
        due=[end for end in self.pending if end+2.<=current+1e-7]
        if not due:
            return None
        # A capture gap can skip several updates. Publish the latest available
        # eligible observation once; never fabricate the absent intervening HRs.
        end=max(due)
        held=self.pending[end]
        matching=(retained if retained is not None and
                  math.isclose(retained["window_end_s"],end,rel_tol=0.,abs_tol=1e-6) else None)
        result=choose_aligned(held,matching)
        selected_compute_ms=_number(result.get("processing_ms"),0.)
        skipped=max(0,round(end-self.last_emitted_end)-1) if self.last_emitted_end is not None else 0
        for key in list(self.pending):
            if key<=end:
                del self.pending[key]
        result.update(variant=VARIANT,event_index=self.events_emitted,reference_used=False,
                      emitted_at_source_s=current,input_delay_s=current-result["window_end_s"],
                      fixed_delay_s=2.,skipped_update_count=skipped,
                      original_selected_processing_ms=selected_compute_ms,
                      processing_ms=1000*(time.perf_counter()-started))
        self.last_emitted_end=end
        self.events_emitted+=1
        return result

import copy
import json
from pathlib import Path
import unittest

from fast_backend import StreamingBackend as Fast
from hybrid_backend import StreamingBackend, choose_aligned
from test_retained_backend import rows

P=Path(str(Path(__file__).resolve().parent.parent))


def events():
    fast=dict(window_start_s=20.,window_end_s=30.,time_s=25.,accepted=True,hr_bpm=185.,
              waveform=dict(time_s=[20.,21.],values=[1.,2.],covered=[True,True]),
              status='candidate_only')
    retained=copy.deepcopy(fast)
    retained.update(hr_bpm=107.,component_route_eligible=True,
                    retained_guard_old_bpm=184.,retained_guard_candidate_bpm=106.,
                    evidence_candidates=[dict(supported_bpm=[105.,106.,107.])])
    retained['waveform']['values']=[3.,4.]
    return fast,retained


class HybridTests(unittest.TestCase):
    def test_guarded_replacement_uses_its_own_measured_wave(self):
        fast,retained=events()
        chosen=choose_aligned(fast,retained)
        self.assertTrue(chosen['hybrid_replaced'])
        self.assertEqual(chosen['hr_bpm'],107.)
        self.assertEqual(chosen['waveform'],retained['waveform'])
        chosen['waveform']['values'][0]=99
        self.assertEqual(retained['waveform']['values'][0],3.)

    def test_each_guard_failure_keeps_exact_r1_hr_and_wave(self):
        mutations=[dict(accepted=False),dict(component_route_eligible=False),
                   dict(retained_guard_old_bpm=175.),dict(retained_guard_old_bpm=None),
                   dict(retained_guard_candidate_bpm=100.),dict(evidence_candidates=[])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                fast,retained=events()
                retained.update(mutation)
                chosen=choose_aligned(fast,retained)
                self.assertFalse(chosen['hybrid_replaced'])
                for key in fast:
                    self.assertEqual(chosen[key],fast[key])

    def test_no_added_acceptance(self):
        fast,retained=events()
        fast.update(accepted=False,hr_bpm=None)
        chosen=choose_aligned(fast,retained)
        self.assertFalse(chosen['accepted'])
        self.assertIsNone(chosen['hr_bpm'])
        self.assertFalse(chosen['hybrid_replaced'])

    def test_wrong_window_rejected(self):
        fast,retained=events()
        retained['window_end_s']+=1
        with self.assertRaises(ValueError):
            choose_aligned(fast,retained)

    def test_absent_retained_is_exact_fallback(self):
        fast,_=events()
        chosen=choose_aligned(fast,None)
        for key in fast:
            self.assertEqual(chosen[key],fast[key])

    def test_live_prefix_and_historical_fallback_alignment(self):
        source=rows(16)
        fast=Fast(P)
        old={}
        for row in source:
            event=fast.append(row)
            if event is not None:
                old[event['window_end_s']]=event
        hybrid=StreamingBackend(P)
        output=[]
        immutable=None
        for row in source:
            event=hybrid.append(row)
            if event is not None:
                output.append(event)
                if immutable is None:
                    immutable=copy.deepcopy(event)
        self.assertEqual(output[0]['emitted_at_source_s'],12.)
        self.assertEqual(output[0]['window_end_s'],10.)
        self.assertEqual(output[0],immutable)
        self.assertEqual(len(hybrid.pending),2)
        for event in output:
            target=old[event['window_end_s']]
            self.assertEqual(event['accepted'],target['accepted'])
            self.assertEqual(event['input_delay_s'],2.)
            if not event['hybrid_replaced']:
                self.assertEqual(event['hr_bpm'],target['hr_bpm'])
                self.assertEqual(event['waveform'],target['waveform'])
            json.dumps(event,allow_nan=False)
        prefix=StreamingBackend(P)
        shorter=[e for row in source[:421] if (e:=prefix.append(row)) is not None]
        def stable(item):
            return {k:v for k,v in item.items() if k not in ('processing_ms','original_selected_processing_ms')}
        self.assertEqual([stable(e) for e in shorter],[stable(e) for e in output[:len(shorter)]])


if __name__=='__main__':
    unittest.main()

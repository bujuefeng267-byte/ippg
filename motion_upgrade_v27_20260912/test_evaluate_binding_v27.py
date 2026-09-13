"""Synthetic production evaluation bindings; never opens physiological labels."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import evaluate_v27 as evaluation


class BindingContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.home=Path(self.temp.name)
        self.base=self.home/'base';self.root=self.home/'result';self.case='data1'
        self.variant='early_median';self.front=self.root/'frontend'/self.case
        self.signals=self.root/'signals'/self.case;self.out=self.root/self.variant/self.case
        for folder in (self.base/self.case/'inference',self.front,self.signals,self.out):folder.mkdir(parents=True)
        self.overrides=patch.multiple(evaluation,B=self.base,ROOT=self.root)
        self.overrides.start()
        self.fixed=dict(source_hashes={'fixture.py':'frozen-source'},frontend_config={'fixture':True},patch_hr_config={'fixture':True})
        self.write(self.root/'protocol_before_inference.json',{'synthetic':True})
        video=self.home/'video.avi'
        self.write(self.base/'inputs_manifest.json',[dict(case=self.case,video=dict(path=str(video),bytes=1,sha256='video-hash'))])
        self.write(self.base/self.case/'inference/frame_trace.json',dict(n_frames=2,fps=30.))
        names=('patch_trace.csv','frame_trace.csv','patch_anchors.json','landmark_trace.npz')
        for name in names:(self.front/name).write_bytes(name.encode())
        metadata=dict(frames=2,fps=30.,video_path=str(video),video_sha256='video-hash',video_bytes=1,
            max_seconds=None,reference_used=False,temporal_RGB_filling=False,RGB_aggregation_across_patches=False,
            anchors_reinitialized_after_gap=False,config=self.fixed['frontend_config'],source_hashes=self.fixed['source_hashes'],
            output_hashes={name:evaluation.sha(self.front/name) for name in names})
        self.write(self.front/'metadata.json',metadata)
        binding=dict(case=self.case,frames=2,fps=30.,video=str(video),video_sha256='video-hash',
            source_hashes=self.fixed['source_hashes'],metadata_sha256=evaluation.sha(self.front/'metadata.json'),
            fixed_protocol_sha256=evaluation.sha(self.root/'protocol_before_inference.json'))
        self.write(self.front/'batch_binding.json',binding)
        self.write(self.signals/'early_signals.json',{'spatial_mode':'early'})
        for name in ('anchored_patch_trace.csv','early_patch_waveforms.csv'):(self.signals/name).write_bytes(name.encode())
        for name in ('waveform.csv','heart_rate.csv','fusion_heart_rate.csv','patch_diagnostics.csv'):(self.out/name).write_bytes(name.encode())
        sources=list(self.front.iterdir())+list(self.signals.iterdir())
        self.manifest=dict(case=self.case,variant=self.variant,status='complete',reference_used=False,frames=2,fps=30.,
            source_hashes=self.fixed['source_hashes'],patch_hr_config=self.fixed['patch_hr_config'],
            patch_waveforms_path=str(self.signals/'early_patch_waveforms.csv'),
            patch_waveforms_sha256=evaluation.sha(self.signals/'early_patch_waveforms.csv'),
            patch_signal_metadata={'spatial_mode':'early'},source_files={str(p):evaluation.sha(p) for p in sources},
            output_hashes={p.name:evaluation.sha(p) for p in self.out.iterdir()},
            primary_hr_source='patch_median',motion_on_primary=False,temporal_correction_on_primary=False)
        self.write(self.out/'manifest.json',self.manifest)

    def tearDown(self):
        self.overrides.stop();self.temp.cleanup()

    @staticmethod
    def write(path,data):path.write_text(json.dumps(data))

    def verify(self):return evaluation.validate_run_binding(self.case,self.variant,self.fixed)

    def test_complete_same_video_binding_is_accepted(self):
        self.assertEqual(self.verify()['primary_hr_source'],'patch_median')

    def test_incomplete_reference_used_or_wrong_source_is_rejected(self):
        for key,value in [('status','running'),('reference_used',True),('primary_hr_source','fused_wave_dp'),('frames',3)]:
            bad={**self.manifest,key:value};self.write(self.out/'manifest.json',bad)
            with self.subTest(key=key),self.assertRaises(AssertionError):self.verify()
        self.write(self.out/'manifest.json',self.manifest)

    def test_output_and_source_cache_tampering_are_rejected(self):
        for path in (self.out/'heart_rate.csv',self.front/'landmark_trace.npz',self.signals/'early_patch_waveforms.csv'):
            original=path.read_bytes();path.write_bytes(original+b'changed')
            with self.subTest(path=path.name),self.assertRaises(AssertionError):self.verify()
            path.write_bytes(original)

    def test_missing_provenance_hash_cannot_silently_pass(self):
        bad=copy.deepcopy(self.manifest);del bad['source_files'][str(self.front/'batch_binding.json')]
        self.write(self.out/'manifest.json',bad)
        with self.assertRaises(AssertionError):self.verify()

    def test_fusion_readout_requires_full_finite_source_window(self):
        finite=np.ones(20,bool);starts=np.array([0,10]);accepted=np.array([True,False])
        evaluation.assert_fused_support(finite,accepted,starts,10,'diagnostic')
        finite[3]=False
        with self.assertRaises(AssertionError):evaluation.assert_fused_support(finite,accepted,starts,10,'diagnostic')
        evaluation.assert_fused_support(finite,np.array([False,True]),starts,10,'diagnostic')


if __name__=='__main__':unittest.main()

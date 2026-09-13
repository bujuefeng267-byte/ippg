"""Post-development ablation: preserve V24 waveform, replace only HR evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json, subprocess, sys
from run_stages import HERE, BASE, ROOT, sha, emit, run
from analyze_motion_v2 import source_hashes

stage=dict(name='stage4_preserve_waveform',motion_evidence='legacy',hr_mode='evidence',routing_mode='legacy')
frozen=source_hashes()
original=json.loads((ROOT/'protocol_before_validation.json').read_text())
assert original['source_hashes']==frozen
data=json.loads((BASE/'inputs_manifest.json').read_text())
for d in data:
    c=d['case']
    assert sha(BASE/c/'evaluation/paired_windows.csv')==original['baseline_reference_hashes'][c]
    assert sha(BASE/c/'inference/frame_trace.csv')==original['caches'][c]['csv']
    assert sha(BASE/c/'inference/frame_trace.json')==original['caches'][c]['metadata']
assert sha(HERE/'evaluate_stages.py')==original['evaluator_sha256']
assert sha(HERE/'run_stages.py')==original['runner_sha256']
folder=ROOT/stage['name']
if folder.exists():
    assert not any(folder.iterdir()), 'Do not overwrite an existing experimental run'
else:folder.mkdir()
record=dict(created_utc=datetime.now(timezone.utc).isoformat(),stage=stage,
    source_hashes=frozen,runner_sha256=sha(Path(__file__)),
    original_protocol_sha256=sha(ROOT/'protocol_before_validation.json'),
    evaluator_sha256=sha(HERE/'evaluate_stages.py'),
    selection_context='Post-development ablation after inspecting stages 1-3; not preregistered or held-out validation.',
    reason='Stage 1 lost data6 waveform coverage; stage 2 improved HR. Isolate evidence HR while preserving original real waveform and gates. Motion evidence remains active inside evidence HR.',
    parameters_changed=False,original_adoption_gate_unchanged=True,
    observed_stage_reports={s:sha(ROOT/s/'stage_evaluation.json') for s in ('v24_replay','stage1_motion','stage2_hr','stage3_routing')})
(folder/'protocol_before_run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
emit(stage=stage['name'],event='ablation_start')
with ThreadPoolExecutor(max_workers=2) as pool:
    results=[f.result() for f in as_completed([pool.submit(run,stage,d) for d in data])]
(folder/'runs.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
assert source_hashes()==frozen
assert not any(r['returncode'] for r in results)
subprocess.run([sys.executable,'-B',str(HERE/'evaluate_stages.py'),'--stage',stage['name']],check=True)
import numpy as np
import pandas as pd
for d in data:
    c=d['case'];new=pd.read_csv(folder/c/'fusion_waveform.csv');old=pd.read_csv(BASE/c/'inference/fusion_waveform.csv')
    np.testing.assert_allclose(new.base,old.base,atol=1e-8,rtol=0,equal_nan=True)
    nh=pd.read_csv(folder/c/'fusion_heart_rate.csv');oh=pd.read_csv(BASE/c/'inference/fusion_heart_rate.csv')
    np.testing.assert_array_equal(nh.accepted,oh.accepted)
emit(stage=stage['name'],waveform_and_accepted_masks_preserved=True,source_unchanged=source_hashes()==frozen)

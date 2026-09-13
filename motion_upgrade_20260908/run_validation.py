"""Run two frontends on a reference-labelled and an unlabelled video."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
project = Path('/home/fengbujue/项目/rppg识别')


def run(case):
    name, frontend = case
    video = project / ('videos/ubfc_subject1/vid.avi' if name == 'ubfc' else
                       'videos/user_0907/Video_20260907_171742478.avi')
    output = here / 'validation_final' / (name + '_' + frontend)
    cmd = [sys.executable, str(here / 'analyze_rppg_motion.py'), str(video),
           '--frontend', frontend, '--output', str(output), '--nlms']
    if name == 'ubfc':
        cmd += ['--reference-ubfc', str(project / 'videos/ubfc_subject1/ground_truth.txt')]
    log = here / 'validation_final' / (name + '_' + frontend + '.log')
    with log.open('w') as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(name, frontend, 'exit', code, flush=True)
    return code


if __name__ == '__main__':
    (here / 'validation_final').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(run, [('ubfc','baseline'), ('ubfc','robust'),
                                   ('user0907','baseline'), ('user0907','robust')]))
    sys.exit(int(any(codes)))

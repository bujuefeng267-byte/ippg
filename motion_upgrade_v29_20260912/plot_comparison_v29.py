"""Plot saved V28/V29 signals without smoothing or filling missing outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path('/home/fengbujue/项目/rppg识别')
R28 = P/'results/data1_6_v28_20260912/direct_guard'
R29 = P/'results/data1_6_v29_20260912'
OUT = R29/'figures'
SERIES = [('V28: 10 s', R28, '#777777', '--'),
          ('V29: 6 s, original gates', R29/'short6', '#cb8a21', '-.'),
          ('V29: 6 s, relaxed ambiguity', R29/'short6_relaxed', '#285aa7', '-')]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frames(case):
    return [(name, pd.read_csv(root/case/'waveform.csv'),
             pd.read_csv(root/case/'heart_rate.csv'), color, style)
            for name, root, color, style in SERIES]

def reference(case):
    return pd.read_csv(R29/'reference'/f'{case}.csv')

def heart_rates(ax, case, legend=False):
    ref = reference(case)
    ax.plot(ref.time_s, ref.reference_bpm, color='#222222', lw=1.8, label='Polar reference (~1 Hz)')
    ax.fill_between(ref.time_s, ref.reference_bpm-5, ref.reference_bpm+5,
                    color='#555555', alpha=.10, label='Reference +/-5 bpm')
    data = frames(case)
    for name, wave, hr, color, style in data:
        # Keep every planned timestamp so rejected windows break the line.
        ax.plot(hr.time_s, hr.ridge_bpm.where(hr.accepted), color=color,
                ls=style, lw=1.4, label=name)
    ax.set(xlim=(0, max(w.time_s.iloc[-1] for _, w, _, _, _ in data)),
           ylim=(39, 213), ylabel='HR (bpm)')
    ax.grid(alpha=.18)
    if legend:
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, fontsize=8)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    case = 'data6'
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios': [1, 1, 1, 1.65]})
    for ax, (name, wave, hr, color, style) in zip(axes[:3], frames(case)):
        ax.plot(wave.time_s, wave.base, color=color, lw=.8)
        coverage = np.isfinite(wave.base).mean()*100
        ax.set(title=f'{name} | waveform coverage {coverage:.2f}%', ylabel='Signal (a.u.)')
        ax.grid(alpha=.18)
    heart_rates(axes[3], case, legend=True)
    axes[3].set_xlabel('Video elapsed time (s); HR points mark window centers')
    fig.suptitle('data6: saved PPG signals and heart rate\n'
                 'Offline outputs; blanks remain missing. Waveform amplitude scales are independent; morphology is unvalidated.',
                 fontsize=12)
    fig.savefig(OUT/'data6_ppg_hr_short_window_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.subplots_adjust(top=.84, bottom=.065, left=.06, right=.985, hspace=.40, wspace=.13)
    for case, ax in zip([f'data{i}' for i in range(1, 7)], axes.flat):
        heart_rates(ax, case)
        ax.set_title(case)
        ax.set_xlabel('Video elapsed time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3, fontsize=9)
    fig.suptitle('Six-video HR outputs: 10 s versus 6 s windows\n'
                 'Different native windows; use the common 10 s table for like-for-like accuracy comparison. Missing values stay blank.',
                 fontsize=12, y=.995)
    fig.savefig(OUT/'six_video_hr_short_window_comparison.png', dpi=150)
    plt.close(fig)
    source_paths = [R29/'reference'/f'data{i}.csv' for i in range(1, 7)]
    source_paths += [root/f'data{i}'/filename for _, root, _, _ in SERIES
                    for i in range(1, 7) for filename in ('waveform.csv', 'heart_rate.csv')]
    (OUT/'manifest.json').write_text(json.dumps(dict(
        source_hashes={str(path):sha(path) for path in source_paths},
        outputs={path.name:sha(path) for path in OUT.glob('*.png')},
        plot_policy='Saved samples only; NaN preserved; no extra waveform filtering or HR interpolation.'), indent=2))
    print(str(OUT))

if __name__ == '__main__':
    main()

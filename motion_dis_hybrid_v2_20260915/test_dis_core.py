import unittest
import numpy as np
import pandas as pd
from dis_core import REGIONS, extract_all, fill_bounded, project, select_regions


def synthetic_trace(fps=30., seconds=12.):
    t = np.arange(round(fps*seconds))/fps
    pulse = np.sin(2*np.pi*1.2*t)
    table = pd.DataFrame({'time_s': t})
    for i, region in enumerate(REGIONS):
        for channel, amplitude in zip('rgb', [.3, .8, .5]):
            table[region+'_'+channel] = 130+2*amplitude*pulse
        table[region+'_dx'] = .002*np.sin(2*np.pi*2.2*t+.1*i)
        table[region+'_dy'] = .001*np.cos(2*np.pi*1.8*t+.1*i)
        table[region+'_valid'] = True
    return table


class TestDIS(unittest.TestCase):
    def test_projection_no_motion_reduces_to_pbv(self):
        rng = np.random.default_rng(81)
        c = rng.normal(size=(6, 300, 3))
        c -= c.mean(axis=1, keepdims=True)
        m = np.zeros((6, 300, 2))
        np.testing.assert_allclose(project(c, m, 'dis'), project(c, m, 'pbv'), atol=1e-10)

    def test_joint_projection_removes_injected_motion(self):
        t = np.arange(300)/30
        pulse = np.sin(2*np.pi*1.2*t)
        a, b = np.sin(2*np.pi*2.1*t), np.cos(2*np.pi*1.7*t)
        c = np.stack([.2*pulse+3*a, .6*pulse+2*b, .1*pulse+a-b], axis=1)[None]
        m = np.stack([a, b], axis=1)[None]
        dis, pbv = project(c, m, 'dis')[0], project(c, m, 'pbv')[0]
        self.assertGreater(abs(np.corrcoef(dis, pulse)[0, 1]), .99)
        self.assertGreater(abs(np.corrcoef(dis, pulse)[0, 1]), abs(np.corrcoef(pbv, pulse)[0, 1]))

    def test_actual_fps_keeps_frequency_in_seconds(self):
        for fps in (30., 60., 180.):
            waves, _ = extract_all(synthetic_trace(fps), fps)
            for method, wave in waves.items():
                self.assertTrue(wave.covered.iloc[round(fps):round(11*fps)].all())
                x = wave.base.iloc[round(fps):round(11*fps)].to_numpy()
                frequencies = np.fft.rfftfreq(len(x), 1/fps)
                peak = frequencies[np.argmax(abs(np.fft.rfft(x-x.mean()))) ]*60
                self.assertAlmostEqual(peak, 72., delta=1.)

    def test_long_gap_and_initial_missing_stay_missing(self):
        trace = synthetic_trace()
        for region in REGIONS:
            cols = [region+'_'+c for c in ('r', 'g', 'b', 'dx', 'dy')]
            trace.loc[:5, cols] = np.nan
            trace.loc[:5, region+'_valid'] = False
            trace.loc[170:185, cols] = np.nan
            trace.loc[170:185, region+'_valid'] = False
        products, _ = extract_all(trace, 30.)
        for wave in products.values():
            self.assertTrue(wave.base.iloc[:6].isna().all())
            self.assertTrue(wave.base.iloc[170:186].isna().all())
            np.testing.assert_array_equal(wave.covered, np.isfinite(wave.base))

    def test_only_bounded_short_input_gaps_interpolate(self):
        a = np.arange(20.)[:, None]*np.ones((1, 5))
        a[[0, 5, 6, 10, 11, 12, 13, 19]] = np.nan
        out, filled = fill_bounded(a, 3)
        np.testing.assert_array_equal(np.flatnonzero(filled), [5, 6])
        self.assertTrue(np.isnan(out[[0, 10, 11, 12, 13, 19]]).all())

    def test_fewer_than_five_regions_rejected(self):
        trace = synthetic_trace()
        for region in REGIONS[:2]:
            trace[[region+'_'+c for c in ('r', 'g', 'b', 'dx', 'dy')]] = np.nan
            trace[region+'_valid'] = False
        products, _ = extract_all(trace, 30.)
        self.assertTrue(all(wave.base.isna().all() for wave in products.values()))

    def test_missing_flag_cannot_claim_finite_data(self):
        trace = synthetic_trace()
        trace.loc[10, REGIONS[0]+'_valid'] = False
        with self.assertRaises(ValueError):
            extract_all(trace, 30.)

    def test_top_five_uses_distinct_regions(self):
        t = np.arange(150)/30.
        p = np.stack([np.sin(2*np.pi*1.2*t) for _ in range(6)])
        selected, h = select_regions(p)
        self.assertEqual(len(set(selected.tolist())), 5)
        self.assertTrue(np.isfinite(h).all())


if __name__ == '__main__':
    unittest.main()

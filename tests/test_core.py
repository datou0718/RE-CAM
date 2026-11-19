import itertools
import tempfile
from pathlib import Path
import unittest

import numpy as np

from recam import Noise, Waveforms, load_waveforms, target_values
from recam.adc import train_adc
from recam.evaluation import category_probabilities, noise_bank, sample_predictions
from recam.experiments import Settings, fit_method
from recam.noise import fit_readout
from recam.nonuniform_adc import train_nonuniform_adc
from recam.optimization import train
from recam.ordered import FirstZeroReadout, clean_initialization, first_zero_categories
from recam.staged import train_staged
from recam.timing import available_ticks, tick_schedules


def capture(states=5):
    t = np.linspace(0, 10e-9, 2001)
    return Waveforms(np.arange(float(states)),
                     [(t, np.exp(-t/(.2e-9+i*.2e-9))) for i in range(states)])


class WaveformTests(unittest.TestCase):
    def test_bundled_similarity_order(self):
        w = load_waveforms()
        np.testing.assert_array_equal(w.x, np.arange(33))
        voltage = w.voltages([.25e-9])[0]
        self.assertTrue(np.all(np.diff(voltage) > 0))
        self.assertGreater(voltage[-1], 1.1)
        with self.assertRaises(ValueError):
            w.voltages([w.window[1]+1])

    def test_csv_padding_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'waveform.csv'
            path.write_text('t,v,t,v\n0,1,0,1\n1,0,1,.5\n,,2,.25\n')
            w = load_waveforms(path)
            np.testing.assert_array_equal(w.voltages([.5]), [[.5, .75]])
            path.write_text('t,v,t,v\n0,1,0,1\n,,1,.5\n1,0,2,.25\n')
            with self.assertRaises(ValueError):
                load_waveforms(path)

    def test_paper_functions(self):
        x = np.array([0., 16., 32.])
        np.testing.assert_array_equal(target_values('linear', x), x)
        np.testing.assert_array_equal(target_values('piecewise-linear', x), [-8., 0., 16.])
        np.testing.assert_allclose(target_values('sigmoid', x), [1/(1+np.exp(8)), .5, 1/(1+np.exp(-8))])
        np.testing.assert_allclose(target_values('exponential', x), np.exp([0, 2, 4]))

    def test_ticks_and_latency(self):
        w = capture()
        np.testing.assert_array_equal(available_ticks(w, 4e9, 70e-12, 500e6), np.arange(1, 8))
        np.testing.assert_array_equal(available_ticks(w, 4e9, 70e-12, 1000e6), np.arange(1, 4))
        np.testing.assert_array_equal(available_ticks(w, 4e9, 70e-12, 2000e6), [1])
        self.assertEqual(len(available_ticks(w, 8e9, 70e-12, 500e6)), 15)
        self.assertEqual(len(tick_schedules(w, 4e9, 70e-12, 500e6, max_samples=7)), 127)
        self.assertEqual(len(tick_schedules(w, 8e9, 70e-12, 500e6, max_samples=7)), 16383)
        for ticks in tick_schedules(w, 4e9, .6e-9, 500e6):
            self.assertTrue(np.all(np.diff(ticks)/4e9 >= .6e-9))
        with self.assertRaises(ValueError):
            available_ticks(w, 4e9, 70e-12, 0)


class ProbabilityTests(unittest.TestCase):
    def test_first_zero_ignores_later_bits(self):
        bits = [[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
        np.testing.assert_array_equal(first_zero_categories(bits), [2, 1, 0, 4])

    def test_lognormal_and_residual_nodes(self):
        noise = Noise(integration_samples=16384)
        a, offset = noise.nodes()
        self.assertTrue(np.all(a > 0))
        self.assertAlmostEqual(a.mean(), 1., delta=1e-4)
        self.assertAlmostEqual(a.std(), .1, delta=1e-4)
        self.assertAlmostEqual(offset.std(), .001, delta=2e-6)
        with self.assertRaises(ValueError):
            Noise(integration_samples=100)

    def test_shared_variation_is_integrated_after_category_probability(self):
        w = capture()
        readout = FirstZeroReadout(w, [.3e-9, .6e-9], w.x, Noise(.002, .05, .15, 512))
        refs, _ = readout.mapping.references(readout.mapping.project([1.2, 1.8]))
        _, p = readout.conditional(refs)
        expected = np.stack([1-p[..., 0], p[..., 0]*(1-p[..., 1]), p[..., 0]*p[..., 1]], axis=-1).mean(axis=0)
        np.testing.assert_allclose(readout.distribution(refs), expected)
        self.assertGreater(np.max(np.abs(expected[:, -1]-p.mean(axis=0).prod(axis=1))), 1e-4)

    def test_expected_mse_differs_from_mean_prediction_error(self):
        fit = fit_readout([[.5, .5]], [1.], levels=[0., 2.])
        self.assertEqual(fit['mse'], 1.)
        self.assertEqual(fit['mean_prediction_mse'], 0.)

    def test_closed_form_decoding_and_fp16(self):
        p, y = np.array([[.8, .2, 0.], [.2, .8, 0.]]), np.array([0., 10.])
        fit = fit_readout(p, y)
        np.testing.assert_allclose(fit['levels'], [2., 8., 5.])
        self.assertAlmostEqual(fit['mse'], 16.)
        weighted = fit_readout(p, y, state_weights=[.9, .1], decoder_dtype='float16')
        expected = np.array([.2/.74, .8/.26, 1.], dtype=np.float16).astype(float)
        np.testing.assert_array_equal(weighted['levels'], expected)
        for k in range(2):
            for direction in (-np.inf, np.inf):
                perturbed = weighted['levels'].copy()
                perturbed[k] = np.nextafter(np.float16(perturbed[k]), np.float16(direction))
                value = fit_readout(p, y, levels=perturbed, state_weights=[.9, .1])['mse']
                self.assertGreaterEqual(value+1e-12, weighted['mse'])

    def test_analytic_reference_and_boundary_gradients(self):
        w = capture()
        for q in (None, [1, 2, 7, 3, 1]):
            readout = FirstZeroReadout(w, [.3e-9, .6e-9, 1e-9], [0., 1., 3., 7., 10.],
                                      Noise(.02, .01, .1, 128), state_weights=q)
            c = readout.mapping.project([.73, 2.21, 3.37])
            refs, _ = readout.mapping.references(c)
            for fun, center, step in [(readout.objective, refs, 1e-7), (readout.boundary_objective, c, 1e-6)]:
                numeric = [(fun(center+d)[0]-fun(center-d)[0])/(2*step) for d in np.eye(3)*step]
                np.testing.assert_allclose(fun(center)[1], numeric, rtol=2e-5, atol=3e-7)

    def test_clean_initialization_matches_exhaustive_partitions(self):
        t = np.array([0., 10e-9])
        w = Waveforms(np.arange(5.), [(t, np.array([v, v])) for v in [.05, .2, .4, .7, 1.]])
        y = np.array([0., 0., 1., 4., 12.])
        readout = FirstZeroReadout(w, [.3e-9, .6e-9, 1e-9], y, Noise(.01, 0., 0.))
        _, best = clean_initialization(readout.mapping, y)
        losses = []
        for cuts in itertools.combinations_with_replacement(range(4), 3):
            categories = first_zero_categories(readout.clean_bits(np.array(cuts)+.5))
            prediction = [y[categories == k].mean() for k in categories]
            losses.append(np.mean((prediction-y)**2))
        self.assertAlmostEqual(best, min(losses), places=13)


class TrainingTests(unittest.TestCase):
    def test_fp16_training_and_probability_replay(self):
        w = capture()
        model, _, schedules = train(w, w.x, 2e9, search_hz=1e9,
            noise=Noise(.02, .01, .1, 32), restarts=2, max_iterations=60, polish_samples=128)
        self.assertEqual(len(schedules), 1)
        p = category_probabilities(w, model)
        levels = np.asarray(model['decoding_parameters'])
        np.testing.assert_array_equal(levels, levels.astype(np.float16).astype(float))
        self.assertEqual(len(levels), model['samples']+1)
        np.testing.assert_allclose(((levels-w.x[:, None])**2*p).sum(axis=1).mean(), model['mse'])
        self.assertTrue(np.all(np.diff(model['clean_bits'], axis=1) <= 0))

    def test_u_adc_clean_means_and_empty_bins_ignore_noise(self):
        t = np.array([0., 10e-9])
        w = Waveforms(np.arange(5.), [(t, np.array([v, v])) for v in [.2, .25, .5, .85, 1.]])
        y = np.array([0., 2., 5., 10., 20.])
        for noise in (Noise(.001, 0., 0.), Noise(.03, .01, .1)):
            m, _ = train_adc(w, y, 2e9, noise, search_hz=1e9)
            # Occupied categories 1, 3, 5, 6 have means 1, 5, 10, 20.
            np.testing.assert_array_equal(m['decoding_parameters'], [1, 1, 3, 5, 7.5, 10, 20, 20])
        with self.assertRaises(ValueError):
            train_adc(w, y, 2e9, Noise(), state_weights=np.ones(5))

    def test_nu_adc_nominal_output_quantization(self):
        w = capture(8)
        y = np.arange(8.)**2
        m, _, _ = train_nonuniform_adc(w, y, 2e9, Noise(), search_hz=1e9)
        np.testing.assert_array_equal(m['decoding_parameters'], np.linspace(0, 49, 8))
        voltage = w.voltages([.5e-9])[0]
        cuts = np.searchsorted(y, np.arange(3.5, 49, 7), side='right')
        np.testing.assert_allclose(m['references_v'], (voltage[cuts-1]+voltage[cuts])/2)

    def test_fixed_baselines_never_use_distribution(self):
        w = capture(8)
        config = Settings(search_mhz=2000., nodes=32, refine_nodes=64, restarts=2, iterations=40)
        for method in ('u-adc', 'nu-adc'):
            a, _ = fit_method(w, w.x, method, config)
            b, _ = fit_method(w, w.x, method, config, [0, 1, 2, 8, 3, 2, 1, 0])
            self.assertEqual(a, b)
            self.assertFalse(b['training_data_used'])
        m, _ = fit_method(w, w.x, 'recam-u', config, [0, 1, 2, 8, 3, 2, 1, 0])
        self.assertTrue(m['training_data_used'])

    def test_staged_search_retains_coarser_solution(self):
        w, noise = capture(), Noise(.02, .01, .1, 32)
        options = dict(noise=noise, restarts=2, max_iterations=35, polish_samples=64, max_samples=2)
        seed, _, _ = train(w, w.x, 2e9, **options)
        m, _, rows = train_staged(w, w.x, 4e9, seed, per_count=1, global_count=1,
                                 neighbor_count=1, neighbor_rounds=1, **options)
        self.assertEqual(len(rows), 28)
        self.assertLessEqual(m['mse'], seed['mse']+1e-10)
        self.assertFalse(m['global_optimum_certified'])

    def test_strict_comparator_and_first_zero_sampling(self):
        t = np.array([0., 10e-9])
        w = Waveforms(np.arange(2.), [(t, np.array([v, v])) for v in [.2, .5]])
        model = {'times_s': [.25e-9, .5e-9], 'references_v': [.2, .4], 'signs': [1., 1.],
                 'decoding_parameters': [0., 1., 2.], 'decoder_dtype': 'float16'}
        bank = {'noise_v': np.zeros((2, 4, 15)), 'residual_offset_v': np.zeros((2, 4)),
                'time_scale': np.ones((2, 4))}
        categories, values = sample_predictions(w, model, bank)
        np.testing.assert_array_equal(categories, [[0]*4, [2]*4])
        self.assertEqual(values.dtype, np.dtype('float16'))


if __name__ == '__main__':
    unittest.main()

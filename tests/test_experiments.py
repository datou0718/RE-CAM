import csv
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from recam.datasets import generate_split_pairs
from recam.__main__ import main
from recam.evaluation import evaluate, noise_bank
from recam.experiments import Settings, fit_method, run
from recam.targets import target_values
from recam.waveforms import load_waveforms


class ExperimentTests(unittest.TestCase):
    def test_cli_help_describes_constraints_and_nonidealities(self):
        for command in ('train', 'benchmark'):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                main([command, '--help'])
            self.assertEqual(stopped.exception.code, 0)
            self.assertIn('System Constraints:', output.getvalue())
            self.assertIn('Hardware Nonidealities:', output.getvalue())
            self.assertIn('0.1 means 10%', output.getvalue())

    def test_cli_selects_dataset_and_experiment(self):
        with patch('recam.__main__.run') as run_experiment:
            main(['benchmark', '--dataset', 'distribution', '--experiment', 'comparison-budget',
                  '--output', 'results/check'])
            self.assertEqual(run_experiment.call_args.kwargs['dataset'], 'distribution')
            self.assertEqual(run_experiment.call_args.kwargs['experiment'], 'comparison-budget')
            main(['train', '--method', 'recam', '--function', 'linear', '--output', 'results/check',
                  '--clock-ghz', '8', '--search-mhz', '1000',
                  '--latency-ps', '80', '--budget', '15', '--discharge-std', '.15',
                  '--offset-mv', '2', '--noise-mv', '3'])
            self.assertEqual(run_experiment.call_args.kwargs['dataset'], 'monte-carlo')
            self.assertEqual(run_experiment.call_args.kwargs['experiment'], 'readout')
            self.assertEqual(run_experiment.call_args.kwargs['methods'], ['recam'])
            settings = run_experiment.call_args.args[1]
            self.assertEqual((settings.clock_ghz, settings.search_mhz,
                              settings.latency_ps, settings.budget), (8, 1000, 80, 15))
            self.assertEqual((settings.noise().timing_cv, settings.noise().residual_std_v,
                              settings.noise().sensing_std_v), (.15, .002, .003))

    def test_disjoint_queries_shared_candidates_and_counts(self):
        training, validation = generate_split_pairs(50, 50, 4, 12, 10, 11)
        self.assertTrue({r.tobytes() for r in training['queries']}.isdisjoint(
                        {r.tobytes() for r in validation['queries']}))
        np.testing.assert_array_equal(training['candidates'], validation['candidates'])
        for data in (training, validation):
            actual = 12-np.bitwise_xor(data['queries'][:, None], data['candidates'][None, :]).sum(axis=2)
            np.testing.assert_array_equal(data['similarities'], actual)
            np.testing.assert_array_equal(data['counts'], np.bincount(actual.astype(int).ravel(), minlength=13))

    def test_distribution_evaluation_uses_validation_frequencies(self):
        w = load_waveforms()
        config = Settings(search_mhz=2000., nodes=32, refine_nodes=64, samples=100)
        m, _ = fit_method(w, target_values('linear', w.x), 'u-adc', config)
        q = np.zeros(33); q[10:15] = [.1, .2, .4, .2, .1]
        bank = noise_bank(33, 100, config.noise(), search_hz=config.search_mhz*1e6)
        metrics, arrays = evaluate(w, m, bank, q, evaluation_nodes=128)
        costs = (np.array(m['decoding_parameters'])-np.array(m['target'])[:, None])**2
        expected = sum(q[x]*sum(arrays['probabilities'][x]*costs[x]) for x in range(33))
        self.assertAlmostEqual(metrics['mse'], expected)
        self.assertAlmostEqual(metrics['nrmse'], np.sqrt(expected)/32)

    def test_run_outputs_replay_and_replace_previous_results(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'run'
            output.mkdir()
            (output/'notes.txt').write_text('Keep these notes.')
            settings = Settings(search_mhz=2000., nodes=32, refine_nodes=64, samples=100)
            rows = run(output, settings, functions=['linear'], methods=['u-adc', 'recam-u'])
            config = json.loads((output/'settings.json').read_text())
            self.assertEqual(config['status'], 'completed')
            self.assertEqual(config['dataset'], 'monte-carlo')
            self.assertEqual(config['experiment'], 'readout')
            self.assertNotIn(str(Path(temp)), (output/'settings.json').read_text())
            for row in rows:
                folder = output/'linear'/row['method']
                model = json.loads((folder/'model.json').read_text())
                self.assertEqual(model['cam_search_frequency_hz'], 2e9)
                self.assertTrue(all(0 < t < 1/model['cam_search_frequency_hz'] for t in model['times_s']))
                with np.load(folder/'evaluation.npz') as data:
                    self.assertEqual(data['prediction'].dtype, np.dtype('float16'))
                    error = data['prediction'].astype(float)-np.asarray(model['target'])[:, None]
                    self.assertEqual(row['mse'], float(np.mean(error**2)))
                self.assertAlmostEqual(row['relative_change_percent'],
                                       100*(row['nrmse']/rows[0]['nrmse']-1))
            (output/'linear'/'recam-u'/'notes.txt').write_text('Keep these too.')
            unrelated = output/'other'
            unrelated.mkdir()
            (unrelated/'model.json').write_text('Unrelated model')
            updated = replace(settings, samples=40)
            rerun = run(output, updated, functions=['linear'], methods=['u-adc'])
            fresh = run(Path(temp)/'fresh', updated, functions=['linear'], methods=['u-adc'])
            self.assertEqual(rerun, fresh)
            self.assertEqual((output/'notes.txt').read_text(), 'Keep these notes.')
            self.assertEqual((output/'linear'/'recam-u'/'notes.txt').read_text(), 'Keep these too.')
            self.assertEqual((unrelated/'model.json').read_text(), 'Unrelated model')
            self.assertFalse((output/'linear'/'recam-u'/'model.json').exists())
            with (output/'metrics.csv').open() as stream:
                self.assertEqual([r['method'] for r in csv.DictReader(stream)], ['u-adc'])
            with np.load(output/'linear'/'u-adc'/'evaluation.npz') as data:
                self.assertEqual(data['prediction'].shape, (33, 40))

    def test_comparison_budget_summary_averages_nrmse(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'run'
            settings = Settings(search_mhz=1000., nodes=16, refine_nodes=32, evaluation_nodes=64,
                                restarts=2, iterations=15, samples=30)
            rows = run(output, settings, dataset='distribution', functions=['linear', 'exponential'],
                       experiment='comparison-budget')
            with (output/'overall.csv').open() as stream:
                overall = list(csv.DictReader(stream))
            self.assertEqual([int(r['comparison_budget']) for r in overall], [3, 7, 15])
            for r in overall:
                key = 'budget-'+r['comparison_budget']
                expected = np.mean([row['nrmse_percent'] for row in rows if row['method'] == key])
                self.assertAlmostEqual(float(r['mean_nrmse_percent']), expected)
            run(output, settings, functions=['linear'], methods=['u-adc'])
            for name in ('overall.csv', 'training_dataset.npz', 'validation_dataset.npz',
                         'linear/budget-3', 'linear/budget-7', 'linear/budget-15', 'exponential'):
                self.assertFalse((output/name).exists(), name)
            config = json.loads((output/'settings.json').read_text())
            self.assertEqual((config['dataset'], config['experiment'], config['status']),
                             ('monte-carlo', 'readout', 'completed'))

    def test_invalid_rerun_keeps_existing_results(self):
        settings = Settings(search_mhz=2000., nodes=16, refine_nodes=32, samples=20,
                            restarts=2, iterations=15)
        cases = [(replace(settings, budget=3), 'u-adc'),
                 (replace(settings, clock_ghz=1.), 'recam'),
                 (replace(settings, seed=-1), 'recam')]
        with tempfile.TemporaryDirectory() as temp:
            for index, (invalid, method) in enumerate(cases):
                with self.subTest(settings=invalid):
                    output = Path(temp)/str(index)
                    model = output/'linear'/'recam'/'model.json'
                    model.parent.mkdir(parents=True)
                    model.write_text('previous model')
                    (output/'metrics.csv').write_text('previous metrics')
                    (output/'settings.json').write_text('{"status": "completed"}')
                    before = {p.relative_to(output): p.read_bytes()
                              for p in output.rglob('*') if p.is_file()}
                    with self.assertRaises(ValueError):
                        run(output, invalid, functions=['linear'], methods=[method])
                    after = {p.relative_to(output): p.read_bytes()
                             for p in output.rglob('*') if p.is_file()}
                    self.assertEqual(before, after)

    def test_evaluation_noise_spans_cam_search_period(self):
        noise = Settings().noise()
        for frequency, comparisons in ((250e6, 31), (500e6, 15), (1000e6, 7)):
            bank = noise_bank(2, 3, noise, grid_hz=8e9, search_hz=frequency)
            self.assertEqual(bank['noise_v'].shape, (2, 3, comparisons))
        with self.assertRaises(ValueError):
            noise_bank(2, 3, noise, search_hz=0)

    def test_budget_fifteen_retains_eight_ghz_seven_comparison_fit(self):
        calls = []
        def staged(waveforms, target, clock, seed_model, **options):
            calls.append((clock, seed_model['clock_hz'], seed_model['comparison_budget'], options['max_samples']))
            return {'clock_hz': clock, 'comparison_budget': options['max_samples']}, [], []
        w = load_waveforms()
        with patch('recam.experiments.train_staged', side_effect=staged):
            model, records = fit_method(w, w.x, 'recam', Settings(clock_ghz=8., budget=15),
                seed_model={'clock_hz': 4e9, 'comparison_budget': 7})
        self.assertEqual(calls, [(8e9, 4e9, 7, 7), (8e9, 8e9, 7, 15)])
        self.assertEqual(model['comparison_budget'], 15)
        self.assertEqual(records['seven_comparison_warm_start']['model']['comparison_budget'], 7)

    def test_parallel_functions_match_individual_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = Settings(search_mhz=2000., nodes=16, refine_nodes=32, samples=40,
                                restarts=2, iterations=25)
            functions = ['sigmoid', 'exponential']
            methods = ['u-adc', 'recam-n']
            parallel = run(root/'parallel', settings, functions=functions, methods=methods)
            individual = []
            for function in functions:
                individual.extend(run(root/function, settings, functions=[function], methods=methods))
            self.assertEqual(parallel, individual)
            self.assertEqual(parallel, run(root/'parallel', settings,
                                          functions=functions*2, methods=methods*2))
            config = json.loads((root/'parallel'/'settings.json').read_text())
            self.assertEqual(config['functions'], functions)
            self.assertEqual(config['methods'], methods)


if __name__ == '__main__':
    unittest.main()

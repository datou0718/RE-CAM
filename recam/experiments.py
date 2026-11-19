"""Paper-named readouts and reproducible Monte Carlo/distribution experiments."""
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from recam.adc import train_adc
from recam.datasets import generate_split_pairs
from recam.evaluation import evaluate, noise_bank
from recam.noise import Noise
from recam.nonuniform_adc import train_nonuniform_adc
from recam.optimization import train
from recam.staged import train_staged
from recam.targets import FUNCTIONS, target_values
from recam.timing import available_ticks
from recam.waveforms import default_waveform, load_waveforms


METHODS = {
    'u-adc': 'U-ADC', 'recam-u': 'RE-CAM(U)',
    'nu-adc': 'NU-ADC', 'recam-n': 'RE-CAM(N)',
    'recam': 'RE-CAM', 'finer-sensing': 'Finer Sensing',
}
FIXED_BASELINES = {'u-adc', 'nu-adc'}


@dataclass(frozen=True)
class Settings:
    clock_ghz: float = 4.
    search_mhz: float = 500.
    latency_ps: float = 70.
    budget: int = 7
    discharge_std: float = .1
    offset_mv: float = 1.
    noise_mv: float = 1.
    nodes: int = 128
    refine_nodes: int = 2048
    evaluation_nodes: int = 16384
    restarts: int = 8
    iterations: int = 300
    samples: int = 5000
    seed: int = 0
    search: str = 'auto'

    def noise(self):
        return Noise(self.noise_mv*1e-3, self.offset_mv*1e-3, self.discharge_std, self.nodes)

    def validate(self):
        timing = (self.clock_ghz, self.search_mhz, self.latency_ps)
        if (not all(np.isfinite(v) for v in timing) or min(timing[:2]) <= 0
                or self.latency_ps < 0):
            raise ValueError('Clock/search rates must be positive and finite; latency must be nonnegative and finite.')
        if self.seed < 0:
            raise ValueError('Seed must be nonnegative.')
        noise = self.noise()
        replace(noise, integration_samples=self.refine_nodes)
        replace(noise, integration_samples=self.evaluation_nodes)
        if self.refine_nodes < self.nodes:
            raise ValueError('Refinement must use at least as many nodes as initial training.')
        if self.budget < 1 or self.restarts < 2 or self.iterations < 1 or self.samples < 2:
            raise ValueError('Use a positive budget/iteration limit, at least two starts and two samples.')
        if self.search not in ('auto', 'exhaustive'):
            raise ValueError('Use auto or exhaustive schedule search.')


def fit_method(waveforms, target, method, settings, state_weights=None, seed_model=None):
    """Baselines are always distribution-free; only RE-CAM methods use q(x)."""
    settings.validate()
    if method not in METHODS:
        raise ValueError(f'Unknown implementation: {method}')
    q = None if method in FIXED_BASELINES else state_weights
    noise = settings.noise()
    common = dict(latency_s=settings.latency_ps*1e-12, search_hz=settings.search_mhz*1e6,
                  state_weights=q, decoder_dtype='float16')
    clock = settings.clock_ghz*1e9
    if method in ('u-adc', 'recam-u', 'nu-adc', 'recam-n'):
        if settings.budget != 7:
            raise ValueError('The paper ADC baselines have exactly seven references (budget 7).')
        dense = replace(noise, integration_samples=settings.refine_nodes)
        if method in ('u-adc', 'recam-u'):
            model, candidates = train_adc(waveforms, target, clock, dense,
                decoding_mode='nominal_mean' if method == 'u-adc' else 'posterior_mean', **common)
            trials = []
        else:
            model, candidates, trials = train_nonuniform_adc(waveforms, target, clock, dense,
                reference_selection='nominal_midpoints' if method == 'nu-adc' else 'optimized',
                restarts=settings.restarts, max_iterations=settings.iterations, seed=settings.seed, **common)
        records = {'timestamp_candidates': candidates, 'optimizer_trials': trials}
    else:
        clock *= 2 if method == 'finer-sensing' else 1
        ticks = available_ticks(waveforms, clock, common['latency_s'], common['search_hz'])
        options = dict(noise=noise, polish_samples=settings.refine_nodes, max_samples=settings.budget,
                       restarts=settings.restarts, max_iterations=settings.iterations, seed=settings.seed, **common)
        warm_start = None
        if settings.search == 'auto' and len(ticks) > 7:
            if seed_model is None:
                coarser = replace(settings, clock_ghz=clock/2e9)
                seed_model, _ = fit_method(waveforms, target, 'recam', coarser, q)
            if settings.budget > 7 and seed_model['clock_hz'] < clock:
                # Table III expands the 8 GHz seven-comparison solution from
                # Table II, preserving that feasible fit before adding senses.
                seed_model, seed_trials, seed_schedules = train_staged(
                    waveforms, target, clock, seed_model, **{**options, 'max_samples': 7})
                seed_model['comparison_budget'] = 7
                warm_start = {'model': seed_model, 'optimizer_trials': seed_trials,
                              'schedule_search': seed_schedules}
            model, trials, schedules = train_staged(waveforms, target, clock, seed_model, **options)
        else:
            model, trials, schedules = train(waveforms, target, clock, **options)
        records = {'schedule_search': schedules, 'optimizer_trials': trials}
        if warm_start is not None:
            records['seven_comparison_warm_start'] = warm_start
    model.update(method=method, implementation=METHODS[method], comparison_budget=settings.budget)
    return model, records


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(output):
    """Replace generated results while preserving unrelated files."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ('settings.json', 'metrics.csv', 'overall.csv', 'README.md',
                 'training_dataset.npz', 'validation_dataset.npz'):
        (output/name).unlink(missing_ok=True)
    for function in FUNCTIONS:
        folder = output/function
        (folder/'noise.npz').unlink(missing_ok=True)
        for method in (*METHODS, 'budget-3', 'budget-7', 'budget-15'):
            destination = folder/method
            for name in ('model.json', 'training.json', 'evaluation.npz'):
                (destination/name).unlink(missing_ok=True)
            if destination.is_dir() and not any(destination.iterdir()):
                destination.rmdir()
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def run(output, settings, dataset='monte-carlo', functions=None, methods=None,
        waveform=None, experiment='readout'):
    settings.validate()
    if dataset not in ('monte-carlo', 'distribution') or experiment not in ('readout', 'comparison-budget'):
        raise ValueError('Unknown dataset or experiment.')
    functions = list(dict.fromkeys(FUNCTIONS if functions is None else functions))
    methods = list(dict.fromkeys(METHODS if methods is None else methods))
    if not functions or not methods or any(f not in FUNCTIONS for f in functions) or any(m not in METHODS for m in methods):
        raise ValueError('Select at least one known function and valid implementations.')
    waveform = Path(waveform or default_waveform())
    w = load_waveforms(waveform)
    if len(w.x) != 33:
        raise ValueError('The four paper functions and benchmark use 33 similarity states (dimensionality 32).')
    cases = ({'budget-3': ('recam', replace(settings, clock_ghz=2., budget=3)),
              'budget-7': ('recam', replace(settings, clock_ghz=4., budget=7)),
              'budget-15': ('recam', replace(settings, clock_ghz=8., budget=15))}
             if experiment == 'comparison-budget' else {m: (m, settings) for m in methods})
    # Reject invalid configurations before replacing previous results.
    for method, case_settings in cases.values():
        if method not in ('recam', 'finer-sensing') and case_settings.budget != 7:
            raise ValueError('The paper ADC baselines have exactly seven references (budget 7).')
        clock = case_settings.clock_ghz*1e9*(2 if method == 'finer-sensing' else 1)
        ticks = available_ticks(w, clock, case_settings.latency_ps*1e-12,
                                case_settings.search_mhz*1e6)
        if not len(ticks):
            raise ValueError('No clock tick fits within the waveform capture and CAM search period.')
    output = Path(output)
    prepare_output(output)
    config = {'dataset': dataset, 'experiment': experiment, 'settings': asdict(settings), 'functions': functions,
              'methods': methods, 'waveform_sha256': hashlib.sha256(waveform.read_bytes()).hexdigest(),
              'decoder_dtype': 'float16', 'target_dtype': 'float64', 'status': 'running'}
    write_json(output/'settings.json', config)
    train_q = validation_q = None
    if dataset == 'distribution':
        training, validation = generate_split_pairs(10000, 10000, 32, 32, 20261110, 20261111)
        for name, data in (('training', training), ('validation', validation)):
            np.savez_compressed(output/f'{name}_dataset.npz', **data)
        train_q = training['counts']/training['counts'].sum()
        validation_q = validation['counts']/validation['counts'].sum()
        config.update(training_queries=10000, validation_queries=10000, candidates=32,
                      dataset_seed=20261110, validation_seed=20261111, shared_candidates=True)
    config['cases'] = {key: {'method': m, 'settings': asdict(s)} for key, (m, s) in cases.items()}
    config['noise_grid_hz'] = 8e9 if experiment == 'comparison-budget' else 2*settings.clock_ghz*1e9
    write_json(output/'settings.json', config)
    arguments = (output, w, settings, cases, train_q, validation_q, experiment, config['noise_grid_hz'])
    if len(functions) == 1:
        rows = _run_function(functions[0], *arguments)
    else:
        with ProcessPoolExecutor(max_workers=min(4, len(functions))) as pool:
            jobs = [pool.submit(_run_function, function, *arguments) for function in functions]
            rows = [row for job in jobs for row in job.result()]
    write_csv(output/'metrics.csv', rows)
    if experiment == 'comparison-budget':
        overall = [{'comparison_budget': s.budget, 'clock_ghz': s.clock_ghz,
                    'mean_nrmse_percent': float(np.mean([r['nrmse_percent'] for r in rows if r['method'] == key]))}
                   for key, (_, s) in cases.items()]
        for row in overall:
            row['relative_change_percent'] = 100*(row['mean_nrmse_percent']/overall[0]['mean_nrmse_percent']-1)
        write_csv(output/'overall.csv', overall)
    config['status'] = 'completed'
    write_json(output/'settings.json', config)
    report(output, rows)
    return rows


def _run_function(function, output, w, settings, cases, train_q, validation_q, experiment, grid_hz):
    rows = []
    y = target_values(function, w.x)
    folder = output/function
    folder.mkdir(exist_ok=True)
    bank = noise_bank(len(w.x), settings.samples, settings.noise(), grid_hz=grid_hz,
                      search_hz=settings.search_mhz*1e6,
                      seed=20361110+1000*list(FUNCTIONS).index(function))
    np.savez_compressed(folder/'noise.npz', **bank)
    models = {}
    for key, (method, case_settings) in cases.items():
        print(f'Training {FUNCTIONS[function]} / {key}', flush=True)
        seed_model = models.get('recam') if method == 'finer-sensing' else models.get('budget-7') if key == 'budget-15' else None
        model, records = fit_method(w, y, method, case_settings, train_q, seed_model)
        model.update(function=function)
        models[key] = model
        destination = folder/key
        destination.mkdir(exist_ok=True)
        write_json(destination/'model.json', model)
        write_json(destination/'training.json', records)
        metrics, arrays = evaluate(w, model, bank, validation_q, settings.evaluation_nodes, grid_hz)
        np.savez_compressed(destination/'evaluation.npz', **arrays)
        label = f'Comparison budget {case_settings.budget}' if experiment == 'comparison-budget' else METHODS[method]
        rows.append({'function': function, 'method': key, 'implementation': label, **metrics,
                     'relative_change_percent': None})
        print(f'  NRMSE: {metrics["nrmse_percent"]:.2f}%', flush=True)
    baseline_key = 'budget-3' if experiment == 'comparison-budget' else 'u-adc'
    group = [row for row in rows if row['function'] == function]
    baseline = next((r['nrmse'] for r in group if r['method'] == baseline_key), None)
    for row in group:
        if baseline is not None:
            row['relative_change_percent'] = 100*(row['nrmse']/baseline-1) if baseline else 0.
    return rows


def report(output, rows):
    lines = ['# RE-CAM results', '', 'NRMSE is normalized by the full target output span. '
             'Parentheses show relative change from U-ADC (or budget 3 for the budget sweep); negative is better.', '',
             '| Function | Implementation | NRMSE (relative change) | Training MSE | Evaluation MSE |',
             '|---|---|---:|---:|---:|']
    for r in rows:
        change = '' if r['relative_change_percent'] is None else f' ({r["relative_change_percent"]:+.2f}%)'
        lines.append(f'| {FUNCTIONS[r["function"]]} | {r["implementation"]} | {r["nrmse_percent"]:.2f}%{change} '
                     f'| {r["training_mse"]:.8g} | {r["mse"]:.8g} |')
    lines += ['', 'Models, training traces, evaluation samples and settings are saved beside this report. '
              'Training MSE is computed on captured waveform states. Distribution evaluation uses disjoint '
              'validation queries with shared candidates and fresh noise integration. These are behavioral '
              'waveform-model results, not physical-device measurements.', '']
    (output/'README.md').write_text('\n'.join(lines))

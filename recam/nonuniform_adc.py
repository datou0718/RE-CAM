"""Non-uniform ADC: nominal or optimized references at one common clock tick."""
from dataclasses import replace
import numpy as np

from recam.adc import adc_distribution
from recam.noise import Readout, fit_readout, state_distribution
from recam.ordered import FirstZeroReadout
from recam.timing import available_ticks
from recam.optimization import optimize


class VoltageMap:
    """Direct voltage coordinates with ordered, bounded ADC thresholds."""
    coordinate_name = 'references_v'

    def __init__(self, count, upper_voltage):
        self.lower = np.zeros(count)
        self.upper = np.full(count, upper_voltage)

    def references(self, values):
        return np.asarray(values, dtype=float), np.ones(len(self.lower))

    def project(self, values):
        return np.clip(np.sort(values), self.lower, self.upper)

    def feasible(self, values, tolerance=1e-9):
        values = np.asarray(values)
        return bool(np.isfinite(values).all() and np.all(values >= self.lower-tolerance)
                    and np.all(values <= self.upper+tolerance) and np.all(np.diff(values) >= -tolerance))


class ADCReadout(FirstZeroReadout):
    """Reuse RE-CAM's expected-MSE probabilities and gradient at repeated times."""

    def __init__(self, waveforms, timestamp, target, noise, upper_voltage, state_weights=None,
                 decoder_dtype='float16'):
        Readout.__init__(self, waveforms, [timestamp]*7, target, noise)
        self.signs = np.ones(7)
        self.loss_mode = 'expected_mse'
        self.decoder_dtype = decoder_dtype
        self.state_weights = state_distribution(state_weights, len(target))
        self.mapping = VoltageMap(7, upper_voltage)


def optimize_references(waveforms, timestamp, target, noise, initial, upper_voltage,
                        restarts, max_iterations, seed, state_weights=None, decoder_dtype='float16'):
    """Multistart SLSQP; refine on the supplied final noise grid."""
    if restarts < 2 or max_iterations < 1:
        raise ValueError('Need at least two starts and one iteration.')
    coarse_noise = replace(noise, integration_samples=min(128, noise.integration_samples))
    readout = ADCReadout(waveforms, timestamp, target, coarse_noise, upper_voltage, state_weights, decoder_dtype)
    starts = [initial, np.linspace(0., upper_voltage, 9)[1:-1]]
    rng = np.random.default_rng(seed)
    while len(starts) < restarts:
        point = (rng.uniform(0., upper_voltage, 7) if len(starts) % 2 else
                 initial+rng.normal(0., upper_voltage/8, 7))
        starts.append(readout.mapping.project(point))
    scale = max(float(np.var(target)), np.finfo(float).tiny)
    trials = []

    def run(initial, stage, restart):
        result = optimize(readout, initial, max_iterations, loss_scale=scale)
        result['restart'] = restart
        trials.append({'stage': stage, 'restart': restart, 'iterations': result['iterations'],
            'solver_success': result['solver_success'], 'solver_message': result['solver_message'],
            'function_evaluations': result['function_evaluations'],
            'initial_loss': result['initial_loss'], 'qualified_loss': result['fitted']['loss']})
        return result

    candidates = [run(point, 'initial-integration', i) for i, point in enumerate(starts)]
    if noise.shared and noise.integration_samples > coarse_noise.integration_samples:
        seeds = sorted(candidates, key=lambda result: result['fitted']['loss'])[:3]
        readout = ADCReadout(waveforms, timestamp, target, noise, upper_voltage, state_weights, decoder_dtype)
        candidates = [run(result['references'], 'dense-integration-polish', result['restart'])
                      for result in seeds]
        # Preserve the decoder-only baseline as a feasible starting point on
        # the final grid, even if its coarse optimized trajectory was poor.
        candidates.append(run(initial, 'dense-integration-polish', -1))
    return min(candidates, key=lambda result: result['fitted']['loss']), trials


def output_quantizer(target):
    """Eight endpoint-inclusive levels; exact midpoint ties select the lower level.

    Each cut indexes the first state above a decision boundary. Multiple output
    boundaries may fall in the same gap between discrete input states; retain
    those repeated cuts to keep seven comparisons and eight decoding entries.
    """
    target = np.asarray(target, dtype=float)
    if (target.ndim != 1 or len(target) < 2 or not np.isfinite(target).all()
            or np.any(np.diff(target) < 0) or target[-1] <= target[0]):
        raise ValueError('Need finite, nondecreasing, nonconstant target values.')
    # Interpolate endpoints directly to reduce rounding drift at symmetric ties.
    fraction = np.arange(8)/7
    levels = (1-fraction)*target[0] + fraction*target[-1]
    boundaries = (levels[:-1]+levels[1:])/2
    # Negative output ranges can round an exact midpoint slightly downward.
    # Apply tie tolerance only to this nominal quantizer, never the comparator.
    tolerance = 8*np.finfo(float).eps*max(1., np.max(np.abs(levels)))
    cuts = np.searchsorted(target, boundaries+tolerance, side='right')
    nearest = np.searchsorted(boundaries+tolerance, target, side='left')
    return levels, boundaries, cuts, nearest


def train_nonuniform_adc(waveforms, target, clock_hz, noise, latency_s=70e-12,
                        search_hz=500e6, progress=None,
                        decoding_mode=None, reference_selection='nominal_midpoints',
                        restarts=8, max_iterations=300, seed=0, state_weights=None, decoder_dtype='float16'):
    """Construct references independently at every tick and select noisy MSE.

    Nominal references retain the equal-output quantizer. Optimized references
    use RE-CAM's individual noisy-output MSE with closed-form decoding, under
    ADC ordering and simultaneous-sensing constraints. Optional state weights
    apply to reference/decoder fitting and timestamp selection.
    """
    target = np.asarray(target, dtype=float)
    if target.shape != waveforms.x.shape:
        raise ValueError('Need one target per waveform.')
    weights = state_distribution(state_weights, len(target))
    if reference_selection not in ('nominal_midpoints', 'optimized'):
        raise ValueError('Use nominal_midpoints or optimized reference selection.')
    if decoding_mode is None:
        decoding_mode = 'posterior_mean' if reference_selection == 'optimized' else 'uniform_output'
    if decoding_mode not in ('uniform_output', 'posterior_mean'):
        raise ValueError('Use uniform_output or posterior_mean decoding.')
    if reference_selection == 'optimized' and decoding_mode != 'posterior_mean':
        raise ValueError('Optimized references require closed-form posterior_mean decoding.')
    levels, boundaries, cuts, nearest = output_quantizer(target)
    ticks = available_ticks(waveforms, clock_hz, latency_s, search_hz)
    if not len(ticks):
        raise ValueError('No ADC sampling tick fits the search/capture window.')
    upper_voltage = max(float(v.max()) for _, v in waveforms.curves)
    models, attempts = [], []
    for index, tick in enumerate(ticks):
        timestamp = float(tick/clock_hz)
        voltage = waveforms.voltages([timestamp])[0]
        references = (voltage[cuts-1]+voltage[cuts])/2
        if (np.any(np.diff(references) < 0) or np.any(references < 0)
                or np.any(references > upper_voltage)):
            raise ValueError('Nominal midpoint references must be nondecreasing and within the voltage range.')
        initial = references.copy()
        chosen, trials = None, []
        if reference_selection == 'optimized':
            chosen, trials = optimize_references(waveforms, timestamp, target, noise, initial,
                upper_voltage, restarts, max_iterations, seed+int(tick), weights, decoder_dtype)
            references, fitted = chosen['references'], chosen['fitted']
            distribution = fitted['distribution']
        else:
            distribution = adc_distribution(waveforms, timestamp, references, noise, allow_repeated=True)
            fitted = fit_readout(distribution, target,
                                 levels=levels if decoding_mode == 'uniform_output' else None,
                                 loss_mode='expected_mse', state_weights=weights, decoder_dtype=decoder_dtype)
        bits = voltage[:, None] > references
        categories = np.cumprod(bits, axis=1).sum(axis=1)
        model = {'algorithm': 'normal', 'architecture': 'nonuniform_adc', 'decoder': 'first_zero',
            'loss_mode': 'expected_mse', 'input_distribution': 'uniform similarity states',
            'decoding_mode': decoding_mode, 'decoder_dtype': decoder_dtype, 'target_dtype': 'float64',
            'decoding_rule': ('p[k] = min(f) + k*(max(f)-min(f))/7, k=0..7; nearest level, ties lower'
                              if decoding_mode == 'uniform_output' else 'noise-probability-weighted means'),
            'quantization_levels': levels.tolist(),
            'quantization_tie_atol': float(8*np.finfo(float).eps*max(1., np.max(np.abs(levels)))),
            'empty_category_rule': ('retain nominal output-grid value' if decoding_mode == 'uniform_output'
                                    else 'global target mean when probability mass is zero'),
            'quantization_boundaries': boundaries.tolist(), 'nearest_target_categories': nearest.tolist(),
            'reference_selection': 'nominal_midpoints', 'boundary_state_indices': cuts.tolist(),
            'reference_rule': 'midpoint of adjacent nominal ML voltages straddling each output decision boundary',
            'training_data_used': False, 'timing_selection': 'single_tick_nonuniform_adc',
            'comparison_mode': 'simultaneous thresholds on one waveform realization',
            'optimizer': 'exhaustive timestamp search; nominal midpoint references; '+decoding_mode+' decoding',
            'gradient_optimization_used': False, 'clock_hz': clock_hz,
            'clock_ticks': [int(tick)]*7, 'times_s': [timestamp]*7,
            'available_clock_ticks': ticks.tolist(), 'timestamp_s': timestamp, 'selected_tick': int(tick),
            'references_v': references.tolist(), 'reference_min_v': 0., 'reference_max_v': upper_voltage,
            'reference_bounds_rule': 'zero to maximum voltage in captured nominal waveforms',
            'reference_order': 'nondecreasing; repeated cuts retain repeated references', 'signs': [1.]*7,
            'samples': 7, 'categories': 8, 'timestamps_searched': len(ticks),
            'target': target.tolist(), 'decoding_parameters': fitted['levels'].tolist(),
            'category_probabilities': distribution.tolist(),
            'prediction': fitted['prediction'].tolist(), 'prediction_std': fitted['std'].tolist(),
            'clean_bits': bits.astype(int).tolist(),
            'nominal_mse': float(np.mean((fitted['levels'][categories]-target)**2)),
            **{key: fitted[key] for key in ('loss', 'mse', 'mae', 'rmse', 'mean_prediction_mse', 'mean_output_variance')},
            'noise': noise.to_dict(), 'sensing_latency_s': latency_s, 'cam_search_frequency_hz': search_hz,
            'allow_end_sensing': False, 'result_ready_s': [timestamp+latency_s],
            'selected_trace': [{'iteration': 0, 'loss': fitted['loss'],
                                'references_v': references.tolist(), 'decoding_parameters': fitted['levels'].tolist()}],
            'optimizer_iterations': 0, 'solver_success': True,
            'solver_message': 'deterministic nominal construction; no continuous optimization'}
        if chosen is not None:
            model.update(reference_selection='optimized', gradient_optimization_used=True,
                reference_rule='optimize ordered reference voltages under expected individual-output MSE',
                reference_order='nondecreasing; equal references allowed',
                optimizer='SLSQP reference gradients and closed-form decoding; exhaustive timestamp search',
                optimizer_iterations=chosen['iterations'], selected_trace=chosen['history'],
                solver_success=chosen['solver_success'], solver_message=chosen['solver_message'],
                selected_restart=chosen['restart'], restarts=restarts, seed=seed+int(tick),
                max_iterations=max_iterations, initial_references_v=initial.tolist(),
                initial_integration_samples=min(128, noise.integration_samples),
                optimizer_trials=trials, global_optimum_certified=False)
            for key in ('quantization_boundaries', 'boundary_state_indices', 'nearest_target_categories',
                        'quantization_tie_atol'):
                model.pop(key)
            model['initial_quantization_levels'] = model.pop('quantization_levels')
        if weights is not None:
            model.update(algorithm='dataset_enhanced', training_data_used=True,
                input_distribution='empirical calibration similarity frequencies', state_weights=weights.tolist(),
                uniform_state_mse=fit_readout(distribution, target, fitted['levels'], 'expected_mse')['mse'])
        models.append(model)
        attempts.append({'tick': int(tick), 'timestamp_s': timestamp, 'stage': reference_selection,
                         'nominal_mse': model['nominal_mse'], 'expected_mse': model['mse']})
        if progress:
            progress(index+1, len(ticks), min(m['loss'] for m in models))
    return min(models, key=lambda m: m['loss']), models, attempts

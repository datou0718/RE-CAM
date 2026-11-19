"""Discrete schedules, ordered references and optionally weighted first-zero loss."""
from dataclasses import replace
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from recam.noise import Noise, fit_readout, state_distribution
from recam.ordered import FirstZeroReadout, clean_initialization, first_zero_categories
from recam.timing import available_ticks, tick_schedules


def optimize(readout, initial, max_iterations, loss_scale=1.):
    """Full-batch SLSQP with closed-form decoding; retain the best feasible iterate."""
    mapping = readout.mapping
    initial = mapping.project(initial)
    history, best = [], None
    iteration = 0

    def record(point):
        nonlocal best
        if not mapping.feasible(point, tolerance=1e-7):
            return
        point = mapping.project(point)
        refs, _ = mapping.references(point)
        fitted = readout.fit(refs)
        if best is None or fitted['loss'] < best['fitted']['loss']-1e-15:
            best = {'boundaries': point.copy(), 'references': refs.copy(), 'fitted': fitted}
            history.append({'iteration': iteration,
                            getattr(mapping, 'coordinate_name', 'boundaries_similarity'): point.tolist(),
                            'references_v': refs.tolist(), 'loss': fitted['loss'],
                            'decoding_parameters': fitted['levels'].tolist()})

    def callback(point):
        nonlocal iteration
        iteration += 1
        record(point)

    record(initial)
    senses = len(initial)
    difference = np.diff(np.eye(senses), axis=0)
    constraint = (LinearConstraint(difference, 0., np.inf),) if senses > 1 else ()
    if not np.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError('Loss scale must be positive and finite.')
    def objective(point):
        loss, gradient = readout.boundary_objective(point)
        return loss/loss_scale, gradient/loss_scale
    fit = minimize(objective, initial, jac=True, method='SLSQP',
                   bounds=list(zip(mapping.lower, mapping.upper)), constraints=constraint,
                   callback=callback, options={'maxiter': max_iterations, 'ftol': 1e-12})
    record(fit.x)
    best.update(history=history, solver_success=bool(fit.success), solver_message=str(fit.message),
                iterations=int(fit.nit), function_evaluations=int(fit.nfev),
                initial_loss=history[0]['loss'], final_loss=float(fit.fun*loss_scale))
    return best


def _train_schedule(waveforms, target, clock_hz, ticks, sigma_v=.001, latency_s=70e-12,
          search_hz=500e6, seed=0, restarts=8, max_iterations=300,
          progress=None, noise=None, polish_samples=2048,
          loss_mode='expected_mse', state_weights=None, initial_boundaries=None, decoder_dtype='float16'):
    target = np.asarray(target, dtype=float)
    if target.shape != waveforms.x.shape or not np.isfinite(target).all():
        raise ValueError('Need one finite target per waveform.')
    if restarts < 2 or max_iterations < 1:
        raise ValueError('Need at least two starts and one iteration.')
    times = ticks/clock_hz
    if not len(ticks) or np.any(np.diff(times) < latency_s-1e-23):
        raise ValueError('Selected clock ticks must satisfy sensing latency.')
    noise = noise or Noise(sensing_std_v=sigma_v)
    weights = state_distribution(state_weights, len(target))
    readout = FirstZeroReadout(waveforms, times, target, noise, loss_mode, weights, decoder_dtype)
    mapping = readout.mapping
    clean, clean_loss = clean_initialization(mapping, target, loss_mode, weights, decoder_dtype)
    clean_distribution = np.eye(len(ticks)+1)[first_zero_categories(readout.clean_bits(clean))]
    clean_metrics = fit_readout(clean_distribution, target, loss_mode=loss_mode, state_weights=weights,
                                decoder_dtype=decoder_dtype)
    count = len(ticks)
    rng = np.random.default_rng(seed)
    even = np.linspace(waveforms.x[0], waveforms.x[-1], count+2)[1:-1]
    starts = [clean, mapping.project(even)]
    if initial_boundaries is not None:
        starts.append(mapping.project(initial_boundaries))
        restarts = max(restarts, len(starts))
    while len(starts) < restarts:
        if len(starts) % 2:
            point = rng.uniform(waveforms.x[0], waveforms.x[-1], count)
        else:
            point = clean+rng.normal(0, max(1., np.ptp(waveforms.x)/(count+1)), count)
        starts.append(mapping.project(point))
    trials, candidates = [], []

    def save_trial(result, restart, stage):
        trials.append({'clock_ticks': ticks.tolist(), 'stage': stage, 'restart': restart,
                       'solver_success': result['solver_success'], 'solver_message': result['solver_message'],
                       'iterations': result['iterations'], 'function_evaluations': result['function_evaluations'],
                       'initial_loss': result['initial_loss'], 'final_loss': result['final_loss'],
                       'qualified_loss': result['fitted']['loss']})

    dense_count = min(3, restarts) if noise.shared else 0
    total = restarts+dense_count
    for restart, initial in enumerate(starts):
        result = optimize(readout, initial, max_iterations)
        result['restart'] = restart
        candidates.append(result)
        save_trial(result, restart, 'initial-integration')
        if progress:
            progress(len(trials), total, min(c['fitted']['loss'] for c in candidates))
    candidates.sort(key=lambda c: c['fitted']['loss'])
    coarse_best = candidates[0]['fitted']['loss']
    if noise.shared:
        noise = replace(noise, integration_samples=max(polish_samples, noise.integration_samples))
        readout = FirstZeroReadout(waveforms, times, target, noise, loss_mode, weights, decoder_dtype)
        refined = []
        dense_seeds = candidates[:dense_count]
        if initial_boundaries is not None:
            # Always retain/refine the existing schedule under the final grid.
            refs, _ = mapping.references(initial_boundaries)
            dense_seeds = dense_seeds + [{'boundaries': initial_boundaries, 'restart': -1,
                                         'fitted': readout.fit(refs)}]
        for candidate in dense_seeds:
            result = optimize(readout, candidate['boundaries'], max_iterations)
            result.update(restart=candidate['restart'], coarse_loss=candidate['fitted']['loss'])
            refined.append(result)
            save_trial(result, candidate['restart'], 'dense-integration-polish')
            if progress:
                progress(len(trials), total, min(c['fitted']['loss'] for c in refined))
        candidates = refined
    chosen = min(candidates, key=lambda c: c['fitted']['loss'])
    fitted = chosen['fitted']
    bits = readout.clean_bits(chosen['boundaries'])
    if np.any(np.diff(bits.astype(int), axis=1) > 0):
        raise AssertionError('Selected clean comparisons are not monotonic in time.')
    model = {'algorithm': 'normal', 'decoder': 'first_zero', 'loss_mode': loss_mode,
             'decoder_dtype': decoder_dtype, 'target_dtype': 'float64',
             'timing_selection': 'fixed_clock_subset', 'optimizer': 'SLSQP, full-batch analytic gradients',
             'loss': fitted['loss'], 'mae': fitted['mae'], 'mse': fitted['mse'], 'rmse': fitted['rmse'],
             'mean_output_variance': fitted['mean_output_variance'],
             'mean_prediction_mse': fitted['mean_prediction_mse'],
             'clock_hz': clock_hz, 'clock_ticks': ticks.tolist(), 'times_s': times.tolist(),
             'references_v': chosen['references'].tolist(), 'signs': readout.signs.tolist(),
             'boundaries_similarity': chosen['boundaries'].tolist(),
             'boundary_lower': mapping.lower.tolist(), 'boundary_upper': mapping.upper.tolist(),
             'clean_initial_boundaries': clean.tolist(), 'clean_initial_loss': clean_loss,
             'clean_initial_loss_mode': loss_mode, 'clean_initial_mae': clean_metrics['mae'],
             'clean_initial_mse': clean_metrics['mse'],
             'clean_bits': bits.astype(int).tolist(), 'clean_monotonic': True,
             'decoding_parameters': fitted['levels'].tolist(),
             'category_probabilities': fitted['distribution'].tolist(),
             'prediction': fitted['prediction'].tolist(), 'prediction_std': fitted['std'].tolist(),
             'samples': count, 'categories': count+1, 'target': target.tolist(),
             'selected_restart': chosen['restart'], 'solver_success': chosen['solver_success'],
             'solver_message': chosen['solver_message'], 'selected_trace': chosen['history'],
             'trace_kind': 'best feasible iterate', 'optimizer_iterations': chosen['iterations'],
             'training_data_used': False, 'input_distribution': 'uniform similarity states',
             'noise': noise.to_dict(), 'available_clock_ticks': ticks.tolist(),
             'timing_subsets_searched': 1, 'timing_subsets_optimized': 1,
             'restarts': restarts, 'seed': seed, 'coarse_optimizer_best_loss': coarse_best,
             'sensing_latency_s': latency_s, 'cam_search_frequency_hz': search_hz,
             'result_ready_s': (times+latency_s).tolist(), 'allow_end_sensing': False,
             'global_optimum_certified': False}
    if noise.shared:
        model['coarse_loss'] = chosen['coarse_loss']
    if weights is not None:
        model.update(algorithm='dataset_enhanced', training_data_used=True,
                     input_distribution='empirical calibration similarity frequencies',
                     state_weights=weights.tolist())
        uniform = fit_readout(fitted['distribution'], target, fitted['levels'], loss_mode)
        model['uniform_state_mse'] = uniform['mse']
    return model, trials


def train(waveforms, target, clock_hz, sigma_v=.001, latency_s=70e-12,
          search_hz=500e6, samples=None, seed=0, restarts=8, max_iterations=300,
          progress=None, noise=None, polish_samples=2048,
          loss_mode='expected_mse', schedule_selection='discrete', max_samples=None, state_weights=None,
          decoder_dtype='float16'):
    """Enumerate feasible subsets; fit and densely refine every feasible schedule.

    A discrete enumeration is exhaustive, but each continuous fit is multistart
    local optimization. All schedules use the same integration rule; validation
    noise never participates in fitting or selecting a schedule.
    """
    if schedule_selection not in ('discrete', 'all'):
        raise ValueError('Use discrete or all schedule selection.')
    if loss_mode != 'expected_mse':
        raise ValueError('Only expected individual-output MSE is supported.')
    target = np.asarray(target, dtype=float)
    if target.shape != waveforms.x.shape or not np.isfinite(target).all():
        raise ValueError('Need one finite target per waveform.')
    if restarts < 2 or max_iterations < 1:
        raise ValueError('Need at least two starts and one iteration.')
    weights = state_distribution(state_weights, len(target))
    ticks = available_ticks(waveforms, clock_hz, latency_s, search_hz)
    if schedule_selection == 'discrete':
        schedules = tick_schedules(waveforms, clock_hz, latency_s, search_hz,
                                  samples=samples, max_samples=max_samples)
    else:
        if max_samples is not None and (type(max_samples) is not int or max_samples < len(ticks)):
            raise ValueError('All-tick mode exceeds the comparison budget.')
        if samples is not None and (type(samples) is not int or samples != len(ticks)):
            raise ValueError('All-tick mode requires the full available sensing count.')
        if not len(ticks) or np.any(np.diff(ticks)/clock_hz < latency_s-1e-23):
            raise ValueError('All-tick mode cannot satisfy the capture, CAM search period, or sensing latency.')
        schedules = [ticks]
    best, trials, rows = None, [], []
    for index, schedule in enumerate(schedules):
        try:
            model, attempts = _train_schedule(
                waveforms, target, clock_hz, schedule, sigma_v=sigma_v,
                latency_s=latency_s, search_hz=search_hz, seed=seed, restarts=restarts,
                max_iterations=max_iterations, noise=noise, polish_samples=polish_samples,
                loss_mode=loss_mode, state_weights=weights,
                decoder_dtype=decoder_dtype)
        except ValueError as error:
            # Physical infeasibility excludes only this subset. Bad experiment
            # configuration or missing waveform coverage must not be hidden.
            if not any(message in str(error) for message in (
                'no usable voltage envelope', 'Waveforms cross inside',
                'would not separate a contiguous', 'No ordered boundaries',
                'No ordered clean partition')):
                raise
            rows.append({'clock_ticks': schedule.tolist(), 'qualified': False,
                         'reason': str(error)})
        else:
            trials.extend(attempts)
            rows.append({key: model[key] for key in (
                'clock_ticks', 'times_s', 'samples', 'categories', 'references_v',
                'decoding_parameters', 'boundaries_similarity', 'loss', 'mse', 'mae',
                'mean_prediction_mse', 'mean_output_variance', 'solver_success',
                'solver_message', 'optimizer_iterations', 'selected_restart')})
            rows[-1]['qualified'] = True
            if best is None or model['loss'] < best['loss']:
                best = model
        if progress:
            progress(index+1, len(schedules), None if best is None else best['loss'])
    if best is None:
        raise ValueError('No schedule has feasible ordered references: '+str(rows))
    best.update(available_clock_ticks=ticks.tolist(),
                timing_selection='discrete_clock_subsets' if schedule_selection == 'discrete' else 'all_clock_ticks',
                timing_subsets_searched=len(schedules),
                timing_subsets_optimized=sum(row['qualified'] for row in rows),
                requested_samples=samples,
                max_samples=max_samples,
                schedule_selection_loss='same integration rule after refinement of each schedule')
    for row in rows:
        row['selected'] = row['clock_ticks'] == best['clock_ticks']
    return best, trials, rows

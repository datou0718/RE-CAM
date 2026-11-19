"""Screen every legal schedule, refine a shortlist, then explore local exchanges."""
import numpy as np

from recam.noise import Noise, state_distribution
from recam.ordered import FirstZeroReadout, clean_initialization
from recam.timing import available_ticks, tick_schedules
from recam.optimization import _train_schedule


ROW_KEYS = ('clock_ticks', 'times_s', 'samples', 'categories', 'references_v',
            'decoding_parameters', 'boundaries_similarity', 'loss', 'mse', 'mae',
            'mean_prediction_mse', 'mean_output_variance', 'solver_success',
            'solver_message', 'optimizer_iterations', 'selected_restart')


def neighbors(schedule, ticks, budget):
    current = set(schedule)
    choices = {tuple(sorted(current-{old}|{new})) for old in current for new in ticks if new not in current}
    if len(current) < budget:
        choices.update(tuple(sorted(current|{new})) for new in ticks if new not in current)
    if len(current) > 1:
        choices.update(tuple(sorted(current-{old})) for old in current)
    return sorted(choices)


def train_staged(waveforms, target, clock_hz, seed_model, max_samples=7,
                 noise=None, polish_samples=2048, restarts=8, max_iterations=300, seed=0,
                 latency_s=70e-12, search_hz=500e6,
                 per_count=4, global_count=16, neighbor_count=12, neighbor_rounds=2, progress=None,
                 state_weights=None, decoder_dtype='float16'):
    """Noisy-MSE screening only; held-out noise is never used for selection.

    Each screening score uses the better of clean-DP and evenly spaced boundary
    seeds. Refined schedules use the existing multistart SLSQP + dense polishing.
    The coarser-clock seed is explicitly retained and refined on the same final
    integration grid, so the selected training MSE cannot exceed its loss.
    """
    noise = noise or Noise(.001, .001, .1, 128)
    target = np.asarray(target, dtype=float)
    np.testing.assert_array_equal(seed_model['target'], target)
    weights = state_distribution(state_weights, len(target))
    assert seed_model['loss_mode'] == 'expected_mse'
    assert seed_model.get('decoder_dtype', 'float64') == decoder_dtype
    if weights is None:
        assert seed_model['algorithm'] == 'normal'
    else:
        assert seed_model['algorithm'] == 'dataset_enhanced'
        np.testing.assert_allclose(seed_model['state_weights'], weights, atol=1e-15, rtol=0)
    for field in ('sensing_std_v', 'residual_std_v', 'timing_cv', 'integration_seed'):
        assert seed_model['noise'][field] == noise.to_dict()[field]
    assert seed_model['noise']['integration_samples'] == max(noise.integration_samples, polish_samples)
    ticks = available_ticks(waveforms, clock_hz, latency_s, search_hz)
    schedules = tick_schedules(waveforms, clock_hz, latency_s, search_hz,
                              max_samples=max_samples)
    seed_ticks = tuple(np.rint(np.asarray(seed_model['times_s'])*clock_hz).astype(int))
    np.testing.assert_allclose(np.array(seed_ticks)/clock_hz, seed_model['times_s'], atol=1e-23, rtol=0)
    legal = {tuple(s) for s in schedules}
    if seed_ticks not in legal:
        raise ValueError('The seed schedule must fit the new clock, CAM search period, and comparison budget.')
    rows, lookup = [], {}
    for i, schedule in enumerate(schedules):
        row = {'clock_ticks': schedule.tolist(), 'qualified': False, 'refined': False, 'selected': False}
        try:
            readout = FirstZeroReadout(waveforms, schedule/clock_hz, target, noise, 'expected_mse', weights, decoder_dtype)
            clean, _ = clean_initialization(readout.mapping, target, 'expected_mse', weights, decoder_dtype)
            even = readout.mapping.project(np.linspace(waveforms.x[0], waveforms.x[-1], len(schedule)+2)[1:-1])
            starts = [clean, even]
            if tuple(schedule) == seed_ticks:
                starts.append(readout.mapping.project(seed_model['boundaries_similarity']))
            scores = [readout.fit(readout.mapping.references(point)[0])['loss'] for point in starts]
            best_index = int(np.argmin(scores))
            row.update(screen_feasible=True, screening_loss=float(scores[best_index]),
                       screening_boundaries=starts[best_index].tolist())
        except ValueError as error:
            if not any(text in str(error) for text in ('no usable voltage envelope', 'Waveforms cross inside',
                'would not separate a contiguous', 'No ordered boundaries', 'No ordered clean partition')):
                raise
            row.update(screen_feasible=False, reason=str(error))
        rows.append(row); lookup[tuple(schedule)] = row
        if progress and ((i+1) % 512 == 0 or i+1 == len(schedules)):
            progress('screening', i+1, len(schedules), None)
    feasible = sorted((row for row in rows if row['screen_feasible']),
                      key=lambda row: (row['screening_loss'], row['clock_ticks']))
    shortlist = {tuple(row['clock_ticks']) for row in feasible[:global_count]}
    for count in range(1, max_samples+1):
        group = [row for row in feasible if len(row['clock_ticks']) == count]
        shortlist.update(tuple(row['clock_ticks']) for row in group[:per_count])
    shortlist.add(seed_ticks)
    best, trials, fitted = None, [], {}

    def refine(schedule, initial, stage):
        nonlocal best
        model, attempts = _train_schedule(waveforms, target, clock_hz, np.array(schedule),
            noise=noise, polish_samples=polish_samples, restarts=restarts, max_iterations=max_iterations,
            seed=seed, latency_s=latency_s, search_hz=search_hz,
            loss_mode='expected_mse', initial_boundaries=np.array(initial), state_weights=weights,
            decoder_dtype=decoder_dtype)
        for attempt in attempts:
            attempt['schedule_search_stage'] = stage
        trials.extend(attempts)
        fitted[schedule] = model
        lookup[schedule].update({key: model[key] for key in ROW_KEYS})
        lookup[schedule].update(qualified=True, refined=True, refinement_stage=stage)
        if best is None or model['loss'] < best['loss']:
            best = model

    # Preserve the seed before heuristically ranking other schedules.
    refine(seed_ticks, seed_model['boundaries_similarity'], 'coarser_clock_seed')
    ordered = sorted(shortlist-{seed_ticks}, key=lambda s: (lookup[s]['screening_loss'], s))
    for i, schedule in enumerate(ordered):
        refine(schedule, lookup[schedule]['screening_boundaries'], 'screening_shortlist')
        if progress:
            progress('shortlist', i+1, len(ordered), best['loss'])
    for round_index in range(neighbor_rounds):
        previous = tuple(best['clock_ticks'])
        candidates = []
        for schedule in neighbors(previous, ticks, max_samples):
            if schedule not in legal or schedule in fitted or not lookup[schedule]['screen_feasible']:
                continue
            readout = FirstZeroReadout(waveforms, np.array(schedule)/clock_hz, target, noise, 'expected_mse', weights, decoder_dtype)
            warm = readout.mapping.project(np.interp(np.array(schedule)/clock_hz,
                best['times_s'], best['boundaries_similarity']))
            warm_score = readout.fit(readout.mapping.references(warm)[0])['loss']
            row = lookup[schedule]
            if warm_score < row['screening_loss']:
                score, initial = warm_score, warm
            else:
                score, initial = row['screening_loss'], row['screening_boundaries']
            candidates.append((float(score), schedule, initial))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for i, (_, schedule, initial) in enumerate(candidates[:neighbor_count]):
            refine(schedule, initial, f'neighbor_round_{round_index+1}')
            if progress:
                progress(f'neighbors_{round_index+1}', i+1, min(neighbor_count,len(candidates)), best['loss'])
        if tuple(best['clock_ticks']) == previous:
            break
    assert best['loss'] <= seed_model['loss']+1e-10
    best.update(available_clock_ticks=ticks.tolist(), timing_selection='staged_clock_subsets',
        timing_subsets_searched=len(schedules), timing_subsets_optimized=len(fitted), max_samples=max_samples,
        requested_samples=None, seed_clock_hz=seed_model['clock_hz'], seed_clock_ticks=[int(t) for t in seed_ticks],
        seed_loss=seed_model['loss'], global_optimum_certified=False,
        schedule_selection_loss='minimum densely refined loss among screened shortlist, seed, and local exchanges',
        staged_search={'screening_nodes': noise.integration_samples, 'per_count': per_count,
            'global_count': global_count, 'neighbor_count': neighbor_count, 'neighbor_rounds': neighbor_rounds,
            'all_legal_schedules_screened': True, 'all_legal_schedules_optimized': False})
    for row in rows:
        row['selected'] = row['clock_ticks'] == best['clock_ticks']
    return best, trials, rows

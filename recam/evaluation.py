"""Fresh noisy searches, first-zero decoding, and range-normalized RMSE."""
from dataclasses import replace
import math

import numpy as np

from recam.noise import Noise, Readout, state_distribution
from recam.ordered import first_zero_categories, first_zero_probabilities


def noise_bank(states, samples, noise, grid_hz=8e9, search_hz=500e6, seed=20361110):
    """Pair methods by search and physical clock tick; ADC noises stay independent."""
    if samples < 2:
        raise ValueError('Use at least two evaluation samples per similarity.')
    if not math.isfinite(search_hz) or search_hz <= 0:
        raise ValueError('CAM search frequency must be positive and finite.')
    rng = np.random.default_rng(seed)
    shape = (states, samples)
    width = max(7, math.ceil(grid_hz/search_hz-1e-12)-1)
    return {
        'noise_v': rng.normal(0, noise.sensing_std_v, (*shape, width)),
        'residual_offset_v': rng.normal(0, noise.residual_std_v, shape),
        'time_scale': np.exp(rng.normal(0, noise.log_sigma, shape)-.5*noise.log_sigma**2),
    }


def sample_predictions(waveforms, model, bank, grid_hz=8e9):
    times = np.asarray(model['times_s'])
    columns = (np.arange(len(times)) if model.get('architecture') else
               np.rint(times*grid_hz).astype(int)-1)
    if not model.get('architecture'):
        np.testing.assert_allclose((columns+1)/grid_hz, times, rtol=0, atol=1e-22)
    if len(set(columns)) != len(times) or np.any(columns < 0) or np.any(columns >= bank['noise_v'].shape[2]):
        raise ValueError('Need one distinct noise column per comparison.')
    warped = times[None, None, :]/bank['time_scale'][:, :, None]
    if np.any(warped < waveforms.window[0]) or np.any(warped > waveforms.window[1]):
        raise ValueError('A noisy sampling time lies outside the waveform capture.')
    voltage = np.array([np.interp(warped[i], t, v) for i, (t, v) in enumerate(waveforms.curves)])
    bits = voltage+bank['noise_v'][:, :, columns] > np.asarray(model['references_v'])+bank['residual_offset_v'][:, :, None]
    bits = np.where(np.asarray(model['signs']) > 0, bits, ~bits)
    category = first_zero_categories(bits)
    prediction = np.asarray(model['decoding_parameters'], dtype=model['decoder_dtype'])[category]
    return category, prediction


def category_probabilities(waveforms, model, noise=None):
    readout = Readout(waveforms, model['times_s'], model['target'], noise or Noise(**model['noise']))
    _, p = readout.conditional(model['references_v'])
    return first_zero_probabilities(p).mean(axis=0)


def evaluate(waveforms, model, bank, state_weights=None, evaluation_nodes=16384, grid_hz=8e9):
    """Uniform direct Monte Carlo, or held-out-frequency expected noisy MSE.

    Distribution evaluation uses an independent integration seed. Direct hard
    samples also supply a sampling check and per-similarity prediction intervals.
    """
    category, prediction = sample_predictions(waveforms, model, bank, grid_hz)
    target = np.asarray(model['target'], dtype=float)
    span = float(np.ptp(target))
    if span <= 0:
        raise ValueError('NRMSE needs a nonconstant target.')
    error = (prediction.astype(float)-target[:, None])**2
    weights = state_distribution(state_weights, len(target))
    q = np.full(len(target), 1/len(target)) if weights is None else weights
    sample_mse = float(q @ error.mean(axis=1))
    sample_se = float(np.sqrt(np.sum(q**2*error.var(axis=1, ddof=1)/error.shape[1])))
    arrays = {'category': category, 'prediction': prediction,
              'sample_mean': prediction.astype(float).mean(axis=1),
              'percentile_2_5': np.percentile(prediction.astype(float), 2.5, axis=1),
              'percentile_97_5': np.percentile(prediction.astype(float), 97.5, axis=1)}
    mse = float(error.mean())
    if weights is not None:
        noise = replace(Noise(**model['noise']), integration_samples=evaluation_nodes, integration_seed=11729)
        probability = category_probabilities(waveforms, model, noise)
        cost = (np.asarray(model['decoding_parameters'])-target[:, None])**2
        state_mse = (probability*cost).sum(axis=1)
        mse = float(q @ state_mse)
        arrays.update(probabilities=probability, state_mse=state_mse)
    metrics = {'mse': mse, 'nrmse': float(np.sqrt(mse)/span),
               'nrmse_percent': float(100*np.sqrt(mse)/span),
               'training_mse': model['mse'], 'uniform_training_mse': model.get('uniform_state_mse', model['mse']),
               'sample_mse': sample_mse, 'sample_mse_se': sample_se,
               'sampling_check_within_six_se': abs(sample_mse-mse) <= 6*sample_se+1e-12,
               'samples_per_similarity': error.shape[1], 'comparisons': model['samples'],
               'categories': model['categories']}
    return metrics, arrays

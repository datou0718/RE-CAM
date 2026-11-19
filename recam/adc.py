"""Uniform-voltage ADC: simultaneous threshold comparisons at one clock tick."""
import numpy as np
from scipy.special import ndtr

from recam.noise import Readout, fit_readout, state_distribution
from recam.ordered import first_zero_probabilities
from recam.timing import available_ticks


ADC_REFERENCES_V = .15*np.arange(1,8)


def adc_distribution(waveforms, timestamp_s, references_v, noise, allow_repeated=False):
    """Integrate common discharge/offset after forming the first-zero categories.

    All threshold comparisons see the same instantaneous waveform realization,
    but have independent sensing noise. Fixed ladder references may produce empty
    bins; they are not constrained to each tick's useful waveform envelope.
    """
    references = np.asarray(references_v, dtype=float)
    if references.ndim != 1 or not len(references) or not np.isfinite(references).all():
        raise ValueError('Need a finite reference-voltage vector.')
    if np.any(np.diff(references) < 0 if allow_repeated else np.diff(references) <= 0):
        raise ValueError('ADC references must be ordered (strictly increasing unless repeats are allowed).')
    readout = Readout(waveforms,[timestamp_s],waveforms.x,noise)
    probabilities = ndtr((readout.conditional_voltage-references)/noise.sensing_std_v)
    return first_zero_probabilities(probabilities).mean(axis=0)


def train_adc(waveforms, target, clock_hz, noise, latency_s=70e-12,
              search_hz=500e6, references_v=None,
              decoding_mode='nominal_mean', state_weights=None, decoder_dtype='float16'):
    """U-ADC nominal-bin means, or RE-CAM(U) noisy posterior means.

    Both search every eligible clock tick using noisy expected MSE. U-ADC uses
    uniform similarity weights and nominal bin membership for its fixed decoder.
    """
    target = np.asarray(target, dtype=float)
    references = ADC_REFERENCES_V.copy() if references_v is None else np.asarray(references_v,dtype=float)
    if target.shape != waveforms.x.shape or not np.isfinite(target).all():
        raise ValueError('Need one finite target per waveform.')
    if decoding_mode not in ('nominal_mean', 'posterior_mean'):
        raise ValueError('Use nominal_mean or posterior_mean decoding.')
    weights = state_distribution(state_weights, len(target))
    if weights is not None and decoding_mode == 'nominal_mean':
        raise ValueError('Fixed nominal_mean decoding requires uniform training.')
    if references.ndim != 1 or len(references) < 2 or not np.isfinite(references).all():
        raise ValueError('Need at least two finite ADC references.')
    if np.any(np.diff(references) <= 0) or not np.allclose(np.diff(references),np.diff(references)[0],rtol=1e-12,atol=1e-15):
        raise ValueError('ADC references must be increasing and evenly spaced.')
    ticks = available_ticks(waveforms,clock_hz,latency_s,search_hz)
    if not len(ticks):
        raise ValueError('No ADC sampling tick fits the search/capture window.')
    count = len(references)
    models = []
    for tick in ticks:
        timestamp = tick/clock_hz
        distribution = adc_distribution(waveforms,timestamp,references,noise)
        bits = waveforms.voltages([timestamp]).T > references
        assert np.all(np.diff(bits.astype(int),axis=1) <= 0)
        categories = np.cumprod(bits, axis=1).sum(axis=1)
        nominal_counts = np.bincount(categories, minlength=count+1)
        levels = None
        if decoding_mode == 'nominal_mean':
            totals = np.bincount(categories, weights=target, minlength=count+1)
            occupied = np.flatnonzero(nominal_counts)
            levels = np.interp(np.arange(count+1), occupied, totals[occupied]/nominal_counts[occupied])
        fit = fit_readout(distribution,target,levels=levels,loss_mode='expected_mse',state_weights=weights,
                          decoder_dtype=decoder_dtype)
        models.append({
            'algorithm':'normal','architecture':'uniform_adc','decoder':'first_zero',
            'loss_mode':'expected_mse','timing_selection':'single_tick_uniform_adc',
            'comparison_mode':'simultaneous thresholds on one waveform realization',
            'optimizer':'fixed reference ladder; '+decoding_mode+' decoding; exhaustive noisy-MSE tick selection',
            'decoding_mode':decoding_mode, 'decoder_dtype':decoder_dtype, 'target_dtype':'float64',
            'decoding_rule':('mean f(x) within each strict nominal-voltage bin; uniform similarity states'
                             if decoding_mode == 'nominal_mean' else 'noise-probability-weighted means'),
            'nominal_category_counts':nominal_counts.tolist(),
            'nominal_mse':float(np.mean((fit['levels'][categories]-target)**2)),
            'empty_category_rule':('interpolate occupied-bin means by category index; clamp outside occupied range'
                                   if decoding_mode == 'nominal_mean' else 'global target mean when probability mass is zero'),
            'reference_start_v':float(references[0]),'reference_step_v':float(references[1]-references[0]),
            'clock_hz':clock_hz,'clock_ticks':[int(tick)]*count,'times_s':[float(timestamp)]*count,
            'available_clock_ticks':ticks.tolist(),'timestamp_s':float(timestamp),'selected_tick':int(tick),
            'references_v':references.tolist(),'signs':[1.]*count,
            'samples':count,'categories':count+1,'timestamps_searched':len(ticks),
            'target':target.tolist(),'decoding_parameters':fit['levels'].tolist(),
            'category_probabilities':distribution.tolist(),'prediction':fit['prediction'].tolist(),
            'prediction_std':fit['std'].tolist(),'clean_bits':bits.astype(int).tolist(),
            **{key:fit[key] for key in ('loss','mse','mae','rmse','mean_prediction_mse','mean_output_variance')},
            'noise':noise.to_dict(),'input_distribution':'uniform similarity states','training_data_used':False,
            'sensing_latency_s':latency_s,'cam_search_frequency_hz':search_hz,
            'allow_end_sensing':False,'result_ready_s':[float(timestamp+latency_s)],
            'selected_trace':[{'iteration':0,'loss':fit['loss']}],
            'optimizer_iterations':0,'solver_success':True})
        if weights is not None:
            models[-1].update(algorithm='dataset_enhanced', training_data_used=True,
                input_distribution='empirical calibration similarity frequencies', state_weights=weights.tolist(),
                uniform_state_mse=fit_readout(distribution, target, fit['levels'], 'expected_mse')['mse'])
    best = min(models,key=lambda m:m['loss'])
    return best, models

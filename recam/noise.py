"""Conditional sensing probabilities and closed-form MSE decoding."""
import math
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc


@dataclass(frozen=True)
class Noise:
    sensing_std_v: float = .001
    residual_std_v: float = .001
    timing_cv: float = .1
    integration_samples: int = 128
    integration_seed: int = 1729

    def __post_init__(self):
        values = (self.sensing_std_v, self.residual_std_v, self.timing_cv)
        if not all(np.isfinite(v) for v in values) or self.sensing_std_v <= 0 or min(values) < 0:
            raise ValueError('Noise STDs must be finite and nonnegative; sensing STD must be positive.')
        n = self.integration_samples
        if type(n) is not int or n < 2 or n & (n-1):
            raise ValueError('Integration sample count must be a power of two, at least 2.')

    @property
    def shared(self):
        return self.residual_std_v > 0 or self.timing_cv > 0

    @property
    def log_sigma(self):
        return math.sqrt(math.log1p(self.timing_cv**2))

    def scale(self, standard_normal):
        # Mean exactly 1 and relative standard deviation exactly timing_cv.
        return np.exp(self.log_sigma*standard_normal - .5*self.log_sigma**2)

    def nodes(self):
        if not self.shared:
            return np.ones(1), np.zeros(1)
        uniform = qmc.Sobol(2, scramble=True, seed=self.integration_seed).random_base2(
            int(math.log2(self.integration_samples)))
        normal = ndtri(uniform)
        return self.scale(normal[:,0]), self.residual_std_v*normal[:,1]

    def to_dict(self):
        return asdict(self)


def state_distribution(weights, size):
    """Normalize empirical frequencies; None preserves the uniform objective."""
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (size,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Need one nonnegative frequency per state and positive total mass.')
    return weights / weights.sum()


def quantize_decoder(levels, decoder_dtype='float64'):
    """Round LUT entries; promote back to FP64 for probabilities and error arithmetic."""
    if decoder_dtype not in ('float16', 'float64'):
        raise ValueError('Decoder dtype must be float16 or float64.')
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.asarray(levels, dtype=decoder_dtype).astype(float)
    if not np.isfinite(values).all():
        raise ValueError('Decoder values must be finite and representable in '+decoder_dtype+'.')
    return values


def fit_readout(distribution, target, levels=None, loss_mode='expected_mse', state_weights=None,
                decoder_dtype='float64'):
    """Closed-form S+1 decoding parameters for expected individual-output MSE."""
    if loss_mode != 'expected_mse':
        raise ValueError('Only expected individual-output MSE is supported.')
    distribution = np.asarray(distribution, dtype=float)
    target = np.asarray(target, dtype=float)
    weights = state_distribution(state_weights, len(target))
    joint = distribution if weights is None else distribution * weights[:, None]
    if levels is None:
        mass = joint.sum(axis=0)
        fallback = target.mean() if weights is None else weights @ target
        levels = np.divide(joint.T @ target, mass,
                           out=np.full(len(mass), fallback), where=mass > np.finfo(float).tiny)
    # For squared error, the nearest representable value to the posterior mean
    # is the exact optimum over FP16 values. Ground truth is never rounded.
    levels = quantize_decoder(levels, decoder_dtype)
    mean = distribution @ levels
    error = levels - target[:, None]
    variance = np.sum(distribution * (levels - mean[:, None])**2, axis=1)
    def average(values):
        return float(np.mean(values) if weights is None else weights @ values)
    mae = average(np.sum(distribution * np.abs(error), axis=1))
    mse = average(np.sum(distribution * error**2, axis=1))
    return {"loss": mse,
            "mae": mae, "mse": mse, "rmse": math.sqrt(mse),
            "mean_output_variance": average(variance),
            "mean_prediction_mse": average((mean-target)**2),
            "levels": levels, "distribution": distribution, "prediction": mean,
            "std": np.sqrt(variance)}


class Readout:
    """Probability model for one clock schedule; distribution fitting is handled by the decoder."""

    def __init__(self, waveforms, times, target, sigma_v):
        self.noise = sigma_v if isinstance(sigma_v, Noise) else Noise(sensing_std_v=sigma_v)
        self.voltage = waveforms.voltages(times).T
        self.target = np.asarray(target, dtype=float)
        self.sigma = self.noise.sensing_std_v
        trend = np.dot(waveforms.x-waveforms.x.mean(), self.voltage.mean(axis=1))
        self.signs = np.full(len(times), 1. if trend >= 0 else -1.)
        scales, offsets = self.noise.nodes()
        warped = np.asarray(times)[None,:] / scales[:,None]
        if np.any(warped < waveforms.window[0]) or np.any(warped > waveforms.window[1]):
            raise ValueError('Time variation extends beyond the captured waveforms; provide a longer capture.')
        self.conditional_voltage = np.array([np.interp(warped, t, v) for t,v in waveforms.curves]).transpose(1,0,2)
        self.conditional_voltage -= offsets[:,None,None]

    def conditional(self, references):
        z = self.signs*(self.conditional_voltage-references)/self.sigma
        return z, ndtr(z)

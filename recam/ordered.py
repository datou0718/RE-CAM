"""Ordered similarity boundaries and a first-zero (leading-ones) readout."""
import math
import numpy as np

from recam.noise import Readout, fit_readout, state_distribution, quantize_decoder


def first_zero_probabilities(p):
    """P(k leading ones): later bits after the first zero are marginalized out."""
    p = np.asarray(p, dtype=float)
    prefix = np.concatenate((np.ones((*p.shape[:-1], 1)), np.cumprod(p, axis=-1)), axis=-1)
    return np.concatenate((prefix[..., :-1]*(1-p), prefix[..., -1:]), axis=-1)


def first_zero_categories(bits):
    """Strictly count the leading ones, including the all-ones category S."""
    return np.cumprod(np.asarray(bits, dtype=int), axis=-1).sum(axis=-1)


class BoundaryMap:
    """Interpolate captured voltages in similarity; never sort or alter curves.

    Restrict references to an ordered part of the nominal voltage envelope,
    leaving one sensing-noise STD at each edge. Crossings inside this usable
    region are rejected rather than replaced by fictitious monotone voltages.
    """

    def __init__(self, x, voltage, signs, margin):
        self.x = np.asarray(x, dtype=float)
        self.voltage = np.asarray(voltage, dtype=float)
        if np.any(np.diff(self.x) <= 0):
            raise ValueError('Similarity states must be strictly increasing.')
        lower, upper = [], []
        for j, sign in enumerate(signs):
            v = sign*self.voltage[:, j]
            lo, hi = v.min()+margin, v.max()-margin
            if lo >= hi:
                raise ValueError('A required tick has no usable voltage envelope.')
            start, end = np.flatnonzero(v <= lo)[-1], np.flatnonzero(v >= hi)[0]
            if start >= end or np.any(np.diff(v[start:end+1]) <= 0):
                raise ValueError('Waveforms cross inside the usable boundary range.')
            if np.max(v[:start+1]) > lo or np.min(v[end:]) < hi:
                raise ValueError('A boundary would not separate a contiguous similarity interval.')
            lower.append(np.interp(lo, v[start:end+1], self.x[start:end+1]))
            upper.append(np.interp(hi, v[start:end+1], self.x[start:end+1]))
        # Propagate necessary bounds implied by c[j] <= c[j+1].
        self.lower = np.maximum.accumulate(lower)
        self.upper = np.minimum.accumulate(np.asarray(upper)[::-1])[::-1]
        if np.any(self.lower > self.upper):
            raise ValueError('No ordered boundaries fit all required clock ticks.')

    def references(self, boundaries):
        c = np.asarray(boundaries, dtype=float)
        if c.shape != self.lower.shape or not np.isfinite(c).all():
            raise ValueError('Need one finite similarity boundary per comparison.')
        index = np.clip(np.searchsorted(self.x, c, side='right')-1, 0, len(self.x)-2)
        j = np.arange(len(c))
        slopes = (self.voltage[index+1, j]-self.voltage[index, j])/(self.x[index+1]-self.x[index])
        return self.voltage[index, j]+slopes*(c-self.x[index]), slopes

    def feasible(self, boundaries, tolerance=1e-9):
        c = np.asarray(boundaries)
        return bool(np.isfinite(c).all() and np.all(c >= self.lower-tolerance)
                    and np.all(c <= self.upper+tolerance) and np.all(np.diff(c) >= -tolerance))

    def project(self, boundaries):
        # Both bound vectors are nondecreasing, so clipping a sorted vector
        # preserves ordering, including equality/repeated boundaries.
        return np.clip(np.sort(boundaries), self.lower, self.upper)


def clean_initialization(mapping, target, loss_mode='expected_mse', state_weights=None,
                         decoder_dtype='float64'):
    """Clean MSE partition DP, with ordered and repeated cuts."""
    if loss_mode != 'expected_mse':
        raise ValueError('Only expected individual-output MSE is supported.')
    quantize_decoder([0.], decoder_dtype)
    y, x = np.asarray(target), mapping.x
    n, senses = len(y), len(mapping.lower)
    weights = state_distribution(state_weights, n)
    cost = np.zeros((n+1, n+1))
    for a in range(n):
        for b in range(a+1, n+1):
            region = y[a:b]
            if weights is None:
                mean = region.mean()
                if decoder_dtype == 'float16':
                    mean = float(np.float16(mean))
                cost[a, b] = ((region-mean)**2).sum()
            elif weights[a:b].sum() > 0:
                mean = np.average(region, weights=weights[a:b])
                if decoder_dtype == 'float16':
                    mean = float(np.float16(mean))
                cost[a, b] = weights[a:b] @ (region-mean)**2
    candidates = []
    for lo, hi in zip(mapping.lower, mapping.upper):
        row = {}
        for cut in range(n-1):
            left, right = max(lo, x[cut]), min(hi, np.nextafter(x[cut+1], -np.inf))
            if left <= right:
                row[cut] = np.clip((x[cut]+x[cut+1])/2, left, right)
        candidates.append(row)
    previous = {cut: (cost[0, cut+1], (cut,)) for cut in candidates[0]}
    for row in candidates[1:]:
        current = {}
        for cut in row:
            options = [(value+cost[old+1, cut+1], path+(cut,))
                       for old, (value, path) in previous.items() if old <= cut]
            if options:
                current[cut] = min(options)
        previous = current
    if not previous:
        raise ValueError('No ordered clean partition fits the required comparisons.')
    total, path = min((value+cost[cut+1, n], path) for cut, (value, path) in previous.items())
    boundaries = np.array([row[cut] for row, cut in zip(candidates, path)])
    if not mapping.feasible(boundaries):
        raise AssertionError('Clean initialization did not preserve boundary order.')
    return boundaries, float(total/n if weights is None else total)


class FirstZeroReadout(Readout):
    """Shared-noise integration with an O(S) first-zero forward/backward pass."""

    fixed_levels = None

    def __init__(self, waveforms, times, target, noise, loss_mode='expected_mse', state_weights=None,
                 decoder_dtype='float64'):
        if loss_mode != 'expected_mse':
            raise ValueError('Only expected individual-output MSE is supported.')
        super().__init__(waveforms, times, target, noise)
        self.loss_mode = loss_mode
        self.decoder_dtype = decoder_dtype
        self.state_weights = state_distribution(state_weights, len(self.target))
        self.mapping = BoundaryMap(waveforms.x, self.voltage, self.signs, self.sigma)

    def distribution(self, references):
        _, p = self.conditional(references)
        return first_zero_probabilities(p).mean(axis=0)

    def fit(self, references, levels=None):
        if levels is None:
            levels = self.fixed_levels
        return fit_readout(self.distribution(references), self.target, levels, self.loss_mode,
                           self.state_weights, self.decoder_dtype)

    def objective(self, references):
        z, p = self.conditional(references)
        prefix = np.concatenate((np.ones((*p.shape[:-1], 1)), np.cumprod(p, axis=-1)), axis=-1)
        fitted = fit_readout(first_zero_probabilities(p).mean(axis=0), self.target,
                             levels=self.fixed_levels, loss_mode=self.loss_mode, state_weights=self.state_weights,
                             decoder_dtype=self.decoder_dtype)
        error = self.target[:, None]-fitted['levels']
        cost = error**2
        future = np.broadcast_to(cost[:, -1], p.shape[:2]).copy()
        gradient = np.empty(p.shape[-1])
        density = np.exp(-z*z/2)/(self.sigma*math.sqrt(2*math.pi))
        for j in reversed(range(p.shape[-1])):
            derivative_p = prefix[:, :, j]*(future-cost[:, j])
            terms = -self.signs[j]*density[:, :, j]*derivative_p
            gradient[j] = (np.mean(terms) if self.state_weights is None
                           else np.mean(terms, axis=0) @ self.state_weights)
            future = (1-p[:, :, j])*cost[:, j]+p[:, :, j]*future
        return fitted['loss'], gradient

    def boundary_objective(self, boundaries):
        refs, slopes = self.mapping.references(boundaries)
        loss, gradient = self.objective(refs)
        return loss, gradient*slopes

    def clean_bits(self, boundaries):
        refs, _ = self.mapping.references(boundaries)
        bits = self.voltage > refs
        return np.where(self.signs > 0, bits, ~bits)

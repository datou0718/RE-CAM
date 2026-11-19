"""Positive clock ticks strictly before the CAM search period ends."""
from itertools import combinations
import math
import numpy as np

def available_ticks(waveforms, clock_hz, latency_s, search_frequency_hz):
    if not all(map(math.isfinite, (clock_hz, latency_s, search_frequency_hz))):
        raise ValueError("Timing settings must be finite.")
    if min(clock_hz, search_frequency_hz) <= 0 or latency_s < 0:
        raise ValueError("Rates must be positive and latency nonnegative.")
    lo, hi = waveforms.window
    cycle = 1 / search_frequency_hz
    end = min(hi, cycle)
    first = max(1, math.ceil(lo * clock_hz - 1e-12))
    last = math.floor(end * clock_hz + 1e-12)
    ticks = np.arange(first, last + 1, dtype=int)
    times = ticks / clock_hz
    tolerance = 32 * np.finfo(float).eps * cycle
    # Endpoint exclusion concerns the sampling instant, not completion latency.
    before_search_end = times < cycle - tolerance
    return ticks[(times >= lo) & (times <= hi) & before_search_end]

def tick_schedules(waveforms, clock_hz, latency_s, search_frequency_hz, samples=None,
                   max_samples=None):
    ticks = available_ticks(waveforms, clock_hz, latency_s, search_frequency_hz)
    stride = max(1, math.ceil(latency_s * clock_hz - 1e-12))
    if samples is not None and (type(samples) is not int or not 1 <= samples <= len(ticks)):
        raise ValueError("The requested sensing count does not fit the available clock ticks.")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError('The comparison budget must be a positive integer.')
    if samples is not None and max_samples is not None and samples > max_samples:
        raise ValueError('The exact sensing count exceeds the comparison budget.')
    limit = len(ticks) if max_samples is None else min(max_samples, len(ticks))
    schedules = [np.array(choice, dtype=int)
                 for count in ([samples] if samples is not None else range(1, limit + 1))
                 for choice in combinations(ticks, count)
                 if all(b - a >= stride for a, b in zip(choice, choice[1:]))]
    if not schedules:
        raise ValueError("No clock-tick sensing schedule fits the capture, latency and search cycle.")
    return schedules

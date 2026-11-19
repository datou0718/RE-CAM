"""Nominal matchline waveforms indexed by similarity, in seconds and volts."""
import csv
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np


@dataclass
class Waveforms:
    x: np.ndarray
    curves: list

    @property
    def window(self):
        return max(t[0] for t, _ in self.curves), min(t[-1] for t, _ in self.curves)

    def voltages(self, times):
        times = np.asarray(times, dtype=float)
        lo, hi = self.window
        if not np.isfinite(times).all() or np.any(times < lo) or np.any(times > hi):
            raise ValueError('Sensing outside the common waveform window; extrapolation is disabled.')
        return np.array([np.interp(times, t, v) for t, v in self.curves]).T


def default_waveform():
    return Path(str(files('recam').joinpath('data/10T_06V_CAM.csv')))


def load_waveforms(path=None):
    """Load paired (time, voltage) columns in increasing similarity order.

    The bundled CSV has 33 pairs for similarity 0 through 32. Each pair may
    have its own time grid; paired trailing blank cells are allowed.
    """
    with Path(path or default_waveform()).open(newline='') as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 3 or len(rows[0]) < 4 or len(rows[0]) % 2:
        raise ValueError('Need a header and at least two time/voltage pairs.')
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        raise ValueError('CSV rows have inconsistent column counts.')
    curves = []
    for pair in range(width // 2):
        points, padded = [], False
        for row in rows[1:]:
            a, b = (value.strip() for value in row[2*pair:2*pair+2])
            if not a and not b:
                padded = True
                continue
            if padded or not a or not b:
                raise ValueError('Missing values must be paired trailing blanks.')
            points.append((float(a), float(b)))
        values = np.asarray(points, dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('Each waveform needs at least two finite points.')
        t, v = values.T
        if t[0] < 0 or np.any(np.diff(t) <= 0):
            raise ValueError('Time must be nonnegative and strictly increasing.')
        curves.append((t, v))
    result = Waveforms(np.arange(len(curves), dtype=float), curves)
    if result.window[1] <= result.window[0]:
        raise ValueError('Waveforms have no common time window.')
    return result

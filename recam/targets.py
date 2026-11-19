"""The four target functions in the paper; x denotes similarity."""
import numpy as np


FUNCTIONS = {
    'linear': 'Linear',
    'piecewise-linear': 'Piecewise-Linear',
    'sigmoid': 'Sigmoid',
    'exponential': 'Exponential',
}


def target_values(name, x):
    x = np.asarray(x, dtype=float)
    if name == 'linear':
        return x.copy()
    if name == 'piecewise-linear':
        return .5*np.minimum(x-16, 0)+np.maximum(x-16, 0)
    if name == 'sigmoid':
        return 1/(1+np.exp(-(x-16)/2))
    if name == 'exponential':
        return np.exp(x/8)
    raise ValueError(f'Unknown target function: {name}')

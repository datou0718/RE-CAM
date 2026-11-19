"""Disjoint random query splits with shared binary candidates."""
import numpy as np


def generate_pairs(query_count, candidate_count, dimensionality, seed):
    """IID Bernoulli(0.5) bits; use every query/candidate pair exactly once."""
    if any(type(n) is not int or n < 1 for n in (query_count, candidate_count, dimensionality)):
        raise ValueError('Vector counts and dimensionality must be positive integers.')
    rng = np.random.default_rng(seed)
    queries = rng.integers(0, 2, (query_count, dimensionality), dtype=np.uint8)
    candidates = rng.integers(0, 2, (candidate_count, dimensionality), dtype=np.uint8)
    similarities = np.count_nonzero(queries[:, None, :] == candidates[None, :, :], axis=2)
    counts = np.bincount(similarities.ravel(), minlength=dimensionality+1)
    return dict(queries=queries, candidates=candidates, similarities=similarities, counts=counts)


def generate_split_pairs(query_count, validation_count, candidate_count, dimensionality,
                         training_seed, validation_seed):
    """Separate query sets, with shared candidates and no cross-split duplicates.

    Both splits use IID Bernoulli(0.5) bits and independent random generators.
    A validation query identical to any training query is rejected and redrawn.
    """
    if type(validation_count) is not int or validation_count < 1:
        raise ValueError('Validation query count must be a positive integer.')
    if training_seed == validation_seed:
        raise ValueError('Training and validation must use different seeds.')
    training = generate_pairs(query_count, candidate_count, dimensionality, training_seed)
    excluded = {row.tobytes() for row in training['queries']}
    if len(excluded) == 2**dimensionality:
        raise ValueError('Training queries cover the entire binary input space.')
    rng = np.random.default_rng(validation_seed)
    queries = rng.integers(0, 2, (validation_count, dimensionality), dtype=np.uint8)
    while True:
        duplicate = np.array([row.tobytes() in excluded for row in queries])
        if not duplicate.any():
            break
        queries[duplicate] = rng.integers(0, 2, (int(duplicate.sum()), dimensionality), dtype=np.uint8)
    candidates = training['candidates'].copy()
    similarities = np.count_nonzero(queries[:, None, :] == candidates[None, :, :], axis=2)
    validation = dict(queries=queries, candidates=candidates, similarities=similarities,
                      counts=np.bincount(similarities.ravel(), minlength=dimensionality+1))
    return training, validation

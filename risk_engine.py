"""Terminal simulation and empirical tail-risk calculations."""
import numpy as np


def simulate_terminal(mu, sigma, correlation, weights, shocks, initial, days):
    """Exact terminal distribution under IID normal logs; buy and hold only."""
    factor = np.linalg.cholesky(correlation)
    total_logs = days * mu + np.sqrt(days) * (shocks @ factor.T) * sigma
    return initial * (np.exp(total_logs) @ weights)


def historical_terminal(log_returns, weights, paths, days, initial, rng, block_days=5):
    """Resample contiguous blocks, selecting the same days for every stock."""
    history = np.asarray(log_returns, dtype=float)
    if history.ndim != 2 or not np.isfinite(history).all():
        raise ValueError('Historical returns must be a finite two-dimensional array.')
    if not 1 <= block_days <= min(len(history), days):
        raise ValueError('Block length must be between 1 and the horizon/history length.')
    ends = np.empty(paths)
    for first in range(0, paths, 4096):
        count = min(4096, paths-first)
        total_logs = np.zeros((count, history.shape[1]))
        for day in range(0, days, block_days):
            length = min(block_days, days-day)
            starts = rng.integers(0, len(history)-length+1, size=count)
            total_logs += history[starts[:, None] + np.arange(length)].sum(axis=1)
        ends[first:first+count] = initial * (np.exp(total_logs) @ weights)
    return ends


def risk(values, initial, confidence=.95):
    """Inverse empirical CDF VaR and exact upper-tail probability-mass ES."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Ending values must be a nonempty finite one-dimensional array.')
    if not np.isfinite(initial) or initial <= 0 or not 0 < confidence < 1:
        raise ValueError('Initial value must be positive; confidence must be between 0 and 1.')
    losses = initial-values
    ordered = np.sort(losses)
    boundary = confidence * len(losses)
    var = float(ordered[int(np.ceil(boundary))-1])
    index = int(np.floor(boundary))
    mass = len(losses)-boundary
    # Exactly the requested tail mass, not every observation tied at VaR.
    es = float((ordered[index+1:].sum() + (index+1-boundary)*ordered[index])/mass)
    return {'confidence':confidence, 'VaR_dollars':var, 'ES_dollars':es,
            'VaR_fraction':var/initial, 'ES_fraction':es/initial,
            'tail_observations_equivalent':mass,
            'mean_return':float(np.mean(values)/initial-1),
            'loss_probability':float(np.mean(losses > 0))}


def monte_carlo_intervals(values, initial, confidence, rng, resamples=200):
    """Approximate 95% percentile-bootstrap intervals for MC noise ONLY.

    Resample simulated outcomes, not historical parameters. These intervals
    exclude uncertainty about the fitted inputs and the model's correctness.
    """
    values = np.asarray(values)
    estimates = np.empty((resamples, 2))
    for i in range(resamples):
        sample = values[rng.integers(0, len(values), len(values))]
        metrics = risk(sample, initial, confidence)
        estimates[i] = metrics['VaR_dollars'], metrics['ES_dollars']
    bounds = np.quantile(estimates, [.025, .975], axis=0)
    return {'VaR_mc_low':float(bounds[0,0]), 'VaR_mc_high':float(bounds[1,0]),
            'ES_mc_low':float(bounds[0,1]), 'ES_mc_high':float(bounds[1,1])}

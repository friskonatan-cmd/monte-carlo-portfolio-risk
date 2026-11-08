"""Compare only the old/new Gaussian simulation kernels, not IO or charts."""
import argparse
import json
import statistics
import time
import tracemalloc
import numpy as np
from portfolio_risk import simulate, simulate_terminal


def measure(fn):
    fn()  # warmup
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter()-started)
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'median_seconds':statistics.median(timings), 'traced_peak_MiB':peak/1024**2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=int, default=10000)
    args = parser.parse_args()
    n, days = 5, 21
    mu, sigma, w = np.full(n, .0005), np.full(n, .02), np.full(n, 1/n)
    corr = .3*np.ones((n,n)) + .7*np.eye(n)

    def old():
        z = np.random.default_rng(42).standard_normal((args.paths, days, n))
        return simulate(mu, sigma, corr, w, z, 10000)[:, -1]

    def new():
        z = np.random.default_rng(42).standard_normal((args.paths, n))
        return simulate_terminal(mu, sigma, corr, w, z, 10000, days)

    before, after = measure(old), measure(new)
    print(json.dumps({'paths':args.paths, 'days':days, 'stocks':n, 'old':before, 'new':after,
                      'speedup':before['median_seconds']/after['median_seconds'],
                      'scope':'One Gaussian scenario including random draws; excludes plotting, bootstrap intervals, historical model, and file IO. Traced memory is not total process memory.'}, indent=2))


if __name__ == '__main__':
    main()

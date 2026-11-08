"""Five steps from historical prices to a Monte Carlo portfolio loss distribution."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from risk_engine import (simulate_terminal, historical_terminal, risk,
                         monte_carlo_intervals)

ROOT = Path(__file__).resolve().parent


def load_prices(tickers, start, end, cached=False):
    """1. Download adjusted prices, aligning dates without inventing missing prices."""
    cache = ROOT / 'data' / f'{"_".join(tickers)}_{start}_{end}.csv'
    if cached:
        if not cache.exists():
            raise ValueError(f'No saved data at {cache}; run without --cached first.')
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        yf.set_tz_cache_location(str(ROOT / 'data' / 'yf_cache'))
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            raise ValueError('Download returned no data. No synthetic data substituted.')
        prices = raw['Close'].reindex(columns=tickers)
    prices = prices.sort_index()
    before = len(prices)
    prices = prices.dropna(how='any')
    if len(prices) < 450 or (prices.index[-1] - prices.index[0]).days < 700:
        raise ValueError('Need approximately two years of common trading history for all tickers.')
    if prices.index.has_duplicates or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Prices must be finite, positive, and have unique dates.')
    if not cached:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    print(f'Prices: {len(prices)} aligned days, {prices.index[0].date()} to {prices.index[-1].date()}; removed {before-len(prices)} incomplete rows.')
    return prices, cache


def estimate(prices):
    """2. Use daily log returns for simulation; annualized figures are for display."""
    simple = prices.pct_change(fill_method=None).dropna()
    logs = np.log(prices / prices.shift(1)).dropna()
    stats = pd.DataFrame({
        'daily_mean_simple': simple.mean(),
        'annualized_arithmetic_mean_simple': simple.mean() * 252,
        'annualized_volatility_simple': simple.std(ddof=1) * np.sqrt(252),
        'daily_mean_log': logs.mean(),
        'daily_volatility_log': logs.std(ddof=1),
    })
    if (logs.std(ddof=1) <= 0).any():
        raise ValueError('Each stock must have nonzero return variance.')
    return logs, stats


def stress_correlation(correlation, strength):
    """Move towards perfect positive correlation, preserving a valid matrix."""
    if not 0 <= strength < 1:
        raise ValueError('Stress strength must be in [0, 1).')
    n = len(correlation)
    # Tiny diagonal shrinkage prevents singular sample correlations.
    base = (1 - 1e-8) * correlation + 1e-8 * np.eye(n)
    return (1-strength) * base + strength * np.ones((n, n))


def simulate(mu, sigma, correlation, weights, shocks, initial):
    """3. Cholesky correlates shocks; compound each stock before adding holdings."""
    factor = np.linalg.cholesky(correlation)
    correlated = shocks @ factor.T
    daily_log_returns = mu + correlated * sigma
    # Shapes: paths x days x stocks. Log returns add through time.
    stock_growth = np.exp(np.cumsum(daily_log_returns, axis=1))
    # Buy and hold: starting allocations stay invested, with drifting weights.
    values = initial * (stock_growth @ weights)
    return np.column_stack([np.full(len(shocks), initial), values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM'])
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2026-01-01', help='Exclusive; fixed completed historical period.')
    parser.add_argument('--paths', type=int, default=20000)
    parser.add_argument('--days', type=int, default=21)
    parser.add_argument('--initial', type=float, default=20000)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--resamples', type=int, default=200, help='MC interval resamples; 0 disables.')
    parser.add_argument('--compare-historical', action='store_true')
    parser.add_argument('--block-days', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    args.tickers = [t.upper() for t in args.tickers]
    n = len(args.tickers)
    if n < 3 or n > 5 or len(set(args.tickers)) != n:
        parser.error('Choose 3–5 distinct tickers.')
    if args.paths < 1000 or args.days < 1 or not np.isfinite(args.initial) or args.initial <= 0:
        parser.error('Use at least 1000 paths, a positive horizon, and positive initial value.')
    if not 0 < args.confidence < 1 or args.seed < 0:
        parser.error('Confidence must be between 0 and 1; seed must be nonnegative.')
    if args.resamples != 0 and args.resamples < 100:
        parser.error('Use at least 100 interval resamples, or 0 to disable.')
    if args.compare_historical and not 1 <= args.block_days <= args.days:
        parser.error('Historical block length must be between 1 and --days.')
    weights = np.asarray(args.weights if args.weights else [1/n]*n)
    if len(weights) != n or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        parser.error('Provide one nonnegative weight per ticker, summing to 1.')
    weights = weights/weights.sum()
    prices, cache = load_prices(args.tickers, args.start, args.end, args.cached)
    logs, stats = estimate(prices)
    mu, sigma = logs.mean().to_numpy(), logs.std(ddof=1).to_numpy()
    corr = logs.corr().to_numpy()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output / 'parameters.csv')
    logs.to_csv(output / 'daily_log_returns.csv')
    logs.corr().to_csv(output / 'correlation.csv')
    # Separate streams keep optional features from changing baseline estimates.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(5)]
    shocks = streams[0].standard_normal((args.paths, n))
    rows = []
    for strength in [0, .25, .5, .75, .95]:
        stressed = stress_correlation(corr, strength)
        values = simulate_terminal(mu, sigma, stressed, weights, shocks, args.initial, args.days)
        metrics = risk(values, args.initial, args.confidence)
        rows.append({'stress_strength': strength,
                     'mean_pairwise_correlation': stressed[np.triu_indices(n, 1)].mean(), **metrics})
        if strength == 0:
            baseline = values
    example_paths = simulate(mu, sigma, stress_correlation(corr, 0), weights,
                             streams[1].standard_normal((100, args.days, n)), args.initial)
    models = [{'model':'Gaussian log returns', **risk(baseline, args.initial, args.confidence)}]
    if args.resamples:
        models[0].update(monte_carlo_intervals(baseline, args.initial, args.confidence, streams[2], args.resamples))
    if args.compare_historical:
        historical = historical_terminal(logs.to_numpy(), weights, args.paths, args.days,
                                         args.initial, streams[3], args.block_days)
        model = {'model':f'Historical blocks ({args.block_days} days)', **risk(historical, args.initial, args.confidence)}
        if args.resamples:
            model.update(monte_carlo_intervals(historical, args.initial, args.confidence, streams[4], args.resamples))
        models.append(model)
    pd.DataFrame(models).to_csv(output/'model_comparison.csv', index=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'sensitivity.csv', index=False)
    pd.DataFrame({'ending_value':baseline, 'return':baseline/args.initial-1,
                  'loss':args.initial-baseline}).to_csv(output / 'simulated_outcomes.csv', index=False)
    pd.DataFrame(example_paths.T).to_csv(output / 'example_paths.csv', index_label='day')
    metadata = {**vars(args), 'output':output.name, 'weights': weights.tolist(), 'price_file': cache.name,
                'first_observation':str(prices.index[0].date()), 'last_observation':str(prices.index[-1].date()),
                'model':'IID multivariate normal daily log returns; buy and hold adjusted-price growth',
                'correlation_shrinkage':1e-8, 'sampler':'direct terminal normal v2',
                'VaR_method':'empirical inverse CDF', 'ES_method':'exact empirical upper-tail mass'}
    (output/'settings.json').write_text(json.dumps(metadata, indent=2))
    b=rows[0]
    report = f'''# Portfolio risk experiment

Data: Yahoo Finance adjusted closes via yfinance, {metadata['first_observation']} through {metadata['last_observation']}.
Tickers: {', '.join(args.tickers)}. Starting weights: {weights.tolist()}.
Initial value: ${args.initial:,.2f}. Horizon: {args.days} trading days. Paths: {args.paths:,}. Seed: {args.seed}.

## Baseline model estimates
- {args.confidence:.1%} VaR: ${b['VaR_dollars']:,.2f} ({b['VaR_fraction']:.2%}). Roughly {1-args.confidence:.1%} of modeled losses exceed this threshold.
- {args.confidence:.1%} Expected Shortfall: ${b['ES_dollars']:,.2f} ({b['ES_fraction']:.2%}). Average of the worst {1-args.confidence:.1%} probability mass, fractionally weighting the boundary.
- Mean simulated return: {b['mean_return']:.2%}.
- Fraction of simulated outcomes below the starting value: {b['loss_probability']:.2%}.

## Correlation sensitivity
All scenarios use identical random shocks and unchanged stock means and volatilities.
Stress strength moves correlations towards +1; it is not the correlation itself.

'''
    for row in rows:
        report += f"- Stress {row['stress_strength']:.0%}: mean correlation {row['mean_pairwise_correlation']:.3f}, VaR ${row['VaR_dollars']:,.2f}, ES ${row['ES_dollars']:,.2f}.\n"
    report += '\n## Model comparison and Monte Carlo precision\n'
    for model in models:
        report += f"- {model['model']}: VaR ${model['VaR_dollars']:,.2f}, ES ${model['ES_dollars']:,.2f}.\n"
        if args.resamples:
            report += f"  Approximate 95% MC intervals: VaR [${model['VaR_mc_low']:,.2f}, ${model['VaR_mc_high']:,.2f}]; ES [${model['ES_mc_low']:,.2f}, ${model['ES_mc_high']:,.2f}].\n"
    if args.resamples:
        report += f'Intervals use {args.resamples} percentile-bootstrap resamples of simulated outcomes. They measure simulation noise ONLY, not historical-parameter uncertainty or real-market loss bounds.\n'
    if args.compare_historical:
        report += 'Historical blocks retain joint observed returns and within-block time order. Block boundaries break time dependence; block length is a modeling choice. This cannot generate unseen daily shocks. A higher ES does not establish a more accurate model.\n'
    report += f"Effective tail observations: {b['tail_observations_equivalent']:.1f}.\n"
    if b['tail_observations_equivalent'] < 100:
        report += 'CAUTION: Fewer than 100 effective tail observations; increase --paths for more stable estimates.\n'
    report += '''
## Assumptions and limits
These are conditional model estimates, not forecasts or maximum losses. No out-of-sample validation is claimed.
The Gaussian model assumes independent daily log returns with constant means, volatilities, and correlations; extreme moves and volatility clustering can be understated.
The terminal shortcut preserves that model's distribution but uses different random draws from the old daily-path sampler, so the same seed need not reproduce old estimates. VaR now uses the empirical inverse CDF.
Estimated means are noisy; results depend on the historical window and selected surviving stocks.
Adjusted prices approximate reinvested distributions, not raw executable share prices. No tax, fees, rebalancing, or market impact is modeled.
Annualized mean simple returns are arithmetic summaries, not realized compound annual growth rates.
Increasing positive correlations reduces diversification in this long-only model, but weights and individual volatilities also drive risk.
Finite simulation results contain sampling noise. Repeat with different seeds and more paths before interpreting small differences.
'''
    (output/'REPORT.md').write_text(report, encoding='utf-8')
    plot(example_paths, baseline, corr, table, args, output, models)
    print(report)
    print(f'Results saved to {output}')


def plot(values, terminal, corr, table, args, output, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].plot(values[:100].T, alpha=.18, linewidth=.8)
    ax[0,0].set(title='100 independent illustrative paths', xlabel='Trading day', ylabel='Portfolio value ($)')
    losses = args.initial-terminal
    ax[0,1].hist(losses, bins=60, color='#187e91', alpha=.8)
    for label, col in [('VaR','#d04b36'),('ES','#723fa0')]:
        ax[0,1].axvline(table.iloc[0][f'{label}_dollars'], color=col, label=f'{args.confidence:.1%} {label}')
    ax[0,1].set(title='Simulated ending losses (negative = gain)', xlabel='Loss ($)', ylabel='Number of paths')
    ax[0,1].legend()
    im=ax[1,0].imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax[1,0].set_xticks(range(len(args.tickers)), args.tickers)
    ax[1,0].set_yticks(range(len(args.tickers)), args.tickers)
    ax[1,0].set_title('Historical daily log-return correlations')
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax[1,0].text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr[i,j])>.65 else 'black')
    fig.colorbar(im, ax=ax[1,0], shrink=.8)
    for label in ['VaR','ES']:
        ax[1,1].plot(table['mean_pairwise_correlation'],table[f'{label}_dollars'],marker='o',label=label)
    ax[1,1].set(title='Sensitivity: correlations stressed towards +1',xlabel='Mean pairwise correlation',ylabel=f'{args.confidence:.1%} loss measure ($)')
    ax[1,1].legend()
    fig.suptitle(f'Portfolio risk | {args.paths:,} terminal simulations | {args.days} trading days',fontsize=16)
    fig.tight_layout()
    fig.savefig(output/'dashboard.png',dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, metric in zip(axes, ['VaR', 'ES']):
        for i, model in enumerate(models):
            axis.scatter(i, model[f'{metric}_dollars'], s=60)
            if f'{metric}_mc_low' in model:
                axis.vlines(i, model[f'{metric}_mc_low'], model[f'{metric}_mc_high'], linewidth=2)
        axis.set_xticks(range(len(models)), [m['model'] for m in models], fontsize=9)
        axis.set(title=f'{args.confidence:.1%} {metric}', ylabel='Loss ($)', xlim=(-.5, len(models)-.5))
    title = ('Approximate 95% intervals show Monte Carlo noise only\nModel and historical-parameter uncertainty are excluded'
             if args.resamples else 'Model comparison | Monte Carlo intervals disabled')
    fig.suptitle(title, fontsize=11)
    fig.savefig(output/'model_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

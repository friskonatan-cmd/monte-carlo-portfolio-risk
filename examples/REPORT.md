# Portfolio risk experiment

Data: Yahoo Finance adjusted closes via yfinance, 2023-01-03 through 2025-12-31.
Tickers: AAPL, MSFT, JPM, JNJ, XOM. Starting weights: [0.2, 0.2, 0.2, 0.2, 0.2].
Initial value: $20,000.00. Horizon: 21 trading days. Paths: 20,000. Seed: 10.

## Baseline model estimates
- 95.0% VaR: $901.69 (4.51%). Roughly 5.0% of modeled losses exceed this threshold.
- 95.0% Expected Shortfall: $1,216.14 (6.08%). Average of the worst 5.0% probability mass, fractionally weighting the boundary.
- Mean simulated return: 1.88%.
- Fraction of simulated outcomes below the starting value: 32.49%.

## Correlation sensitivity
All scenarios use identical random shocks and unchanged stock means and volatilities.
Stress strength moves correlations towards +1; it is not the correlation itself.

- Stress 0%: mean correlation 0.189, VaR $901.69, ES $1,216.14.
- Stress 25%: mean correlation 0.392, VaR $1,165.54, ES $1,519.86.
- Stress 50%: mean correlation 0.594, VaR $1,385.59, ES $1,773.99.
- Stress 75%: mean correlation 0.797, VaR $1,558.37, ES $1,995.78.
- Stress 95%: mean correlation 0.959, VaR $1,699.79, ES $2,157.80.

## Model comparison and Monte Carlo precision
- Gaussian log returns: VaR $901.69, ES $1,216.14.
  Approximate 95% MC intervals: VaR [$885.42, $922.52]; ES [$1,194.86, $1,242.53].
- Historical blocks (5 days): VaR $908.56, ES $1,483.83.
  Approximate 95% MC intervals: VaR [$869.58, $940.37]; ES [$1,433.18, $1,529.42].
Intervals use 200 percentile-bootstrap resamples of simulated outcomes. They measure simulation noise ONLY, not historical-parameter uncertainty or real-market loss bounds.
Historical blocks retain joint observed returns and within-block time order. Block boundaries break time dependence; block length is a modeling choice. This cannot generate unseen daily shocks. A higher ES does not establish a more accurate model.
Effective tail observations: 1000.0.

## Assumptions and limits
These are conditional model estimates, not forecasts or maximum losses. No out-of-sample validation is claimed.
The Gaussian model assumes independent daily log returns with constant means, volatilities, and correlations; extreme moves and volatility clustering can be understated.
The terminal shortcut preserves that model's distribution but uses different random draws from the old daily-path sampler, so the same seed need not reproduce old estimates. VaR now uses the empirical inverse CDF.
Estimated means are noisy; results depend on the historical window and selected surviving stocks.
Adjusted prices approximate reinvested distributions, not raw executable share prices. No tax, fees, rebalancing, or market impact is modeled.
Annualized mean simple returns are arithmetic summaries, not realized compound annual growth rates.
Increasing positive correlations reduces diversification in this long-only model, but weights and individual volatilities also drive risk.
Finite simulation results contain sampling noise. Repeat with different seeds and more paths before interpreting small differences.

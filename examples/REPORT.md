# Portfolio risk experiment

Data: Yahoo Finance adjusted closes via yfinance, 2023-01-03 through 2025-12-31.
Tickers: AAPL, MSFT, JPM, JNJ, XOM. Starting weights: [0.2, 0.2, 0.2, 0.2, 0.2].
Initial value: $10,000.00. Horizon: 21 trading days. Paths: 10,000. Seed: 42.

## Baseline model estimates
- 95% VaR: $452.06 (4.52%). Roughly 5% of modeled losses exceed this threshold.
- 95% Expected Shortfall: $592.33 (5.92%). Average loss in the worst 5% tail (including ties).
- Mean simulated return: 1.84%.
- Fraction of simulated outcomes below the starting value: 32.74%.

## Correlation sensitivity
All scenarios use identical random shocks and unchanged stock means and volatilities.
Stress strength moves correlations towards +1; it is not the correlation itself.

- Stress 0%: mean correlation 0.189, VaR $452.06, ES $592.33.
- Stress 25%: mean correlation 0.392, VaR $570.44, ES $739.80.
- Stress 50%: mean correlation 0.594, VaR $679.04, ES $866.03.
- Stress 75%: mean correlation 0.797, VaR $770.60, ES $979.26.
- Stress 95%: mean correlation 0.959, VaR $842.75, ES $1,064.84.

## Assumptions and limits
These are conditional model estimates, not forecasts or maximum losses. No out-of-sample validation is claimed.
Daily log returns are independent over time with constant means, volatilities, and correlations; extreme moves and volatility clustering can be understated.
Estimated means are noisy; results depend on the historical window and selected surviving stocks.
Adjusted prices approximate reinvested distributions, not raw executable share prices. No tax, fees, rebalancing, or market impact is modeled.
Annualized mean simple returns are arithmetic summaries, not realized compound annual growth rates.
Increasing positive correlations reduces diversification in this long-only model, but weights and individual volatilities also drive risk.
Finite simulation results contain sampling noise. Repeat with different seeds and more paths before interpreting small differences.

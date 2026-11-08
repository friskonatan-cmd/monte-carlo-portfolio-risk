# Portfolio VaR and Expected Shortfall Estimator

Python implementation for estimating portfolio Value at Risk (VaR) and Expected Shortfall (ES) using correlated Monte Carlo simulations and historical block resampling. Includes correlation stress scenarios, simulation uncertainty intervals, visualizations, and numerical tests.

Default configuration: AAPL, MSFT, JPM, JNJ, and XOM at 20% each; $20,000 initial value; 20,000 simulations; 21 trading days; seed 10. Historical inputs are adjusted daily closing prices from January 2023 through December 2025, downloaded through yfinance. No live prices are used.

## Installation and usage

Open this `portfolio-risk` folder in VS Code (File > Open Folder). Open Terminal > New Terminal. For a fresh download of this repository, follow the installation commands below first.

```powershell
.\.venv\Scripts\python.exe portfolio_risk.py --cached
```

The cached command reuses the downloaded prices. To download them again, omit `--cached`. The results folder is overwritten on each run. Open `results/REPORT.md` for numbers and `results/dashboard.png` for charts. The included VS Code launch configuration also runs the cached version. Use `--output results/my-experiment` to keep a separate run.

On a new computer, install Python, then run these commands inside the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe portfolio_risk.py
```

## Example output

![Example simulation dashboard](examples/dashboard.png)

See [the example report](examples/REPORT.md) for the corresponding settings and results. Downloaded prices and generated run files are excluded from Git.

## Methods and configuration

```powershell
# Compare normal log returns with resampled blocks of historical days
.\.venv\Scripts\python.exe portfolio_risk.py --cached --compare-historical

# A more extreme tail, with more simulations and a separate output folder
.\.venv\Scripts\python.exe portfolio_risk.py --cached --confidence .99 --paths 100000 --compare-historical --output results/99-percent
```

- **Speed:** `simulate_terminal()` draws ending values directly. Under the independent normal-log-return assumption, 21 daily log returns sum to a normal variable with mean `21 * mu` and volatility `sqrt(21) * sigma`. We only generate 100 daily paths for illustration. The shortcut is exact for this model's terminal buy-and-hold distribution; it cannot compute maximum drawdowns or daily rebalancing outcomes.
- **Tail calculation:** `risk()` uses empirical inverse-CDF VaR and averages exactly the worst tail probability mass for ES. A fractional boundary observation is weighted appropriately. Repeated losses at the threshold no longer cause ES to include too much of the sample.
- **Confidence:** `--confidence .95` examines the worst 5%, while `.99` examines the worst 1%. The report warns when there are fewer than 100 effective tail observations. This is a useful caution threshold, not a statistical guarantee.
- **Simulation uncertainty:** 200 percentile-bootstrap resamples of simulated outcomes provide approximate 95% intervals around VaR and ES. These measure **Monte Carlo sampling noise only**, conditional on fitted inputs. They exclude uncertainty in historical estimates and model assumptions. `--resamples 1000` improves the stability of interval endpoints; `--resamples 0` skips this work. The interval's 95% coverage level is separate from your chosen VaR/ES level.
- **Alternative model:** `--compare-historical` samples five-day contiguous blocks of observed log returns. All stocks use the same selected days, preserving within-day relationships and some short-term patterns. `--block-days 1` selects independent days. Blocks do not wrap at the end of the dataset. Block boundaries break dependence; five days is a starting assumption, not an optimized choice. Unseen daily shocks cannot be generated from this finite history. A higher ES does not prove a model is more accurate.
- **Reproducibility:** separate random streams mean changing plots, resample counts, or enabling the historical comparison does not alter the Gaussian point estimates. The new sampler changes the draws compared with version 1, so the same seed no longer produces the original $452/$592 estimates. The quantile convention also changed slightly. This is not evidence of a change in underlying portfolio risk.

`model_comparison.png` shows the estimates with Monte Carlo uncertainty ranges. `model_comparison.csv` stores their numbers. Correlation stress scenarios still apply to the Gaussian model only. Changing block length or running more simulations does not constitute out-of-sample validation.

CSV risk columns are now `VaR_dollars`, `ES_dollars`, `VaR_fraction`, and `ES_fraction`, with an explicit `confidence` column, instead of names hard-coded to 95%.

Benchmark with `.\.venv\Scripts\python.exe benchmark.py`. It reports median time over five repetitions and traced peak allocations for one scenario, including random generation. It excludes chart rendering, file operations, uncertainty intervals, and the historical model; it is not an end-to-end speed claim. On the development machine, the 10,000-outcome/21-day/five-stock kernel ran about 31 times faster, with traced peak allocations falling from about 40 MiB to 1.2 MiB. Your timing may differ.

## Risk definitions

Loss is initial portfolio value minus terminal value; negative loss denotes a gain. VaR is the empirical inverse-CDF loss quantile at the selected confidence level. ES averages exactly the upper-tail probability mass, using a fractional boundary weight when necessary. Neither measure represents a maximum possible loss.

The portfolio is buy and hold: starting allocations are fixed, but weights drift as prices change. The Gaussian model estimates daily log-return means, volatilities, and correlations from aligned adjusted prices. Cholesky decomposition produces correlated shocks. Terminal log returns use horizon-scaled means and square-root-of-horizon-scaled volatilities.

Annualized arithmetic mean returns and volatilities are reported as descriptive statistics; daily log-return parameters drive the simulation. Annualized arithmetic mean is not realized CAGR. The 100 daily paths in the dashboard are an independent illustrative sample and are not used to calculate VaR or ES.

## Files and checks

- `portfolio_risk.py`: data loading, settings, experiment orchestration, and charts.
- `data/`: downloaded, aligned adjusted closes; cached data permits offline runs.
- `results/parameters.csv`: estimated daily and annualized statistics, in decimal units.
- `results/correlation.csv`: historical log-return correlations.
- `results/daily_log_returns.csv`: inputs used to estimate the model.
- `results/simulated_outcomes.csv`: all terminal values, returns, and losses.
- `results/example_paths.csv`: 100 independently sampled illustrative paths, not used for the reported risk estimates.
- `results/sensitivity.csv`: results for each correlation stress.
- `results/settings.json`: parameters and data dates for the latest run.
- `results/REPORT.md` and `results/dashboard.png`: readable summary and charts.
- `results/model_comparison.csv` and `results/model_comparison.png`: model estimates and optional MC intervals.
- `risk_engine.py`: terminal samplers, VaR/ES calculations, and Monte Carlo intervals.
- `benchmark.py`: reproducible timing/memory comparison of daily and terminal Gaussian simulation.

Run checks with `.\.venv\Scripts\python.exe -m unittest -v`. The 12 tests include known and tied loss tails, fractional ES boundaries, 99% confidence, invalid inputs, compounding, an analytic single-stock lognormal VaR/ES benchmark, equality with the original daily sampler using matched shocks, joint historical sampling, interval reproducibility/scaling, generated correlations, and valid stressed matrices.

The Gaussian model assumes independent daily normal log returns and stable parameters. Real markets can have heavier tails, changing volatility, and changing correlations. Historical mean returns are noisy. More simulations cannot fix these assumptions. The historical comparison is now implemented; the most important next extension is evaluating both models' risk estimates on later, unseen data using only information available at each forecast date. Until then, neither model is claimed to be empirically validated.

## Documentation

- [yfinance download arguments](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html): adjusted prices are explicitly enabled and the end date is exclusive.
- [NumPy normal random draws](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html).
- [NumPy quantile conventions](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html): the empirical inverse CDF differs from the previous linear interpolation convention.

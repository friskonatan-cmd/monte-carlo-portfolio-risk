# Learn Monte Carlo portfolio risk

The question: **If I start with $10,000 spread across five stocks, what range of values could my portfolio have after 21 trading days, under a model fitted to historical data?**

This project uses AAPL, MSFT, JPM, JNJ, and XOM as examples, with 20% ($2,000) initially in each stock. These are learning examples, not investment recommendations. The default historical window is January 2023 through December 2025: three completed calendar years. It does not use live prices.

## Run it

Open this `portfolio-risk` folder in VS Code (File > Open Folder). Open Terminal > New Terminal. For a fresh download of this repository, follow the installation commands below first.

```powershell
.\.venv\Scripts\python.exe portfolio_risk.py --cached
```

The cached command reuses the downloaded prices. To download them again, omit `--cached`. The results folder is overwritten on each run. Open `results/REPORT.md` for numbers and `results/dashboard.png` for charts. The included VS Code launch configuration also runs the cached version.

On a new computer, install Python, then run these commands inside the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe portfolio_risk.py
```

## Example output

![Example simulation dashboard](examples/dashboard.png)

See [the example report](examples/REPORT.md) for the corresponding settings and results. Downloaded prices and generated run files are excluded from Git.

## Learn in this order

### 1. A return is a percentage change

A price moving from $100 to $103 gives a simple return of `103 / 100 - 1 = 0.03`, or 3%. A $2,000 holding becomes $2,060. Returns let us compare stocks with different share prices.

In `estimate()`, `prices.pct_change()` calculates this for every stock and day. The average daily return describes the historical average; the standard deviation measures how much daily returns varied. A larger standard deviation means greater volatility.

Annualized arithmetic mean = daily mean times 252. Annualized volatility = daily standard deviation times square root of 252, assuming independent daily returns with stable variance. These annualized figures are summaries; the simulation uses daily inputs. The annualized arithmetic mean is not the compounded return you actually earned.

**Try first:** Calculate returns for prices 100, 110, 99. The daily returns are +10% and -10%, but the overall return is -1%. This is why we compound returns instead of just adding percentages.

### 2. Log returns make compounding convenient

The simulation uses `log(today / yesterday)`. Log returns add across days. To convert the accumulated log return back into a growth multiplier, use `exp(sum_of_log_returns)`.

We estimate the mean and standard deviation of historical log returns directly. Consequently, we do not subtract another half-variance adjustment from that estimated log mean.

Read `estimate()` and compare its simple-return and log-return calculations. Both start from the same adjusted prices.

### 3. Correlation links the stocks

Correlation measures whether returns tend to move together. Near +1 means strongly together; near 0 means little linear relationship; near -1 means strongly opposite. Zero correlation does not generally mean independence.

If all your stocks fall together, holding five names may provide little protection. Risk depends on **weights, individual volatilities, and correlations**, not correlation alone.

`logs.corr()` estimates the correlation matrix. Cholesky decomposition produces a matrix `L` such that `L @ L.T` equals that correlation matrix. Multiplying independent standard normal draws by `L.T` gives shocks with the desired correlations in the model. A finite sample will only approximate those correlations.

Read these lines in `simulate()`:

```python
factor = np.linalg.cholesky(correlation)
correlated = shocks @ factor.T
daily_log_returns = mu + correlated * sigma
```

`mu` is each stock's daily log mean. `sigma` is its daily log volatility. The random part supplies the day's surprise. This is a modeling assumption, not a prediction of tomorrow's return.

### 4. Simulate many possible months

The random array has shape `(10000, 21, 5)`: 10,000 possible futures, 21 days each, five stocks each day. Each path compounds stock growth and adds the value of the five holdings. Starting weights are equal, but they drift as stocks move: this is buy and hold, not daily rebalancing.

The seed makes random draws reproducible. Changing it produces a different sample from the same model; it does not change the underlying assumptions.

### 5. Turn ending values into losses

`loss = starting_value - ending_value`. An ending value of $9,400 means a $600 loss; $10,600 means a negative $600 loss (a gain).

For 10,000 paths, the worst 5% is approximately 500 outcomes:

- **95% VaR:** the loss threshold exceeded in approximately 5% of simulated outcomes.
- **95% Expected Shortfall (ES):** the average loss in that worst 5% tail.

For example, a hypothetical VaR of $700 and ES of $950 means about 5% of modeled months lose more than $700, and those bad months lose $950 on average. Neither number is a maximum possible loss. Read `risk()` to connect these definitions to the code. The empirical tail includes ties at the threshold.

### 6. Change one assumption

`stress_correlation()` moves all off-diagonal correlations toward +1 while preserving a valid correlation matrix. Stress strength 0.5 means halfway from the historical matrix to perfect positive correlation; it does not mean every correlation becomes 0.5.

The scenarios reuse the same random draws and keep stock means and volatilities fixed. This makes their differences easier to attribute to correlation. The stress experiment describes a hypothetical change, not a prediction that correlations will rise.

## Small experiments to do yourself

Run each command, then read the report before proceeding:

```powershell
# Same assumptions, different random sample
.\.venv\Scripts\python.exe portfolio_risk.py --cached --seed 7

# More paths: reduce simulation sampling noise, not model error
.\.venv\Scripts\python.exe portfolio_risk.py --cached --paths 50000

# Double the starting amount: dollar VaR and ES should double
.\.venv\Scripts\python.exe portfolio_risk.py --cached --initial 20000

# Concentrate 60% in the first stock; keep the total at 100%
.\.venv\Scripts\python.exe portfolio_risk.py --cached --weights .6 .1 .1 .1 .1
```

Before running each experiment, write down what you expect and why. Afterwards, explain any difference. Do not choose weights merely because they look best in the same historical sample.

## Files and checks

- `portfolio_risk.py`: the five-step implementation and chart generation.
- `data/`: downloaded, aligned adjusted closes; cached data permits offline runs.
- `results/parameters.csv`: estimated daily and annualized statistics, in decimal units.
- `results/correlation.csv`: historical log-return correlations.
- `results/daily_log_returns.csv`: inputs used to estimate the model.
- `results/simulated_outcomes.csv`: all terminal values, returns, and losses.
- `results/example_paths.csv`: 100 illustrative paths, not the entire sample.
- `results/sensitivity.csv`: results for each correlation stress.
- `results/settings.json`: parameters and data dates for the latest run.
- `results/REPORT.md` and `results/dashboard.png`: readable summary and charts.

Run checks with `.\.venv\Scripts\python.exe -m unittest -v`. They check known loss quantiles, compounding, constant-price behavior, generated correlations, and valid stressed matrices.

The model assumes independent daily normal log returns and stable parameters. Real markets can have heavier tails, changing volatility, and changing correlations. Historical mean returns are noisy. More simulations cannot fix these assumptions. A useful next extension is comparing this model with resampled historical returns and evaluating risk estimates on later, unseen data.

## Documentation

- [yfinance download arguments](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html): adjusted prices are explicitly enabled and the end date is exclusive.
- [NumPy normal random draws](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html).

This is an original learning implementation; no third-party repository code was copied.




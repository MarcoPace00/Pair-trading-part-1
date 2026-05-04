# Pair Trading Backtesting Framework part-1

This project is the beginning of an implementation of the pair-trading
strategy framework proposed in:

> **Moraes Sarmento, S. & Horta, N. (2021).**
> *A Machine Learning based Pairs Trading Investment Strategy.*
> SpringerBriefs in Applied Sciences and Technology, Springer.

The current code covers the **classical (non-ML) building blocks** of the
framework — the cointegration-based pair selection, spread mean-reversion
checks, and a quarterly walk-forward backtester — on top of which the
machine-learning components described in the book (OPTICS-based pair
search, ARMA / LSTM / LSTM-encoder-decoder forecasting models) will be
added in subsequent iterations.

## Repository contents

| File | Purpose |
|------|---------|
| `pair_trading_lib.py` | Universe fetchers (S&P 500, top ETFs by AUM), price downloader, cointegration test, Hurst exponent, half-life and mean-crossing checks. |
| `backtesting_lib.py` | `validate_and_backtest_pair` — runs the full validation pipeline on a pair and simulates a 2σ mean-reverting trade on the out-of-sample window. |
| `backtest_sp500_1.py` | Quarterly walk-forward backtest on the S&P 500 universe. |
| `backtest_etf_50.py` | Quarterly walk-forward backtest on the top ETFs by AUM. |

## Method
For each of the last 4 quarters:

1. Take the preceding 3 years as training window.
2. For every pair in the universe:
   - Engle–Granger cointegration in **both** directions (`p < 0.01`).
   - Hurst exponent of the spread `< 0.5`.
   - Half-life `< 180` days and a minimum number of mean-crossings.
3. Estimate the hedge ratio β by OLS on the training spread.
4. Trade the spread out-of-sample with ±2σ entry / mean-reversion exit,
   dollar-neutral on each leg.
5. Equal-weight the pairs that pass validation and report the trimester P&L.

Final output: per-quarter P&L plus the global P&L (sum and compounded).

## Requirements

```
numpy
pandas
yfinance
statsmodels
requests
```

## Usage

```bash
python backtest_sp500_1.py
python backtest_etf_50.py
```

## Reference

Moraes Sarmento, S., Horta, N. (2021). *A Machine Learning based Pairs
Trading Investment Strategy.* Springer.
ISBN 978-3-030-47250-4 — DOI [10.1007/978-3-030-47251-1](https://doi.org/10.1007/978-3-030-47251-1).

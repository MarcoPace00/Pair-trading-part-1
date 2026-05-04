"""
Quarterly pair-trading backtest on S&P 500.

For each of the 4 trimesters of the last year:
  - take the preceding 3 years as training window + the trimester as test
  - run validate_and_backtest_pair on every pair
  - equal-weight the pairs that pass all checks
  - report trimester P&L
At the end print the global P&L (sum and compounded).
"""

from datetime import datetime, timedelta
from itertools import combinations

import numpy as np
import pandas as pd

from pair_trading_lib import fetch_sp500_tickers, download_prices
from backtesting_lib import validate_and_backtest_pair

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_TRIMESTERS = 4                  # last 4 quarters
TRAIN_DAYS = 3 * 252              # ~3 years of trading days
TEST_DAYS = 63                    # ~3 months of trading days
P_VAL = 0.01                      # cointegration threshold
MAX_PAIRS_PER_TRIMESTER = 100     # maximum number of pairs to test per trimester

# Calendar window: 4 years back from today (a bit of buffer for non-trading days)
end_date = datetime.today()
start_date = end_date - timedelta(days=int(4 * 365.25) + 15)

# ---------------------------------------------------------------------------
# 1) Download data once
# ---------------------------------------------------------------------------
print("Fetching S&P 500 tickers ...")
tickers = fetch_sp500_tickers()
print(f"Downloading {len(tickers)} tickers from {start_date.date()} to {end_date.date()} ...")
prices = download_prices(
    tickers,
    start_date.strftime("%Y-%m-%d"),
    end_date.strftime("%Y-%m-%d"),
)
print(f"Clean price matrix: {prices.shape[0]} rows x {prices.shape[1]} tickers")

n_rows = len(prices)
needed = TRAIN_DAYS + N_TRIMESTERS * TEST_DAYS
if n_rows < needed:
    raise RuntimeError(f"Not enough data: have {n_rows} rows, need {needed}")

# ---------------------------------------------------------------------------
# 2) Walk through the 4 trimesters of the last year
# ---------------------------------------------------------------------------
trimester_pnls = []

for q in range(N_TRIMESTERS):
    # window indices: train of TRAIN_DAYS, then a TEST_DAYS test slice
    test_end   = n_rows - (N_TRIMESTERS - 1 - q) * TEST_DAYS
    test_start = test_end - TEST_DAYS
    train_start = test_start - TRAIN_DAYS

    window = prices.iloc[train_start:test_end]
    test_window = prices.iloc[test_start:test_end]
    t0, t1 = test_window.index[0].date(), test_window.index[-1].date()
    print(f"\n=== Trimester {q+1}: test {t0} -> {t1} "
          f"(train rows={TRAIN_DAYS}, test rows={TEST_DAYS}) ===")

    cols = window.columns.tolist()
    pair_returns = []

    curr_it = 0

    for a, b in combinations(cols, 2):
        if curr_it == MAX_PAIRS_PER_TRIMESTER:
            print(f"  stopping after {curr_it} pairs")
            break
        try:
            res = validate_and_backtest_pair(window[a], window[b], p_val=P_VAL)
        except Exception:
            continue
        if res.get("ok"):
            curr_it += 1
            # print(f"  pair {a} vs {b} passed validation, test return: {res['return_pct']:+.2f}%")
            pair_returns.append((a, b, res["return_pct"]))

    if not pair_returns:
        print("  no pairs passed validation -> P&L = 0%")
        trimester_pnls.append(0.0)
        continue

    rets = np.array([r for _, _, r in pair_returns])
    pnl = rets.mean()  # equal-weighted across selected pairs
    trimester_pnls.append(pnl)

    print(f"  pairs selected: {len(pair_returns)}")
    print(f"  trimester P&L (equal-weighted): {pnl:+.2f}%")
    # show the 5 best/worst contributors for inspection
    pair_returns.sort(key=lambda x: x[2], reverse=True)
    print("  top 5:", [(a, b, f"{r:+.2f}%") for a, b, r in pair_returns[:5]])
    print("  bot 5:", [(a, b, f"{r:+.2f}%") for a, b, r in pair_returns[-5:]])

# ---------------------------------------------------------------------------
# 3) Global P&L
# ---------------------------------------------------------------------------
print("\n================= SUMMARY =================")
for i, p in enumerate(trimester_pnls, 1):
    print(f"  Trimester {i}: {p:+.2f}%")

total_sum = float(np.sum(trimester_pnls))
total_compound = float(np.prod([1 + p / 100 for p in trimester_pnls]) - 1) * 100
print(f"\nGlobal P&L (sum of trimesters, fresh capital each Q): {total_sum:+.2f}%")
print(f"Global P&L (compounded, reinvested):                  {total_compound:+.2f}%")
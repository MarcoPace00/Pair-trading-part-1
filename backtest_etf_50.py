"""
Quarterly pair-trading backtest on the 50 biggest ETFs by AUM.

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

from pair_trading_lib import (
    fetch_etf_tickers,
    group_etfs_by_type,
    download_prices,
)
from backtesting_lib import validate_and_backtest_pair

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ETFS = 50
N_TRIMESTERS = 4
YEARS_TRAIN = 3
TRAIN_DAYS = YEARS_TRAIN * 252
P_VAL = 0.01
TEST_DAYS = 63

end_date = datetime.today()
start_date = end_date - timedelta(days=int(4 * 365.25) + 15)

# ---------------------------------------------------------------------------
# 1) Fetch top-50 ETFs by AUM and inspect their asset classes
# ---------------------------------------------------------------------------
print(f"Fetching top {N_ETFS} ETFs by AUM ...")
# tickers = fetch_etf_tickers(max_n=N_ETFS, asset_classes=["Equity"])
tickers = fetch_etf_tickers(max_n=N_ETFS)
print("Universe:", tickers)

groups = group_etfs_by_type(tickers)
print("\nBreakdown by asset class:")
for cls, ts in groups.items():
    print(f"  {cls:20s} ({len(ts):2d}): {ts}")

# ---------------------------------------------------------------------------
# 2) Download prices once
# ---------------------------------------------------------------------------
print(f"\nDownloading prices {start_date.date()} -> {end_date.date()} ...")
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
# 3) Walk through the 4 trimesters of the last year
# ---------------------------------------------------------------------------
trimester_pnls = []

for q in range(N_TRIMESTERS):
    test_end   = n_rows - (N_TRIMESTERS - 1 - q) * TEST_DAYS
    test_start = test_end - TEST_DAYS
    train_start = test_start - TRAIN_DAYS

    window = prices.iloc[train_start:test_end]
    test_window = prices.iloc[test_start:test_end]
    t0, t1 = test_window.index[0].date(), test_window.index[-1].date()
    print(f"\n=== Trimester {q+1}: test {t0} -> {t1} ===")

    cols = window.columns.tolist()
    pair_returns = []

    for a, b in combinations(cols, 2):
        try:
            res = validate_and_backtest_pair(window[a], window[b], 
                p_val=P_VAL, TRAIN_YEARS=YEARS_TRAIN, TEST_DAYS=TEST_DAYS)
        except Exception:
            continue
        if res.get("ok"):
            pair_returns.append((a, b, res["return_pct"]))

    if not pair_returns:
        print("  no pairs passed validation -> P&L = 0%")
        trimester_pnls.append(0.0)
        continue

    rets = np.array([r for _, _, r in pair_returns])
    pnl = rets.mean()
    trimester_pnls.append(pnl)

    print(f"  pairs selected: {len(pair_returns)}")
    print(f"  trimester P&L (equal-weighted): {pnl:+.2f}%")
    pair_returns.sort(key=lambda x: x[2], reverse=True)
    print("  top 5:", [(a, b, f"{r:+.2f}%") for a, b, r in pair_returns[:5]])
    print("  bot 5:", [(a, b, f"{r:+.2f}%") for a, b, r in pair_returns[-5:]])

# ---------------------------------------------------------------------------
# 4) Global P&L
# ---------------------------------------------------------------------------
print("\n================= SUMMARY =================")
for i, p in enumerate(trimester_pnls, 1):
    print(f"  Trimester {i}: {p:+.2f}%")

total_sum = float(np.sum(trimester_pnls))
total_compound = float(np.prod([1 + p / 100 for p in trimester_pnls]) - 1) * 100
print(f"\nGlobal P&L (sum of trimesters, fresh capital each Q): {total_sum:+.2f}%")
print(f"Global P&L (compounded, reinvested):                  {total_compound:+.2f}%")
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from pair_trading_lib import (
    check_reciprocal_cointegration,
    hurst_mean_revertion,
    half_life_checks,
)



def validate_and_backtest_pair(s1, s2, p_val=0.01, TRAIN_YEARS = 3, TEST_DAYS = 63):  
    """
    s1, s2: price series of length ~3y + 3m (aligned dates).
    Returns dict with validation outcome and, if all checks pass,
    the simulated % return on the last 3 months.
    """
    # align & clean
    s1, s2 = s1.align(s2, join="inner")
    df = pd.concat([s1, s2], axis=1).dropna()
    s1, s2 = df.iloc[:, 0], df.iloc[:, 1]

    # 0) split: train = first 3y, test = last 3m
    if len(s1) <= TEST_DAYS:
        return {"ok": False, "reason": "not enough data"}
    s1_tr, s1_te = s1.iloc[:-TEST_DAYS], s1.iloc[-TEST_DAYS:]
    s2_tr, s2_te = s2.iloc[:-TEST_DAYS], s2.iloc[-TEST_DAYS:]

    # 1) reciprocal cointegration on training window
    if not check_reciprocal_cointegration(s1_tr, s2_tr, p_val=p_val):
        return {"ok": False, "reason": "not cointegrated"}

    # hedge ratio (beta) from OLS on training set: s1 = a + beta*s2
    beta = OLS(s1_tr.values, add_constant(s2_tr.values)).fit().params[1]
    spread_tr = s1_tr - beta * s2_tr

    # 2) Hurst < 0.5 on the spread
    if not hurst_mean_revertion(spread_tr):
        return {"ok": False, "reason": "Hurst >= 0.5"}

    # 3) half-life + mean-crossing checks on the spread
    if not half_life_checks(spread_tr, max_hl=180, min_cross_over=12*TRAIN_YEARS):
        return {"ok": False, "reason": "half-life / crossings failed"}

    # 4) thresholds
    mu = spread_tr.mean()
    sd = spread_tr.std()
    H, L = mu + 2 * sd, mu - 2 * sd

    # build out-of-sample spread with the SAME beta
    spread_te = s1_te - beta * s2_te

    ret_pct = simulated_pair_trading(s1_te, s2_te, spread_te, beta, mu, H, L)

    return {
        "ok": True,
        "beta": beta,
        "mean": mu,
        "H": H,
        "L": L,
        "return_pct": ret_pct,
    }


def simulated_pair_trading(s1, s2, spread, beta, mean_spread, H, L):
    """
    Mean-reverting strategy on spread = s1 - beta*s2.
      - spread < L  -> long spread  (long s1, short beta*s2)
      - spread > H  -> short spread (short s1, long beta*s2)
      - exit when spread crosses mean_spread.
    Dollar-neutral: same $ invested in each leg.
    Returns cumulative % return (sum of per-trade pct P&L).
    """
    pos = 0          # 0 flat, +1 long spread, -1 short spread
    e1 = e2 = 0.0    # entry prices
    total = 0.0

    def trade_pct(p1, p2, side):
        # equal $ in each leg => P&L per $1 of gross capital
        # = 0.5 * (return_leg1 - return_leg2) for long spread
        r1 = (p1 - e1) / e1
        r2 = (p2 - e2) / e2
        return 0.5 * side * (r1 - r2)

    for i in range(len(spread)):
        sp = spread.iloc[i]
        p1, p2 = s1.iloc[i], s2.iloc[i]

        if pos == 0:
            if sp < L:
                pos, e1, e2 = +1, p1, p2
            elif sp > H:
                pos, e1, e2 = -1, p1, p2
        elif pos == +1 and sp >= mean_spread:
            total += trade_pct(p1, p2, +1)
            pos = 0
        elif pos == -1 and sp <= mean_spread:
            total += trade_pct(p1, p2, -1)
            pos = 0

    # close any open trade at the last bar
    if pos != 0:
        total += trade_pct(s1.iloc[-1], s2.iloc[-1], pos)

    return total * 100.0
"""
pair_trading_lib.py
Pair trading strategy backtester utilities.
"""

from logging import DEBUG

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from numpy import subtract, var, log10, polyfit
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# ---------------------------------------------------------------------------
# 1. Ticker fetchers
# ---------------------------------------------------------------------------
def fetch_sp500_tickers():
    """Fetch current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    table = pd.read_html(StringIO(html))[0]
    tickers = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return tickers


def _parse_aum(x):
    """Parse strings like '3.17B', '42.36M', '500.96K', '1.2T' into a float (USD)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("$", "").replace(",", "")
    if not s or s == "-":
        return np.nan
    mult = 1.0
    last = s[-1].upper()
    if last == "T":
        mult, s = 1e12, s[:-1]
    elif last == "B":
        mult, s = 1e9, s[:-1]
    elif last == "M":
        mult, s = 1e6, s[:-1]
    elif last == "K":
        mult, s = 1e3, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def _fetch_etf_table():
    """
    Download the full ETF table from stockanalysis.com.
    Returns a DataFrame with columns: ticker, asset_class, aum (USD float).
    """
    url = "https://stockanalysis.com/etf/"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    df = pd.read_html(StringIO(html))[0]
    df.columns = [str(c).strip() for c in df.columns]

    # locate columns robustly
    sym_col = next(c for c in df.columns if "symbol" in c.lower())
    cls_col = next(c for c in df.columns if "asset class" in c.lower() or c.lower() == "class")
    aum_col = next(c for c in df.columns if c.lower() in ("assets", "aum", "total assets"))

    out = pd.DataFrame({
        "ticker": df[sym_col].astype(str).str.upper().str.strip(),
        "asset_class": df[cls_col].astype(str).str.strip(),
        "aum": df[aum_col].apply(_parse_aum),
    })
    return out.dropna(subset=["ticker"]).reset_index(drop=True)


def fetch_etf_tickers(max_n=200, asset_classes=None):
    """
    Fetch the top `max_n` ETF tickers by AUM from stockanalysis.com.

    Parameters
    ----------
    max_n : int
        Maximum number of tickers to return.
    asset_classes : list[str] | None
        Optional list of asset classes to keep (e.g. ['Equity', 'Fixed Income']).
        Comparison is case-insensitive. If None, no filter is applied.

    Returns
    -------
    list[str]: tickers sorted by AUM descending.
    """
    df = _fetch_etf_table()
    if asset_classes:
        wanted = {c.lower() for c in asset_classes}
        df = df[df["asset_class"].str.lower().isin(wanted)]
    df = df.sort_values("aum", ascending=False, na_position="last")
    return df["ticker"].head(max_n).tolist()


def group_etfs_by_type(tickers):
    """
    Group a list of ETF tickers by their asset class.

    Parameters
    ----------
    tickers : iterable of str

    Returns
    -------
    dict[str, list[str]]: {asset_class: [tickers]}
    Tickers not found on stockanalysis.com end up under the key 'Unknown'.
    """
    df = _fetch_etf_table()
    wanted = {t.upper() for t in tickers}
    sub = df[df["ticker"].isin(wanted)]

    result = {}
    for cls, g in sub.groupby("asset_class"):
        result[cls] = g.sort_values("aum", ascending=False)["ticker"].tolist()

    found = set(sub["ticker"])
    missing = [t for t in wanted if t not in found]
    if missing:
        result["Unknown"] = sorted(missing)
    return result


# ---------------------------------------------------------------------------
# 2. Price downloader
# ---------------------------------------------------------------------------
def download_prices(ticker_list, start_date, end_date):
    """
    Download adjusted close prices for `ticker_list`.
    Drops any ticker that has at least one NaN inside the window
    (so ETFs/stocks not traded every day are discarded).
    Returns a DataFrame of close prices indexed by date.
    """
    data = yf.download(
        ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        avail = data.columns.get_level_values(0).unique()
        closes = pd.DataFrame(
            {t: data[t]["Close"] for t in ticker_list if t in avail}
        )
    else:
        closes = pd.DataFrame({ticker_list[0]: data["Close"]})

    closes = closes.dropna(axis=1, how="any")
    return closes


# ---------------------------------------------------------------------------
# 3. Cointegration test (bidirectional)
# ---------------------------------------------------------------------------
def check_reciprocal_cointegration(price_serie_1, price_serie_2, p_val=0.01, DEBUG=False):
    """
    Engle-Granger test in both directions.
    Returns True only if BOTH directions reject the unit-root null at `p_val`.
    """
    s1 = pd.Series(price_serie_1).dropna()
    s2 = pd.Series(price_serie_2).dropna()
    s1, s2 = s1.align(s2, join="inner")
    if len(s1) < 30:
        return False

    _, p1, _ = coint(s1, s2)
    _, p2, _ = coint(s2, s1)
    if DEBUG:
        print(f"cointegration p-values: {p1:.4f} (s1~s2)  {p2:.4f} (s2~s1)")
    return bool((p1 < p_val) and (p2 < p_val))


# ---------------------------------------------------------------------------
# 4. Hurst exponent (Ernie Chan formulation)
# ---------------------------------------------------------------------------
def _hurst_ernie_chan(p, lags=range(2, 100)):
    p = np.asarray(p, dtype=float)
    variancetau, tau = [], []
    for lag in lags:
        tau.append(lag)
        pp = subtract(p[lag:], p[:-lag])
        variancetau.append(var(pp))
    m = polyfit(log10(tau), log10(variancetau), 1)
    return m[0] / 2.0


def hurst_mean_revertion(spread_ts):
    """True if Hurst exponent of the spread is < 0.5 (mean-reverting)."""
    s = pd.Series(spread_ts).dropna().values
    if len(s) < 110:
        return False
    h = _hurst_ernie_chan(s)
    return h < 0.5


# ---------------------------------------------------------------------------
# 5. Half-life and mean-crossing checks
# ---------------------------------------------------------------------------
def _half_life(spread_ts):
    """Half-life of mean reversion via OU AR(1) regression."""
    s = pd.Series(spread_ts).dropna()
    s_lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    s_lag, delta = s_lag.align(delta, join="inner")
    X = add_constant(s_lag.values)
    res = OLS(delta.values, X).fit()
    lam = res.params[1]
    if lam >= 0:
        return np.inf
    return -np.log(2) / lam


def _mean_crossings(spread_ts):
    """Count how many times the spread crosses its own mean."""
    s = pd.Series(spread_ts).dropna().values
    centered = s - s.mean()
    signs = np.sign(centered)
    signs[signs == 0] = 1
    return int((np.diff(signs) != 0).sum())


def half_life_checks(spread_ts, max_hl=180, min_cross_over=12):
    """
    Returns True if:
      - half-life of the spread < max_hl (days), AND
      - the spread crosses its mean at least `min_cross_over` times.
    Otherwise returns False.
    """
    hl = _half_life(spread_ts)
    if not np.isfinite(hl) or hl <= 0 or hl >= max_hl:
        return False
    crossings = _mean_crossings(spread_ts)
    return crossings >= min_cross_over
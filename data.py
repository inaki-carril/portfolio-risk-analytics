"""
data.py — Central data layer for the VaR Engine project.

Handles all Yahoo Finance fetching, price splitting, return computation,
and market-value calculation. Both VaR_engine.py and portfolio_analytics.py
import from here — neither should re-implement these primitives.
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKERS: dict[str, str] = {
    "SP500":         "^GSPC",
    "Nasdaq100":     "^NDX",
    "Gold":          "GC=F",
    "Alibaba":       "BABA",
    "JD":            "JD",
    "Baidu":         "BIDU",
    "Alphabet_A":    "GOOGL",
    "Tencent_Music": "TME",
    "TSMC":          "TSM",
    "Bitcoin":       "BTC-USD",
    "Ethereum":      "ETH-USD",
    "XRP":           "XRP-USD",
    "Solana":        "SOL-USD",
}

DEFAULT_POSITIONS: dict[str, float] = {
    "SP500":           66.19,
    "Nasdaq100":        3.93,
    "Gold":             9.02,
    "Alibaba":        275.14,
    "JD":            1020.06,
    "Baidu":          188.42,
    "Alphabet_A":      55.71,
    "Tencent_Music": 2188.18,
    "TSMC":            34.35,
    "Bitcoin":          1.33,
    "Ethereum":        24.84,
    "XRP":          37492.16,
    "Solana":         279.28,
}

# Assets treated as crypto for frontier constraints
CRYPTO_ASSETS: list[str] = ["Bitcoin", "Ethereum", "XRP", "Solana"]

# Assets eligible as benchmarks in Portfolio Analytics
BENCHMARK_OPTIONS: list[str] = ["SP500", "Nasdaq100", "Gold"]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_prices(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Download adjusted closing prices for all assets from Yahoo Finance.

    Returns a DataFrame with a ``Date`` column plus one column per friendly
    asset name (keys of TICKERS).  Rows that are entirely NaN are dropped.
    """
    symbols = list(TICKERS.values())
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].copy()
    # Map ticker symbols back to friendly names
    prices.columns = [
        next(k for k, v in TICKERS.items() if v == col)
        for col in prices.columns
    ]
    prices = prices.reset_index()
    prices.dropna(how="all", inplace=True)
    return prices


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_prices(prices: pd.DataFrame):
    """Split a prices DataFrame into (historical, latest_prices, base_date).

    Returns
    -------
    historical    : DataFrame — all rows strictly before the last complete row
    latest_prices : dict {asset: price} — last row with no NaNs
    base_date     : Timestamp — date of that last row
    """
    last_complete_idx = prices.dropna().index[-1]
    base_date = prices.loc[last_complete_idx, "Date"]
    latest_prices = prices.loc[last_complete_idx].drop("Date").to_dict()
    historical = prices[prices["Date"] < base_date].copy()
    return historical, latest_prices, base_date


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def compute_log_returns(historical: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns from a historical price DataFrame.

    The ``Date`` column is stripped before computation.  Rows where ALL assets
    are NaN are dropped; rows where only *some* assets are NaN are kept so that
    assets with different start dates are handled correctly.
    """
    asset_cols = [c for c in historical.columns if c != "Date"]
    px = historical[asset_cols]
    log_returns = np.log(px / px.shift(1))
    log_returns = log_returns.dropna(how="all")
    return log_returns


# ---------------------------------------------------------------------------
# Market values
# ---------------------------------------------------------------------------

def compute_market_values(positions: dict, latest_prices: dict) -> dict:
    """Return {asset: market_value} = position_size * latest_price."""
    return {t: positions[t] * latest_prices[t] for t in positions}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices = fetch_prices(start="2022-01-01")
    hist, lp, bd = split_prices(prices)
    ret = compute_log_returns(hist)
    mv = compute_market_values(DEFAULT_POSITIONS, lp)
    print(f"Base date  : {pd.Timestamp(bd).date()}")
    print(f"Return rows: {len(ret)}")
    print(f"Assets     : {list(ret.columns)}")

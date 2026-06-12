"""
portfolio_analytics.py — Risk-adjusted performance, market exposure and
efficient-frontier functions for the VaR Engine project.

All functions accept:
  returns  : pd.DataFrame of daily log-returns (columns = asset names)
  mv       : np.ndarray of dollar market values aligned to returns.columns

They are completely independent of VaR_engine.py — only data.py constants
(CRYPTO_ASSETS) are imported for the frontier constraint helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from data import CRYPTO_ASSETS


# ===========================================================================
# Internal helpers
# ===========================================================================

def _portfolio_daily_returns(returns: pd.DataFrame, mv: np.ndarray) -> pd.Series:
    """Dollar-weight daily portfolio return series (scalar per day)."""
    portfolio_value = mv.sum()
    pnl = (returns.fillna(0) * mv).sum(axis=1)
    return pnl / portfolio_value


def _cumulative_value(daily_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Compound daily returns into a cumulative value series."""
    return start_value * (1 + daily_returns).cumprod()


def _drawdown_series(cum_value: pd.Series) -> pd.Series:
    """Drawdown as a fraction of running peak (always <= 0)."""
    running_max = cum_value.cummax()
    return (cum_value - running_max) / running_max


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Longest consecutive number of days spent below the previous peak."""
    max_dur, current_dur = 0, 0
    for below in (drawdown < 0):
        if below:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


# ===========================================================================
# 1. Risk-adjusted performance
# ===========================================================================

def compute_performance_metrics(
    returns: pd.DataFrame,
    mv: np.ndarray,
    risk_free_rate: float = 0.045,
    benchmark_col: str = "SP500",
) -> dict:
    """Return a flat dict of annualised performance and risk metrics.

    Parameters
    ----------
    returns        : DataFrame of daily log-returns
    mv             : dollar market values aligned to returns.columns
    risk_free_rate : annual risk-free rate (e.g. 0.045 for 4.5 %)
    benchmark_col  : column name used for beta / alpha calculations

    Returns
    -------
    dict with keys:
        ann_return, ann_vol, sharpe, sortino, calmar,
        max_drawdown, max_drawdown_duration_days, total_return, daily_rf,
        treynor, jensens_alpha, m2
    """
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    port_ret = _portfolio_daily_returns(returns, mv)
    n = len(port_ret)

    ann_return = (1 + port_ret).prod() ** (252 / n) - 1
    ann_vol    = port_ret.std() * np.sqrt(252)

    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

    # Sortino: downside deviation uses only negative excess returns
    excess = port_ret - daily_rf
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = (ann_return - risk_free_rate) / downside_std if downside_std and downside_std > 0 else np.nan

    cum = _cumulative_value(port_ret)
    dd  = _drawdown_series(cum)
    max_dd   = float(dd.min())
    dd_dur   = _max_drawdown_duration(dd)
    total_ret = float(cum.iloc[-1] - 1)

    # ------------------------------------------------------------------
    # Treynor ratio, Jensen's Alpha, Modigliani-squared (M2)
    # ------------------------------------------------------------------
    treynor      = np.nan
    jensens_alpha = np.nan
    m2           = np.nan

    if benchmark_col in returns.columns:
        bm_ret  = returns[benchmark_col].fillna(0)
        bm_var  = bm_ret.var()

        if bm_var > 0:
            # Portfolio beta via Cov(r_p, r_bm) / Var(r_bm)
            port_beta = port_ret.cov(bm_ret) / bm_var

            # Annualised benchmark return
            bm_ann = (1 + bm_ret).prod() ** (252 / len(bm_ret)) - 1

            # Treynor ratio: (R_p - R_f) / beta_p
            if port_beta != 0:
                treynor = (ann_return - risk_free_rate) / port_beta

            # Jensen's Alpha: R_p - [R_f + beta_p * (R_bm - R_f)]
            jensens_alpha = ann_return - (risk_free_rate + port_beta * (bm_ann - risk_free_rate))

            # Modigliani-Squared (M2): Sharpe_p * sigma_bm + R_f
            # M2 = R_f + Sharpe_p * ann_vol_benchmark
            bm_ann_vol = bm_ret.std() * np.sqrt(252)
            if not np.isnan(sharpe) and bm_ann_vol > 0:
                m2 = risk_free_rate + sharpe * bm_ann_vol

    return {
        "ann_return":                 ann_return,
        "ann_vol":                    ann_vol,
        "sharpe":                     sharpe,
        "sortino":                    sortino,
        "calmar":                     (ann_return / abs(max_dd)) if max_dd != 0 else np.nan,
        "max_drawdown":               max_dd,
        "max_drawdown_duration_days": dd_dur,
        "total_return":               total_ret,
        "daily_rf":                   daily_rf,
        "treynor":                    treynor,
        "jensens_alpha":              jensens_alpha,
        "m2":                         m2,
    }


def compute_drawdown_series(
    returns: pd.DataFrame,
    mv: np.ndarray,
    start_value: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Return (cumulative_value, drawdown) series.

    Both series share the same index as ``returns``.
    """
    port_ret = _portfolio_daily_returns(returns, mv)
    cum = _cumulative_value(port_ret, start_value)
    dd  = _drawdown_series(cum)
    return cum, dd


def compute_asset_sharpes(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.045,
) -> pd.Series:
    """Per-asset annualised Sharpe ratio."""
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    ann_ret  = (1 + returns.fillna(0)).prod() ** (252 / len(returns)) - 1
    ann_vol  = returns.std() * np.sqrt(252)
    return (ann_ret - risk_free_rate) / ann_vol


# ===========================================================================
# 2. Beta and market exposure
# ===========================================================================

def compute_betas(
    returns: pd.DataFrame,
    benchmark_col: str = "SP500",
) -> pd.DataFrame:
    """Per-asset OLS beta, R-squared, systematic and idiosyncratic risk %.

    Returns a DataFrame indexed by asset name with columns:
        beta, r_squared, systematic_risk_pct, idiosyncratic_risk_pct
    """
    if benchmark_col not in returns.columns:
        raise ValueError(f"Benchmark column '{benchmark_col}' not in returns DataFrame.")

    bm  = returns[benchmark_col].fillna(0)
    bm_var = bm.var()
    rows = []
    for col in returns.columns:
        if col == benchmark_col:
            continue
        asset_r = returns[col].fillna(0)
        cov_val  = asset_r.cov(bm)
        beta     = cov_val / bm_var if bm_var > 0 else np.nan
        corr_val = asset_r.corr(bm)
        r_sq     = corr_val ** 2
        sys_pct  = r_sq * 100
        idio_pct = (1 - r_sq) * 100
        rows.append({
            "asset":                  col,
            "beta":                   beta,
            "r_squared":              r_sq,
            "systematic_risk_pct":    sys_pct,
            "idiosyncratic_risk_pct": idio_pct,
        })
    return pd.DataFrame(rows).set_index("asset")


def compute_portfolio_beta(
    returns: pd.DataFrame,
    mv: np.ndarray,
    benchmark_col: str = "SP500",
) -> float:
    """Scalar portfolio beta computed directly from portfolio vs benchmark returns.

    Uses Cov(r_portfolio, r_benchmark) / Var(r_benchmark) over the full window,
    exactly like compute_rolling_beta but without a rolling window.  This avoids
    any column-order alignment issues that arise from the weighted-average approach.
    """
    port_ret = _portfolio_daily_returns(returns, mv)
    bm_ret   = returns[benchmark_col].fillna(0)
    cov_val  = port_ret.cov(bm_ret)
    bm_var   = bm_ret.var()
    return cov_val / bm_var if bm_var > 0 else np.nan


def compute_rolling_beta(
    returns: pd.DataFrame,
    mv: np.ndarray,
    benchmark_col: str = "SP500",
    window: int = 60,
) -> pd.Series:
    """Rolling portfolio beta over a given window (days)."""
    port_ret = _portfolio_daily_returns(returns, mv)
    bm_ret   = returns[benchmark_col].fillna(0)

    roll_cov = port_ret.rolling(window).cov(bm_ret)
    roll_var = bm_ret.rolling(window).var()
    return (roll_cov / roll_var).rename("rolling_beta")


def compute_rolling_correlation(
    returns: pd.DataFrame,
    benchmark_col: str = "SP500",
    window: int = 60,
) -> pd.DataFrame:
    """Rolling correlation of each asset vs benchmark.

    Returns a DataFrame with one column per asset (excluding benchmark).
    """
    bm = returns[benchmark_col].fillna(0)
    result = {}
    for col in returns.columns:
        if col == benchmark_col:
            continue
        result[col] = returns[col].fillna(0).rolling(window).corr(bm)
    return pd.DataFrame(result)


def compute_risk_decomposition(
    returns: pd.DataFrame,
    mv: np.ndarray,
    benchmark_col: str = "SP500",
) -> pd.DataFrame:
    """Per-asset dollar risk decomposition: total, systematic, idiosyncratic.

    Uses annualised volatility * market_value as the dollar risk proxy.

    Returns a DataFrame indexed by asset with columns:
        market_value, weight_pct, ann_vol_pct,
        total_risk_usd, systematic_risk_usd, idiosyncratic_risk_usd
    """
    beta_df = compute_betas(returns, benchmark_col)
    portfolio_value = mv.sum()
    assets = list(returns.columns)
    rows = []
    for i, asset in enumerate(assets):
        if asset == benchmark_col or asset not in beta_df.index:
            continue
        ann_vol  = returns[asset].fillna(0).std() * np.sqrt(252)
        total_usd = mv[i] * ann_vol
        sys_frac  = beta_df.loc[asset, "systematic_risk_pct"] / 100
        rows.append({
            "asset":                  asset,
            "market_value":           mv[i],
            "weight_pct":             mv[i] / portfolio_value * 100,
            "ann_vol_pct":            ann_vol * 100,
            "total_risk_usd":         total_usd,
            "systematic_risk_usd":    total_usd * sys_frac,
            "idiosyncratic_risk_usd": total_usd * (1 - sys_frac),
        })
    return pd.DataFrame(rows).set_index("asset")


# ===========================================================================
# 3. Efficient frontier
# ===========================================================================

def compute_efficient_frontier(
    returns: pd.DataFrame,
    n_portfolios: int = 3000,
    risk_free_rate: float = 0.045,
    max_crypto_weight: float = 0.30,
    seed: int = 99,
) -> pd.DataFrame:
    """Monte-Carlo efficient frontier with an optional crypto weight cap.

    Generates ``n_portfolios`` random long-only weight vectors.  If the sum of
    crypto asset weights exceeds ``max_crypto_weight``, weights are rescaled so
    that the constraint is exactly binding while preserving the relative
    non-crypto weights.

    Returns
    -------
    DataFrame with columns:
        volatility, ann_return, sharpe, *<asset>_weight  (one per asset)
    """
    rng    = np.random.default_rng(seed)
    assets = list(returns.columns)
    n_a    = len(assets)

    # Clean returns: fill NaN with 0 at computation step
    ret_clean = returns.fillna(0).values          # shape (T, n_a)
    mean_r    = ret_clean.mean(axis=0)            # (n_a,)
    cov_m     = np.cov(ret_clean.T)               # (n_a, n_a)

    crypto_idx     = [i for i, a in enumerate(assets) if a in CRYPTO_ASSETS]
    non_crypto_idx = [i for i in range(n_a) if i not in crypto_idx]

    # Draw Dirichlet samples — already sums to 1
    weights_raw = rng.dirichlet(np.ones(n_a), size=n_portfolios)

    # Apply crypto constraint
    for k in range(n_portfolios):
        crypto_sum = weights_raw[k, crypto_idx].sum() if crypto_idx else 0.0
        if crypto_sum > max_crypto_weight and crypto_idx:
            scale = max_crypto_weight / crypto_sum
            weights_raw[k, crypto_idx] *= scale
            remaining = 1.0 - max_crypto_weight
            nc_sum = weights_raw[k, non_crypto_idx].sum()
            if nc_sum > 0:
                weights_raw[k, non_crypto_idx] *= remaining / nc_sum
            else:
                weights_raw[k, non_crypto_idx] = remaining / len(non_crypto_idx)

    # Annualised stats for each portfolio
    ann_ret = (weights_raw @ mean_r) * 252
    ann_vol = np.sqrt(np.einsum("ij,jk,ik->i", weights_raw, cov_m, weights_raw) * 252)
    sharpe  = (ann_ret - risk_free_rate) / np.where(ann_vol > 0, ann_vol, np.nan)

    df = pd.DataFrame({
        "volatility": ann_vol,
        "ann_return": ann_ret,
        "sharpe":     sharpe,
    })
    for i, a in enumerate(assets):
        df[f"{a}_weight"] = weights_raw[:, i]

    return df.dropna(subset=["sharpe"])


def compute_tangency_portfolio(frontier_df: pd.DataFrame) -> pd.Series:
    """Row of frontier_df with the maximum Sharpe ratio."""
    return frontier_df.loc[frontier_df["sharpe"].idxmax()]


def compute_min_variance_portfolio(frontier_df: pd.DataFrame) -> pd.Series:
    """Row of frontier_df with the minimum volatility."""
    return frontier_df.loc[frontier_df["volatility"].idxmin()]


def compute_current_portfolio_point(
    returns: pd.DataFrame,
    mv: np.ndarray,
    risk_free_rate: float = 0.045,
) -> tuple[float, float, float]:
    """Return (annualised_vol, annualised_return, sharpe) for the current portfolio."""
    port_ret = _portfolio_daily_returns(returns, mv)
    ann_ret  = (1 + port_ret).prod() ** (252 / len(port_ret)) - 1
    ann_vol  = port_ret.std() * np.sqrt(252)
    sharpe   = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    return ann_vol, ann_ret, sharpe

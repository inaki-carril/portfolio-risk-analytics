"""
VaR_engine.py — Value-at-Risk computation module.

Data primitives (TICKERS, DEFAULT_POSITIONS, fetch_prices, split_prices,
compute_log_returns, compute_market_values) live in data.py and are
re-exported here for backwards compatibility.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Re-export everything from the data layer so callers can still do:
#   from VaR_engine import DEFAULT_POSITIONS, fetch_prices, ...
from data import (
    TICKERS,
    DEFAULT_POSITIONS,
    CRYPTO_ASSETS,
    BENCHMARK_OPTIONS,
    fetch_prices,
    split_prices,
    compute_log_returns,
    compute_market_values,
)

__all__ = [
    "TICKERS", "DEFAULT_POSITIONS", "CRYPTO_ASSETS", "BENCHMARK_OPTIONS",
    "fetch_prices", "split_prices", "compute_log_returns", "compute_market_values",
    "historical_simulation", "parametric_var", "monte_carlo_var",
    "marginal_var", "run_var_engine",
]


# ---------------------------------------------------------------------------
# VaR / ES calculations
# ---------------------------------------------------------------------------

def historical_simulation(
    returns: pd.DataFrame,
    mv: np.ndarray,
    confidence: float = 0.99,
):
    """Historical-simulation VaR and ES (dollar terms).

    Parameters
    ----------
    returns    : DataFrame of log-returns; columns must align with mv order
    mv         : array of dollar market values in the same column order
    confidence : VaR confidence level (e.g. 0.99)

    Returns
    -------
    VaR_hs, ES_hs, pnl  (scalar, scalar, 1-D array)
    """
    pnl    = (returns.fillna(0) * mv).sum(axis=1)
    VaR_hs = -np.percentile(pnl, (1 - confidence) * 100)
    tail   = pnl[pnl <= -VaR_hs]
    ES_hs  = -tail.mean() if len(tail) > 0 else np.nan
    return VaR_hs, ES_hs, pnl


def parametric_var(
    returns: pd.DataFrame,
    mv: np.ndarray,
    confidence: float = 0.99
):
    """Parametric (variance-covariance) VaR and ES.

    Returns
    -------
    VaR_param, ES_param, portfolio_std
    """
    cov_matrix         = returns.cov().values
    portfolio_variance = mv @ cov_matrix @ mv.T
    portfolio_std      = np.sqrt(portfolio_variance)

    z     = norm.ppf(confidence)
    phi_z = norm.pdf(z)

    VaR_param = z     * portfolio_std
    ES_param  = portfolio_std * phi_z / (1 - confidence)
    return VaR_param, ES_param, portfolio_std


def monte_carlo_var(
    returns: pd.DataFrame,
    mv: np.ndarray,
    confidence: float = 0.99,
    n_simulations: int = 10_000,
    seed: int = 42,
):
    """Monte-Carlo VaR and ES using correlated normal draws.

    Returns
    -------
    VaR_mc, ES_mc, pnl_mc
    """
    rng      = np.random.default_rng(seed)
    mean_r   = returns.mean().values
    cov_m    = returns.cov().values
    L        = np.linalg.cholesky(cov_m)
    Z        = rng.standard_normal((n_simulations, len(mean_r)))
    corr_r   = Z @ L.T + mean_r
    pnl_mc   = corr_r @ mv
    VaR_mc   = -np.percentile(pnl_mc, (1 - confidence) * 100)
    tail     = pnl_mc[pnl_mc <= -VaR_mc]
    ES_mc    = -tail.mean() if len(tail) > 0 else np.nan
    return VaR_mc, ES_mc, pnl_mc


# ---------------------------------------------------------------------------
# Marginal VaR
# ---------------------------------------------------------------------------

def marginal_var(
    returns: pd.DataFrame,
    market_values: dict,
    confidence: float = 0.99,
    bump_pct: float = 0.01,
):
    """Compute marginal VaR for each asset by bumping its position by bump_pct.

    Returns a dict {asset: delta_VaR} using historical simulation.
    """
    assets   = list(market_values.keys())
    base_mv  = np.array([market_values[a] for a in assets])
    base_var, _, _ = historical_simulation(returns[assets], base_mv, confidence)

    marginals = {}
    for i, asset in enumerate(assets):
        bumped_mv     = base_mv.copy()
        bumped_mv[i] *= (1 + bump_pct)
        bumped_var, _, _ = historical_simulation(returns[assets], bumped_mv, confidence)
        marginals[asset] = bumped_var - base_var
    return marginals


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_var_engine(
    positions: dict | None = None,
    start: str = "2020-01-01",
    confidence: float = 0.99,
    n_simulations: int = 10_000,
) -> dict:
    """Fetch data and compute all VaR metrics.  Returns a results bundle dict."""
    if positions is None:
        positions = DEFAULT_POSITIONS

    prices                         = fetch_prices(start=start)
    historical, latest_prices, base_date = split_prices(prices)
    returns                        = compute_log_returns(historical)

    assets  = [a for a in positions if a in returns.columns]
    returns = returns[assets]

    market_values = compute_market_values({a: positions[a] for a in assets}, latest_prices)
    mv            = np.array([market_values[a] for a in assets])
    portfolio_value = mv.sum()

    VaR_hs,    ES_hs,    pnl_hs  = historical_simulation(returns, mv, confidence)
    VaR_param, ES_param, port_std = parametric_var(returns, mv, confidence)
    VaR_mc,    ES_mc,    pnl_mc  = monte_carlo_var(returns, mv, confidence, n_simulations)
    marginals = marginal_var(returns, market_values, confidence, bump_pct=0.01)

    return {
        "prices": prices, "historical": historical,
        "latest_prices": latest_prices, "base_date": base_date,
        "returns": returns, "assets": assets,
        "market_values": market_values, "mv": mv,
        "portfolio_value": portfolio_value, "confidence": confidence,
        "VaR_hs": VaR_hs, "ES_hs": ES_hs, "pnl_hs": pnl_hs,
        "VaR_param": VaR_param, "ES_param": ES_param, "port_std": port_std,
        "VaR_mc": VaR_mc, "ES_mc": ES_mc, "pnl_mc": pnl_mc,
        "marginals": marginals,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    res = run_var_engine()
    pv  = res["portfolio_value"]
    print(f"Portfolio value : ${pv:,.2f}")
    print(f"Base date       : {res['base_date'].date()}")
    print(f"\n{'Method':<22} {'VaR (99%)':<18} {'ES (99%)':<18}")
    print("-" * 58)
    for label, var, es in [
        ("Historical Sim", res["VaR_hs"],    res["ES_hs"]),
        ("Parametric",     res["VaR_param"], res["ES_param"]),
        ("Monte Carlo",    res["VaR_mc"],    res["ES_mc"]),
    ]:
        print(f"{label:<22} ${var:>14,.2f}   ${es:>14,.2f}")

"""
streamlit_app.py — Main dashboard for the VaR Engine project.
Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, skew, kurtosis

from data import DEFAULT_POSITIONS, BENCHMARK_OPTIONS, CRYPTO_ASSETS
from VaR_engine import (
    fetch_prices,
    split_prices,
    compute_log_returns,
    compute_market_values,
    historical_simulation,
    parametric_var,
    monte_carlo_var,
    marginal_var,
)
from portfolio_analytics import (
    compute_performance_metrics,
    compute_drawdown_series,
    compute_asset_sharpes,
    compute_betas,
    compute_portfolio_beta,
    compute_rolling_beta,
    compute_rolling_correlation,
    compute_risk_decomposition,
    compute_efficient_frontier,
    compute_tangency_portfolio,
    compute_min_variance_portfolio,
    compute_current_portfolio_point,
)

# ============================================================
# Page config & theme colours
# ============================================================

st.set_page_config(
    page_title="Portfolio Risk Analytics",
    page_icon="📉",
    layout="wide",
)

C_PRIMARY = "#4F8BF9"
C_DANGER  = "#EF4444"
C_WARNING = "#F59E0B"
C_SUCCESS = "#10B981"
C_PURPLE  = "#8B5CF6"
C_TEAL    = "#14B8A6"

# ============================================================
# Shared helpers
# ============================================================

def fmt_dollar(v):
    return f"${v:,.2f}"

def fmt_pct(v, decimals=4):
    return f"{v * 100:.{decimals}f}%"

def fmt_x(v, decimals=2):
    return f"{v:.{decimals}f}x"

# ============================================================
# Sidebar — global controls
# ============================================================

with st.sidebar:
    st.title("⚙️ Settings")

    page = st.radio(
        "Page",
        ["VaR Analysis", "Portfolio Analytics"],
        horizontal=False,
        key="page_sel",
    )

    st.divider()
    st.subheader("Data")
    start_date = st.date_input(
        "Historical data start", value=pd.Timestamp("2020-01-01")
    )
    run_btn = st.button("🔄 Fetch / Refresh", type="primary", use_container_width=True)

    # ---- VaR-specific controls (always visible) ----
    st.divider()
    st.subheader("VaR Settings")
    confidence = st.slider(
        "Confidence level", 0.90, 0.999, 0.99, 0.001, format="%.3f"
    )
    n_sims = st.number_input(
        "Monte-Carlo simulations", 1000, 100000, 10000, 1000
    )

    # ---- Portfolio Analytics controls (visible only on that page) ----
    if page == "Portfolio Analytics":
        st.divider()
        st.subheader("Portfolio Analytics Settings")
        risk_free_rate = st.number_input(
            "Risk-free rate (%)", 0.0, 20.0, 4.5, 0.1, format="%.1f"
        ) / 100.0
        benchmark_col = st.selectbox(
            "Benchmark", options=BENCHMARK_OPTIONS, index=0
        )
        rolling_window = st.slider(
            "Rolling window (days)", 20, 252, 60, 5
        )
    else:
        risk_free_rate        = 0.045
        benchmark_col         = "SP500"
        rolling_window        = 60
        n_frontier_portfolios = 3000
        max_crypto_pct        = 0.30

    st.divider()
    st.subheader("Position sizes (units)")
    custom_positions = {}
    for asset, default_qty in DEFAULT_POSITIONS.items():
        custom_positions[asset] = st.number_input(
            asset, value=float(default_qty), min_value=0.0, step=0.01,
            format="%.4f", key=f"pos_{asset}",
        )

# ============================================================
# Data loading (cached on start-date string)
# ============================================================

@st.cache_data(show_spinner="Fetching prices from Yahoo Finance...")
def load_data(start: str):
    prices = fetch_prices(start=start)
    historical, latest_prices, base_date = split_prices(prices)
    returns = compute_log_returns(historical)
    return prices, historical, latest_prices, base_date, returns

if "loaded" not in st.session_state or run_btn:
    st.session_state["loaded"] = True
    st.session_state["start"]  = str(start_date)

prices, historical, latest_prices, base_date, returns = load_data(
    st.session_state.get("start", "2020-01-01")
)

# Align assets to what was actually downloaded
assets          = [a for a in custom_positions if a in returns.columns]
returns_aligned = returns[assets]
market_values   = compute_market_values(
    {a: custom_positions[a] for a in assets}, latest_prices
)
mv              = np.array([market_values[a] for a in assets])
portfolio_value = mv.sum()

# ============================================================
# ============================================================
#  PAGE 1 — VaR Analysis
# ============================================================
# ============================================================

if page == "VaR Analysis":

    # Pre-compute VaR
    VaR_hs,    ES_hs,    pnl_hs   = historical_simulation(returns_aligned, mv, confidence)
    VaR_param, ES_param, port_std = parametric_var(returns_aligned, mv, confidence)
    VaR_mc,    ES_mc,    pnl_mc   = monte_carlo_var(
        returns_aligned, mv, confidence, int(n_sims)
    )

    # ---- Header KPIs ----
    st.title("📉 Value at Risk Dashboard")
    st.caption(
        f"Base date: **{pd.Timestamp(base_date).date()}**  |  "
        f"Confidence: **{confidence*100:.1f}%**  |  "
        f"Portfolio value: **{fmt_dollar(portfolio_value)}**"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Portfolio Value",   fmt_dollar(portfolio_value))
    k2.metric("VaR – Hist Sim",    fmt_dollar(VaR_hs),
              f"{VaR_hs/portfolio_value*100:.2f}% of NAV")
    k3.metric("ES  – Hist Sim",    fmt_dollar(ES_hs),
              f"{ES_hs/portfolio_value*100:.2f}% of NAV")
    k4.metric("VaR – Parametric",  fmt_dollar(VaR_param),
              f"{VaR_param/portfolio_value*100:.2f}% of NAV")
    k5.metric("VaR – Monte Carlo", fmt_dollar(VaR_mc),
              f"{VaR_mc/portfolio_value*100:.2f}% of NAV")
    k6.metric("ES  – Monte Carlo", fmt_dollar(ES_mc),
              f"{ES_mc/portfolio_value*100:.2f}% of NAV")

    st.divider()

    # ============================================================
    # Section 1 — Data & Descriptive Analysis
    # ============================================================
    st.header("Data & Descriptive Analysis")

    with st.expander("Price History", expanded=True):
        asset_sel_price = st.multiselect(
            "Assets to display", options=assets, default=assets[:4],
            key="price_sel"
        )
        chart_mode = st.radio(
            "Chart mode",
            ["Rebased to 100 (comparable)", "Raw price (USD)"],
            horizontal=True,
            key="price_mode",
        )
        if asset_sel_price:
            fig_price = go.Figure()
            for a in asset_sel_price:
                s = prices.set_index("Date")[a].dropna()
                if chart_mode.startswith("Rebased"):
                    s = s / s.iloc[0] * 100
                fig_price.add_trace(go.Scatter(x=s.index, y=s, name=a, mode="lines"))
            y_label = "Rebased value (start = 100)" if chart_mode.startswith("Rebased") else "Price (USD)"
            title   = "Relative Performance (Rebased to 100)" if chart_mode.startswith("Rebased") else "Adjusted Closing Price"
            fig_price.update_layout(
                title=title, xaxis_title="Date",
                yaxis_title=y_label, legend=dict(orientation="h"), height=400,
            )
            st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Single-Asset Descriptive Analysis")
    col_left, col_right = st.columns([1, 3])
    with col_left:
        asset_sel = st.selectbox("Select asset", options=assets, key="desc_sel")

    r = returns_aligned[asset_sel].dropna()
    desc_stats = pd.DataFrame({
        "Statistic": [
            "Observations", "Mean daily return", "Std dev (daily)",
            "Ann. Volatility", "Min return", "Max return",
            "Skewness", "Excess Kurtosis",
            f"VaR {confidence*100:.0f}% (asset)",
        ],
        "Value": [
            f"{len(r):,}",
            fmt_pct(r.mean()),
            fmt_pct(r.std()),
            fmt_pct(r.std() * np.sqrt(252)),
            fmt_pct(r.min()),
            fmt_pct(r.max()),
            f"{skew(r):.4f}",
            f"{kurtosis(r):.4f}",
            fmt_pct(-np.percentile(r, (1 - confidence) * 100)),
        ],
    })
    with col_left:
        st.dataframe(desc_stats, hide_index=True, use_container_width=True)
    with col_right:
        x_range    = np.linspace(r.min(), r.max(), 300)
        normal_pdf = norm.pdf(x_range, r.mean(), r.std())
        var99_a    = -np.percentile(r, (1 - confidence) * 100)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=r, nbinsx=80, histnorm="probability density",
            name="Empirical", marker_color=C_PRIMARY, opacity=0.7,
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=normal_pdf,
            name="Normal fit", line=dict(color=C_WARNING, width=2),
        ))
        fig_hist.add_vline(
            x=-var99_a, line_dash="dash", line_color=C_DANGER,
            annotation_text=f"VaR {confidence*100:.0f}%",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            title=f"{asset_sel} — Daily Log-Return Distribution",
            xaxis_title="Log-return", yaxis_title="Density",
            legend=dict(orientation="h"), height=380,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Return Correlation Matrix"):
        corr = returns_aligned.corr()
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            text_auto=".2f", title="Pairwise Return Correlation", height=520,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Portfolio Composition by Market Value"):
        mv_series = pd.Series(market_values).sort_values(ascending=False)
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(
                values=mv_series.values, names=mv_series.index,
                title="Market Value Weights", hole=0.4,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            fig_bar_mv = go.Figure(go.Bar(
                x=mv_series.index, y=mv_series.values, marker_color=C_PRIMARY,
                text=[fmt_dollar(v) for v in mv_series.values],
                textposition="outside",
            ))
            fig_bar_mv.update_layout(
                title="Market Value per Asset (USD)",
                xaxis_title="Asset", yaxis_title="Market Value (USD)", height=400,
            )
            st.plotly_chart(fig_bar_mv, use_container_width=True)

    st.divider()

    # ============================================================
    # Section 2 — VaR Analysis
    # ============================================================
    st.header("VaR Analysis")

    st.subheader("VaR / ES Comparison Table")
    var_df = pd.DataFrame([
        {
            "Method":      "Historical Simulation",
            "VaR (1-day)": fmt_dollar(VaR_hs),
            "VaR % NAV":   fmt_pct(VaR_hs / portfolio_value),
            "ES (1-day)":  fmt_dollar(ES_hs),
            "ES % NAV":    fmt_pct(ES_hs / portfolio_value),
        },
        {
            "Method":      "Parametric (Var-Cov)",
            "VaR (1-day)": fmt_dollar(VaR_param),
            "VaR % NAV":   fmt_pct(VaR_param / portfolio_value),
            "ES (1-day)":  fmt_dollar(ES_param),
            "ES % NAV":    fmt_pct(ES_param / portfolio_value),
        },
        {
            "Method":      f"Monte Carlo ({int(n_sims):,} sims)",
            "VaR (1-day)": fmt_dollar(VaR_mc),
            "VaR % NAV":   fmt_pct(VaR_mc / portfolio_value),
            "ES (1-day)":  fmt_dollar(ES_mc),
            "ES % NAV":    fmt_pct(ES_mc / portfolio_value),
        },
    ])
    st.dataframe(var_df, hide_index=True, use_container_width=True)

    methods  = ["Hist Sim", "Parametric", "Monte Carlo"]
    var_vals = [VaR_hs, VaR_param, VaR_mc]
    es_vals  = [ES_hs,  ES_param,  ES_mc]
    fig_vc = go.Figure()
    fig_vc.add_trace(go.Bar(
        name="VaR", x=methods, y=var_vals, marker_color=C_DANGER,
        text=[fmt_dollar(v) for v in var_vals], textposition="outside",
    ))
    fig_vc.add_trace(go.Bar(
        name="ES",  x=methods, y=es_vals,  marker_color=C_WARNING,
        text=[fmt_dollar(v) for v in es_vals], textposition="outside",
    ))
    fig_vc.update_layout(
        barmode="group", title="VaR vs ES by Method",
        yaxis_title="USD", height=380,
    )
    st.plotly_chart(fig_vc, use_container_width=True)

    st.subheader("P&L Distribution")
    col_hs, col_mc_col = st.columns(2)

    def pnl_histogram(pnl_arr, var_val, es_val, title):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=pnl_arr, nbinsx=100, histnorm="probability density",
            marker_color=C_PRIMARY, opacity=0.75, name="P&L",
        ))
        fig.add_vline(x=-var_val, line_dash="dash", line_color=C_DANGER,
                      annotation_text=f"VaR {fmt_dollar(var_val)}",
                      annotation_position="top left")
        fig.add_vline(x=-es_val, line_dash="dot", line_color=C_WARNING,
                      annotation_text=f"ES {fmt_dollar(es_val)}",
                      annotation_position="bottom left")
        fig.update_layout(title=title, xaxis_title="Daily P&L (USD)",
                          yaxis_title="Density", height=380)
        return fig

    with col_hs:
        st.plotly_chart(
            pnl_histogram(pnl_hs, VaR_hs, ES_hs,
                          f"Hist Sim — {confidence*100:.1f}% VaR & ES"),
            use_container_width=True,
        )
    with col_mc_col:
        st.plotly_chart(
            pnl_histogram(pnl_mc, VaR_mc, ES_mc,
                          f"Monte Carlo — {confidence*100:.1f}% VaR & ES"),
            use_container_width=True,
        )

    st.subheader("Marginal VaR — Impact of a Position Bump")
    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        bump_asset = st.selectbox(
            "Asset to bump", options=assets,
            index=assets.index("Bitcoin") if "Bitcoin" in assets else 0,
            key="bump_asset",
        )
    with ctrl2:
        bump_pct = st.slider(
            "Bump size (%)", 1, 100, 10, 1, format="%d%%", key="bump_slider",
        ) / 100.0

    bumped_mv_dict = dict(market_values)
    bumped_mv_dict[bump_asset] = market_values[bump_asset] * (1 + bump_pct)
    bumped_mv = np.array([bumped_mv_dict[a] for a in assets])

    VaR_hs_b,    ES_hs_b,    _ = historical_simulation(returns_aligned, bumped_mv, confidence)
    VaR_param_b, ES_param_b, _ = parametric_var(returns_aligned, bumped_mv, confidence)
    VaR_mc_b,    ES_mc_b,    _ = monte_carlo_var(
        returns_aligned, bumped_mv, confidence, int(n_sims)
    )
    delta_hs    = VaR_hs_b    - VaR_hs
    delta_param = VaR_param_b - VaR_param
    delta_mc    = VaR_mc_b    - VaR_mc
    delta_es    = ES_hs_b     - ES_hs
    delta_pv    = bumped_mv_dict[bump_asset] - market_values[bump_asset]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{bump_asset} MV", fmt_dollar(bumped_mv_dict[bump_asset]),
              f"+{fmt_dollar(delta_pv)} ({bump_pct*100:.0f}%)")
    m2.metric("VaR – Hist Sim",    fmt_dollar(VaR_hs_b),    f"{delta_hs:+,.2f}")
    m3.metric("VaR – Parametric",  fmt_dollar(VaR_param_b), f"{delta_param:+,.2f}")
    m4.metric("VaR – Monte Carlo", fmt_dollar(VaR_mc_b),    f"{delta_mc:+,.2f}")
    m5.metric("ES  – Hist Sim",    fmt_dollar(ES_hs_b),     f"{delta_es:+,.2f}")

    fig_mvar = go.Figure()
    fig_mvar.add_trace(go.Bar(
        name="Base VaR", x=methods, y=var_vals, marker_color=C_PRIMARY,
        text=[fmt_dollar(v) for v in var_vals], textposition="outside",
    ))
    fig_mvar.add_trace(go.Bar(
        name=f"After +{bump_pct*100:.0f}% {bump_asset}",
        x=methods, y=[VaR_hs_b, VaR_param_b, VaR_mc_b],
        marker_color=C_DANGER,
        text=[fmt_dollar(v) for v in [VaR_hs_b, VaR_param_b, VaR_mc_b]],
        textposition="outside",
    ))
    fig_mvar.update_layout(
        barmode="group",
        title=f"Portfolio VaR — Base vs +{bump_pct*100:.0f}% {bump_asset} ({confidence*100:.1f}%)",
        yaxis_title="VaR (USD)", height=400,
    )
    st.plotly_chart(fig_mvar, use_container_width=True)

    delta_df = pd.DataFrame([
        {
            "Method": "Historical Simulation",
            "Base VaR": fmt_dollar(VaR_hs), "Bumped VaR": fmt_dollar(VaR_hs_b),
            "Delta VaR": fmt_dollar(delta_hs),
            "Delta % NAV": fmt_pct(delta_hs / portfolio_value),
            "Delta % VaR": f"{delta_hs / VaR_hs * 100:.2f}%",
        },
        {
            "Method": "Parametric",
            "Base VaR": fmt_dollar(VaR_param), "Bumped VaR": fmt_dollar(VaR_param_b),
            "Delta VaR": fmt_dollar(delta_param),
            "Delta % NAV": fmt_pct(delta_param / portfolio_value),
            "Delta % VaR": f"{delta_param / VaR_param * 100:.2f}%",
        },
        {
            "Method": "Monte Carlo",
            "Base VaR": fmt_dollar(VaR_mc), "Bumped VaR": fmt_dollar(VaR_mc_b),
            "Delta VaR": fmt_dollar(delta_mc),
            "Delta % NAV": fmt_pct(delta_mc / portfolio_value),
            "Delta % VaR": f"{delta_mc / VaR_mc * 100:.2f}%",
        },
    ])
    st.dataframe(delta_df, hide_index=True, use_container_width=True)

    with st.expander("Rolling 1-Day Historical Simulation VaR"):
        roll_w = st.slider("Rolling window (days)", 60, 504, 252, 21)
        rv     = returns_aligned.values
        pnl_series = (returns_aligned.fillna(0).values @ mv)  # full P&L series
        roll_vars = [
            -np.percentile(pnl_series[i - roll_w: i], (1 - confidence) * 100)
            for i in range(roll_w, len(pnl_series))
        ]
        hist_dates = pd.to_datetime(historical["Date"].values)
        offset     = len(hist_dates) - len(returns_aligned)
        ret_index = hist_dates[offset: offset + len(returns_aligned)]
        roll_dates = ret_index[roll_w:]
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=roll_dates, y=roll_vars, mode="lines",
            name=f"Rolling {roll_w}d VaR", line=dict(color=C_PRIMARY),
        ))
        fig_roll.add_hline(y=VaR_hs, line_dash="dash", line_color=C_DANGER,
                           annotation_text="Full-period VaR")
        fig_roll.update_layout(
            title=f"Rolling {roll_w}-Day VaR ({confidence*100:.1f}%)",
            xaxis_title="Date", yaxis_title="VaR (USD)", height=380,
        )
        st.plotly_chart(fig_roll, use_container_width=True)
# ============================================================
#  PAGE 2 — Portfolio Analytics
# ============================================================

elif page == "Portfolio Analytics":

    st.title("📊 Portfolio Analytics")
    st.caption(
        f"Base date: **{pd.Timestamp(base_date).date()}**  |  "
        f"Benchmark: **{benchmark_col}**  |  "
        f"Risk-free rate: **{risk_free_rate*100:.1f}%**  |  "
        f"Portfolio value: **{fmt_dollar(portfolio_value)}**"
    )

    if benchmark_col not in returns_aligned.columns:
        st.error("Benchmark not found in returns. Ensure it has a non-zero position.")
        st.stop()

    perf = compute_performance_metrics(returns_aligned, mv, risk_free_rate, benchmark_col)
    cum_value, drawdown = compute_drawdown_series(
        returns_aligned, mv, start_value=portfolio_value
    )
    port_beta = compute_portfolio_beta(returns_aligned, mv, benchmark_col)
    beta_df   = compute_betas(returns_aligned, benchmark_col)

    hist_dates_pa = pd.to_datetime(historical["Date"].values)
    offset_pa     = len(hist_dates_pa) - len(returns_aligned)
    ret_dates     = hist_dates_pa[offset_pa: offset_pa + len(returns_aligned)]
    cum_value.index = ret_dates
    drawdown.index  = ret_dates

    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Ann. Return",    fmt_pct(perf["ann_return"], 2))
    h2.metric("Ann. Vol",       fmt_pct(perf["ann_vol"], 2))
    h3.metric("Sharpe",         f"{perf['sharpe']:.2f}")
    h4.metric("Sortino",        f"{perf['sortino']:.2f}")
    h5.metric("Max Drawdown",   fmt_pct(perf["max_drawdown"], 2))
    h6.metric("Portfolio Beta", f"{port_beta:.3f}")

    st.divider()
    tab_perf, tab_exposure = st.tabs(
        ["Performance", "Market Exposure"]
    )

    # ----------------------------------------------------------
    # TAB 1 — Performance
    # ----------------------------------------------------------
    with tab_perf:
        st.subheader("Risk-Adjusted Performance Summary")

        # Three new ratio KPIs
        r1, r2, r3 = st.columns(3)
        treynor_val = perf.get("treynor", float("nan"))
        alpha_val   = perf.get("jensens_alpha", float("nan"))
        m2_val      = perf.get("m2", float("nan"))
        r1.metric(
            "Treynor Ratio",
            f"{treynor_val:.4f}" if not np.isnan(treynor_val) else "N/A",
            help="Excess return per unit of systematic risk (beta). "
                 "Higher is better; benchmark-relative.",
        )
        r2.metric(
            "Jensen's Alpha",
            fmt_pct(alpha_val, 2) if not np.isnan(alpha_val) else "N/A",
            help="Annualised return above/below CAPM expectation. "
                 "Positive α = manager skill or factor tilt.",
        )
        r3.metric(
            "Modigliani-Squared (M²)",
            fmt_pct(m2_val, 2) if not np.isnan(m2_val) else "N/A",
            help="Portfolio return if its volatility were rescaled to match "
                 "the benchmark. Directly comparable across strategies.",
        )
        st.divider()

        perf_table = pd.DataFrame([
            {"Metric": "Annualised Return",     "Value": fmt_pct(perf["ann_return"], 2)},
            {"Metric": "Annualised Volatility", "Value": fmt_pct(perf["ann_vol"], 2)},
            {"Metric": "Total Return (period)", "Value": fmt_pct(perf["total_return"], 2)},
            {"Metric": "Sharpe Ratio",          "Value": f"{perf['sharpe']:.4f}"},
            {"Metric": "Sortino Ratio",         "Value": f"{perf['sortino']:.4f}"},
            {"Metric": "Calmar Ratio",          "Value": f"{perf['calmar']:.4f}"},
            {"Metric": "Treynor Ratio",         "Value": f"{treynor_val:.4f}" if not np.isnan(treynor_val) else "N/A"},
            {"Metric": "Jensen's Alpha (ann.)", "Value": fmt_pct(alpha_val, 2) if not np.isnan(alpha_val) else "N/A"},
            {"Metric": "Modigliani-Squared (M²)", "Value": fmt_pct(m2_val, 2) if not np.isnan(m2_val) else "N/A"},
            {"Metric": "Max Drawdown",          "Value": fmt_pct(perf["max_drawdown"], 2)},
            {"Metric": "Max DD Duration",
             "Value": f"{perf['max_drawdown_duration_days']} days"},
        ])
        st.dataframe(perf_table, hide_index=True, use_container_width=True)
        st.divider()

        st.subheader("Cumulative Portfolio Value")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum_value.index, y=cum_value.values, mode="lines",
            line=dict(color=C_PRIMARY, width=2),
            fill="tozeroy", fillcolor="rgba(79,139,249,0.08)",
        ))
        fig_cum.update_layout(
            xaxis_title="Date", yaxis_title="Value (USD)", height=350, showlegend=False,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("Drawdown (Underwater Curve)")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100, mode="lines",
            line=dict(color=C_DANGER, width=1),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.20)",
        ))
        fig_dd.update_layout(
            xaxis_title="Date", yaxis_title="Drawdown (%)",
            height=250, showlegend=False, yaxis=dict(ticksuffix="%"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Per-Asset Sharpe Ratio")
        asset_sharpes = compute_asset_sharpes(
            returns_aligned, risk_free_rate
        ).sort_values(ascending=False)
        colours_sh = [C_SUCCESS if v >= 0 else C_DANGER for v in asset_sharpes.values]
        fig_sh = go.Figure(go.Bar(
            x=asset_sharpes.index, y=asset_sharpes.values,
            marker_color=colours_sh,
            text=[f"{v:.2f}" for v in asset_sharpes.values],
            textposition="outside",
        ))
        fig_sh.add_hline(y=0, line_dash="dash", line_color="grey")
        fig_sh.update_layout(
            title="Individual Asset Sharpe Ratios (Annualised)",
            xaxis_title="Asset", yaxis_title="Sharpe Ratio", height=380,
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    # ----------------------------------------------------------
    # TAB 2 — Market Exposure
    # ----------------------------------------------------------
    with tab_exposure:
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric(
            f"Portfolio Beta vs {benchmark_col}", f"{port_beta:.4f}",
            help="Beta > 1 amplifies benchmark moves; Beta < 1 dampens them",
        )
        bc2.metric("Benchmark",      benchmark_col)
        bc3.metric("Rolling window", f"{rolling_window} days")
        st.divider()

        st.subheader("Per-Asset Beta and Risk Decomposition")
        risk_dec = compute_risk_decomposition(returns_aligned, mv, benchmark_col)
        disp = beta_df.copy().reset_index()
        disp.columns = ["Asset", "Beta", "R-Squared",
                         "Systematic Risk %", "Idiosyncratic Risk %"]
        disp["Beta"]                 = disp["Beta"].round(4)
        disp["R-Squared"]            = disp["R-Squared"].apply(lambda v: f"{v:.4f}")
        disp["Systematic Risk %"]    = disp["Systematic Risk %"].apply(
            lambda v: f"{v:.1f}%")
        disp["Idiosyncratic Risk %"] = disp["Idiosyncratic Risk %"].apply(
            lambda v: f"{v:.1f}%")
        if not risk_dec.empty:
            rd = risk_dec[
                ["total_risk_usd", "systematic_risk_usd", "idiosyncratic_risk_usd"]
            ].copy().reset_index()
            rd.columns = [
                "Asset", "Total Risk $", "Systematic Risk $", "Idiosyncratic Risk $"
            ]
            for c in ["Total Risk $", "Systematic Risk $", "Idiosyncratic Risk $"]:
                rd[c] = rd[c].apply(fmt_dollar)
            disp = disp.merge(rd, on="Asset", how="left")
        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.divider()

        st.subheader(
            f"Rolling Portfolio Beta vs {benchmark_col} ({rolling_window}-day window)"
        )
        roll_beta_s = compute_rolling_beta(
            returns_aligned, mv, benchmark_col, rolling_window
        )
        roll_beta_s.index = ret_dates[:len(roll_beta_s)]
        fig_rb = go.Figure()
        fig_rb.add_trace(go.Scatter(
            x=roll_beta_s.index, y=roll_beta_s.values, mode="lines",
            name="Rolling Beta", line=dict(color=C_PRIMARY, width=2),
        ))
        fig_rb.add_hline(y=1.0, line_dash="dash", line_color="grey",
                          annotation_text="Beta = 1")
        fig_rb.add_hline(y=port_beta, line_dash="dot", line_color=C_WARNING,
                          annotation_text="Full-period Beta")
        fig_rb.update_layout(xaxis_title="Date", yaxis_title="Beta", height=350)
        st.plotly_chart(fig_rb, use_container_width=True)
        st.divider()

        st.subheader(
            f"Rolling Asset Correlation vs {benchmark_col} ({rolling_window}-day window)"
        )
        roll_corr_df = compute_rolling_correlation(
            returns_aligned, benchmark_col, rolling_window
        )
        roll_corr_df.index = ret_dates[:len(roll_corr_df)]
        col_chart, col_opts = st.columns([4, 1])
        with col_opts:
            highlight_crypto = st.checkbox("Bold crypto assets", value=True)
            corr_sel = st.multiselect(
                "Show assets",
                options=list(roll_corr_df.columns),
                default=list(roll_corr_df.columns),
                key="corr_sel",
            )
        with col_chart:
            fig_rc = go.Figure()
            for col in corr_sel:
                lw = 2.5 if (col in CRYPTO_ASSETS and highlight_crypto) else 1.2
                fig_rc.add_trace(go.Scatter(
                    x=roll_corr_df.index, y=roll_corr_df[col],
                    mode="lines", name=col, line=dict(width=lw),
                ))
            fig_rc.add_hline(y=0, line_dash="dash", line_color="grey")
            fig_rc.update_layout(
                title=f"Rolling {rolling_window}d Correlation vs {benchmark_col}",
                xaxis_title="Date", yaxis_title="Correlation",
                yaxis=dict(range=[-1.05, 1.05]),
                legend=dict(orientation="h", y=-0.25), height=420,
            )
            st.plotly_chart(fig_rc, use_container_width=True)

        with st.expander("Correlation snapshot — Last 60 vs Last 252 trading days"):
            sc1, sc2 = st.columns(2)
            for col_ref, n_days, title in [
                (sc1, 60, "Last 60 Days"), (sc2, 252, "Last 252 Days")
            ]:
                snap = returns_aligned.iloc[-n_days:].corr()
                fig_snap = px.imshow(
                    snap, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, text_auto=".2f",
                    title=f"Correlation — {title}", height=480,
                )
                col_ref.plotly_chart(fig_snap, use_container_width=True)

# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "Data sourced from Yahoo Finance via yfinance.  "
    "All VaR figures are 1-day holding-period estimates.  "
    "Not financial advice."
)

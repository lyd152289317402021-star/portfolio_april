from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay

TRADING_DAYS = 252
COMMON_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX",
    "AVGO", "AMD", "JPM", "V", "MA", "COST", "XOM", "JNJ", "UNH",
    "SPY", "QQQ", "DIA", "VOO", "BRK-B", "BABA", "TSM", "PDD", "NIO",
    "PLTR", "SMCI", "ARM", "MSTR", "TLT", "GLD", "SLV", "XLF", "XLK"
]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="April的观测站", page_icon="📈", layout="wide")


def normalize_ticker(value):
    return str(value).strip().upper().replace(" ", "")


def normalize_tickers(values):
    out = []
    for value in values:
        ticker = normalize_ticker(value)
        if ticker and ticker not in out:
            out.append(ticker)
    return out


@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date, end_date):
    tickers = normalize_tickers(tickers)
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(1):
            prices = raw.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame()
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices.columns = [normalize_ticker(c) for c in prices.columns]
    prices = prices.sort_index().ffill().dropna(how="all")
    return prices


def compute_rebalance_mask(index, mode):
    index = pd.DatetimeIndex(index)
    mask = pd.Series(False, index=index)
    if len(index) == 0:
        return mask
    mask.iloc[0] = True
    if mode == "Buy & Hold":
        return mask
    if mode == "Weekly":
        marker = index.to_period("W")
    elif mode == "Monthly":
        marker = index.to_period("M")
    elif mode == "Quarterly":
        marker = index.to_period("Q")
    else:
        return mask
    mask.iloc[1:] = marker[1:] != marker[:-1]
    return mask


@st.cache_data(show_spinner=False)
def build_portfolio_history(prices, asset_weights, benchmark, rebalance_mode, initial_capital):
    assets = list(asset_weights.index)
    price_assets = prices[assets].copy()
    bench_prices = prices[benchmark].copy()
    rebalance_mask = compute_rebalance_mask(prices.index, rebalance_mode)

    shares = (initial_capital * asset_weights / price_assets.iloc[0]).astype(float)
    portfolio_values = []
    benchmark_values = []
    weight_rows = []
    rebalance_dates = []

    bench_shares = initial_capital / bench_prices.iloc[0]

    for i, dt in enumerate(prices.index):
        row_prices = price_assets.loc[dt]
        port_value = float((shares * row_prices).sum())
        bench_value = float(bench_shares * bench_prices.loc[dt])

        if i > 0 and rebalance_mask.iloc[i]:
            shares = (port_value * asset_weights / row_prices).astype(float)
            port_value = float((shares * row_prices).sum())
            rebalance_dates.append(dt)

        weights_now = (shares * row_prices) / port_value if port_value > 0 else pd.Series(0.0, index=assets)
        portfolio_values.append(port_value)
        benchmark_values.append(bench_value)
        weight_rows.append(weights_now.values)

    history = pd.DataFrame(index=prices.index)
    history["Portfolio Value"] = portfolio_values
    history[f"{benchmark} Value"] = benchmark_values
    history["Portfolio"] = history["Portfolio Value"] / initial_capital
    history[benchmark] = history[f"{benchmark} Value"] / initial_capital
    history["Portfolio Daily Return"] = history["Portfolio"].pct_change()
    history[f"{benchmark} Daily Return"] = history[benchmark].pct_change()
    history["Active Return"] = history["Portfolio Daily Return"] - history[f"{benchmark} Daily Return"]
    history["Portfolio Drawdown"] = history["Portfolio"] / history["Portfolio"].cummax() - 1
    history[f"{benchmark} Drawdown"] = history[benchmark] / history[benchmark].cummax() - 1

    weights_over_time = pd.DataFrame(weight_rows, index=prices.index, columns=assets)
    returns = prices[assets].pct_change()
    contribution = weights_over_time.shift(1).mul(returns, axis=0)
    contribution = contribution.fillna(0.0)
    cumulative_contribution = contribution.cumsum()

    return history, weights_over_time, contribution, cumulative_contribution, rebalance_dates


@st.cache_data(show_spinner=False)
def rolling_stats(history, benchmark, window, rf_rate):
    port = history["Portfolio Daily Return"].dropna()
    bench = history[f"{benchmark} Daily Return"].dropna()
    active = history["Active Return"].dropna()
    rf_daily = (1 + rf_rate) ** (1 / TRADING_DAYS) - 1

    out = pd.DataFrame(index=history.index)
    out["Portfolio Rolling Vol"] = port.rolling(window).std() * math.sqrt(TRADING_DAYS)
    out[f"{benchmark} Rolling Vol"] = bench.rolling(window).std() * math.sqrt(TRADING_DAYS)
    out["Portfolio Rolling Sharpe"] = ((port.rolling(window).mean() - rf_daily) * TRADING_DAYS) / (port.rolling(window).std() * math.sqrt(TRADING_DAYS))
    out["Rolling Tracking Error"] = active.rolling(window).std() * math.sqrt(TRADING_DAYS)
    out["Rolling Active Return"] = active.rolling(window).mean() * TRADING_DAYS
    return out.dropna(how="all")


def performance_metrics(portfolio_returns, benchmark_returns, portfolio_curve, benchmark_curve, rf_rate):
    portfolio_returns = portfolio_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    joined = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if joined.empty:
        return {}

    pr = joined.iloc[:, 0]
    br = joined.iloc[:, 1]
    curve = portfolio_curve.loc[joined.index]

    n = len(pr)
    rf_daily = (1 + rf_rate) ** (1 / TRADING_DAYS) - 1
    total_return = float(curve.iloc[-1] - 1)
    cagr = float(curve.iloc[-1] ** (TRADING_DAYS / n) - 1)
    annual_vol = float(pr.std() * math.sqrt(TRADING_DAYS))
    downside_vol = float(pr[pr < 0].std() * math.sqrt(TRADING_DAYS)) if (pr < 0).any() else np.nan
    sharpe = (cagr - rf_rate) / annual_vol if annual_vol and not np.isnan(annual_vol) else np.nan
    sortino = (cagr - rf_rate) / downside_vol if downside_vol and not np.isnan(downside_vol) else np.nan
    drawdown = curve / curve.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd and not np.isnan(max_dd) else np.nan
    beta = pr.cov(br) / br.var() if br.var() not in [0, np.nan] else np.nan
    corr = float(pr.corr(br)) if len(pr) > 1 else np.nan
    tracking_error = float((pr - br).std() * math.sqrt(TRADING_DAYS))
    info_ratio = ((pr - br).mean() * TRADING_DAYS / tracking_error) if tracking_error and not np.isnan(tracking_error) else np.nan
    alpha_daily = (pr.mean() - rf_daily) - beta * (br.mean() - rf_daily) if not np.isnan(beta) else np.nan
    alpha = alpha_daily * TRADING_DAYS if not np.isnan(alpha_daily) else np.nan
    win_rate = float((pr > 0).mean())
    best_day = float(pr.max())
    worst_day = float(pr.min())
    var_95 = float(pr.quantile(0.05))
    cvar_95 = float(pr[pr <= var_95].mean()) if (pr <= var_95).any() else np.nan

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Beta": beta,
        "Alpha": alpha,
        "Correlation": corr,
        "Tracking Error": tracking_error,
        "Information Ratio": info_ratio,
        "Win Rate": win_rate,
        "Best Day": best_day,
        "Worst Day": worst_day,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
    }


@st.cache_data(show_spinner=False)
def build_projection(portfolio_returns, benchmark_returns, portfolio_last, benchmark_last, horizon_days, n_sims, seed, benchmark):
    pr = portfolio_returns.dropna()
    br = benchmark_returns.dropna()
    if pr.empty or br.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    port_mu = pr.mean()
    port_sigma = pr.std()
    bench_mu = br.mean()
    bench_sigma = br.std()

    port_random = rng.normal(port_mu, port_sigma, size=(horizon_days, n_sims))
    bench_random = rng.normal(bench_mu, bench_sigma, size=(horizon_days, n_sims))

    port_paths = portfolio_last * np.cumprod(1 + port_random, axis=0)
    bench_paths = benchmark_last * np.cumprod(1 + bench_random, axis=0)

    future_dates = pd.bdate_range(start=pr.index[-1] + BDay(1), periods=horizon_days)
    return pd.DataFrame(
        {
            "Portfolio P10": np.percentile(port_paths, 10, axis=1),
            "Portfolio Median": np.percentile(port_paths, 50, axis=1),
            "Portfolio P90": np.percentile(port_paths, 90, axis=1),
            f"{benchmark} P10": np.percentile(bench_paths, 10, axis=1),
            f"{benchmark} Median": np.percentile(bench_paths, 50, axis=1),
            f"{benchmark} P90": np.percentile(bench_paths, 90, axis=1),
        },
        index=future_dates,
    )


@st.cache_data(show_spinner=False)
def simulate_frontier(prices, n_points, rf_rate):
    returns = prices.pct_change().dropna()
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()

    mean_returns = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    rng = np.random.default_rng(42)
    weight_matrix = rng.random((n_points, returns.shape[1]))
    weight_matrix = weight_matrix / weight_matrix.sum(axis=1, keepdims=True)
    exp_returns = weight_matrix @ mean_returns.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weight_matrix, cov.values, weight_matrix))
    sharpes = np.where(vols > 0, (exp_returns - rf_rate) / vols, np.nan)

    out = pd.DataFrame(weight_matrix, columns=returns.columns)
    out["Expected Return"] = exp_returns
    out["Volatility"] = vols
    out["Sharpe"] = sharpes
    return out


def monthly_return_table(curve):
    monthly = curve.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"Date": monthly.index, "Return": monthly.values})
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    pivot = df.pivot(index="Year", columns="Month", values="Return")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = MONTH_NAMES
    return pivot.sort_index(ascending=False)


def format_metric_value(metric, value):
    if pd.isna(value):
        return "-"
    percent_metrics = {
        "Total Return", "CAGR", "Annualized Volatility", "Max Drawdown", "Alpha",
        "Tracking Error", "Win Rate", "Best Day", "Worst Day", "VaR 95%", "CVaR 95%"
    }
    return f"{value:.2%}" if metric in percent_metrics else f"{value:.2f}"


def default_portfolio_rows():
    return pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "NVDA"],
            "Weight": [0.35, 0.35, 0.30],
        }
    )


if "portfolio_table" not in st.session_state:
    st.session_state.portfolio_table = default_portfolio_rows()
if "saved_presets" not in st.session_state:
    st.session_state.saved_presets = {}

st.title("📈 April的观测站")
st.caption("Yahoo Finance portfolio dashboard：选股、配权、回测、风险分析、情景模拟，一页看完。")

with st.sidebar:
    st.header("控制面板")
    today = date.today()
    default_start = today - timedelta(days=365 * 3)

    benchmark = st.text_input("Benchmark", value="SPY").strip().upper()
    start_date = st.date_input("开始日期", value=default_start)
    end_date = st.date_input("结束日期", value=today)
    initial_capital = st.number_input("初始资金", min_value=1000.0, value=10000.0, step=1000.0)
    rf_rate = st.number_input("无风险利率（年化）", min_value=0.0, max_value=0.2, value=0.04, step=0.005)
    rebalance_mode = st.selectbox("再平衡方式", ["Buy & Hold", "Weekly", "Monthly", "Quarterly"], index=0)
    rolling_window = st.slider("Rolling Window", min_value=21, max_value=252, value=63, step=21)
    horizon_days = st.slider("未来模拟天数", min_value=21, max_value=252, value=126, step=21)
    n_sims = st.slider("Monte Carlo 次数", min_value=200, max_value=5000, value=1200, step=100)
    frontier_points = st.slider("Frontier 随机组合数", min_value=500, max_value=5000, value=2000, step=250)
    seed = st.number_input("随机种子", min_value=0, value=42, step=1)
    auto_normalize = st.checkbox("自动归一化权重", value=True)

if start_date >= end_date:
    st.error("结束日期必须晚于开始日期。")
    st.stop()

add_col, preset_col = st.columns([1.15, 0.85])
with add_col:
    quick_add = st.multiselect(
        "Quick add tickers",
        options=COMMON_TICKERS,
        default=[],
        placeholder="快速加入常见 ticker",
    )
    if st.button("加入到组合", use_container_width=True):
        current = st.session_state.portfolio_table.copy()
        existing = normalize_tickers(current["Ticker"].tolist())
        new_items = [t for t in normalize_tickers(quick_add) if t not in existing]
        if new_items:
            extra = pd.DataFrame({"Ticker": new_items, "Weight": [0.0] * len(new_items)})
            st.session_state.portfolio_table = pd.concat([current, extra], ignore_index=True)
        st.rerun()

with preset_col:
    preset_name = st.text_input("Preset 名称", value="")
    p1, p2 = st.columns(2)
    with p1:
        if st.button("保存 Preset", use_container_width=True):
            name = preset_name.strip()
            if name:
                st.session_state.saved_presets[name] = st.session_state.portfolio_table.copy()
    with p2:
        preset_options = [""] + list(st.session_state.saved_presets.keys())
        chosen = st.selectbox("加载 Preset", options=preset_options)
        if st.button("载入", use_container_width=True) and chosen:
            st.session_state.portfolio_table = st.session_state.saved_presets[chosen].copy()
            st.rerun()

st.subheader("Portfolio Builder")
portfolio_table = st.data_editor(
    st.session_state.portfolio_table,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
        "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, step=0.01, format="%.4f"),
    },
    key="portfolio_editor",
)
st.session_state.portfolio_table = portfolio_table.copy()

clean_table = portfolio_table.copy()
clean_table["Ticker"] = clean_table["Ticker"].astype(str).map(normalize_ticker)
clean_table["Weight"] = pd.to_numeric(clean_table["Weight"], errors="coerce")
clean_table = clean_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ticker", "Weight"])
clean_table = clean_table[clean_table["Ticker"] != ""]
clean_table = clean_table.groupby("Ticker", as_index=False)["Weight"].sum()
clean_table = clean_table[clean_table["Weight"] >= 0]

if clean_table.empty:
    st.info("先在上面的表格里至少输入一只股票。")
    st.stop()

if clean_table["Weight"].sum() <= 0:
    st.error("权重总和必须大于 0。")
    st.stop()

if auto_normalize:
    clean_table["Weight"] = clean_table["Weight"] / clean_table["Weight"].sum()
else:
    if not np.isclose(clean_table["Weight"].sum(), 1.0, atol=1e-6):
        st.warning(f"当前权重和 = {clean_table['Weight'].sum():.4f}。建议总和为 1。")

weights_df = clean_table.copy()
weights_df["Weight %"] = weights_df["Weight"] * 100
asset_weights = pd.Series(weights_df["Weight"].values, index=weights_df["Ticker"].tolist())
all_tickers = normalize_tickers(weights_df["Ticker"].tolist() + [benchmark])

with st.spinner("正在从 Yahoo Finance 下载数据..."):
    prices = download_prices(all_tickers, start_date, end_date + timedelta(days=1))

if prices.empty:
    st.error("没有拿到价格数据，请检查 ticker 或日期范围。")
    st.stop()

missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.warning("这些 ticker 没有成功返回数据：" + ", ".join(missing))

valid_assets = [t for t in asset_weights.index if t in prices.columns]
if benchmark not in prices.columns:
    st.error(f"Benchmark {benchmark} 无法获取数据。")
    st.stop()
if not valid_assets:
    st.error("你选择的股票都没有成功返回价格数据。")
    st.stop()

asset_weights = asset_weights.loc[valid_assets]
asset_weights = asset_weights / asset_weights.sum()
prices = prices[valid_assets + [benchmark]].ffill().dropna()
if len(prices) < 40:
    st.error("有效历史数据太少，建议扩大时间范围或更换 ticker。")
    st.stop()

history, weights_over_time, contribution, cumulative_contribution, rebalance_dates = build_portfolio_history(
    prices,
    asset_weights,
    benchmark,
    rebalance_mode,
    float(initial_capital),
)
portfolio_returns = history["Portfolio Daily Return"]
benchmark_returns = history[f"{benchmark} Daily Return"]
portfolio_curve = history["Portfolio"]
benchmark_curve = history[benchmark]
projection = build_projection(
    portfolio_returns,
    benchmark_returns,
    portfolio_curve.iloc[-1],
    benchmark_curve.iloc[-1],
    horizon_days,
    n_sims,
    int(seed),
    benchmark,
)
rolling = rolling_stats(history, benchmark, rolling_window, rf_rate)
frontier = simulate_frontier(prices[valid_assets], frontier_points, rf_rate)
port_metrics = performance_metrics(portfolio_returns, benchmark_returns, portfolio_curve, benchmark_curve, rf_rate)
bench_metrics = performance_metrics(benchmark_returns, benchmark_returns, benchmark_curve, benchmark_curve, rf_rate)

headline = st.columns(6)
headline[0].metric("Portfolio Return", format_metric_value("Total Return", port_metrics["Total Return"]))
headline[1].metric(f"{benchmark} Return", format_metric_value("Total Return", bench_metrics["Total Return"]))
headline[2].metric("CAGR", format_metric_value("CAGR", port_metrics["CAGR"]))
headline[3].metric("Sharpe", format_metric_value("Sharpe Ratio", port_metrics["Sharpe Ratio"]))
headline[4].metric("Max Drawdown", format_metric_value("Max Drawdown", port_metrics["Max Drawdown"]))
headline[5].metric("Win Rate", format_metric_value("Win Rate", port_metrics["Win Rate"]))

left, right = st.columns([1.05, 0.95])
with left:
    show_table = weights_df.copy()
    show_table["Weight %"] = show_table["Weight %"].map(lambda x: f"{x:.2f}%")
    st.dataframe(show_table, use_container_width=True, hide_index=True)
with right:
    pie_fig = px.pie(weights_df, names="Ticker", values="Weight", title="Portfolio Weights")
    pie_fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(pie_fig, use_container_width=True)

if rebalance_dates:
    st.caption(f"再平衡方式：{rebalance_mode}｜样本期内触发 {len(rebalance_dates)} 次再平衡。")
else:
    st.caption(f"再平衡方式：{rebalance_mode}｜样本期内没有额外再平衡。")

overview_tab, risk_tab, forecast_tab, diagnostics_tab = st.tabs([
    "Overview", "Risk & Rolling", "Projection", "Diagnostics"
])

with overview_tab:
    growth_fig = go.Figure()
    growth_fig.add_trace(go.Scatter(x=history.index, y=history["Portfolio"], mode="lines", name="Portfolio"))
    growth_fig.add_trace(go.Scatter(x=history.index, y=history[benchmark], mode="lines", name=benchmark))
    growth_fig.update_layout(title="Growth of $1", xaxis_title="Date", yaxis_title="Value", hovermode="x unified")
    st.plotly_chart(growth_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        contrib_fig = go.Figure()
        for col in cumulative_contribution.columns:
            contrib_fig.add_trace(go.Scatter(x=cumulative_contribution.index, y=cumulative_contribution[col], mode="lines", stackgroup="one", name=col))
        contrib_fig.update_layout(title="Cumulative Return Contribution by Asset", xaxis_title="Date", yaxis_title="Contribution", hovermode="x unified")
        st.plotly_chart(contrib_fig, use_container_width=True)
    with c2:
        monthly_table = monthly_return_table(portfolio_curve)
        if not monthly_table.empty:
            heatmap = go.Figure(data=go.Heatmap(
                z=monthly_table.values,
                x=monthly_table.columns,
                y=monthly_table.index.astype(str),
                text=np.vectorize(lambda x: "" if pd.isna(x) else f"{x:.1%}")(monthly_table.values),
                texttemplate="%{text}",
                hovertemplate="Year %{y}<br>Month %{x}<br>Return %{z:.2%}<extra></extra>",
            ))
            heatmap.update_layout(title="Portfolio Monthly Returns Heatmap", xaxis_title="Month", yaxis_title="Year")
            st.plotly_chart(heatmap, use_container_width=True)
        else:
            st.info("当前样本期不足以生成月度收益热力图。")

with risk_tab:
    risk_left, risk_right = st.columns(2)
    with risk_left:
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(x=history.index, y=history["Portfolio Drawdown"], mode="lines", name="Portfolio"))
        dd_fig.add_trace(go.Scatter(x=history.index, y=history[f"{benchmark} Drawdown"], mode="lines", name=benchmark))
        dd_fig.update_layout(title="Drawdown Comparison", xaxis_title="Date", yaxis_title="Drawdown", hovermode="x unified")
        st.plotly_chart(dd_fig, use_container_width=True)
    with risk_right:
        if not rolling.empty:
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Scatter(x=rolling.index, y=rolling["Portfolio Rolling Vol"], mode="lines", name="Portfolio"))
            vol_fig.add_trace(go.Scatter(x=rolling.index, y=rolling[f"{benchmark} Rolling Vol"], mode="lines", name=benchmark))
            vol_fig.update_layout(title=f"{rolling_window}D Rolling Volatility", xaxis_title="Date", yaxis_title="Annualized Vol", hovermode="x unified")
            st.plotly_chart(vol_fig, use_container_width=True)
        else:
            st.info("Rolling 数据不足。")

    risk_left2, risk_right2 = st.columns(2)
    with risk_left2:
        if not rolling.empty:
            sharpe_fig = go.Figure()
            sharpe_fig.add_trace(go.Scatter(x=rolling.index, y=rolling["Portfolio Rolling Sharpe"], mode="lines", name="Portfolio"))
            sharpe_fig.add_hline(y=0)
            sharpe_fig.update_layout(title=f"{rolling_window}D Rolling Sharpe", xaxis_title="Date", yaxis_title="Sharpe", hovermode="x unified")
            st.plotly_chart(sharpe_fig, use_container_width=True)
    with risk_right2:
        if not rolling.empty:
            te_fig = go.Figure()
            te_fig.add_trace(go.Scatter(x=rolling.index, y=rolling["Rolling Tracking Error"], mode="lines", name="Tracking Error"))
            te_fig.update_layout(title=f"{rolling_window}D Rolling Tracking Error", xaxis_title="Date", yaxis_title="Tracking Error", hovermode="x unified")
            st.plotly_chart(te_fig, use_container_width=True)

with forecast_tab:
    if not projection.empty:
        proj_fig = go.Figure()
        proj_fig.add_trace(go.Scatter(x=projection.index, y=projection["Portfolio P90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        proj_fig.add_trace(go.Scatter(x=projection.index, y=projection["Portfolio P10"], mode="lines", line=dict(width=0), fill="tonexty", name="Portfolio 10-90%"))
        proj_fig.add_trace(go.Scatter(x=projection.index, y=projection["Portfolio Median"], mode="lines", name="Portfolio Median"))
        proj_fig.add_trace(go.Scatter(x=projection.index, y=projection[f"{benchmark} P90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        proj_fig.add_trace(go.Scatter(x=projection.index, y=projection[f"{benchmark} P10"], mode="lines", line=dict(width=0), fill="tonexty", name=f"{benchmark} 10-90%"))
        proj_fig.add_trace(go.Scatter(x=projection.index, y=projection[f"{benchmark} Median"], mode="lines", name=f"{benchmark} Median"))
        proj_fig.update_layout(title="Illustrative Scenario Projection", xaxis_title="Future Date", yaxis_title="Projected Value", hovermode="x unified")
        st.plotly_chart(proj_fig, use_container_width=True)
        st.caption("这个 projection 是基于历史收益率均值和波动率的情景模拟，不是严格意义上的未来预测。")
    else:
        st.info("Projection 数据不足。")

    if len(valid_assets) >= 2 and not frontier.empty:
        current_return = portfolio_returns.dropna().mean() * TRADING_DAYS
        current_vol = portfolio_returns.dropna().std() * math.sqrt(TRADING_DAYS)
        frontier_fig = px.scatter(
            frontier,
            x="Volatility",
            y="Expected Return",
            color="Sharpe",
            title="Randomized Efficient Frontier",
            hover_data=valid_assets,
        )
        frontier_fig.add_trace(go.Scatter(
            x=[current_vol], y=[current_return], mode="markers", name="Current Portfolio", marker=dict(size=12, symbol="star")
        ))
        frontier_fig.update_layout(xaxis_title="Annualized Volatility", yaxis_title="Expected Return")
        st.plotly_chart(frontier_fig, use_container_width=True)
    else:
        st.info("至少需要两只有效资产才能生成 frontier。")

with diagnostics_tab:
    d1, d2 = st.columns(2)
    with d1:
        corr = prices[valid_assets].pct_change().dropna().corr()
        if not corr.empty:
            corr_fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            ))
            corr_fig.update_layout(title="Asset Correlation Heatmap")
            st.plotly_chart(corr_fig, use_container_width=True)
    with d2:
        latest_weights = weights_over_time.iloc[-1].sort_values(ascending=False).reset_index()
        latest_weights.columns = ["Ticker", "Current Weight"]
        latest_weights["Current Weight"] = latest_weights["Current Weight"].map(lambda x: f"{x:.2%}")
        st.dataframe(latest_weights, use_container_width=True, hide_index=True)

    summary_metrics = list(port_metrics.keys())
    summary_table = pd.DataFrame(
        {
            "Metric": summary_metrics,
            "Portfolio": [format_metric_value(m, port_metrics[m]) for m in summary_metrics],
            benchmark: [format_metric_value(m, bench_metrics[m]) for m in summary_metrics],
        }
    )
    st.subheader("Performance Summary")
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

export_history = history.copy()
export_history.columns = [c.replace(" ", "_") for c in export_history.columns]
export_weights = weights_over_time.copy()
export_weights.columns = [f"Weight_{c}" for c in export_weights.columns]
export_contrib = cumulative_contribution.copy()
export_contrib.columns = [f"CumContribution_{c}" for c in export_contrib.columns]
export_all = pd.concat([export_history, export_weights, export_contrib], axis=1)

st.download_button(
    "下载结果 CSV",
    data=export_all.to_csv().encode("utf-8"),
    file_name="april_observatory_results.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("**这版新增：** editable portfolio table、preset、再平衡回测、贡献图、月度热力图、frontier、相关性热力图、更多风险指标。")

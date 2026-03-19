from datetime import date, timedelta
import math
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.offsets import BDay

try:
    import akshare as ak
except Exception:
    ak = None

try:
    import yfinance as yf
except Exception:
    yf = None

TRADING_DAYS = 252
COMMON_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "NFLX",
    "AVGO", "AMD", "JPM", "V", "MA", "COST", "XOM", "JNJ", "UNH",
    "SPY", "QQQ", "DIA", "VOO", "BRK-B", "BABA", "TSM", "PDD", "NIO",
    "PLTR", "SMCI", "ARM", "MSTR", "TLT", "GLD", "SLV", "XLF", "XLK",
    "ASML", "SOXX", "INTC", "MU", "CRM", "ADBE", "UBER",
    "0700.HK", "9988.HK", "3690.HK",
    "600519", "601318", "600036", "000001", "000858", "300750", "000300", "399006", "510300", "159915"
]
A_SHARE_INDEX_ALIASES = {
    "上证指数": "000001",
    "上证50": "000016",
    "沪深300": "000300",
    "中证500": "000905",
    "中证1000": "000852",
    "科创50": "000688",
    "深证成指": "399001",
    "创业板指": "399006",
}
MANUAL_NAME_ENTRIES = [
    {"代码": "600519", "名称": "贵州茅台", "类型": "A股", "别名": "茅台"},
    {"代码": "300750", "名称": "宁德时代", "类型": "A股", "别名": ""},
    {"代码": "000001", "名称": "平安银行", "类型": "A股", "别名": ""},
    {"代码": "601318", "名称": "中国平安", "类型": "A股", "别名": ""},
    {"代码": "600036", "名称": "招商银行", "类型": "A股", "别名": ""},
    {"代码": "000858", "名称": "五粮液", "类型": "A股", "别名": ""},
    {"代码": "510300", "名称": "沪深300ETF", "类型": "ETF", "别名": "300etf"},
    {"代码": "159915", "名称": "创业板ETF", "类型": "ETF", "别名": ""},
    {"代码": "^HSI", "名称": "恒生指数", "类型": "指数", "别名": "恒生,恒指"},
    {"代码": "0700.HK", "名称": "腾讯控股", "类型": "港股", "别名": "腾讯"},
    {"代码": "9988.HK", "名称": "阿里巴巴", "类型": "港股", "别名": "阿里"},
    {"代码": "3690.HK", "名称": "美团", "类型": "港股", "别名": "美团点评"},
    {"代码": "AAPL", "名称": "Apple", "类型": "美股", "别名": "苹果"},
    {"代码": "MSFT", "名称": "Microsoft", "类型": "美股", "别名": "微软"},
    {"代码": "NVDA", "名称": "NVIDIA", "类型": "美股", "别名": "英伟达"},
    {"代码": "TSM", "名称": "Taiwan Semiconductor", "类型": "美股", "别名": "台积电"},
]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(
    page_title="April的观测站",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #121a2b;
        --panel-2: #172033;
        --line: rgba(255,255,255,0.08);
        --text: #e5e7eb;
        --muted: #94a3b8;
        --accent: #3b82f6;
        --accent-2: #22c55e;
    }
    .stApp {
        background: linear-gradient(180deg, #09111d 0%, #0b1220 100%);
        color: var(--text);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.2rem;
        max-width: 1620px;
    }
    [data-testid="stSidebar"] {
        background: #0d1524;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
        color: var(--text);
    }
    .topbar {
        background: linear-gradient(135deg, rgba(59,130,246,.18), rgba(34,197,94,.10));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,.16);
    }
    .topbar-title {
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: .2px;
        line-height: 1.15;
        margin-bottom: 4px;
        color: #f8fafc;
    }
    .topbar-subtitle {
        font-size: 0.80rem;
        color: var(--muted);
        margin: 0;
    }
    .panel-title {
        font-size: 0.92rem;
        font-weight: 700;
        letter-spacing: .15px;
        margin: 0 0 0.55rem 0;
        color: #f8fafc;
    }
    .panel-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin: -0.15rem 0 0.55rem 0;
    }
    .compact-caption {
        color: var(--muted);
        font-size: 0.76rem;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(18,26,43,.92), rgba(13,19,33,.92));
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        min-height: 82px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.02);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.76rem !important;
        letter-spacing: .2px;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.42rem !important;
        line-height: 1.05 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(18,26,43,.84);
        border: 1px solid var(--line);
        border-radius: 18px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 12px;
        background: rgba(18,26,43,.68);
        border: 1px solid var(--line);
        color: var(--muted);
        padding: 0 16px;
        font-size: 0.82rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,.14) !important;
        color: #f8fafc !important;
        border-color: rgba(59,130,246,.35) !important;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid var(--line);
        background: linear-gradient(180deg, #1a2336 0%, #111827 100%);
        color: #f8fafc;
        font-weight: 600;
        min-height: 42px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(59,130,246,.45);
        color: #ffffff;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stDateInput > div > div,
    .stNumberInput > div > div {
        background: #0f172a;
        border-color: var(--line);
    }
    .stMultiSelect [data-baseweb="tag"] {
        background: rgba(59,130,246,.18);
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border-radius: 14px;
        overflow: hidden;
    }
    .stMarkdown p {
        color: var(--text);
    }
    .st-emotion-cache-16txtl3 h3, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h1 {
        color: #f8fafc;
    }
    
    .stat-card {
        background: linear-gradient(180deg, rgba(18,26,43,.92), rgba(13,19,33,.92));
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
        min-height: 92px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.02);
    }
    .stat-card-title {
        color: var(--muted);
        font-size: 0.74rem;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }
    .stat-card-value {
        color: #f8fafc;
        font-size: 1.18rem;
        font-weight: 800;
        line-height: 1.25;
        white-space: normal;
        word-break: break-word;
    }
    .stat-card-sub {
        color: var(--muted);
        font-size: 0.72rem;
        line-height: 1.3;
        margin-top: 0.28rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def normalize_ticker(value, market_hint="智能识别"):
    raw = str(value).strip()
    if not raw:
        return ""
    if raw in A_SHARE_INDEX_ALIASES:
        return A_SHARE_INDEX_ALIASES[raw]
    ticker = raw.upper().replace(" ", "")
    ticker = ticker.replace("，", "")
    if ticker in A_SHARE_INDEX_ALIASES:
        return A_SHARE_INDEX_ALIASES[ticker]
    if ticker.startswith(("SH", "SZ", "BJ")) and ticker[2:].isdigit() and len(ticker[2:]) == 6:
        return ticker[2:]
    if "." in ticker:
        left, right = ticker.rsplit(".", 1)
        if right in {"SS", "SZ", "BJ", "SH"} and left.isdigit() and len(left) == 6:
            return left
        if right == "HK" and left.isdigit():
            return f"{left.zfill(4)}.HK"
        return ticker
    if ticker.isdigit() and 1 <= len(ticker) <= 5 and market_hint in {"港股", "智能识别"}:
        return f"{ticker.zfill(4)}.HK"
    return ticker


def normalize_tickers(values, market_hint="智能识别"):
    out = []
    for value in values:
        ticker = normalize_ticker(value, market_hint=market_hint)
        if ticker and ticker not in out:
            out.append(ticker)
    return out


def parse_manual_tickers(text, market_hint="智能识别"):
    if not text:
        return []
    raw = text.replace("\n", ",").replace(";", ",").replace("，", ",")
    return normalize_tickers(raw.split(","), market_hint=market_hint)


def contains_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))



@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_name_universe():
    def _std_frame(df, default_type, code_candidates=None, name_candidates=None, alias_candidates=None, type_candidates=None):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=["代码", "名称", "类型", "别名"])
        code_candidates = code_candidates or ["代码", "code", "证券代码", "基金代码"]
        name_candidates = name_candidates or ["名称", "name", "证券简称", "基金简称"]
        alias_candidates = alias_candidates or ["别名", "英文名称", "拼音缩写"]
        type_candidates = type_candidates or ["类型", "基金类型", "security_type"]

        code_col = next((c for c in code_candidates if c in df.columns), None)
        name_col = next((c for c in name_candidates if c in df.columns), None)
        alias_col = next((c for c in alias_candidates if c in df.columns), None)
        type_col = next((c for c in type_candidates if c in df.columns), None)
        if code_col is None or name_col is None:
            return pd.DataFrame(columns=["代码", "名称", "类型", "别名"])

        out = pd.DataFrame({
            "代码": df[code_col].astype(str).str.strip(),
            "名称": df[name_col].astype(str).str.strip(),
            "类型": df[type_col].astype(str).str.strip() if type_col else default_type,
            "别名": df[alias_col].astype(str).str.strip() if alias_col else "",
        })
        out = out[(out["代码"] != "") & (out["名称"] != "")]
        return out

    frames = []

    manual_index = pd.DataFrame(
        [{"代码": code, "名称": name, "类型": "指数", "别名": ""} for name, code in A_SHARE_INDEX_ALIASES.items()]
    )
    frames.append(manual_index)
    frames.append(pd.DataFrame(MANUAL_NAME_ENTRIES))

    if ak is not None:
        fetchers = [
            (lambda: ak.stock_info_a_code_name(), "A股"),
            (lambda: ak.stock_zh_a_spot_em(), "A股"),
            (lambda: ak.fund_etf_spot_em(), "ETF"),
            (lambda: ak.fund_lof_spot_em(), "LOF"),
        ]
        for fetcher, default_type in fetchers:
            try:
                frames.append(_std_frame(fetcher(), default_type=default_type))
            except Exception:
                pass

    if not frames:
        return pd.DataFrame(columns=["代码", "名称", "类型", "别名", "search_name", "search_code", "search_alias", "search_blob"])

    out = pd.concat(frames, ignore_index=True)
    out["代码"] = out["代码"].astype(str).str.strip()
    out["名称"] = out["名称"].astype(str).str.strip()
    out["类型"] = out["类型"].astype(str).str.strip().replace({"": "其他"})
    out["别名"] = out["别名"].astype(str).str.strip()
    out = out[(out["代码"] != "") & (out["名称"] != "")]
    out = out.drop_duplicates(subset=["代码", "名称"], keep="first")

    def _norm_text(x):
        return (
            str(x).lower()
            .replace(" ", "")
            .replace("　", "")
            .replace("-", "")
            .replace("_", "")
        )

    out["search_name"] = out["名称"].map(_norm_text)
    out["search_code"] = out["代码"].map(_norm_text)
    out["search_alias"] = out["别名"].fillna("").map(_norm_text)
    out["search_blob"] = (
        out["search_name"] + "|" + out["search_alias"] + "|" + out["search_code"]
    )
    return out.reset_index(drop=True)


def search_name_universe(query, limit=20):
    q = str(query).strip()
    if not q:
        return pd.DataFrame(columns=["代码", "名称", "类型", "别名", "score"])

    universe = load_name_universe()
    if universe.empty:
        return pd.DataFrame(columns=["代码", "名称", "类型", "别名", "score"])

    q_norm = (
        q.lower()
        .replace(" ", "")
        .replace("　", "")
        .replace("-", "")
        .replace("_", "")
    )

    mask = universe["search_blob"].str.contains(q_norm, regex=False, na=False)
    out = universe.loc[mask, ["代码", "名称", "类型", "别名", "search_name", "search_code", "search_alias"]].copy()
    if out.empty:
        return pd.DataFrame(columns=["代码", "名称", "类型", "别名", "score"])

    out["score"] = 0
    out.loc[out["名称"] == q, "score"] += 120
    out.loc[out["代码"].str.upper() == q.upper(), "score"] += 140
    out.loc[out["search_alias"] == q_norm, "score"] += 110
    out.loc[out["search_name"].str.startswith(q_norm, na=False), "score"] += 70
    out.loc[out["search_alias"].str.startswith(q_norm, na=False), "score"] += 65
    out.loc[out["search_code"].str.startswith(q_norm, na=False), "score"] += 45
    out.loc[out["search_name"].str.contains(q_norm, regex=False, na=False), "score"] += 25
    out.loc[out["search_alias"].str.contains(q_norm, regex=False, na=False), "score"] += 20
    out.loc[out["类型"] == "A股", "score"] += 8
    out.loc[out["类型"] == "ETF", "score"] += 6
    out.loc[out["类型"] == "指数", "score"] += 4

    out = out.sort_values(["score", "代码"], ascending=[False, True])[["代码", "名称", "类型", "别名", "score"]]
    return out.head(limit).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def build_exact_name_map():
    universe = load_name_universe()
    alias_map = {}
    if universe.empty:
        return alias_map
    for _, row in universe.iterrows():
        code = str(row["代码"]).strip()
        name = str(row["名称"]).strip()
        if name and name not in alias_map:
            alias_map[name] = code
        for alias in str(row.get("别名", "")).split(","):
            alias = alias.strip()
            if alias and alias not in alias_map:
                alias_map[alias] = code
    for name, code in A_SHARE_INDEX_ALIASES.items():
        alias_map.setdefault(name, code)
    return alias_map

def resolve_name_or_ticker(value, market_hint="智能识别"):
    raw = str(value).strip()
    if not raw:
        return ""

    exact_name_map = build_exact_name_map()
    if raw in exact_name_map:
        return str(exact_name_map[raw])

    normalized = normalize_ticker(raw, market_hint=market_hint)
    if raw in A_SHARE_INDEX_ALIASES:
        return normalized

    if contains_chinese(raw) or raw != normalized:
        hits = search_name_universe(raw, limit=8)
        if not hits.empty:
            exact = hits[(hits["名称"] == raw) | (hits["别名"].fillna("").str.contains(raw, regex=False))]
            if not exact.empty:
                return str(exact.iloc[0]["代码"])
            if len(hits) == 1:
                return str(hits.iloc[0]["代码"])
            starts = hits[hits["名称"].str.startswith(raw, na=False)]
            if len(starts) == 1:
                return str(starts.iloc[0]["代码"])

    return normalized


def parse_manual_inputs(text, market_hint="智能识别"):
    if not text:
        return []
    raw = text.replace("\n", ",").replace(";", ",").replace("，", ",")
    out = []
    for item in raw.split(","):
        ticker = resolve_name_or_ticker(item, market_hint=market_hint)
        if ticker and ticker not in out:
            out.append(ticker)
    return out


def detect_market(ticker, market_hint="智能识别"):
    normalized = normalize_ticker(ticker, market_hint=market_hint)
    if market_hint in {"美股", "港股", "A股"}:
        return market_hint
    if normalized.endswith(".HK"):
        return "港股"
    if normalized.isdigit() and len(normalized) == 6:
        return "A股"
    return "美股"


def a_share_adjust_code(label):
    mapping = {"前复权": "qfq", "后复权": "hfq", "不复权": ""}
    return mapping.get(label, "qfq")


def to_yfinance_symbol(ticker, market_hint="智能识别"):
    normalized = normalize_ticker(ticker, market_hint=market_hint)
    market = detect_market(normalized, market_hint=market_hint)
    if market == "A股" and normalized.isdigit() and len(normalized) == 6:
        if normalized.startswith(("60", "68", "90")):
            return f"{normalized}.SS"
        if normalized.startswith(("00", "30")):
            return f"{normalized}.SZ"
        if normalized.startswith(("43", "83", "87", "92")):
            return f"{normalized}.BJ"
    if market == "港股" and normalized.isdigit():
        return f"{normalized.zfill(4)}.HK"
    return normalized


def to_akshare_symbol(ticker):
    normalized = normalize_ticker(ticker, market_hint="A股")
    if normalized.startswith(("SH", "SZ", "BJ")) and normalized[2:].isdigit():
        return normalized[2:]
    if "." in normalized:
        left, right = normalized.rsplit(".", 1)
        if right in {"SS", "SZ", "BJ", "SH"} and left.isdigit():
            return left
    return normalized


def _extract_close_series(df, ticker):
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    date_col = next((c for c in ["日期", "date", "时间", "净值日期"] if c in df.columns), None)
    close_col = next((c for c in ["收盘", "close", "单位净值", "累计净值", "收盘价", "最新价"] if c in df.columns), None)
    if date_col is None or close_col is None:
        return pd.Series(dtype=float, name=ticker)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    ).dropna()
    if out.empty:
        return pd.Series(dtype=float, name=ticker)
    series = out.drop_duplicates(subset=["date"]).set_index("date")["close"].sort_index()
    series.name = ticker
    return series


@st.cache_data(show_spinner=False)
def fetch_yfinance_series(ticker, start_date, end_date, market_hint):
    if yf is None:
        return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint=market_hint))
    symbol = to_yfinance_symbol(ticker, market_hint=market_hint)
    raw = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint=market_hint))
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        else:
            return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint=market_hint))
        if isinstance(prices, pd.DataFrame):
            series = prices.iloc[:, 0]
        else:
            series = prices
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if close_col is None:
            return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint=market_hint))
        series = raw[close_col]
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series.name = normalize_ticker(ticker, market_hint=market_hint)
    return series


@st.cache_data(show_spinner=False)
def fetch_akshare_stock_series(ticker, start_date, end_date, adjust_label):
    if ak is None:
        return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint="A股"))
    df = ak.stock_zh_a_hist(
        symbol=to_akshare_symbol(ticker),
        period="daily",
        start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        adjust=a_share_adjust_code(adjust_label),
    )
    return _extract_close_series(df, normalize_ticker(ticker, market_hint="A股"))


@st.cache_data(show_spinner=False)
def fetch_akshare_index_series(ticker, start_date, end_date):
    if ak is None:
        return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint="A股"))
    df = ak.index_zh_a_hist(
        symbol=to_akshare_symbol(ticker),
        period="daily",
        start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
    )
    return _extract_close_series(df, normalize_ticker(ticker, market_hint="A股"))


@st.cache_data(show_spinner=False)
def fetch_akshare_etf_series(ticker, start_date, end_date, adjust_label):
    if ak is None:
        return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint="A股"))
    df = ak.fund_etf_hist_em(
        symbol=to_akshare_symbol(ticker),
        period="daily",
        start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        adjust=a_share_adjust_code(adjust_label),
    )
    return _extract_close_series(df, normalize_ticker(ticker, market_hint="A股"))


@st.cache_data(show_spinner=False)
def fetch_akshare_lof_series(ticker, start_date, end_date, adjust_label):
    if ak is None:
        return pd.Series(dtype=float, name=normalize_ticker(ticker, market_hint="A股"))
    df = ak.fund_lof_hist_em(
        symbol=to_akshare_symbol(ticker),
        period="daily",
        start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        adjust=a_share_adjust_code(adjust_label),
    )
    return _extract_close_series(df, normalize_ticker(ticker, market_hint="A股"))


def fetch_one_series(ticker, start_date, end_date, market_hint, data_source_mode, a_share_adjust, role="asset"):
    normalized = normalize_ticker(ticker, market_hint=market_hint)
    market = detect_market(normalized, market_hint=market_hint)
    errors = []

    def _try(label, fn):
        try:
            series = fn()
            if series is not None and not series.empty:
                return series, label
        except Exception as e:
            errors.append(f"{label}: {e}")
        return None, None

    if market == "A股":
        if data_source_mode == "仅 Yahoo Finance":
            plans = [("Yahoo Finance", lambda: fetch_yfinance_series(normalized, start_date, end_date, market))]
        elif role == "benchmark":
            plans = [
                ("AKShare-指数", lambda: fetch_akshare_index_series(normalized, start_date, end_date)),
                ("AKShare-A股股票", lambda: fetch_akshare_stock_series(normalized, start_date, end_date, a_share_adjust)),
                ("AKShare-ETF", lambda: fetch_akshare_etf_series(normalized, start_date, end_date, a_share_adjust)),
                ("AKShare-LOF", lambda: fetch_akshare_lof_series(normalized, start_date, end_date, a_share_adjust)),
                ("Yahoo Finance", lambda: fetch_yfinance_series(normalized, start_date, end_date, market)),
            ]
        else:
            plans = [
                ("AKShare-A股股票", lambda: fetch_akshare_stock_series(normalized, start_date, end_date, a_share_adjust)),
                ("AKShare-ETF", lambda: fetch_akshare_etf_series(normalized, start_date, end_date, a_share_adjust)),
                ("AKShare-LOF", lambda: fetch_akshare_lof_series(normalized, start_date, end_date, a_share_adjust)),
                ("AKShare-指数", lambda: fetch_akshare_index_series(normalized, start_date, end_date)),
                ("Yahoo Finance", lambda: fetch_yfinance_series(normalized, start_date, end_date, market)),
            ]
    else:
        plans = [("Yahoo Finance", lambda: fetch_yfinance_series(normalized, start_date, end_date, market))]

    for label, fn in plans:
        series, source = _try(label, fn)
        if series is not None and source is not None:
            return series, source, errors

    return pd.Series(dtype=float, name=normalized), None, errors


@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date, end_date, benchmark, market_hint, data_source_mode, a_share_adjust):
    tickers = normalize_tickers(tickers, market_hint=market_hint)
    if not tickers:
        return pd.DataFrame(), {}, {}

    series_map = {}
    source_map = {}
    error_map = {}

    normalized_benchmark = normalize_ticker(benchmark, market_hint=market_hint)
    for ticker in tickers:
        role = "benchmark" if ticker == normalized_benchmark else "asset"
        series, source, errors = fetch_one_series(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            market_hint=market_hint,
            data_source_mode=data_source_mode,
            a_share_adjust=a_share_adjust,
            role=role,
        )
        if series is not None and not series.empty:
            series_map[ticker] = series
            source_map[ticker] = source
        elif errors:
            error_map[ticker] = errors[-1]

    if not series_map:
        return pd.DataFrame(), source_map, error_map

    prices = pd.concat(series_map.values(), axis=1)
    prices.columns = list(series_map.keys())
    prices = prices.sort_index().ffill().dropna(how="all")
    return prices, source_map, error_map


def compute_rebalance_mask(index, mode):

    index = pd.DatetimeIndex(index)
    mask = pd.Series(False, index=index)
    if len(index) == 0:
        return mask
    mask.iloc[0] = True
    if mode == "买入并持有":
        return mask
    if mode == "每周":
        marker = index.to_period("W")
    elif mode == "每月":
        marker = index.to_period("M")
    elif mode == "每季":
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


def apply_dark_figure(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(size=12, color="#e5e7eb"),
        hovermode="x unified",
        height=height,
        margin=dict(l=18, r=18, t=42, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def draw_line_chart(df, title, y_title, columns, height=360):
    fig = go.Figure()
    for col in columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title=None, yaxis_title=y_title)
    return apply_dark_figure(fig, height=height)




def local_currency_for_market(market):
    return {"美股": "USD", "港股": "HKD", "A股": "CNY"}.get(market, "USD")


def ordered_markets(markets):
    order = {"美股": 0, "港股": 1, "A股": 2}
    return sorted(list(markets), key=lambda x: order.get(x, 99))


def display_metric_name(metric):
    mapping = {
        "Total Return": "累计收益",
        "CAGR": "年化收益",
        "Annualized Volatility": "年化波动率",
        "Sharpe Ratio": "Sharpe",
        "Sortino Ratio": "Sortino",
        "Max Drawdown": "最大回撤",
        "Calmar Ratio": "Calmar",
        "Beta": "Beta",
        "Alpha": "Alpha",
        "Correlation": "相关性",
        "Tracking Error": "跟踪误差",
        "Information Ratio": "信息比率",
        "Win Rate": "胜率",
        "Best Day": "最好单日",
        "Worst Day": "最差单日",
        "VaR 95%": "VaR 95%",
        "CVaR 95%": "CVaR 95%",
    }
    return mapping.get(metric, metric)


def render_stat_card(title, value, subtitle=""):
    subtitle_html = f"<div class='stat-card-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-card-title'>{title}</div>
            <div class='stat-card-value'>{value}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def fetch_direct_yf_series(symbol, start_date, end_date):
    if yf is None:
        return pd.Series(dtype=float, name=symbol)
    raw = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=symbol)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            data = raw["Close"]
        elif "Adj Close" in raw.columns.get_level_values(0):
            data = raw["Adj Close"]
        else:
            return pd.Series(dtype=float, name=symbol)
        series = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if close_col is None:
            return pd.Series(dtype=float, name=symbol)
        series = raw[close_col]
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series.name = symbol
    return series


@st.cache_data(show_spinner=False)
def download_prices_with_roles(tickers, benchmark_like_tickers, start_date, end_date, market_hint, data_source_mode, a_share_adjust):
    tickers = normalize_tickers(list(tickers), market_hint=market_hint)
    benchmark_like_tickers = set(normalize_tickers(list(benchmark_like_tickers), market_hint=market_hint))
    if not tickers:
        return pd.DataFrame(), {}, {}

    series_map = {}
    source_map = {}
    error_map = {}

    for ticker in tickers:
        role = "benchmark" if ticker in benchmark_like_tickers else "asset"
        series, source, errors = fetch_one_series(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            market_hint=market_hint,
            data_source_mode=data_source_mode,
            a_share_adjust=a_share_adjust,
            role=role,
        )
        if series is not None and not series.empty:
            series_map[ticker] = series
            source_map[ticker] = source
        elif errors:
            error_map[ticker] = errors[-1]

    if not series_map:
        return pd.DataFrame(), source_map, error_map

    prices = pd.concat(series_map.values(), axis=1)
    prices.columns = list(series_map.keys())
    prices = prices.sort_index().dropna(how="all")
    return prices, source_map, error_map


def choose_anchor_index(raw_prices, tickers, ticker_market_map, anchor_mode):
    available_cols = [c for c in tickers if c in raw_prices.columns]
    if not available_cols:
        return pd.DatetimeIndex([])

    def _index_for(cols):
        if not cols:
            return pd.DatetimeIndex([])
        frame = raw_prices[cols]
        return frame.dropna(how="all").index

    us_cols = [c for c in available_cols if ticker_market_map.get(c) == "美股"]
    hk_cols = [c for c in available_cols if ticker_market_map.get(c) == "港股"]
    cn_cols = [c for c in available_cols if ticker_market_map.get(c) == "A股"]
    all_idx = _index_for(available_cols)

    if anchor_mode == "所有交易日":
        return all_idx
    if anchor_mode == "美股收盘日":
        idx = _index_for(us_cols)
        return idx if len(idx) else all_idx
    if anchor_mode == "A股收盘日":
        idx = _index_for(cn_cols)
        return idx if len(idx) else all_idx
    if anchor_mode == "港股收盘日":
        idx = _index_for(hk_cols)
        return idx if len(idx) else all_idx

    for cols in [us_cols, hk_cols, cn_cols]:
        idx = _index_for(cols)
        if len(idx):
            return idx
    return all_idx


def build_fx_frame(index, base_currency, required_currencies, start_date, end_date):
    index = pd.DatetimeIndex(index)
    fx = pd.DataFrame(index=index)
    for cur in required_currencies:
        fx[cur] = np.nan
    if not len(index):
        return fx
    if base_currency not in required_currencies:
        required_currencies = set(required_currencies) | {base_currency}
    if base_currency == "USD":
        fx["USD"] = 1.0
        if "CNY" in required_currencies:
            usdcny = fetch_direct_yf_series("USDCNY=X", start_date, end_date).reindex(index).ffill().bfill()
            fx["CNY"] = 1 / usdcny
        if "HKD" in required_currencies:
            hkdusd = fetch_direct_yf_series("HKDUSD=X", start_date, end_date).reindex(index).ffill().bfill()
            fx["HKD"] = hkdusd
    elif base_currency == "CNY":
        fx["CNY"] = 1.0
        usdcny = fetch_direct_yf_series("USDCNY=X", start_date, end_date).reindex(index).ffill().bfill()
        if "USD" in required_currencies:
            fx["USD"] = usdcny
        if "HKD" in required_currencies:
            hkdusd = fetch_direct_yf_series("HKDUSD=X", start_date, end_date).reindex(index).ffill().bfill()
            fx["HKD"] = hkdusd * usdcny
    else:
        fx[base_currency] = 1.0
    return fx


def convert_prices_to_base(prices, ticker_market_map, base_currency, fx_frame):
    out = prices.copy()
    for col in out.columns:
        market = ticker_market_map.get(col, detect_market(col))
        cur = local_currency_for_market(market)
        if cur == base_currency:
            continue
        if cur not in fx_frame.columns:
            out[col] = np.nan
        else:
            out[col] = out[col] * fx_frame[cur].reindex(out.index).ffill().bfill()
    return out


def build_market_weight_table(asset_weights, market_map):
    rows = []
    for ticker, weight in asset_weights.items():
        rows.append({"市场": market_map.get(ticker, detect_market(ticker)), "权重": float(weight)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby("市场", as_index=False)["权重"].sum().sort_values("权重", ascending=False)

if "portfolio_table" not in st.session_state:
    st.session_state.portfolio_table = default_portfolio_rows()
if "saved_presets" not in st.session_state:
    st.session_state.saved_presets = {}
if "quick_add_manual" not in st.session_state:
    st.session_state.quick_add_manual = ""
if "name_search_query" not in st.session_state:
    st.session_state.name_search_query = ""

with st.sidebar:
    st.markdown("### 参数")

    today = date.today()
    default_start = today - timedelta(days=365 * 3)

    with st.expander("基础", expanded=True):
        market_mode = st.selectbox("市场模式", ["智能识别", "美股", "港股", "A股"], index=0, help="混合组合建议选择“智能识别”；纯A股组合建议选“A股”。")
        data_source_mode = st.selectbox("数据源", ["智能混合（推荐）", "仅 Yahoo Finance"], index=0, help="智能混合会对A股优先使用 AKShare，对美股/港股继续使用 Yahoo Finance。")
        a_share_adjust = st.selectbox("A股复权", ["前复权", "后复权", "不复权"], index=0, help="A股使用 AKShare 时生效。")
        benchmark = normalize_ticker(
            st.text_input("基准代码", value="SPY", help="A股可直接输入：沪深300 / 000300 / 399006；美股如 SPY；港股如 0700.HK。").strip(),
            market_hint=market_mode,
        )
        start_date = st.date_input("开始", value=default_start)
        end_date = st.date_input("结束", value=today)
        initial_capital = st.number_input("初始资金", min_value=1000.0, value=10000.0, step=1000.0)
        rf_rate = st.number_input("无风险利率", min_value=0.0, max_value=0.2, value=0.04, step=0.005)
        rebalance_mode = st.selectbox("再平衡", ["买入并持有", "每周", "每月", "每季"], index=0)
        auto_normalize = st.checkbox("自动归一化权重", value=True)

    with st.expander("高级", expanded=False):
        rolling_window = st.slider("滚动窗口", min_value=21, max_value=252, value=63, step=21)
        horizon_days = st.slider("未来模拟天数", min_value=21, max_value=252, value=126, step=21)
        n_sims = st.slider("蒙特卡洛次数", min_value=200, max_value=5000, value=1200, step=100)
        frontier_points = st.slider("前沿随机组合数", min_value=500, max_value=5000, value=2000, step=250)
        seed = st.number_input("随机种子", min_value=0, value=42, step=1)

    if data_source_mode == "智能混合（推荐）" and ak is None:
        st.warning("当前环境未安装 AKShare；A股会自动回退到 Yahoo Finance。")


st.markdown(
    """
    <div class="topbar">
        <div class="topbar-title">April 的观测站 · 组合终端</div>
        <p class="topbar-subtitle">选股 · 配权 · 回测 · 风险 · 情景模拟 · 支持美股 / 港股 / A股</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if start_date >= end_date:
    st.error("结束日期必须晚于开始日期。")
    st.stop()

left_col, right_col = st.columns([1.25, 0.75])

with left_col:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>1) 搜索并添加股票</div>", unsafe_allow_html=True)
        query_input = st.text_input(
            "输入代码或名称",
            value=st.session_state.quick_add_manual,
            placeholder="输入代码或名称，例如：AAPL、TSM、0700.HK、600519、贵州茅台、沪深300",
            label_visibility="collapsed",
        )
        st.session_state.quick_add_manual = query_input
        st.session_state.name_search_query = query_input

        name_hits = search_name_universe(query_input, limit=15) if query_input.strip() else pd.DataFrame()
        hit_map = {}
        hit_options = []
        if isinstance(name_hits, pd.DataFrame) and not name_hits.empty:
            for _, row in name_hits.iterrows():
                label = f"{row['名称']} ｜ {row['代码']} ｜ {row['类型']}"
                hit_options.append(label)
                hit_map[label] = str(row["代码"])

        if query_input.strip() and hit_options:
            st.caption("搜索结果")
            selected_name_hits = st.multiselect(
                "搜索结果",
                options=hit_options,
                default=[],
                placeholder="搜索结果会显示在这里，可直接勾选加入",
                label_visibility="collapsed",
            )
        else:
            selected_name_hits = []
            if query_input.strip():
                st.caption("未找到匹配名称；你仍可直接加入输入的代码或名称。")
            else:
                st.caption("支持代码直接输入；名称搜索会在下方自动出现结果。")

        add_col, reset_col, clear_col = st.columns(3)
        if add_col.button("加入", use_container_width=True):
            manual_items = parse_manual_inputs(st.session_state.quick_add_manual, market_hint=market_mode)
            search_items = [hit_map[label] for label in selected_name_hits if label in hit_map]
            pending = normalize_tickers(manual_items + search_items, market_hint=market_mode)
            current = st.session_state.portfolio_table.copy()
            existing = [resolve_name_or_ticker(x, market_hint=market_mode) for x in current["Ticker"].tolist()]
            existing = normalize_tickers(existing, market_hint=market_mode)
            new_items = [t for t in pending if t not in existing]
            if new_items:
                extra = pd.DataFrame({"Ticker": new_items, "Weight": [0.0] * len(new_items)})
                st.session_state.portfolio_table = pd.concat([current, extra], ignore_index=True)
            st.session_state.quick_add_manual = ""
            st.session_state.name_search_query = ""
            st.rerun()
        if reset_col.button("恢复默认", use_container_width=True):
            st.session_state.portfolio_table = default_portfolio_rows()
            st.rerun()
        if clear_col.button("清空", use_container_width=True):
            st.session_state.portfolio_table = pd.DataFrame({"Ticker": [], "Weight": []})
            st.rerun()

with right_col:
    with st.container(border=True):
        st.markdown("<div class='panel-title'>2) 方案</div>", unsafe_allow_html=True)
        preset_name = st.text_input("方案名称", value="", placeholder="例如：美股科技 / 红利 / A股白马")
        preset_options = list(st.session_state.saved_presets.keys())
        chosen = st.selectbox("方案", options=[""] + preset_options, label_visibility="collapsed")
        p1, p2 = st.columns(2)
        if p1.button("保存", use_container_width=True):
            name = preset_name.strip()
            if name:
                st.session_state.saved_presets[name] = st.session_state.portfolio_table.copy()
        if p2.button("载入", use_container_width=True) and chosen:
            st.session_state.portfolio_table = st.session_state.saved_presets[chosen].copy()
            st.rerun()
        p3, p4 = st.columns(2)
        if p3.button("删除", use_container_width=True) and chosen:
            st.session_state.saved_presets.pop(chosen, None)
            st.rerun()
        if p4.button("清空方案", use_container_width=True):
            st.session_state.saved_presets = {}
            st.rerun()

with st.container(border=True):
    st.markdown("<div class='panel-title'>持仓编辑</div>", unsafe_allow_html=True)
    portfolio_table = st.data_editor(
        st.session_state.portfolio_table,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        height=240,
        column_config={
            "Ticker": st.column_config.TextColumn("代码", width="medium"),
            "Weight": st.column_config.NumberColumn("权重", min_value=0.0, step=0.01, format="%.4f"),
        },
        key="portfolio_editor",
    )
    st.session_state.portfolio_table = portfolio_table.copy()


clean_table = portfolio_table.copy()
clean_table["Ticker"] = clean_table["Ticker"].astype(str).map(lambda x: resolve_name_or_ticker(x, market_hint=market_mode))
clean_table["Weight"] = pd.to_numeric(clean_table["Weight"], errors="coerce")
clean_table = clean_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["Ticker", "Weight"])
clean_table = clean_table[clean_table["Ticker"] != ""]
clean_table = clean_table.groupby("Ticker", as_index=False)["Weight"].sum()
clean_table = clean_table[clean_table["Weight"] >= 0]

if clean_table.empty:
    st.info("请先输入至少一只股票。")
    st.stop()

if clean_table["Weight"].sum() <= 0:
    st.error("权重总和必须大于 0。")
    st.stop()

raw_weight_sum = float(clean_table["Weight"].sum())
if auto_normalize:
    clean_table["Weight"] = clean_table["Weight"] / clean_table["Weight"].sum()
else:
    if not np.isclose(raw_weight_sum, 1.0, atol=1e-6):
        st.warning(f"当前权重合计 = {raw_weight_sum:.4f}。")

with st.sidebar:
    with st.expander("全局视图", expanded=False):
        base_currency = st.selectbox("统一计价货币", ["USD", "CNY"], index=0, help="跨市场组合会先折算到统一货币，再计算组合净值与风险。")
        anchor_mode = st.selectbox(
            "日期对齐",
            ["自动（混合组合优先美股）", "美股收盘日", "A股收盘日", "港股收盘日", "所有交易日"],
            index=0,
            help="混合组合默认优先用美股收盘日；A股使用同日收盘价并向前填充非交易日。",
        )
        us_market_benchmark = normalize_ticker(st.text_input("美股市场基准", value="VOO"), market_hint="美股")
        hk_market_benchmark = normalize_ticker(st.text_input("港股市场基准", value="2800.HK"), market_hint="港股")
        cn_market_benchmark = normalize_ticker(st.text_input("A股市场基准", value="000300"), market_hint="A股")

weights_df = clean_table.copy()
weights_df["Weight %"] = weights_df["Weight"] * 100
asset_weights = pd.Series(weights_df["Weight"].values, index=weights_df["Ticker"].tolist())
asset_market_map = {ticker: detect_market(ticker, market_hint=market_mode) for ticker in asset_weights.index}
present_markets = ordered_markets(set(asset_market_map.values()))
market_benchmark_map = {
    "美股": us_market_benchmark,
    "港股": hk_market_benchmark,
    "A股": cn_market_benchmark,
}
comparison_benchmarks = {m: normalize_ticker(market_benchmark_map[m], market_hint=m) for m in present_markets}

all_tickers = normalize_tickers(
    weights_df["Ticker"].tolist() + [benchmark] + list(comparison_benchmarks.values()),
    market_hint=market_mode,
)
benchmark_like = normalize_tickers([benchmark] + list(comparison_benchmarks.values()), market_hint=market_mode)

summary_cols = st.columns(4)
with summary_cols[0]:
    render_stat_card("组合股票数", f"{len(weights_df)}")
with summary_cols[1]:
    render_stat_card("权重合计", f"{weights_df['Weight'].sum():.2%}")
with summary_cols[2]:
    render_stat_card("全局基准", benchmark, f"统一计价：{base_currency}")
with summary_cols[3]:
    render_stat_card("回测区间", f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

spinner_text = "正在获取行情数据..."
if data_source_mode == "智能混合（推荐）":
    spinner_text = "正在获取行情数据（A股优先使用 AKShare）..."
with st.spinner(spinner_text):
    raw_prices, source_map, error_map = download_prices_with_roles(
        all_tickers,
        benchmark_like,
        start_date,
        end_date + timedelta(days=1),
        market_mode,
        data_source_mode,
        a_share_adjust,
    )

if raw_prices.empty:
    st.error("没有拿到价格数据，请检查代码、日期范围，或确认本地已安装可用数据源。")
    st.stop()

all_market_map = {ticker: detect_market(ticker, market_hint="智能识别") for ticker in all_tickers}
anchor_key = {
    "自动（混合组合优先美股）": "自动",
    "美股收盘日": "美股收盘日",
    "A股收盘日": "A股收盘日",
    "港股收盘日": "港股收盘日",
    "所有交易日": "所有交易日",
}[anchor_mode]
anchor_index = choose_anchor_index(raw_prices, all_tickers, all_market_map, anchor_key)
if len(anchor_index) == 0:
    st.error("无法生成有效的时间轴，请检查代码与日期范围。")
    st.stop()

required_currencies = {local_currency_for_market(all_market_map[t]) for t in all_tickers}
fx_frame = build_fx_frame(anchor_index, base_currency, required_currencies, start_date, end_date + timedelta(days=1))
if len(required_currencies - {base_currency}) and fx_frame.drop(columns=[base_currency], errors="ignore").isna().all().all():
    st.error("跨市场货币折算失败，请检查 yfinance 网络连接，或暂时切换到单一市场组合。")
    st.stop()

aligned_local = raw_prices.reindex(anchor_index).sort_index().ffill()
prices_full = convert_prices_to_base(aligned_local, all_market_map, base_currency, fx_frame)

if source_map:
    source_count = pd.Series(source_map).value_counts()
    st.caption("数据源分布：" + " ｜ ".join([f"{k}：{v}只" for k, v in source_count.items()]))

if error_map:
    with st.expander("查看抓取失败详情", expanded=False):
        error_df = pd.DataFrame({"代码": list(error_map.keys()), "错误": list(error_map.values())})
        st.dataframe(error_df, use_container_width=True, hide_index=True)

missing = [t for t in all_tickers if t not in prices_full.columns or prices_full[t].dropna().empty]
if missing:
    st.warning("这些代码没有成功返回数据：" + ", ".join(missing))

valid_assets = [t for t in asset_weights.index if t in prices_full.columns and prices_full[t].dropna().any()]
if benchmark not in prices_full.columns or prices_full[benchmark].dropna().empty:
    st.error(f"全局基准 {benchmark} 无法获取数据。")
    st.stop()
if not valid_assets:
    st.error("你选择的股票都没有成功返回价格数据。")
    st.stop()

asset_weights = asset_weights.loc[valid_assets]
asset_weights = asset_weights / asset_weights.sum()
prices = prices_full[valid_assets + [benchmark]].dropna()
if len(prices) < 40:
    st.error("有效历史数据太少，建议扩大时间范围或更换代码。")
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
market_weight_table = build_market_weight_table(asset_weights, asset_market_map)

market_rows = []
market_histories = {}
market_return_series = {}
for market in present_markets:
    market_assets = [t for t in valid_assets if asset_market_map.get(t) == market]
    if not market_assets:
        continue
    market_benchmark = comparison_benchmarks.get(market)
    if market_benchmark not in prices_full.columns or prices_full[market_benchmark].dropna().empty:
        continue
    market_prices = prices_full[market_assets + [market_benchmark]].dropna()
    if len(market_prices) < 25:
        continue
    sleeve_weight = float(asset_weights.loc[market_assets].sum())
    sleeve_weights = asset_weights.loc[market_assets] / sleeve_weight
    m_history, _, _, _, _ = build_portfolio_history(
        market_prices,
        sleeve_weights,
        market_benchmark,
        rebalance_mode,
        1.0,
    )
    m_metrics = performance_metrics(
        m_history["Portfolio Daily Return"],
        m_history[f"{market_benchmark} Daily Return"],
        m_history["Portfolio"],
        m_history[market_benchmark],
        rf_rate,
    )
    market_histories[market] = {
        "history": m_history,
        "benchmark": market_benchmark,
        "assets": market_assets,
        "weight": sleeve_weight,
        "metrics": m_metrics,
    }
    market_return_series[market] = m_history["Portfolio Daily Return"].rename(market)
    market_rows.append(
        {
            "市场": market,
            "市场权重": sleeve_weight,
            "股票数": len(market_assets),
            "组合收益": m_metrics.get("Total Return", np.nan),
            "市场基准收益": performance_metrics(
                m_history[f"{market_benchmark} Daily Return"],
                m_history[f"{market_benchmark} Daily Return"],
                m_history[market_benchmark],
                m_history[market_benchmark],
                rf_rate,
            ).get("Total Return", np.nan),
            "超额收益": m_metrics.get("Total Return", np.nan) - performance_metrics(
                m_history[f"{market_benchmark} Daily Return"],
                m_history[f"{market_benchmark} Daily Return"],
                m_history[market_benchmark],
                m_history[market_benchmark],
                rf_rate,
            ).get("Total Return", np.nan),
            "Sharpe": m_metrics.get("Sharpe Ratio", np.nan),
            "最大回撤": m_metrics.get("Max Drawdown", np.nan),
            "基准代码": market_benchmark,
        }
    )

market_rank_df = pd.DataFrame(market_rows)
if not market_rank_df.empty:
    market_rank_df = market_rank_df.sort_values(["Sharpe", "组合收益"], ascending=[False, False]).reset_index(drop=True)

market_corr = pd.DataFrame()
if market_return_series:
    market_return_df = pd.concat(market_return_series.values(), axis=1).dropna(how="all")
    if not market_return_df.empty:
        market_corr = market_return_df.corr()

core_kpi_cols = st.columns(6)
with core_kpi_cols[0]:
    render_stat_card("组合收益", format_metric_value("Total Return", port_metrics["Total Return"]))
with core_kpi_cols[1]:
    render_stat_card(f"{benchmark} 收益", format_metric_value("Total Return", bench_metrics["Total Return"]))
with core_kpi_cols[2]:
    render_stat_card("年化收益", format_metric_value("CAGR", port_metrics["CAGR"]))
with core_kpi_cols[3]:
    render_stat_card("Sharpe", format_metric_value("Sharpe Ratio", port_metrics["Sharpe Ratio"]))
with core_kpi_cols[4]:
    render_stat_card("年化波动率", format_metric_value("Annualized Volatility", port_metrics["Annualized Volatility"]))
with core_kpi_cols[5]:
    render_stat_card("最大回撤", format_metric_value("Max Drawdown", port_metrics["Max Drawdown"]))

if rebalance_dates:
    st.caption(f"再平衡：{rebalance_mode} ｜ 次数：{len(rebalance_dates)} ｜ 货币统一到 {base_currency} ｜ 日期对齐：{anchor_mode}")
else:
    st.caption(f"再平衡：{rebalance_mode} ｜ 样本期内无额外再平衡 ｜ 货币统一到 {base_currency} ｜ 日期对齐：{anchor_mode}")

portfolio_tab, attribution_tab, risk_tab, scenario_tab, diagnostics_tab = st.tabs([
    "组合总览", "市场分片", "风险", "情景模拟", "诊断"
])

with portfolio_tab:
    top_left, top_right = st.columns([1.3, 0.7])
    with top_left:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>全局净值曲线</div>", unsafe_allow_html=True)
            curve_fig = draw_line_chart(history, f"组合 vs {benchmark}", f"净值（{base_currency}）", ["Portfolio", benchmark], height=360)
            st.plotly_chart(curve_fig, use_container_width=True)
    with top_right:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>市场配置</div>", unsafe_allow_html=True)
            if not market_weight_table.empty:
                alloc_fig = px.bar(
                    market_weight_table.sort_values("权重", ascending=False),
                    x="市场",
                    y="权重",
                    text="权重",
                )
                alloc_fig.update_traces(texttemplate="%{text:.1%}")
                alloc_fig.update_layout(xaxis_title=None, yaxis_title="权重")
                st.plotly_chart(apply_dark_figure(alloc_fig, height=360), use_container_width=True)
            else:
                st.info("暂无市场配置数据。")

    mid_left, mid_right = st.columns([1.05, 0.95])
    with mid_left:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>跨市场相关性矩阵</div>", unsafe_allow_html=True)
            if not market_corr.empty:
                corr_fig = go.Figure(data=go.Heatmap(
                    z=market_corr.values,
                    x=market_corr.columns,
                    y=market_corr.index,
                    zmin=-1,
                    zmax=1,
                    text=np.round(market_corr.values, 2),
                    texttemplate="%{text}",
                    hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                    colorscale="RdBu",
                    reversescale=True,
                ))
                st.plotly_chart(apply_dark_figure(corr_fig, height=340), use_container_width=True)
            else:
                st.info("至少需要两个市场分片才能生成相关性矩阵。")
    with mid_right:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>全局核心指标</div>", unsafe_allow_html=True)
            global_summary = pd.DataFrame(
                {
                    "指标": ["累计收益", "年化收益", "年化波动率", "Sharpe", "最大回撤", "跟踪误差", "信息比率", "胜率"],
                    "组合": [
                        format_metric_value("Total Return", port_metrics.get("Total Return", np.nan)),
                        format_metric_value("CAGR", port_metrics.get("CAGR", np.nan)),
                        format_metric_value("Annualized Volatility", port_metrics.get("Annualized Volatility", np.nan)),
                        format_metric_value("Sharpe Ratio", port_metrics.get("Sharpe Ratio", np.nan)),
                        format_metric_value("Max Drawdown", port_metrics.get("Max Drawdown", np.nan)),
                        format_metric_value("Tracking Error", port_metrics.get("Tracking Error", np.nan)),
                        format_metric_value("Information Ratio", port_metrics.get("Information Ratio", np.nan)),
                        format_metric_value("Win Rate", port_metrics.get("Win Rate", np.nan)),
                    ],
                    f"{benchmark}": [
                        format_metric_value("Total Return", bench_metrics.get("Total Return", np.nan)),
                        format_metric_value("CAGR", bench_metrics.get("CAGR", np.nan)),
                        format_metric_value("Annualized Volatility", bench_metrics.get("Annualized Volatility", np.nan)),
                        format_metric_value("Sharpe Ratio", bench_metrics.get("Sharpe Ratio", np.nan)),
                        format_metric_value("Max Drawdown", bench_metrics.get("Max Drawdown", np.nan)),
                        format_metric_value("Tracking Error", bench_metrics.get("Tracking Error", np.nan)),
                        format_metric_value("Information Ratio", bench_metrics.get("Information Ratio", np.nan)),
                        format_metric_value("Win Rate", bench_metrics.get("Win Rate", np.nan)),
                    ],
                }
            )
            st.dataframe(global_summary, use_container_width=True, hide_index=True, height=340)

    with st.container(border=True):
        st.markdown("<div class='panel-title'>累计收益贡献</div>", unsafe_allow_html=True)
        contrib_fig = go.Figure()
        for col in cumulative_contribution.columns:
            contrib_fig.add_trace(go.Scatter(x=cumulative_contribution.index, y=cumulative_contribution[col], mode="lines", stackgroup="one", name=col))
        contrib_fig.update_layout(yaxis_title="累计贡献")
        st.plotly_chart(apply_dark_figure(contrib_fig, height=340), use_container_width=True)

with attribution_tab:
    upper_l, upper_r = st.columns([0.9, 1.1])
    with upper_l:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>市场分片排名</div>", unsafe_allow_html=True)
            if not market_rank_df.empty:
                show_rank = market_rank_df.copy()
                for c in ["市场权重", "组合收益", "市场基准收益", "超额收益", "最大回撤"]:
                    show_rank[c] = show_rank[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
                show_rank["Sharpe"] = show_rank["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                st.dataframe(show_rank, use_container_width=True, hide_index=True, height=320)
            else:
                st.info("当前还没有足够的数据生成市场分片。")
    with upper_r:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>市场相对表现</div>", unsafe_allow_html=True)
            if market_histories:
                rel_fig = go.Figure()
                for market in ordered_markets(market_histories.keys()):
                    item = market_histories[market]
                    m_hist = item["history"]
                    bench_code = item["benchmark"]
                    rel_fig.add_trace(go.Scatter(
                        x=m_hist.index,
                        y=m_hist["Portfolio"] - m_hist[bench_code],
                        mode="lines",
                        name=f"{market} 超额",
                    ))
                rel_fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.6)")
                rel_fig.update_layout(yaxis_title="相对净值差")
                st.plotly_chart(apply_dark_figure(rel_fig, height=320), use_container_width=True)
            else:
                st.info("暂无市场相对表现数据。")

    for market in ordered_markets(market_histories.keys()):
        item = market_histories[market]
        m_hist = item["history"]
        bench_code = item["benchmark"]
        m_metrics = item["metrics"]
        m_cols = st.columns([1.25, 0.75])
        with m_cols[0]:
            with st.container(border=True):
                st.markdown(f"<div class='panel-title'>{market}：组合 vs {bench_code}</div>", unsafe_allow_html=True)
                m_fig = draw_line_chart(m_hist, f"{market} 分片", "净值", ["Portfolio", bench_code], height=300)
                st.plotly_chart(m_fig, use_container_width=True)
        with m_cols[1]:
            with st.container(border=True):
                st.markdown(f"<div class='panel-title'>{market} 关键指标</div>", unsafe_allow_html=True)
                m_table = pd.DataFrame(
                    {
                        "指标": ["市场权重", "股票数", "累计收益", "年化收益", "Sharpe", "最大回撤", "胜率"],
                        "数值": [
                            f"{item['weight']:.2%}",
                            str(len(item["assets"])),
                            format_metric_value("Total Return", m_metrics.get("Total Return", np.nan)),
                            format_metric_value("CAGR", m_metrics.get("CAGR", np.nan)),
                            format_metric_value("Sharpe Ratio", m_metrics.get("Sharpe Ratio", np.nan)),
                            format_metric_value("Max Drawdown", m_metrics.get("Max Drawdown", np.nan)),
                            format_metric_value("Win Rate", m_metrics.get("Win Rate", np.nan)),
                        ],
                    }
                )
                st.dataframe(m_table, use_container_width=True, hide_index=True, height=300)

with risk_tab:
    risk_l1, risk_r1 = st.columns(2)
    with risk_l1:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>回撤</div>", unsafe_allow_html=True)
            dd_fig = draw_line_chart(history, "回撤走势", "回撤", ["Portfolio Drawdown", f"{benchmark} Drawdown"], height=330)
            st.plotly_chart(dd_fig, use_container_width=True)
    with risk_r1:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>滚动波动率</div>", unsafe_allow_html=True)
            if not rolling.empty:
                vol_fig = draw_line_chart(rolling, f"{rolling_window}日滚动波动率", "年化波动率", ["Portfolio Rolling Vol", f"{benchmark} Rolling Vol"], height=330)
                st.plotly_chart(vol_fig, use_container_width=True)
            else:
                st.info("滚动窗口数据不足。")

    risk_l2, risk_r2 = st.columns(2)
    with risk_l2:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>滚动 Sharpe</div>", unsafe_allow_html=True)
            if not rolling.empty:
                sharpe_fig = draw_line_chart(rolling, f"{rolling_window}日滚动 Sharpe", "Sharpe", ["Portfolio Rolling Sharpe"], height=330)
                sharpe_fig.add_hline(y=0, line_dash="dot", line_color="rgba(148,163,184,0.6)")
                st.plotly_chart(sharpe_fig, use_container_width=True)
            else:
                st.info("滚动窗口数据不足。")
    with risk_r2:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>跟踪误差</div>", unsafe_allow_html=True)
            if not rolling.empty:
                te_fig = draw_line_chart(rolling, f"{rolling_window}日滚动跟踪误差", "跟踪误差", ["Rolling Tracking Error"], height=330)
                st.plotly_chart(te_fig, use_container_width=True)
            else:
                st.info("滚动窗口数据不足。")

    with st.container(border=True):
        st.markdown("<div class='panel-title'>月度收益热力图</div>", unsafe_allow_html=True)
        monthly_table = monthly_return_table(portfolio_curve)
        if not monthly_table.empty:
            heatmap = go.Figure(data=go.Heatmap(
                z=monthly_table.values,
                x=monthly_table.columns,
                y=monthly_table.index.astype(str),
                text=np.vectorize(lambda x: "" if pd.isna(x) else f"{x:.1%}")(monthly_table.values),
                texttemplate="%{text}",
                hovertemplate="Year %{y}<br>Month %{x}<br>Return %{z:.2%}<extra></extra>",
                colorscale="RdYlGn",
                zmid=0,
            ))
            heatmap.update_layout(xaxis_title="月份", yaxis_title="年份")
            st.plotly_chart(apply_dark_figure(heatmap, height=360), use_container_width=True)
        else:
            st.info("当前样本区间不足以生成月度收益热力图。")

with scenario_tab:
    proj_l, proj_r = st.columns(2)
    with proj_l:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>蒙特卡洛模拟</div>", unsafe_allow_html=True)
            if not projection.empty:
                proj_fig = go.Figure()
                proj_fig.add_trace(go.Scatter(x=projection.index, y=projection["Portfolio P90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                proj_fig.add_trace(go.Scatter(x=projection.index, y=projection["Portfolio P10"], mode="lines", line=dict(width=0), fill="tonexty", name="组合 10-90%"))
                proj_fig.add_trace(go.Scatter(x=projection.index, y=projection["Portfolio Median"], mode="lines", name="组合中位数", line=dict(width=2.2)))
                proj_fig.add_trace(go.Scatter(x=projection.index, y=projection[f"{benchmark} P90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                proj_fig.add_trace(go.Scatter(x=projection.index, y=projection[f"{benchmark} P10"], mode="lines", line=dict(width=0), fill="tonexty", name=f"{benchmark} 10-90%"))
                proj_fig.add_trace(go.Scatter(x=projection.index, y=projection[f"{benchmark} Median"], mode="lines", name=f"{benchmark} 中位数", line=dict(width=2.2)))
                proj_fig.update_layout(xaxis_title=None, yaxis_title="模拟净值")
                st.plotly_chart(apply_dark_figure(proj_fig, height=360), use_container_width=True)
            else:
                st.info("模拟数据不足。")
    with proj_r:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>有效前沿</div>", unsafe_allow_html=True)
            if len(valid_assets) >= 2 and not frontier.empty:
                current_return = portfolio_returns.dropna().mean() * TRADING_DAYS
                current_vol = portfolio_returns.dropna().std() * math.sqrt(TRADING_DAYS)
                frontier_fig = px.scatter(frontier, x="Volatility", y="Expected Return", color="Sharpe", hover_data=valid_assets)
                frontier_fig.add_trace(go.Scatter(
                    x=[current_vol], y=[current_return], mode="markers", name="当前组合", marker=dict(size=13, symbol="star")
                ))
                frontier_fig.update_layout(xaxis_title="波动率", yaxis_title="预期收益")
                st.plotly_chart(apply_dark_figure(frontier_fig, height=360), use_container_width=True)
            else:
                st.info("至少需要两只有效资产才能生成有效前沿。")

with diagnostics_tab:
    diag_l, diag_r = st.columns([1.05, 0.95])
    with diag_l:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>资产相关性</div>", unsafe_allow_html=True)
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
                    colorscale="RdBu",
                    reversescale=True,
                ))
                st.plotly_chart(apply_dark_figure(corr_fig, height=360), use_container_width=True)
            else:
                st.info("相关性数据不足。")
    with diag_r:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>市场与数据源</div>", unsafe_allow_html=True)
            market_info = pd.DataFrame(
                {
                    "代码": valid_assets,
                    "市场": [asset_market_map.get(t, detect_market(t)) for t in valid_assets],
                    "数据源": [source_map.get(t, "-") for t in valid_assets],
                }
            )
            st.dataframe(market_info, use_container_width=True, hide_index=True, height=360)

    with st.container(border=True):
        st.markdown("<div class='panel-title'>指标总览</div>", unsafe_allow_html=True)
        summary_metrics = list(port_metrics.keys())
        summary_table = pd.DataFrame(
            {
                "指标": [display_metric_name(m) for m in summary_metrics],
                "组合": [format_metric_value(m, port_metrics[m]) for m in summary_metrics],
                benchmark: [format_metric_value(m, bench_metrics[m]) for m in summary_metrics],
            }
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True, height=390)

export_history = history.copy()
export_history.columns = [c.replace(" ", "_") for c in export_history.columns]
export_weights = weights_over_time.copy()
export_weights.columns = [f"Weight_{c}" for c in export_weights.columns]
export_contrib = cumulative_contribution.copy()
export_contrib.columns = [f"CumContribution_{c}" for c in export_contrib.columns]

market_export_list = []
for market in ordered_markets(market_histories.keys()):
    m_hist = market_histories[market]["history"].copy()
    m_hist.columns = [f"{market}_{c.replace(' ', '_')}" for c in m_hist.columns]
    market_export_list.append(m_hist)

export_all = pd.concat([export_history, export_weights, export_contrib] + market_export_list, axis=1)

st.download_button(
    "导出结果 CSV",
    data=export_all.to_csv().encode("utf-8"),
    file_name="april_observatory_results.csv",
    mime="text/csv",
)

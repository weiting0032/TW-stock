# ================================
# 台股 V6 指揮中心（官方資料源版）
# ================================

import streamlit as st
import gspread
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime

# ----------------
# 基本設定
# ----------------
PORTFOLIO_SHEET_TITLE = "Streamlit TW Stock"

st.set_page_config(
    page_title="台股 V6 指揮中心（官方資料）",
    layout="wide",
    page_icon="📈"
)

# ----------------
# Google Sheet
# ----------------
@st.cache_data(ttl=600)
def load_portfolio():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        df = pd.DataFrame(sh.sheet1.get_all_records())
        df["Symbol"] = df["Symbol"].astype(str).str.zfill(4)
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0).astype(int)
        df["Cost"] = pd.to_numeric(df["Cost"], errors="coerce").fillna(0.0)
        return df
    except:
        return pd.DataFrame(columns=["Symbol","Name","Cost","Shares","Note"])

# ----------------
# 台股官方資料
# ----------------
def _roc_to_ad(date_str):
    y, m, d = date_str.split("/")
    return f"{int(y)+1911}-{m}-{d}"

@st.cache_data(ttl=3600)
def fetch_tw_price(symbol, days=500):
    year = datetime.today().year
    is_twse = int(symbol) < 8000

    if is_twse:
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={year}0101&stockNo={symbol}"
        r = requests.get(url, timeout=10).json()
        rows = r.get("data", [])
        df = pd.DataFrame(rows, columns=[
            "Date","Vol","Amt","Open","High","Low","Close","Chg","Cnt"
        ])
        df["Date"] = pd.to_datetime(df["Date"].apply(_roc_to_ad))
    else:
        url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d={year}/01"
        r = requests.get(url, timeout=10).json()
        rows = [x for x in r["aaData"] if x[0] == symbol]
        df = pd.DataFrame(rows, columns=[
            "Symbol","Name","Close","Chg","Open","High","Low","Vol","Amt","Cnt"
        ])
        df["Date"] = pd.to_datetime(r["reportDate"])

    for c in ["Open","High","Low","Close"]:
        df[c] = df[c].astype(str).str.replace(",","").astype(float)

    df = df.sort_values("Date").set_index("Date")
    return df.tail(days)

def enforce_tick(df):
    for c in ["Open","High","Low","Close"]:
        df[c] = (df[c] / 0.05).round() * 0.05
    return df

# ----------------
# 技術指標（V6）
# ----------------
def fetch_data_v6(symbol):
    df = fetch_tw_price(symbol)
    if df.empty or len(df) < 60:
        return None

    df = enforce_tick(df)

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()
    df["SMA240"] = df["Close"].rolling(240).mean()

    std = df["Close"].rolling(20).std()
    df["BB_pos"] = (df["Close"] - (df["SMA20"] - 2*std)) / (4*std + 1e-9) * 100

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain/(loss+1e-9)))

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = df["EMA12"] - df["EMA26"]
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["DIF"] - df["DEA"]

    return df

def get_v6_strategy(df):
    r, p = df.iloc[-1], df.iloc[-2]
    bull = r["Close"] > r["SMA240"] if not pd.isna(r["SMA240"]) else r["Close"] > r["SMA60"]

    score = 0
    if r["RSI"] < (40 if bull else 30): score += 1
    if r["BB_pos"] < 15: score += 1
    if r["Hist"] > p["Hist"] and r["DIF"] > 0: score += 1
    if bull: score += 1

    if score >= 3:
        return "強力買進", "#2e7d32", score
    if score == 2:
        return "分批佈局", "#43a047", score
    return ("多頭續抱" if bull else "觀望"), ("#1976d2" if bull else "#757575"), score

# ----------------
# 繪圖
# ----------------
def plot_v6_chart(df, name):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55,0.2,0.25],
        subplot_titles=(f"{name} 股價","RSI","MACD")
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="K線"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="月線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA240"], name="年線"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", row=2, col=1)

    colors = ["green" if v>=0 else "red" for v in df["Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["Hist"], marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DIF"], name="DIF"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DEA"], name="DEA"), row=3, col=1)

    fig.update_layout(height=750, xaxis_rangeslider_visible=False)
    return fig

# ----------------
# 共用分析頁
# ----------------
def render_detail(symbol):
    df = fetch_data_v6(symbol)
    if df is None:
        st.error("資料不足")
        return

    adv, col, sc = get_v6_strategy(df)
    last = df.iloc[-1]

    st.title(f"🎯 {symbol} 個股完整分析")

    c1,c2,c3 = st.columns(3)
    c1.metric("收盤價", f"{last['Close']:.2f}")
    c2.metric("RSI", f"{last['RSI']:.1f}")
    c3.metric("策略評分", sc)

    st.markdown(f"<b style='color:{col}'>{adv}</b>", unsafe_allow_html=True)
    st.plotly_chart(plot_v6_chart(df, symbol), use_container_width=True)

# ----------------
# 主畫面
# ----------------
portfolio = load_portfolio()
st.title("📊 台股 V6 指揮中心（官方資料）")

tab1, tab2 = st.tabs(["💰 低基期快篩","🔍 免庫存個股"])

with tab1:
    code = st.text_input("輸入股票代碼")
    if st.button("分析"):
        render_detail(code)

with tab2:
    code2 = st.text_input("輸入股票代碼", key="q2")
    if st.button("個股診斷"):
        render_detail(code2)

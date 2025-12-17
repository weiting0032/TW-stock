# =================================================
# 台股 V6 指揮中心（官方資料 × 跨年度完整版）
# =================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

st.set_page_config(page_title="台股 V6 指揮中心", layout="wide")

# =================================================
# 台股官方資料（跨年度）
# =================================================

def roc_to_ad(s):
    y,m,d = s.split("/")
    return f"{int(y)+1911}-{m}-{d}"

@st.cache_data(ttl=3600)
def fetch_tw_year(symbol, year):
    is_twse = int(symbol) < 8000

    try:
        if is_twse:
            url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={year}0101&stockNo={symbol}"
            r = requests.get(url, timeout=10).json()
            rows = r.get("data", [])
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=[
                "Date","Vol","Amt","Open","High","Low","Close","Chg","Cnt"
            ])
            df["Date"] = pd.to_datetime(df["Date"].apply(roc_to_ad))

        else:
            url = f"https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d={year}/01"
            r = requests.get(url, timeout=10).json()
            rows = [x for x in r["aaData"] if x[0] == symbol]
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=[
                "Symbol","Name","Close","Chg","Open","High","Low","Vol","Amt","Cnt"
            ])
            df["Date"] = pd.to_datetime(r["reportDate"])

        for c in ["Open","High","Low","Close"]:
            df[c] = df[c].astype(str).str.replace(",","").astype(float)

        return df

    except:
        return pd.DataFrame()

def enforce_tick(df):
    for c in ["Open","High","Low","Close"]:
        df[c] = (df[c] / 0.05).round() * 0.05
    return df

@st.cache_data(ttl=3600)
def fetch_tw_full(symbol, years=5):
    dfs = []
    now = datetime.today().year

    for y in range(now-years+1, now+1):
        d = fetch_tw_year(symbol, y)
        if not d.empty:
            dfs.append(d)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs)
    df = df.drop_duplicates("Date").sort_values("Date")
    df = df.set_index("Date")
    return enforce_tick(df)

# =================================================
# V6 技術指標
# =================================================

def fetch_data_v6(symbol):
    df = fetch_tw_full(symbol)
    if df.empty or len(df) < 300:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()
    df["SMA240"] = df["Close"].rolling(240).mean()

    std = df["Close"].rolling(20).std()
    df["BB_pos"] = (df["Close"]-(df["SMA20"]-2*std))/(4*std+1e-9)*100

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - 100/(1+gain/(loss+1e-9))

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = df["EMA12"] - df["EMA26"]
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["Hist"] = df["DIF"] - df["DEA"]

    return df

def get_v6_strategy(df):
    r,p = df.iloc[-1], df.iloc[-2]
    bull = r["Close"] > r["SMA240"]

    score = 0
    if r["RSI"] < (40 if bull else 30): score+=1
    if r["BB_pos"] < 20: score+=1
    if r["Hist"] > p["Hist"]: score+=1
    if bull: score+=1

    if score>=3: return "強力買進","green",score
    if score==2: return "分批布局","orange",score
    return ("多頭續抱" if bull else "觀望"),("blue" if bull else "gray"),score

# =================================================
# 圖表
# =================================================

def plot_chart(df, sym):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55,0.2,0.25])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"]), row=1,col=1)

    fig.add_trace(go.Scatter(x=df.index,y=df["SMA20"],name="SMA20"),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["SMA240"],name="SMA240"),row=1,col=1)

    fig.add_trace(go.Scatter(x=df.index,y=df["RSI"]),row=2,col=1)
    fig.add_hline(y=70,row=2,col=1); fig.add_hline(y=30,row=2,col=1)

    fig.add_trace(go.Bar(x=df.index,y=df["Hist"]),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["DIF"]),row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["DEA"]),row=3,col=1)

    fig.update_layout(height=750,xaxis_rangeslider_visible=False)
    return fig

# =================================================
# UI
# =================================================

st.title("📊 台股 V6 指揮中心（官方資料 × 完整版）")

tab1, tab2 = st.tabs(["💰 低基期 / 單股分析","🔍 快速個股查詢"])

def render(symbol):
    df = fetch_data_v6(symbol)
    if df is None:
        st.error("歷史資料不足（需約 300 根 K）")
        return

    adv,col,sc = get_v6_strategy(df)
    last = df.iloc[-1]

    c1,c2,c3 = st.columns(3)
    c1.metric("收盤價",f"{last['Close']:.2f}")
    c2.metric("RSI",f"{last['RSI']:.1f}")
    c3.metric("策略評分",sc)

    st.markdown(f"### <span style='color:{col}'>{adv}</span>",unsafe_allow_html=True)
    st.plotly_chart(plot_chart(df,symbol),use_container_width=True)

with tab1:
    s = st.text_input("股票代碼（如 3047）")
    if st.button("開始分析"):
        render(s)

with tab2:
    s2 = st.text_input("股票代碼",key="q")
    if st.button("查詢"):
        render(s2)

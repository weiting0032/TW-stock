import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import random
import numpy as np
import os
import math

# --- 0. 基礎設定與快取解決方案 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' 
st.set_page_config(page_title="台股 V6 戰情指揮中心", layout="wide", page_icon="🚀")

# 自訂 CSS
st.markdown("""
    <style>
    .metric-container { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    .stock-card { border: 1px solid #ddd; padding: 15px; border-radius: 12px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 資料存取函數 ---
def get_gsheets_client():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        return gc, sh
    except Exception:
        return None, None

@st.cache_data(ttl=600)
def load_portfolio():
    gc, sh = get_gsheets_client()
    if not sh: return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])
    try:
        df = pd.DataFrame(sh.sheet1.get_all_records())
        df.columns = ['Symbol', 'Name', 'Cost', 'Shares', 'Note']
        df['Symbol'] = df['Symbol'].astype(str).str.zfill(4)
        return df
    except:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

# --- 2. 核心數據抓取 (含防封鎖機制) ---
@st.cache_data(ttl=3600)
def get_tw_stock_map():
    url = "https://stock.wespai.com/lists"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        dfs = pd.read_html(res.text)
        for df in dfs:
            if len(df) > 100:
                data = df.iloc[:, [0, 1, 2, 14, 15]].copy()
                data.columns = ['代碼', '名稱', '產業', 'PE', 'PB']
                data['代碼'] = data['代碼'].astype(str).str.zfill(4)
                return data.set_index('代碼').to_dict('index')
    except: return {}

STOCK_MAP = get_tw_stock_map()

def fetch_data_safe(symbol, period="2y"):
    """帶有隨機延遲的數據抓取，減少被 Yahoo 封鎖機率"""
    full_sym = f"{symbol}.TW"
    time.sleep(random.uniform(0.5, 1.5)) # 隨機延遲
    df = yf.Ticker(full_sym).history(period=period)
    if df.empty:
        df = yf.Ticker(f"{symbol}.TWO").history(period=period)
    return df

# --- 3. NVDA V6 策略邏輯 ---
def calculate_v6_strategy(df):
    if len(df) < 240: return "數據不足", "#999", 0, {}
    
    # 指標計算
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['SMA240'] = df['Close'].rolling(240).mean() # 台股年線
    
    std = df['Close'].rolling(20).std()
    df['BB_pos'] = (df['Close'] - (df['SMA20'] - 2*std)) / (4*std) * 100
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['Hist'] = macd - signal

    # 策略判斷
    row = df.iloc[-1]
    prev = df.iloc[-2]
    bull_trend = row['Close'] > row['SMA240']
    
    score = 0
    if row['RSI'] < (40 if bull_trend else 30): score += 1
    if row['BB_pos'] < 15: score += 1
    if row['Hist'] > prev['Hist'] and macd.iloc[-1] > 0: score += 1
    if bull_trend: score += 1
    
    # 建議
    advice, color = "觀望", "#757575"
    if row['Close'] < row['SMA60'] and row['SMA20'] < row['SMA60']:
        advice, color = "趨勢破壞(減碼)", "#d32f2f"
    elif row['RSI'] > (78 if bull_trend else 70) or row['BB_pos'] > 85:
        advice, color = "過熱(分批獲利)", "#ef6c00"
    elif score >= 3:
        advice, color = "強力買進", "#2e7d32"
    elif score == 2:
        advice, color = "分批佈局", "#43a047"
    elif bull_trend:
        advice, color = "多頭續抱", "#1976d2"
        
    return advice, color, score, row.to_dict()

# --- 4. 介面與功能 ---
st.sidebar.title("🎮 V6 策略控制台")
portfolio = load_portfolio()

# A. 側邊欄快篩
with st.sidebar.expander("💰 低基期標的快篩"):
    p_max = st.number_input("PE 上限", value=15.0)
    b_max = st.number_input("PB 上限", value=1.5)
    if st.button("執行篩選"):
        res = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= p_max and float(v['PB']) <= b_max]
        st.write(f"找到 {len(res)} 檔標的")
        st.dataframe(pd.DataFrame([{"代碼": k, "名稱": STOCK_MAP[k]['名稱'], "PE": STOCK_MAP[k]['PE']} for k in res[:20]]))

with st.sidebar.expander("🔍 個股快篩 (免庫存)"):
    qs = st.text_input("輸入代碼 (如: 2330)")
    if qs and st.button("分析標的"):
        q_df = fetch_data_safe(qs)
        adv, col, sc, _ = calculate_v6_strategy(q_df)
        st.metric(f"{qs} 建議", adv)

# B. 總資產儀表板
st.subheader("🏦 投資組合戰情 Bar")
total_mkt, total_cost = 0, 0
valid_stocks = []

if not portfolio.empty:
    with st.spinner('正在同步全球市場數據...'):
        for _, r in portfolio.iterrows():
            try:
                df = fetch_data_safe(r['Symbol'])
                if not df.empty:
                    cp = df['Close'].iloc[-1]
                    total_mkt += cp * r['Shares']
                    total_cost += r['Cost'] * r['Shares']
                    valid_stocks.append({'r': r, 'df': df, 'cp': cp})
            except: continue

    pl = total_mkt - total_cost
    pl_pct = (pl / total_cost * 100) if total_cost > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("總資產市值", f"${total_mkt:,.0f}")
    c2.metric("總未實現損益", f"${pl:,.0f}", f"{pl_pct:.2f}%")
    c3.metric("總投入成本", f"${total_cost:,.0f}")

st.divider()

# C. 投資組合監控牆 (新增 PE/PB)
st.subheader("🚀 個股監控牆")
if valid_stocks:
    cols = st.columns(4)
    for i, item in enumerate(valid_stocks):
        with cols[i % 4]:
            adv, col, sc, last_row = calculate_v6_strategy(item['df'])
            info = STOCK_MAP.get(item['r']['Symbol'], {'PE': '-', 'PB': '-'})
            
            st.markdown(f"""
            <div class="stock-card" style="border-left: 8px solid {col}">
                <h3 style="margin:0">{item['r']['Name']} <small style="color:gray">{item['r']['Symbol']}</small></h3>
                <h2 style="margin:10px 0; color:#333">${item['cp']:.2f}</h2>
                <p style="margin:2px 0; font-size:14px"><b>本益比:</b> {info['PE']} | <b>淨值比:</b> {info['PB']}</p>
                <div style="background:{col}; color:white; padding:5px 10px; border-radius:5px; display:inline-block; margin:10px 0">
                    {adv} (評分:{sc})
                </div>
                <p style="font-size:12px; color:gray">RSI: {last_row.get('RSI',0):.1f} | BB%: {last_row.get('BB_pos',0):.1f}</p>
            </div>
            """, unsafe_allow_html=True)

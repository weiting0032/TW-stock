import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import numpy as np
import os
import math

# --- 0. 基礎設定 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' 
STOCK_MAP_FILE = 'tw_stock_map.csv'

st.set_page_config(page_title="台股戰情指揮中心 V6 (NVDA 策略整合版)", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .stDataFrame { font-size: 16px; }
    .stButton>button { height: 2em; margin: 2px; }
    .custom-table { width: 100%; border-collapse: collapse; font-size: 14px; }
    .custom-table th, .custom-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    .custom-table th { background-color: #f2f2f2; color: #333; }
    .clickable-name { color: #1976D2; cursor: pointer; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- Google Sheets 連線核心 ---
def get_gsheets_client():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        return gc, sh
    except Exception as e:
        st.error(f"⚠️ Google Sheets 連線失敗: {e}")
    return None, None

@st.cache_data(ttl=600)
def load_portfolio():
    gc, sh = get_gsheets_client()
    if sh is None: return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])
    try:
        worksheet = sh.sheet1
        df = pd.DataFrame(worksheet.get_all_records())
        if df.empty or len(df.columns) < 5:
            return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])
        df.columns = ['Symbol', 'Name', 'Cost', 'Shares', 'Note']
        df['Symbol'] = df['Symbol'].astype(str).str.zfill(4)
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').fillna(0.0)
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce').fillna(0).astype(int)
        df['Note'] = df['Note'].astype(str).fillna('')
        return df[(df['Symbol'] != '')].copy().reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

def save_portfolio(df):
    df['Name'] = df['Symbol'].apply(get_stock_name)
    gc, sh = get_gsheets_client()
    if sh is None: return False
    try:
        worksheet = sh.sheet1
        worksheet.clear()
        data_list = [df.columns.values.tolist()] + df.values.tolist()
        worksheet.update(data_list)
        st.toast("✅ 已同步至 Google Sheets！")
        return True
    except Exception as e:
        st.error(f"⚠️ 儲存失敗: {e}")
        return False

# --- 1. 股票資訊管理與快篩 ---
@st.cache_data(ttl=86400)
def get_tw_stock_map():
    url = "https://stock.wespai.com/lists"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            dfs = pd.read_html(response.text)
            for df in dfs:
                if len(df) > 100:
                    data = df.iloc[:, [0, 1, 2, 14, 15]].copy()
                    data.columns = ['代碼', '名稱', '產業類別', 'PE', 'PB']
                    data['代碼'] = data['代碼'].astype(str).str.zfill(4)
                    return data.set_index('代碼').apply(lambda x: x.to_dict(), axis=1).to_dict()
    except: pass
    return {"2330": {"名稱": "台積電", "產業類別": "半導體", "PE": 0, "PB": 0}}

TW_STOCKS = get_tw_stock_map()
STOCK_SEARCH_LIST = [f"{code} {info['名稱']}" for code, info in TW_STOCKS.items()]

def get_stock_name(symbol):
    return TW_STOCKS.get(symbol.split('.')[0], {}).get('名稱', symbol)

def low_base_screening(max_pe, max_pb):
    data_list = []
    for code, info in TW_STOCKS.items():
        if pd.notna(info.get('PE')) and pd.notna(info.get('PB')):
            if 0 < info['PE'] <= max_pe and info['PB'] <= max_pb:
                data_list.append({"代碼": code, "名稱": info['名稱'], "產業": info['產業類別'], "PE": info['PE'], "PB": info['PB']})
    return pd.DataFrame(data_list).sort_values(by=['產業', 'PE'])

# --- 2. 核心 V6 策略指標 ---
@st.cache_data(ttl=3600)
def get_stock_data(symbol_input, period="2y"):
    symbol = symbol_input.split(' ')[0] if ' ' in symbol_input else symbol_input
    full_symbol = symbol if '.' in symbol else f"{symbol}.TW"
    df = yf.Ticker(full_symbol).history(period=period)
    if df.empty and '.' not in symbol:
        df = yf.Ticker(f"{symbol}.TWO").history(period=period)
    return df

def calculate_v6_indicators(df):
    if df.empty or len(df) < 240: return df
    # 均線 (台股年線 240)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['SMA240'] = df['Close'].rolling(240).mean()
    # 布林帶與位置
    std = df['Close'].rolling(20).std()
    df['Upper'] = df['SMA20'] + 2 * std
    df['Lower'] = df['SMA20'] - 2 * std
    df['BB_pos'] = (df['Close'] - df['Lower']) / (df['Upper'] - df['Lower']) * 100
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

def get_v6_advice(df):
    if df.empty or len(df) < 240: return "數據不足", "#9e9e9e", 0
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # V6 趨勢與門檻
    bull_trend = row['Close'] > row['SMA240']
    oversold_rsi = 40 if bull_trend else 30
    overbought_rsi = 78 if bull_trend else 70

    score = 0
    if row['RSI'] < oversold_rsi: score += 1
    if row['BB_pos'] < 15: score += 1
    if row['Hist'] > prev_row['Hist'] and row['MACD'] > 0: score += 1
    if bull_trend: score += 1

    # 決策
    if (row['Close'] < row['SMA60'] and row['SMA20'] < row['SMA60']): return "趨勢破壞(建議減碼)", "#d32f2f", score
    if row['RSI'] > overbought_rsi or row['BB_pos'] > 85: return "高檔過熱(分批獲利)", "#ef6c00", score
    if score >= 3: return "強力買進", "#2e7d32", score
    if score == 2: return "分批佈局", "#43a047", score
    return "多頭續抱" if bull_trend else "觀望整理", "#1976d2" if bull_trend else "#757575", score

# --- 3. 介面渲染 ---
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = load_portfolio()

# A. 側邊欄控制
st.sidebar.title("🎛️ 指揮控制台")
with st.sidebar.expander("💰 低基期標的快篩", expanded=True):
    max_pe = st.number_input("PE 上限", value=15.0)
    max_pb = st.number_input("PB 上限", value=2.0)
    if st.button("執行快篩"):
        st.session_state.screen_df = low_base_screening(max_pe, max_pb)

with st.sidebar.expander("🔍 個股快篩 (免庫存)"):
    qs_input = st.selectbox("搜尋股票", [""] + STOCK_SEARCH_LIST)
    if st.button("分析"):
        st.session_state.qs_sym = qs_input.split(' ')[0]

# B. 主畫面：資產總覽 Bar
portfolio = st.session_state.portfolio_df
total_mkt, total_cost = 0, 0
stock_details = []

if not portfolio.empty:
    for _, r in portfolio.iterrows():
        df = get_stock_data(r['Symbol'])
        if not df.empty:
            cp = df['Close'].iloc[-1]
            total_mkt += cp * r['Shares']
            total_cost += r['Cost'] * r['Shares']
            stock_details.append({'Symbol': r['Symbol'], 'Price': cp, 'df': df})

total_pl = total_mkt - total_cost
pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0

st.subheader("🏦 投資組合總覽")
c1, c2, c3 = st.columns(3)
c1.metric("總資產市值", f"${total_mkt:,.0f}")
c2.metric("總未實現損益", f"${total_pl:,.0f}", f"{pl_pct:.2f}%")
c3.metric("總投入成本", f"${total_cost:,.0f}")
st.divider()

# C. 監控卡片 (新增 PE/PB)
st.subheader("🚀 個股監控牆")
if stock_details:
    cols = st.columns(4)
    for i, item in enumerate(stock_details):
        with cols[i % 4]:
            df_v6 = calculate_v6_indicators(item['df'])
            advice, color, score = get_v6_advice(df_v6)
            info = TW_STOCKS.get(item['Symbol'], {})
            
            st.markdown(f"""
            <div style="border:1px solid #ddd; padding:10px; border-radius:10px; border-left:8px solid {color}">
                <h4 style="margin:0">{get_stock_name(item['Symbol'])} ({item['Symbol']})</h4>
                <p style="font-size:18px; margin:5px 0"><b>現價: {item['Price']:.2f}</b></p>
                <p style="margin:0; font-size:13px; color:#555">PE: {info.get('PE','N/A')} | PB: {info.get('PB','N/A')}</p>
                <p style="color:{color}; font-weight:bold; margin-top:5px">{advice}</p>
                <p style="font-size:11px; color:#888">V6 評分: {score}/4 | RSI: {df_v6['RSI'].iloc[-1]:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("圖表", key=f"btn_{item['Symbol']}"):
                st.session_state.detail_sym = item['Symbol']

# D. 詳情圖表
if 'detail_sym' in st.session_state:
    sym = st.session_state.detail_sym
    df_plot = calculate_v6_indicators(get_stock_data(sym))
    st.subheader(f"📈 {get_stock_name(sym)} 技術分析")
    # ... (此處可加入原本的 plot_stock_chart 邏輯) ...
    st.plotly_chart(go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'])]))

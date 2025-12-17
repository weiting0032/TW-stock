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
    df_to_save = df[df['Shares'] >= 0].copy()
    gc, sh = get_gsheets_client()
    if sh is None: return False
    try:
        worksheet = sh.sheet1
        worksheet.clear()
        data_list = [df_to_save.columns.values.tolist()] + df_to_save.values.tolist()
        worksheet.update(data_list)
        st.toast("✅ 已同步至 Google Sheets！")
        return True
    except Exception as e:
        st.error(f"⚠️ 儲存失敗: {e}")
        return False

# --- 1. 股票資訊管理 ---
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
    base = symbol.split('.')[0]
    return TW_STOCKS.get(base, {}).get('名稱', symbol)

# --- 2. 數據運算與 V6 策略 ---
@st.cache_data(ttl=3600)
def get_stock_data(symbol_input, period="2y"):
    symbol = symbol_input.split(' ')[0] if ' ' in symbol_input else symbol_input
    full_symbol = symbol if '.' in symbol else f"{symbol}.TW"
    stock = yf.Ticker(full_symbol)
    df = stock.history(period=period)
    if df.empty and '.' not in symbol:
        full_symbol = f"{symbol}.TWO"
        df = yf.Ticker(full_symbol).history(period=period)
    return df, full_symbol, get_stock_name(symbol)

def calculate_indicators(df):
    if df.empty or len(df) < 240: return df
    # 均線
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['SMA240'] = df['Close'].rolling(240).mean()
    # 布林帶
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

def get_v6_strategy_suggestion(df):
    if df.empty or len(df) < 240: 
        return ("數據不足", "#9e9e9e", "需要至少 240 日數據")
    
    row = df.iloc[-1]
    prev_row = df.iloc[-2]
    price = row['Close']
    sma240 = row['SMA240']
    sma60 = row['SMA60']
    sma20 = row['SMA20']
    rsi = row['RSI']
    bb_pos = row['BB_pos']
    hist_val = row['Hist']
    prev_hist = prev_row['Hist']

    # 1. 趨勢判斷 (台股以 240MA 為年線)
    bull_trend = price > sma240
    oversold_rsi = 40 if bull_trend else 30
    overbought_rsi = 78 if bull_trend else 70

    # 2. 條件判定
    is_oversold = rsi < oversold_rsi
    is_near_lower = bb_pos < 15
    macd_turn_up = hist_val > prev_hist
    macd_above_zero = row['MACD'] > 0
    
    # 3. 買入評分
    score = 0
    if is_oversold: score += 1
    if is_near_lower: score += 1
    if macd_turn_up and macd_above_zero: score += 1
    if bull_trend: score += 1

    # 4. 決策邏輯
    status = "觀望整理"
    color = "#757575"
    
    # 賣出/防守條件
    trend_break = price < sma60 and sma20 < sma60
    is_overbought = rsi > overbought_rsi or bb_pos > 85
    
    if trend_break:
        status, color = "趨勢轉空 (建議減碼)", "#d32f2f"
    elif is_overbought:
        status, color = "高檔過熱 (建議分批獲利)", "#ef6c00"
    elif score >= 3:
        status, color = "強力買進訊號", "#2e7d32"
    elif score == 2:
        status, color = "分批佈局 (買進)", "#43a047"
    elif bull_trend and price > sma20:
        status, color = "多頭續抱", "#1976d2"

    msg = f"RSI: {rsi:.1f} | BB位置: {bb_pos:.1f}% | 評分: {score}/4 | 年線趨勢: {'多頭' if bull_trend else '空頭'}"
    return status, color, msg

# --- 3. 介面渲染 ---
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = load_portfolio()

# 側邊欄控制
st.sidebar.title("🎛️ 指揮控制台")
with st.sidebar.expander("➕ 新增/更新 監控標的"):
    search_symbol = st.selectbox("搜尋股票", [""] + STOCK_SEARCH_LIST)
    cost = st.number_input("平均成本", value=0.0)
    shares = st.number_input("持有股數", value=0, step=1000)
    note = st.text_input("備註")
    if st.button("更新清單"):
        if search_symbol:
            sym = search_symbol.split(' ')[0]
            df = st.session_state.portfolio_df
            if sym in df['Symbol'].values:
                df.loc[df['Symbol'] == sym, ['Cost', 'Shares', 'Note']] = [cost, shares, note]
            else:
                new_row = pd.DataFrame({'Symbol':[sym], 'Name':[get_stock_name(sym)], 'Cost':[cost], 'Shares':[shares], 'Note':[note]})
                st.session_state.portfolio_df = pd.concat([df, new_row], ignore_index=True)
            save_portfolio(st.session_state.portfolio_df)
            st.rerun()

# 主介面
st.subheader("🏦 投資組合監控")
portfolio = st.session_state.portfolio_df
if not portfolio.empty:
    cols = st.columns(len(portfolio) if len(portfolio) < 5 else 4)
    for i, (_, r) in enumerate(portfolio.iterrows()):
        with cols[i % 4]:
            df_stock, full_sym, name = get_stock_data(r['Symbol'])
            df_stock = calculate_indicators(df_stock)
            curr_price = df_stock['Close'].iloc[-1]
            status, color, detail = get_v6_strategy_suggestion(df_stock)
            
            pl = (curr_price - r['Cost']) * r['Shares']
            pl_pct = ((curr_price / r['Cost']) - 1) * 100 if r['Cost'] > 0 else 0
            
            st.markdown(f"""
            <div style="border:1px solid #ddd; padding:15px; border-radius:10px; border-left:8px solid {color}">
                <h3 style="margin:0">{name} ({r['Symbol']})</h3>
                <p style="font-size:20px; margin:5px 0"><b>現價: {curr_price:.2f}</b></p>
                <p style="color:{'red' if pl>=0 else 'green'}; margin:0">損益: {pl:,.0f} ({pl_pct:.2f}%)</p>
                <hr style="margin:10px 0">
                <p style="font-weight:bold; color:{color}; margin:0">{status}</p>
                <p style="font-size:12px; color:#666">{detail}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"查看圖表 {r['Symbol']}", key=f"btn_{r['Symbol']}"):
                st.session_state.detail_symbol = r['Symbol']

# 詳細分析與圖表
if 'detail_symbol' in st.session_state:
    sym = st.session_state.detail_symbol
    df_an, _, name = get_stock_data(sym)
    df_an = calculate_indicators(df_an)
    
    st.divider()
    st.subheader(f"📈 {name} ({sym}) 技術分析")
    
    chart_data = df_an.tail(150)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)
    
    # K線與均線
    fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data['Open'], high=chart_data['High'], low=chart_data['Low'], close=chart_data['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA20'], line=dict(color='orange'), name='月線(20)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA60'], line=dict(color='cyan'), name='季線(60)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA240'], line=dict(color='purple', width=2), name='年線(240)'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    colors = ['#ef5350' if v < 0 else '#66bb6a' for v in chart_data['Hist']]
    fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

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

# --- 0. 基礎設定 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' 
st.set_page_config(page_title="台股 V6 指揮中心", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 15px; border-radius: 12px; background-color: white; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .metric-bar { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 15px; margin-bottom: 25px; }
    .stButton>button { width: 100%; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 核心數據函數 (含防封鎖機制) ---
@st.cache_data(ttl=600)
def load_portfolio():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        df = pd.DataFrame(sh.sheet1.get_all_records())
        df['Symbol'] = df['Symbol'].astype(str).str.zfill(4)
        return df
    except: return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

@st.cache_data(ttl=86400)
def get_tw_map():
    url = "https://stock.wespai.com/lists"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        df = pd.read_html(res.text)[0]
        data = df.iloc[:, [0, 1, 2, 14, 15]].copy()
        data.columns = ['代碼', '名稱', '產業', 'PE', 'PB']
        data['代碼'] = data['代碼'].astype(str).str.zfill(4)
        return data.set_index('代碼').to_dict('index')
    except: return {}

STOCK_MAP = get_tw_map()

def fetch_data_v6(symbol):
    """抓取數據並計算 V6 指標（包含 MACD）"""
    time.sleep(random.uniform(0.5, 1.2)) 
    full_sym = f"{symbol}.TW"
    df = yf.Ticker(full_sym).history(period="2y")
    if df.empty or len(df) < 5: 
        df = yf.Ticker(f"{symbol}.TWO").history(period="2y")
    
    if df.empty or len(df) < 20: return None

    # 1. 均線與布林
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['SMA240'] = df['Close'].rolling(240).mean()
    std = df['Close'].rolling(20).std()
    df['BB_pos'] = (df['Close'] - (df['SMA20'] - 2*std)) / (4*std + 1e-9) * 100
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
    
    # 3. MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['DIF'] - df['DEA']
    return df

def get_v6_strategy(df):
    if df is None or len(df) < 240: return "數據不足", "#999", 0
    try:
        row = df.iloc[-1]
        prev = df.iloc[-2]
        bull = row['Close'] > row['SMA240']
        
        score = 0
        if row['RSI'] < (40 if bull else 30): score += 1
        if row['BB_pos'] < 15: score += 1
        if row['Hist'] > prev['Hist'] and row['DIF'] > 0: score += 1
        if bull: score += 1
        
        if row['Close'] < row['SMA60'] and df['SMA20'].iloc[-1] < row['SMA60']:
            return "趨勢轉空(減碼)", "#d32f2f", score
        if score >= 3: return "強力買進", "#2e7d32", score
        if score == 2: return "分批佈局", "#43a047", score
        return ("多頭續抱" if bull else "觀望整理"), ("#1976d2" if bull else "#757575"), score
    except: return "計算錯誤", "#999", 0

def plot_v6_chart(df, name):
    """繪製包含 K線、RSI、MACD 的完整圖表"""
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.5, 0.2, 0.3], 
        vertical_spacing=0.03,
        subplot_titles=(f"{name} 股價/均線", "RSI", "MACD")
    )
    
    # Row 1: K線與均線
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=2), name='年線'), row=1, col=1)
    
    # Row 2: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9370DB', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Row 3: MACD
    colors = ['#2E8B57' if v >= 0 else '#CD5C5C' for v in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DIF'], line=dict(color='#FF8C00', width=1), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DEA'], line=dict(color='#1E90FF', width=1), name='DEA'), row=3, col=1)
    
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, showlegend=True, margin=dict(t=50, b=50))
    return fig

# --- 2. 介面渲染 ---
portfolio = load_portfolio()

# A. 頂部資產 Bar
st.markdown('<div class="metric-bar">', unsafe_allow_html=True)
t_mkt, t_cost, monitored_data = 0.0, 0.0, []

if not portfolio.empty:
    with st.spinner('同步數據中...'):
        for _, r in portfolio.iterrows():
            d = fetch_data_v6(r['Symbol'])
            if d is not None:
                cp = d['Close'].iloc[-1]
                t_mkt += cp * r['Shares']
                t_cost += r['Cost'] * r['Shares']
                monitored_data.append({'r': r, 'df': d, 'cp': cp})

pl = t_mkt - t_cost
p1, p2, p3 = st.columns(3)
p1.metric("總資產市值", f"${t_mkt:,.0f}")
p2.metric("總未實現損益", f"${pl:,.0f}", f"{(pl/t_cost*100 if t_cost>0 else 0):.2f}%")
p3.metric("總投入成本", f"${t_cost:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

# B. 佈局核心
main_col, side_col = st.columns([0.65, 0.35])

with main_col:
    st.subheader("🚀 個股監控牆")
    if monitored_data:
        m_cols = st.columns(3)
        for i, item in enumerate(monitored_data):
            with m_cols[i % 3]:
                adv, col, sc = get_v6_strategy(item['df'])
                info = STOCK_MAP.get(item['r']['Symbol'], {'PE':'-', 'PB':'-'})
                st.markdown(f"""
                <div class="stock-card" style="border-top: 5px solid {col}">
                    <h4 style="margin:0">{item['r']['Name']} ({item['r']['Symbol']})</h4>
                    <p style="color:gray; font-size:12px; margin:5px 0">PE: {info['PE']} | PB: {info['PB']}</p>
                    <h2 style="margin:5px 0">${item['cp']:.1f}</h2>
                    <p style="color:{col}; font-weight:bold; margin:0">{adv} (評分:{sc})</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"顯示技術分析", key=f"btn_{item['r']['Symbol']}"):
                    st.session_state.current_plot = (item['df'], item['r']['Name'])

with side_col:
    tab1, tab2 = st.tabs(["💰 低基期快篩", "🔍 免庫存診斷"])
    with tab1:
        pe_val = st.number_input("PE 上限", value=12.0)
        pb_val = st.number_input("PB 上限", value=1.2)
        if st.button("執行篩選"):
            st.session_state.scr_res = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= pe_val and float(v['PB']) <= pb_val][:15]
        
        if 'scr_res' in st.session_state:
            st.write("--- 篩選清單 ---")
            for code in st.session_state.scr_res:
                c1, c2 = st.columns([0.7, 0.3])
                c1.write(f"**{code} {STOCK_MAP[code]['名稱']}**")
                if c2.button("圖表", key=f"scr_{code}"):
                    st.session_state.current_plot = (fetch_data_v6(code), STOCK_MAP[code]['名稱'])

    with tab2:
        target = st.text_input("輸入股票代碼", placeholder="例如: 2303")
        if st.button("診斷分析") and target:
            q_df = fetch_data_v6(target)
            if q_df is not None:
                st.session_state.current_plot = (q_df, f"查詢: {target}")
            else: st.error("查無數據")

# C. 底部圖表顯示區
if 'current_plot' in st.session_state:
    st.divider()
    plot_df, plot_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(plot_df, plot_name), use_container_width=True)

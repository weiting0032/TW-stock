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
    .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 核心數據函數 ---
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
    """抓取數據並計算 V6 指標，包含異常處理避免 ValueError"""
    time.sleep(random.uniform(0.5, 1.2)) # 防封鎖延遲
    full_sym = f"{symbol}.TW"
    df = yf.Ticker(full_sym).history(period="2y")
    if df.empty or len(df) < 5: 
        df = yf.Ticker(f"{symbol}.TWO").history(period="2y")
    
    if df.empty or len(df) < 20: return None

    # V6 指標計算
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['SMA240'] = df['Close'].rolling(240).mean()
    
    std = df['Close'].rolling(20).std()
    df['BB_pos'] = (df['Close'] - (df['SMA20'] - 2*std)) / (4*std + 1e-9) * 100 # 防止除以0
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
    
    macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Hist'] = macd - macd.ewm(span=9).mean()
    return df

def get_v6_strategy(df):
    """NVDA V6 策略核心邏輯"""
    if df is None or len(df) < 240: 
        return "數據不足", "#999", 0
    
    try:
        row = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. 趨勢判斷
        bull = row['Close'] > row['SMA240']
        
        # 2. 評分系統
        score = 0
        if row['RSI'] < (40 if bull else 30): score += 1
        if row['BB_pos'] < 15: score += 1
        if row['Hist'] > prev['Hist'] and row['Hist'] > 0: score += 1
        if bull: score += 1
        
        # 3. 決策輸出
        if row['Close'] < row['SMA60'] and df['SMA20'].iloc[-1] < row['SMA60']:
            return "趨勢轉空(減碼)", "#d32f2f", score
        if score >= 3:
            return "強力買進", "#2e7d32", score
        if score == 2:
            return "分批佈局", "#43a047", score
        return ("多頭續抱" if bull else "觀望整理"), ("#1976d2" if bull else "#757575"), score
    except:
        return "計算錯誤", "#999", 0

def plot_v6_chart(df, name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1), name='月線'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=2), name='年線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='orange'), name='RSI'), row=2, col=1)
    fig.update_layout(height=500, title=f"{name} 技術分析 (NVDA V6 策略視覺化)", xaxis_rangeslider_visible=False)
    return fig

# --- 2. 介面與顯示邏輯 ---
portfolio = load_portfolio()

# A. 頂部資產 Bar
st.markdown('<div class="metric-bar">', unsafe_allow_html=True)
t_mkt, t_cost = 0.0, 0.0
monitored_data = []

if not portfolio.empty:
    with st.spinner('同步市場數據中...'):
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

# B. 功能區塊
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
                    <h3 style="margin:0">{item['r']['Name']} <small style="font-size:12px">({item['r']['Symbol']})</small></h3>
                    <p style="color:gray; font-size:13px; margin:5px 0">PE: {info['PE']} | PB: {info['PB']}</p>
                    <h2 style="margin:5px 0; color:#333">${item['cp']:.1f}</h2>
                    <p style="color:{col}; font-weight:bold; margin:0">{adv} (V6評分:{sc})</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看圖表", key=f"btn_{item['r']['Symbol']}"):
                    st.session_state.current_plot = (item['df'], item['r']['Name'])

with side_col:
    tab1, tab2 = st.tabs(["💰 低基期快篩", "🔍 個股分析"])
    
    with tab1:
        max_pe = st.number_input("PE 上限", value=12.0)
        max_pb = st.number_input("PB 上限", value=1.2)
        if st.button("執行篩選"):
            results = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= max_pe and float(v['PB']) <= max_pb]
            st.session_state.screen_results = results[:15]

        if 'screen_results' in st.session_state:
            st.write("--- 篩選結果 ---")
            for s_code in st.session_state.screen_results:
                c1, c2 = st.columns([0.7, 0.3])
                c1.write(f"**{s_code} {STOCK_MAP[s_code]['名稱']}**")
                if c2.button("圖表", key=f"scr_{s_code}"):
                    st.session_state.current_plot = (fetch_data_v6(s_code), STOCK_MAP[s_code]['名稱'])

    with tab2:
        qs_code = st.text_input("輸入代碼 (免庫存)", placeholder="例如: 2603")
        if qs_code:
            if st.button("診斷分析"):
                q_df = fetch_data_v6(qs_code)
                if q_df is not None:
                    st.session_state.current_plot = (q_df, f"分析: {qs_code}")
                else:
                    st.error("查無數據")

# C. 底部全幅圖表區
if 'current_plot' in st.session_state:
    st.divider()
    plot_df, plot_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(plot_df, plot_name), use_container_width=True)

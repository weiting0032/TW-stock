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
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# --- 0. 基礎設定 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' 
st.set_page_config(page_title="台股戰情指揮中心 V6.3", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 15px; border-radius: 12px; background-color: white; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); margin-bottom: 12px; }
    .metric-bar { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 20px; border-radius: 15px; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .info-label { color: #666; font-size: 0.85em; }
    .info-value { font-weight: bold; color: #333; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 核心數據函數 ---

# 建立一個持久化的 Session 以避開部分 Rate Limit
@st.cache_resource
def get_session():
    session = Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@st.cache_data(ttl=600)
def load_portfolio():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        df = pd.DataFrame(sh.sheet1.get_all_records())
        df['Symbol'] = df['Symbol'].astype(str).str.zfill(4)
        return df
    except:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

@st.cache_data(ttl=86400)
def get_tw_map():
    url = "https://stock.wespai.com/lists"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        df = pd.read_html(res.text)[0]
        data = df.iloc[:, [0, 1, 2, 14, 15]].copy()
        data.columns = ['代碼', '名稱', '產業', 'PE', 'PB']
        data['代碼'] = data['代碼'].astype(str).str.zfill(4)
        return data.set_index('代碼').to_dict('index')
    except:
        return {}

STOCK_MAP = get_tw_map()

def fetch_data_v6(symbol):
    """加入 Session 與 延遲機制 解決 YFRateLimitError"""
    session = get_session()
    # 增加隨機延遲，減少被 Yahoo 判定為爬蟲
    time.sleep(random.uniform(1.0, 2.5)) 
    
    full_sym = f"{symbol}.TW"
    try:
        ticker = yf.Ticker(full_sym, session=session)
        df = ticker.history(period="2y", auto_adjust=False)
        
        if df.empty or len(df) < 10:
            df = yf.Ticker(f"{symbol}.TWO", session=session).history(period="2y", auto_adjust=False)
        
        if df.empty: return None

        # 技術指標計算
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA60'] = df['Close'].rolling(60).mean()
        df['SMA240'] = df['Close'].rolling(240).mean()
        std = df['Close'].rolling(20).std()
        df['BB_pos'] = (df['Close'] - (df['SMA20'] - 2*std)) / (4*std + 1e-9) * 100
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = df['EMA12'] - df['EMA26']
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['DIF'] - df['DEA']
        return df
    except Exception as e:
        st.warning(f"無法取得 {symbol} 數據: {str(e)}")
        return None

def get_v6_strategy(df):
    if df is None or len(df) < 20: return "數據不足", "#999", 0
    row = df.iloc[-1]
    prev = df.iloc[-2]
    bull = row['Close'] > row['SMA240'] if not pd.isna(row['SMA240']) else row['Close'] > row['SMA60']
    score = 0
    if row['RSI'] < (40 if bull else 30): score += 1
    if row['BB_pos'] < 15: score += 1
    if row['Hist'] > prev['Hist']: score += 1
    if bull: score += 1
    if row['Close'] < row['SMA60'] and row['SMA20'] < row['SMA60']:
        return "趨勢轉空", "#d32f2f", score
    if score >= 3: return "強力買進", "#2e7d32", score
    if score == 2: return "分批佈局", "#43a047", score
    return ("多頭續抱" if bull else "觀望整理"), ("#1976d2" if bull else "#757575"), score

def plot_v6_chart(df, name):
    if df is None: return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03,
                        subplot_titles=(f"{name} 股價/均線", "RSI", "MACD"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1.5), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=2), name='年線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9370DB'), name='RSI'), row=2, col=1)
    colors = ['#2E8B57' if v >= 0 else '#CD5C5C' for v in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(t=50, b=20))
    return fig

# --- 2. 頁面邏輯 ---
portfolio = load_portfolio()

# A. 資產總覽
st.markdown('<div class="metric-bar">', unsafe_allow_html=True)
t_mkt, t_cost, monitored_data = 0.0, 0.0, []
if not portfolio.empty:
    with st.spinner('同步市場數據中... (請稍候)'):
        for _, r in portfolio.iterrows():
            d = fetch_data_v6(r['Symbol'])
            if d is not None:
                cp = d['Close'].iloc[-1]
                t_mkt += cp * r['Shares']
                t_cost += r['Cost'] * r['Shares']
                monitored_data.append({'r': r, 'df': d, 'cp': cp})
p1, p2, p3 = st.columns(3)
p1.metric("總市值", f"${t_mkt:,.0f}")
p2.metric("總損益", f"${(t_mkt-t_cost):,.0f}", f"{((t_mkt-t_cost)/t_cost*100 if t_cost>0 else 0):.2f}%")
p3.metric("總成本", f"${t_cost:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

# B. 監控牆與工具
main_col, side_col = st.columns([0.6, 0.4])

with main_col:
    st.subheader("🚀 庫存個股監控 (含 PE/PB)")
    if monitored_data:
        m_cols = st.columns(2)
        for i, item in enumerate(monitored_data):
            with m_cols[i % 2]:
                adv, col, sc = get_v6_strategy(item['df'])
                info = STOCK_MAP.get(item['r']['Symbol'], {'PE': '-', 'PB': '-', '產業': '-'})
                st.markdown(f"""
                <div class="stock-card" style="border-left: 8px solid {col}">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: bold;">{item['r']['Name']} ({item['r']['Symbol']})</span>
                        <span class="info-label">{info['產業']}</span>
                    </div>
                    <div style="margin: 8px 0;">
                        <span style="font-size: 1.8em; font-weight: bold;">${item['cp']:.2f}</span>
                        <span style="margin-left: 10px; color: {col}; font-weight: bold;">{adv}</span>
                    </div>
                    <div style="display: flex; gap: 12px; border-top: 1px solid #f0f0f0; padding-top: 8px; font-size: 0.8em;">
                        <div>PE: <span class="info-value">{info['PE']}</span></div>
                        <div>PB: <span class="info-value">{info['PB']}</span></div>
                        <div style="color:blue;">成本: {item['r']['Cost']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看技術分析", key=f"mon_{item['r']['Symbol']}"):
                    st.session_state.current_plot = (item['df'], item['r']['Name'])

with side_col:
    tab1, tab2 = st.tabs(["💰 低基期快篩", "🔍 免庫存個股"])
    with tab1:
        pe_val = st.number_input("PE 上限", 15.0)
        pb_val = st.number_input("PB 上限", 1.5)
        if st.button("🔍 執行快篩"):
            candidates = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= pe_val and 0 < float(v['PB']) <= pb_val]
            st.session_state.scan_results = random.sample(candidates, min(len(candidates), 10))
        
        if 'scan_results' in st.session_state:
            for code in st.session_state.scan_results:
                with st.expander(f"📌 {code} {STOCK_MAP[code]['名稱']}"):
                    s_df = fetch_data_v6(code)
                    if s_df is not None:
                        adv, col, sc = get_v6_strategy(s_df)
                        st.markdown(f"**建議：<span style='color:{col}'>{adv}</span>** ({sc}分)", unsafe_allow_html=True)
                        st.write(f"現價：{s_df['Close'].iloc[-1]:.2f}")
                        if st.button("看圖", key=f"sc_{code}"):
                            st.session_state.current_plot = (s_df, STOCK_MAP[code]['名稱'])

    with tab2:
        target = st.text_input("輸入代碼")
        if target:
            q_df = fetch_data_v6(target)
            if q_df is not None:
                name = STOCK_MAP.get(target, {'名稱': '未知'})['名稱']
                adv, col, sc = get_v6_strategy(q_df)
                st.markdown(f'<div class="stock-card" style="border-top:4px solid {col}"><h4>{name}</h4><h3>{adv}</h3></div>', unsafe_allow_html=True)
                if st.button("顯示圖表", key="q_btn"):
                    st.session_state.current_plot = (q_df, name)

if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(p_df, p_name), use_container_width=True)

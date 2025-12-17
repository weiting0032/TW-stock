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
st.set_page_config(page_title="台股戰情指揮中心 V6.5", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 20px; border-radius: 15px; background-color: white; box-shadow: 3px 3px 10px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-bar { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; font-weight: bold; }
    .sidebar-btn-active { background-color: #2a5298 !important; color: white !important; }
    .info-label { color: #666; font-size: 0.9em; }
    .info-value { font-weight: bold; color: #333; }
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
    """標準化數據抓取，確保 3047 等標的價格精確至小數點兩位"""
    time.sleep(random.uniform(0.3, 0.8)) # 輕微延遲避開 API 限制
    full_sym = f"{symbol}.TW"
    try:
        ticker = yf.Ticker(full_sym)
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty or len(df) < 10:
            df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
        if df.empty: return None

        # 計算 V6 策略技術指標
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
    except:
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
                        subplot_titles=(f"{name} 股價/均線", "RSI 相對強弱", "MACD 趨勢動能"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1.5), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=2), name='年線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9370DB'), name='RSI'), row=2, col=1)
    colors = ['#2E8B57' if v >= 0 else '#CD5C5C' for v in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(t=50, b=20), showlegend=True)
    return fig

# --- 2. 側邊導覽選單 ---
with st.sidebar:
    st.title("📈 導覽控制")
    if 'menu' not in st.session_state:
        st.session_state.menu = "portfolio"
    
    if st.button("🚀 庫存個股監控", key="btn_p"):
        st.session_state.menu = "portfolio"
    if st.button("💰 低基期快篩", key="btn_s"):
        st.session_state.menu = "screening"
    if st.button("🔍 免庫存診斷", key="btn_d"):
        st.session_state.menu = "diagnosis"
    
    st.divider()
    st.caption("版本: V6.5 Optimized")

# --- 3. 主畫面內容 ---
portfolio = load_portfolio()

# A. 頂部資產 Bar (始終顯示)
st.markdown('<div class="metric-bar">', unsafe_allow_html=True)
t_mkt, t_cost, monitored_data = 0.0, 0.0, []
if not portfolio.empty:
    for _, r in portfolio.iterrows():
        d = fetch_data_v6(r['Symbol'])
        if d is not None:
            cp = d['Close'].iloc[-1]
            t_mkt += cp * r['Shares']
            t_cost += r['Cost'] * r['Shares']
            monitored_data.append({'r': r, 'df': d, 'cp': cp})
p1, p2, p3 = st.columns(3)
p1.metric("總資產市值", f"${t_mkt:,.0f}")
p2.metric("未實現損益", f"${(t_mkt-t_cost):,.0f}", f"{((t_mkt-t_cost)/t_cost*100 if t_cost>0 else 0):.2f}%")
p3.metric("總投入成本", f"${t_cost:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

# B. 功能切換邏輯
if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股即時監控")
    if monitored_data:
        cols = st.columns(3)
        for i, item in enumerate(monitored_data):
            with cols[i % 3]:
                adv, col, sc = get_v6_strategy(item['df'])
                info = STOCK_MAP.get(item['r']['Symbol'], {'PE':'-', 'PB':'-', '產業':'-'})
                st.markdown(f"""
                <div class="stock-card" style="border-top: 5px solid {col}">
                    <div style="font-weight:bold; color:#555;">{item['r']['Name']} ({item['r']['Symbol']})</div>
                    <div style="font-size:2em; font-weight:bold; margin:10px 0;">${item['cp']:.2f}</div>
                    <div style="color:{col}; font-weight:bold;">{adv} (評分:{sc})</div>
                    <hr style="margin:10px 0;">
                    <div style="font-size:0.85em; color:#666;">PE: {info['PE']} | PB: {info['PB']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看圖表", key=f"p_btn_{item['r']['Symbol']}"):
                    st.session_state.current_plot = (item['df'], item['r']['Name'])

elif st.session_state.menu == "screening":
    st.subheader("💰 低基期潛力標的快篩")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("本益比 PE 上限", value=15.0)
    pb_lim = c2.number_input("淨值比 PB 上限", value=1.2)
    if c3.button("開始掃描", use_container_width=True):
        candidates = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= pe_lim and 0 < float(v['PB']) <= pb_lim]
        st.session_state.scan_results = random.sample(candidates, min(len(candidates), 12))
    
    if 'scan_results' in st.session_state:
        st.divider()
        sc_cols = st.columns(3)
        for i, code in enumerate(st.session_state.scan_results):
            with sc_cols[i % 3]:
                s_df = fetch_data_v6(code)
                name = STOCK_MAP[code]['名稱']
                if s_df is not None:
                    adv, col, sc = get_v6_strategy(s_df)
                    st.markdown(f"""
                    <div class="stock-card" style="border-left: 5px solid {col}">
                        <b>{name} ({code})</b><br>
                        現價: {s_df['Close'].iloc[-1]:.2f}<br>
                        <span style="color:{col}; font-weight:bold;">{adv}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"查看技術圖表", key=f"sc_btn_{code}"):
                        st.session_state.current_plot = (s_df, name)

elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 免庫存個股診斷分析")
    target = st.text_input("請輸入台股代碼 (例如: 3047)", placeholder="輸入代碼後按下 Enter")
    if target:
        q_df = fetch_data_v6(target)
        if q_df is not None:
            name = STOCK_MAP.get(target, {'名稱': '未知'})['名稱']
            adv, col, sc = get_v6_strategy(q_df)
            info = STOCK_MAP.get(target, {'PE':'-', 'PB':'-', '產業':'-'})
            st.markdown(f"""
            <div class="stock-card" style="border-top: 8px solid {col}; background-color: #f8f9fa;">
                <h2>{name} ({target}) - {info['產業']}</h2>
                <div style="display:flex; gap:30px;">
                    <div><span class="info-label">目前股價</span><br><span style="font-size:2em; font-weight:bold;">${q_df['Close'].iloc[-1]:.2f}</span></div>
                    <div><span class="info-label">策略評分</span><br><span style="font-size:2em; font-weight:bold; color:{col}">{sc} 分</span></div>
                    <div><span class="info-label">操作建議</span><br><span style="font-size:2em; font-weight:bold; color:{col}">{adv}</span></div>
                </div>
                <div style="margin-top:15px; color:#666;">PE: {info['PE']} | PB: {info['PB']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.current_plot = (q_df, name)
        else:
            st.error("找不到該股票數據，請檢查代碼是否正確。")

# C. 底部圖表顯示區
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(p_df, p_name), use_container_width=True)

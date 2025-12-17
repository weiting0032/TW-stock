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
st.set_page_config(page_title="台股戰情指揮中心 V6.7", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 20px; border-radius: 15px; background-color: white; box-shadow: 3px 3px 10px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-bar { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; font-weight: bold; }
    .group-tag { background-color: #f0f2f6; color: #555; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; margin-left: 5px; vertical-align: middle; }
    .info-label { color: #666; font-size: 0.9em; }
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
# 建立包含族群資訊的選單清單: ["2330 台積電 (半導體業)", ...]
STOCK_OPTIONS = [f"{k} {v['名稱']} ({v['產業']})" for k, v in STOCK_MAP.items()]

def fetch_data_v6(symbol):
    time.sleep(random.uniform(0.1, 0.3)) 
    full_sym = f"{symbol}.TW"
    try:
        ticker = yf.Ticker(full_sym)
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty or len(df) < 10:
            df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
        if df.empty: return None
        
        # 指標計算
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
    except: return None

def get_v6_strategy(df):
    if df is None or len(df) < 20: return "數據不足", "#999", 0
    row, prev = df.iloc[-1], df.iloc[-2]
    bull = row['Close'] > row['SMA240'] if not pd.isna(row['SMA240']) else row['Close'] > row['SMA60']
    score = 0
    if row['RSI'] < (40 if bull else 30): score += 1
    if row['BB_pos'] < 15: score += 1
    if row['Hist'] > prev['Hist']: score += 1
    if bull: score += 1
    if row['Close'] < row['SMA60'] and row['SMA20'] < row['SMA60']: return "趨勢轉空", "#d32f2f", score
    if score >= 3: return "強力買進", "#2e7d32", score
    if score == 2: return "分批佈局", "#43a047", score
    return ("多頭續抱" if bull else "觀望整理"), ("#1976d2" if bull else "#757575"), score

def plot_v6_chart(df, name):
    if df is None: return None
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1.5), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=2), name='年線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9370DB'), name='RSI'), row=2, col=1)
    colors = ['#2E8B57' if v >= 0 else '#CD5C5C' for v in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(t=30, b=20), showlegend=True)
    return fig

# --- 2. 側邊導覽 ---
with st.sidebar:
    st.title("📈 功能導覽")
    if 'menu' not in st.session_state: st.session_state.menu = "portfolio"
    if st.button("🚀 庫存個股監控"): st.session_state.menu = "portfolio"
    if st.button("💰 低基期快篩"): st.session_state.menu = "screening"
    if st.button("🔍 免庫存診斷"): st.session_state.menu = "diagnosis"

# --- 3. 主畫面 ---
portfolio = load_portfolio()
st.markdown('<div class="metric-bar">', unsafe_allow_html=True)
t_mkt, t_cost = 0.0, 0.0
for _, r in portfolio.iterrows():
    ticker = yf.Ticker(f"{r['Symbol']}.TW")
    hist = ticker.history(period="1d")
    if hist.empty: hist = yf.Ticker(f"{r['Symbol']}.TWO").history(period="1d")
    if not hist.empty:
        cp = hist['Close'].iloc[-1]
        t_mkt += cp * r['Shares']
        t_cost += r['Cost'] * r['Shares']
p1, p2, p3 = st.columns(3)
p1.metric("總市值", f"${t_mkt:,.0f}")
p2.metric("總損益", f"${(t_mkt-t_cost):,.0f}", f"{((t_mkt-t_cost)/t_cost*100 if t_cost>0 else 0):.2f}%")
p3.metric("總投入成本", f"${t_cost:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股監控")
    cols = st.columns(3)
    for i, r in portfolio.iterrows():
        d = fetch_data_v6(r['Symbol'])
        if d is not None:
            adv, col, sc = get_v6_strategy(d)
            # 獲取族群資訊
            info = STOCK_MAP.get(r['Symbol'], {'PE':'-', 'PB':'-', '產業': '未知'})
            with cols[i % 3]:
                st.markdown(f"""
                <div class="stock-card" style="border-top:5px solid {col}">
                    <b>{r['Name']} ({r['Symbol']})</b> <span class="group-tag">{info['產業']}</span><br>
                    <span style="font-size:1.6em;font-weight:bold;">${d['Close'].iloc[-1]:.2f}</span><br>
                    <span style="color:{col}; font-weight:bold;">{adv} ({sc}分)</span><br>
                    <small>PE: {info['PE']} | PB: {info['PB']}</small>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看技術圖表", key=f"p_{r['Symbol']}"): st.session_state.current_plot = (d, r['Name'])

elif st.session_state.menu == "screening":
    st.subheader("💰 低基期潛力標的快篩")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 上限", value=15.0)
    pb_lim = c2.number_input("PB 上限", value=1.2)
    if c3.button("開始全面掃描"):
        st.session_state.scan_results = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= pe_lim and 0 < float(v['PB']) <= pb_lim]
    
    if 'scan_results' in st.session_state:
        st.info(f"符合條件標的共 {len(st.session_state.scan_results)} 筆")
        sc_cols = st.columns(3)
        for i, code in enumerate(st.session_state.scan_results):
            with sc_cols[i % 3]:
                name = STOCK_MAP[code]['名稱']
                group = STOCK_MAP[code]['產業']
                st.markdown(f"""
                <div class="stock-card">
                    <b>{code} {name}</b> <br><small>{group}</small><br>
                    <hr style="margin:8px 0;">
                    PE: {STOCK_MAP[code]['PE']} | PB: {STOCK_MAP[code]['PB']}
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"執行診斷 {code}", key=f"sc_{code}"):
                    d = fetch_data_v6(code)
                    if d is not None: st.session_state.current_plot = (d, name)

elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 免庫存個股診斷分析")
    selection = st.selectbox("搜尋標的 (輸入代碼、名稱或族群關鍵字)", options=["請選擇股票..."] + STOCK_OPTIONS)
    if st.button("開始診斷") and selection != "請選擇股票...":
        target_code = selection.split(" ")[0]
        q_df = fetch_data_v6(target_code)
        if q_df is not None:
            name = STOCK_MAP.get(target_code, {'名稱': '未知'})['名稱']
            group = STOCK_MAP.get(target_code, {'產業': '未知'})['產業']
            adv, col, sc = get_v6_strategy(q_df)
            st.markdown(f"""
            <div class="stock-card" style="border-top:8px solid {col}; background-color: #fbfbfb;">
                <div style="font-size:1.8em; font-weight:bold;">{name} ({target_code}) <span style="font-size:0.5em; font-weight:normal; color:#888;">{group}</span></div>
                <hr>
                <div style="display:flex; justify-content: space-around; text-align:center;">
                    <div><small>建議</small><br><b style="font-size:1.5em; color:{col};">{adv}</b></div>
                    <div><small>策略評分</small><br><b style="font-size:1.5em;">{sc} 分</b></div>
                    <div><small>當前價格</small><br><b style="font-size:1.5em;">${q_df['Close'].iloc[-1]:.2f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.current_plot = (q_df, name)

if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(p_df, p_name), use_container_width=True)

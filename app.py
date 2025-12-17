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
st.set_page_config(page_title="台股戰情指揮中心 V6.9", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 20px; border-radius: 15px; background-color: white; box-shadow: 3px 3px 10px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-bar { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; font-weight: bold; }
    .group-tag { background-color: #f0f2f6; color: #555; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; margin-left: 5px; }
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
        data['PE'] = pd.to_numeric(data['PE'], errors='coerce').fillna(999)
        data['PB'] = pd.to_numeric(data['PB'], errors='coerce').fillna(999)
        return data
    except:
        return pd.DataFrame()

STOCK_DF = get_tw_map()

def fetch_data_v6(symbol):
    time.sleep(random.uniform(0.1, 0.3)) 
    full_sym = f"{symbol}.TW"
    try:
        ticker = yf.Ticker(full_sym)
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty or len(df) < 10:
            df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
        if df.empty: return None

        # 技術指標
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

# 🚨 修正：圖形上方加入名稱
def plot_v6_chart(df, name):
    if df is None: return None
    
    # 建立子圖，並透過 subplot_titles 直接定義名稱
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.5, 0.2, 0.3], 
        vertical_spacing=0.08, # 稍微拉開間距讓標題更清楚
        subplot_titles=("■ 股價與均線分析", "■ RSI 相對強弱指標", "■ MACD 趨勢動能指標")
    )

    # 1. 股價圖
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1.2), name='月線(20)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='green', width=1.2), name='季線(60)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=1.8), name='年線(240)'), row=1, col=1)

    # 2. RSI 圖
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9370DB', width=2), name='RSI(14)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # 3. MACD 圖
    colors = ['#2E8B57' if v >= 0 else '#CD5C5C' for v in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱狀體'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DIF'], line=dict(color='blue', width=1), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DEA'], line=dict(color='orange', width=1), name='DEA'), row=3, col=1)

    # 更新佈局設定
    fig.update_layout(
        title=dict(text=f"<b>{name} 技術分析報告</b>", x=0.5, font=dict(size=24)),
        height=900, 
        xaxis_rangeslider_visible=False, 
        margin=dict(t=100, b=50, l=50, r=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # 優化子圖標題字體
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=18, color='#333')
        i['x'] = 0  # 標題靠左對齊
        i['xanchor'] = 'left'

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
# 這裡簡易展示總覽
for _, r in portfolio.iterrows():
    try:
        ticker = yf.Ticker(f"{r['Symbol']}.TW")
        cp = ticker.history(period="1d")['Close'].iloc[-1]
        t_mkt += cp * r['Shares']
        t_cost += r['Cost'] * r['Shares']
    except: pass
p1, p2, p3 = st.columns(3)
p1.metric("總市值", f"${t_mkt:,.0f}")
p2.metric("未實現損益", f"${(t_mkt-t_cost):,.0f}", f"{((t_mkt-t_cost)/t_cost*100 if t_cost>0 else 0):.2f}%")
p3.metric("總成本", f"${t_cost:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

# 功能切換內容 (略，結構同 V6.8，僅調用更新後的 plot_v6_chart)
if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股監控")
    cols = st.columns(3)
    for i, r in portfolio.iterrows():
        d = fetch_data_v6(r['Symbol'])
        if d is not None:
            adv, col, sc = get_v6_strategy(d)
            stock_info = STOCK_DF[STOCK_DF['代碼'] == r['Symbol']]
            group = stock_info['產業'].values[0] if not stock_info.empty else "未知"
            with cols[i % 3]:
                st.markdown(f'<div class="stock-card" style="border-top:5px solid {col}"><b>{r["Name"]} ({r["Symbol"]})</b> <span class="group-tag">{group}</span><br><span style="font-size:1.6em;font-weight:bold;">${d["Close"].iloc[-1]:.2f}</span><br><span style="color:{col}; font-weight:bold;">{adv} ({sc}分)</span></div>', unsafe_allow_html=True)
                if st.button(f"顯示報告", key=f"p_{r['Symbol']}"): st.session_state.current_plot = (d, r['Name'])

elif st.session_state.menu == "screening":
    st.subheader("💰 低基期快篩 (依族群及 PE/PB 排序)")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 上限", value=15.0)
    pb_lim = c2.number_input("PB 上限", value=1.2)
    if c3.button("執行掃描"):
        filtered = STOCK_DF[(STOCK_DF['PE'] > 0) & (STOCK_DF['PE'] <= pe_lim) & (STOCK_DF['PB'] > 0) & (STOCK_DF['PB'] <= pb_lim)].copy()
        st.session_state.scan_results = filtered.sort_values(by=['產業', 'PE', 'PB'])
    
    if 'scan_results' in st.session_state:
        df_res = st.session_state.scan_results
        st.info(f"共找到 {len(df_res)} 筆標的")
        sc_cols = st.columns(3)
        for i, (idx, row) in enumerate(df_res.iterrows()):
            with sc_cols[i % 3]:
                st.markdown(f'<div class="stock-card"><small>{row["產業"]}</small><br><b>{row["代碼"]} {row["名稱"]}</b><br><hr>PE: {row["PE"]} | PB: {row["PB"]}</div>', unsafe_allow_html=True)
                if st.button(f"技術診斷 {row['代碼']}", key=f"sc_{row['代碼']}"):
                    d = fetch_data_v6(row['代碼'])
                    if d is not None: st.session_state.current_plot = (d, row['名稱'])

elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 免庫存個股診斷分析")
    options = [f"{r['代碼']} {r['名稱']} ({r['產業']})" for _, r in STOCK_DF.iterrows()]
    selection = st.selectbox("請選擇或搜尋標的", options=["請選擇..."] + options)
    if st.button("開始分析") and selection != "請選擇...":
        code = selection.split(" ")[0]
        q_df = fetch_data_v6(code)
        if q_df is not None:
            name = selection.split(" ")[1]
            adv, col, sc = get_v6_strategy(q_df)
            st.markdown(f'<div class="stock-card" style="border-top:8px solid {col}"><h2>{name} ({code})</h2><h3>建議：<span style="color:{col}">{adv}</span> ({sc}分)</h3>現價：${q_df["Close"].iloc[-1]:.2f}</div>', unsafe_allow_html=True)
            st.session_state.current_plot = (q_df, name)

if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(p_df, p_name), use_container_width=True)

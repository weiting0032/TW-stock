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
st.set_page_config(page_title="台股戰情指揮中心 V8.0", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 20px; border-radius: 15px; background-color: white; box-shadow: 3px 3px 10px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-bar { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; border-radius: 15px; margin-bottom: 25px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; font-weight: bold; }
    .group-tag { background-color: #f0f2f6; color: #555; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; margin-left: 5px; vertical-align: middle; }
    .profit-up { color: #eb093b; font-weight: bold; }
    .profit-down { color: #00a651; font-weight: bold; }
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
        # 處理 PE/PB 為數值以便快篩比較
        data['PE'] = pd.to_numeric(data['PE'], errors='coerce').fillna(999.0)
        data['PB'] = pd.to_numeric(data['PB'], errors='coerce').fillna(999.0)
        return data.set_index('代碼').to_dict('index')
    except:
        return {}

STOCK_MAP = get_tw_map()
STOCK_OPTIONS = [f"{k} {v['名稱']} ({v['產業']})" for k, v in STOCK_MAP.items()]

def fetch_data_v6(symbol):
    time.sleep(random.uniform(0.1, 0.3)) 
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
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
    colors = ['#eb093b' if v >= 0 else '#00a651' for v in df['Hist']]
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
    if st.button("📝 庫存管理"): st.session_state.menu = "management"

# --- 3. 核心邏輯 ---
portfolio = load_portfolio()

# 計算總指標
t_mkt, t_cost = 0.0, 0.0
stock_details = {} # 暫存現價以供後續使用
for _, r in portfolio.iterrows():
    ticker = yf.Ticker(f"{r['Symbol']}.TW")
    hist = ticker.history(period="1d")
    if hist.empty: hist = yf.Ticker(f"{r['Symbol']}.TWO").history(period="1d")
    if not hist.empty:
        cp = hist['Close'].iloc[-1]
        stock_details[r['Symbol']] = cp
        t_mkt += cp * r['Shares']
        t_cost += r['Cost'] * r['Shares']

# 總資產看板
st.markdown('<div class="metric-bar">', unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)
p1.metric("總市值", f"${t_mkt:,.0f}")
p2.metric("總未實現損益", f"${(t_mkt-t_cost):,.0f}", f"{((t_mkt-t_cost)/t_cost*100 if t_cost>0 else 0):.2f}%")
p3.metric("總投入成本", f"${t_cost:,.0f}")
st.markdown('</div>', unsafe_allow_html=True)

# --- 4. 頁面內容 ---

if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股監控")
    cols = st.columns(3)
    for i, r in portfolio.iterrows():
        d = fetch_data_v6(r['Symbol'])
        if d is not None:
            adv, col, sc = get_v6_strategy(d)
            info = STOCK_MAP.get(r['Symbol'], {'PE':'-', 'PB':'-', '產業': '未知'})
            curr_price = d['Close'].iloc[-1]
            # 計算個股盈虧%
            profit_pct = (curr_p - r['Cost']) / r['Cost'] * 100 if r['Cost'] > 0 else 0
            p_style = "profit-up" if profit_pct >= 0 else "profit-down"
            p_sign = "+" if profit_pct >= 0 else ""

            with cols[i % 3]:
                st.markdown(f"""
                <div class="stock-card" style="border-top:5px solid {col}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b>{r['Name']} ({r['Symbol']})</b> <span class="group-tag">{info['產業']}</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span style="font-size:1.8em;font-weight:bold;">${curr_price:.2f}</span>
                        <span class="{p_style}" style="margin-left:10px;">{p_sign}{profit_pct:.2f}%</span>
                    </div>
                    <div style="color:{col}; font-weight:bold; margin-bottom:5px;">{adv} ({sc}分)</div>
                    <div style="font-size:0.9em; border-top: 1px solid #eee; padding-top:5px;">
                        <span class="info-label">PE:</span> {info['PE']} | 
                        <span class="info-label">PB:</span> {info['PB']} | 
                        <span class="info-label">成本:</span> {r['Cost']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"技術圖表 {r['Symbol']}", key=f"p_{r['Symbol']}"): st.session_state.current_plot = (d, r['Name'])

elif st.session_state.menu == "screening":
    st.subheader("💰 低基期潛力標的快篩 (V6.7版)")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 本益比上限", value=15.0)
    pb_lim = c2.number_input("PB 淨值比上限", value=1.2)
    
    if c3.button("開始全面掃描"):
        # 維持 V6.7 篩選邏輯
        st.session_state.scan_results = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= pe_lim and 0 < float(v['PB']) <= pb_lim]
    
    if 'scan_results' in st.session_state:
        st.info(f"符合低基期條件標的共 {len(st.session_state.scan_results)} 筆")
        sc_cols = st.columns(3)
        for i, code in enumerate(st.session_state.scan_results):
            with sc_cols[i % 3]:
                s_info = STOCK_MAP[code]
                st.markdown(f"""
                <div class="stock-card">
                    <b>{code} {s_info['名稱']}</b> <span class="group-tag">{s_info['產業']}</span><br>
                    <hr style="margin:8px 0;">
                    <div style="display:flex; justify-content:space-between;">
                        <span>PE: <b>{s_info['PE']}</b></span>
                        <span>PB: <b>{s_info['PB']}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"執行診斷 {code}", key=f"sc_{code}"):
                    d = fetch_data_v6(code)
                    if d is not None: st.session_state.current_plot = (d, s_info['名稱'])

elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 免庫存個股診斷分析")
    selection = st.selectbox("搜尋標的", options=["請選擇股票..."] + STOCK_OPTIONS)
    if st.button("開始診斷") and selection != "請選擇股票...":
        target_code = selection.split(" ")[0]
        q_df = fetch_data_v6(target_code)
        if q_df is not None:
            s_info = STOCK_MAP.get(target_code, {'名稱': '未知', '產業': '未知'})
            adv, col, sc = get_v6_strategy(q_df)
            st.markdown(f"""
            <div class="stock-card" style="border-top:8px solid {col}; background-color: #fbfbfb;">
                <div style="font-size:1.8em; font-weight:bold;">{s_info['名稱']} ({target_code}) <span style="font-size:0.5em; font-weight:normal; color:#888;">{s_info['產業']}</span></div>
                <hr>
                <div style="display:flex; justify-content: space-around; text-align:center;">
                    <div><small>建議</small><br><b style="font-size:1.5em; color:{col};">{adv}</b></div>
                    <div><small>策略評分</small><br><b style="font-size:1.5em;">{sc} 分</b></div>
                    <div><small>當前價格</small><br><b style="font-size:1.5em;">${q_df['Close'].iloc[-1]:.2f}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.current_plot = (q_df, s_info['名稱'])

elif st.session_state.menu == "management":
    st.subheader("📝 庫存清單管理")
    # A. 新增庫存區
    with st.expander("➕ 新增個股到庫存", expanded=False):
        new_selection = st.selectbox("選擇標的", options=["請選擇..."] + STOCK_OPTIONS)
        c1, c2, c3 = st.columns(3)
        new_cost = c1.number_input("平均成本", min_value=0.0, step=0.1)
        new_shares = c2.number_input("持有股數", min_value=0, step=1000)
        new_note = c3.text_input("備註", value="-")
        
        if st.button("確認新增"):
            if new_selection != "請選擇..." and new_shares > 0:
                code = new_selection.split(" ")[0]
                name = new_selection.split(" ")[1]
                credentials = st.secrets["gcp_service_account"]
                gc = gspread.service_account_from_dict(credentials)
                sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                sh.append_row([code, name, new_cost, new_shares, new_note])
                st.success(f"已新增 {name} ({code})")
                st.cache_data.clear()
                st.rerun()

    if not portfolio.empty:
        edited_df = st.data_editor(portfolio, use_container_width=True, hide_index=True)
        if st.button("💾 儲存所有變更"):
            credentials = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(credentials)
            sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
            final_df = edited_df[edited_df['Shares'] > 0]
            sh.clear()
            sh.update('A1', [portfolio.columns.tolist()] + final_df.values.tolist())
            st.success("同步成功！")
            st.cache_data.clear()
            st.rerun()

# 底部圖表顯示
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(p_df, p_name), use_container_width=True)

import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import random

# --- 0. 基礎設定 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' 
st.set_page_config(page_title="台股戰情指揮中心 V8.2", layout="wide", page_icon="📈")

# 自訂 CSS：包含看板與卡片設計
st.markdown("""
    <style>
    .stock-card { border: 1px solid #eee; padding: 18px; border-radius: 12px; background-color: white; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-container { display: flex; justify-content: space-around; background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.03); }
    .metric-item { text-align: center; border-right: 1px solid #eee; flex: 1; }
    .metric-item:last-child { border-right: none; }
    .metric-label { font-size: 0.95em; color: #666; margin-bottom: 8px; font-weight: 500; }
    .metric-value { font-size: 2em; font-weight: 800; color: #1a2a6c; }
    .profit-up { color: #eb093b; font-weight: bold; } /* 台灣紅盈 */
    .profit-down { color: #00a651; font-weight: bold; } /* 台灣綠虧 */
    .group-tag { background-color: #f0f2f6; color: #555; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 核心數據處理函數 ---

def get_gsheet_client():
    credentials = st.secrets["gcp_service_account"]
    return gspread.service_account_from_dict(credentials)

@st.cache_data(ttl=300) # 庫存清單快取 5 分鐘
def load_portfolio():
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        df = pd.DataFrame(sh.sheet1.get_all_records())
        df['Symbol'] = df['Symbol'].astype(str).str.zfill(4)
        return df
    except:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

@st.cache_data(ttl=86400) # 市場清單快取 24 小時
def get_tw_map():
    url = "https://stock.wespai.com/lists"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        df = pd.read_html(res.text)[0]
        data = df.iloc[:, [0, 1, 2, 14, 15]].copy()
        data.columns = ['代碼', '名稱', '產業', 'PE', 'PB']
        data['代碼'] = data['代碼'].astype(str).str.zfill(4)
        data['PE'] = pd.to_numeric(data['PE'], errors='coerce').fillna(999.0)
        data['PB'] = pd.to_numeric(data['PB'], errors='coerce').fillna(999.0)
        return data.set_index('代碼').to_dict('index')
    except:
        return {}

STOCK_MAP = get_tw_map()
STOCK_OPTIONS = [f"{k} {v['名稱']} ({v['產業']})" for k, v in STOCK_MAP.items()]

@st.cache_data(ttl=600) # 個股數據快取 10 分鐘，防止頻繁請求 Yahoo
def fetch_stock_data(symbol):
    # 隨機延遲 0.5~1.5 秒，避免被 Yahoo 偵測為爬蟲
    time.sleep(random.uniform(0.5, 1.5)) 
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty:
            df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
        if df.empty: return None
        
        # 指標計算
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA60'] = df['Close'].rolling(60).mean()
        df['SMA240'] = df['Close'].rolling(240).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['Hist'] = (df['EMA12'] - df['EMA26']) - (df['EMA12'] - df['EMA26']).ewm(span=9, adjust=False).mean()
        return df
    except: return None

def get_strategy_advice(df):
    if df is None or len(df) < 20: return "數據不足", "#999", 0
    row, prev = df.iloc[-1], df.iloc[-2]
    bull = row['Close'] > row['SMA240'] if not pd.isna(row['SMA240']) else row['Close'] > row['SMA60']
    score = 0
    if row['RSI'] < (40 if bull else 30): score += 1
    if row['Hist'] > prev['Hist']: score += 1
    if bull: score += 1
    if row['Close'] < row['SMA60'] and row['SMA20'] < row['SMA60']: return "趨勢轉空", "#d32f2f", score
    if score >= 2: return "分批佈局", "#43a047", score
    return ("多頭續抱" if bull else "觀望整理"), ("#1976d2" if bull else "#757575"), score

# --- 2. 主頁面與側邊欄邏輯 ---

with st.sidebar:
    st.title("📈 投資監控系統")
    if 'menu' not in st.session_state: st.session_state.menu = "portfolio"
    if st.button("🚀 庫存個股監控"): st.session_state.menu = "portfolio"
    if st.button("💰 低基期快篩"): st.session_state.menu = "screening"
    if st.button("🔍 免庫存診斷"): st.session_state.menu = "diagnosis"
    if st.button("📝 庫存清單管理"): st.session_state.menu = "management"

portfolio = load_portfolio()

# --- 功能 A: 庫存監控 (含總資產看板) ---
if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股動態監控")
    if not portfolio.empty:
        total_mv, total_cost, display_list = 0.0, 0.0, []
        
        with st.spinner('同步市場數據中，請稍候...'):
            for _, r in portfolio.iterrows():
                df = fetch_stock_data(r['Symbol'])
                if df is not None:
                    curr_p = df['Close'].iloc[-1]
                    total_mv += curr_p * r['Shares']
                    total_cost += r['Cost'] * r['Shares']
                    display_list.append({'r': r, 'df': df, 'cp': curr_p})

        # 1. 總資產看板
        diff = total_mv - total_cost
        p_ratio = (diff / total_cost * 100) if total_cost > 0 else 0
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item"><div class="metric-label">總資產市值</div><div class="metric-value">${total_mv:,.0f}</div></div>
                <div class="metric-item"><div class="metric-label">總未實現損益</div>
                    <div class="metric-value {'profit-up' if diff>=0 else 'profit-down'}">{'+' if diff>=0 else ''}${diff:,.0f}</div>
                    <div class="{'profit-up' if diff>=0 else 'profit-down'}" style="font-weight:bold;">{'+' if diff>=0 else ''}{p_ratio:.2f}%</div>
                </div>
                <div class="metric-item"><div class="metric-label">總投入成本</div><div class="metric-value" style="color:#444;">${total_cost:,.0f}</div></div>
            </div>
        """, unsafe_allow_html=True)

        # 2. 個股卡片展示
        cols = st.columns(3)
        for i, item in enumerate(display_list):
            r, df, cp = item['r'], item['df'], item['cp']
            adv, col, sc = get_strategy_advice(df)
            info = STOCK_MAP.get(r['Symbol'], {'PE':'-', 'PB':'-', '產業': '未知'})
            p_pct = (cp - r['Cost']) / r['Cost'] * 100 if r['Cost'] > 0 else 0
            
            with cols[i % 3]:
                st.markdown(f"""
                <div class="stock-card" style="border-top:5px solid {col}">
                    <div style="display:flex; justify-content:space-between;"><b>{r['Name']} ({r['Symbol']})</b> <span class="group-tag">{info['產業']}</span></div>
                    <div style="margin:10px 0;">
                        <span style="font-size:1.6em;font-weight:bold;">${cp:.2f}</span>
                        <span class="{'profit-up' if p_pct>=0 else 'profit-down'}" style="margin-left:10px;">{'+' if p_pct>=0 else ''}{p_pct:.2f}%</span>
                    </div>
                    <div style="color:{col}; font-weight:bold; margin-bottom:5px;">{adv}</div>
                    <div style="font-size:0.85em; color:#666; border-top:1px dashed #eee; padding-top:8px;">
                        PE: {info['PE']} | PB: {info['PB']} | 成本: {r['Cost']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看分析圖表", key=f"btn_{r['Symbol']}"):
                    st.session_state.current_plot = (df, r['Name'])

# --- 功能 B: 低基期快篩 (維持 V6.7 功能) ---
elif st.session_state.menu == "screening":
    st.subheader("💰 低基期潛力標的快篩 (V6.7)")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 本益比上限", value=15.0)
    pb_lim = c2.number_input("PB 淨值比上限", value=1.2)
    
    if c3.button("啟動掃描"):
        with st.spinner('過濾中...'):
            st.session_state.scan_results = [k for k, v in STOCK_MAP.items() if 0 < float(v['PE']) <= pe_lim and 0 < float(v['PB']) <= pb_lim]
    
    if 'scan_results' in st.session_state:
        st.info(f"符合標的共 {len(st.session_state.scan_results)} 筆")
        sc_cols = st.columns(3)
        for i, code in enumerate(st.session_state.scan_results):
            with sc_cols[i % 3]:
                s_info = STOCK_MAP[code]
                st.markdown(f"""
                <div class="stock-card">
                    <b>{code} {s_info['名稱']}</b> <span class="group-tag">{s_info['產業']}</span><br>
                    <hr style="margin:8px 0; border:0; border-top:1px solid #eee;">
                    PE: {s_info['PE']} | PB: {s_info['PB']}
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"執行診斷 {code}", key=f"sc_{code}"):
                    df = fetch_stock_data(code)
                    if df is not None: st.session_state.current_plot = (df, s_info['名稱'])

# --- 功能 C: 免庫存診斷 ---
elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 免庫存個股診斷分析")
    selection = st.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
    if st.button("開始分析") and selection != "請選擇...":
        code = selection.split(" ")[0]
        name = selection.split(" ")[1]
        df = fetch_stock_data(code)
        if df is not None:
            adv, col, sc = get_strategy_advice(df)
            st.markdown(f"""<div class="stock-card" style="border-left:10px solid {col}"><h3>{name} ({code}) - {adv}</h3><p>目前價格: ${df['Close'].iloc[-1]:.2f}</p></div>""", unsafe_allow_html=True)
            st.session_state.current_plot = (df, name)

# --- 功能 D: 庫存清單管理 ---
elif st.session_state.menu == "management":
    st.subheader("📝 庫存清單管理")
    with st.expander("➕ 新增庫存標的"):
        new_stock = st.selectbox("選擇股票", options=["請選擇..."] + STOCK_OPTIONS)
        c1, c2 = st.columns(2)
        cost = c1.number_input("成本", min_value=0.0)
        shares = c2.number_input("股數", min_value=0, step=1000)
        if st.button("確認新增"):
            if new_stock != "請選擇..." and shares > 0:
                code, name = new_stock.split(" ")[0], new_stock.split(" ")[1]
                gc = get_gsheet_client()
                sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                sh.append_row([code, name, cost, shares, "-"])
                st.success("已新增標的"); st.cache_data.clear(); st.rerun()

    if not portfolio.empty:
        edited = st.data_editor(portfolio, hide_index=True, use_container_width=True)
        if st.button("💾 儲存並同步至 Google Sheets"):
            gc = get_gsheet_client()
            sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
            sh.clear()
            sh.update('A1', [portfolio.columns.tolist()] + edited.values.tolist())
            st.success("同步成功"); st.cache_data.clear(); st.rerun()

# 底部圖表渲染區
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color='orange'), name='20MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA60'], line=dict(color='green'), name='60MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, title=f"{p_name} 技術走勢")
    st.plotly_chart(fig, use_container_width=True)

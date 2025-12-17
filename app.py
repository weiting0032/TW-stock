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
st.set_page_config(page_title="台股戰情指揮中心 V7.0", layout="wide", page_icon="📈")

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

def get_gsheet_client():
    credentials = st.secrets["gcp_service_account"]
    return gspread.service_account_from_dict(credentials)

@st.cache_data(ttl=60)
def load_portfolio():
    try:
        gc = get_gsheet_client()
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
STOCK_OPTIONS = [f"{r['代碼']} {r['名稱']} ({r['產業']})" for _, r in STOCK_DF.iterrows()]

def fetch_data_v6(symbol):
    time.sleep(random.uniform(0.1, 0.2)) 
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty: df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
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
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.08,
                        subplot_titles=("■ 股價與均線分析", "■ RSI 相對強弱指標", "■ MACD 趨勢動能指標"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1.2), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=1.8), name='年線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9370DB', width=2), name='RSI'), row=2, col=1)
    colors = ['#2E8B57' if v >= 0 else '#CD5C5C' for v in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    fig.update_layout(title=dict(text=f"<b>{name} 技術分析</b>", x=0.5), height=850, xaxis_rangeslider_visible=False, margin=dict(t=100))
    return fig

# --- 2. 側邊導覽 ---
with st.sidebar:
    st.title("📈 功能導覽")
    if 'menu' not in st.session_state: st.session_state.menu = "portfolio"
    if st.button("🚀 庫存個股監控"): st.session_state.menu = "portfolio"
    if st.button("💰 低基期快篩"): st.session_state.menu = "screening"
    if st.button("🔍 免庫存診斷"): st.session_state.menu = "diagnosis"
    if st.button("📝 庫存清單管理"): st.session_state.menu = "management"

# --- 3. 主畫面邏輯 ---
portfolio = load_portfolio()

if st.session_state.menu == "management":
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
                gc = get_gsheet_client()
                sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                sh.append_row([code, name, new_cost, new_shares, new_note])
                st.success(f"已新增 {name} ({code}) 至庫存！")
                st.cache_data.clear() # 強制重新讀取
                st.rerun()
            else:
                st.warning("請填寫正確的標的與股數")

    # B. 修改/刪除庫存區
    st.write("---")
    st.write("現有庫存列表 (修改股數為 0 即可踢出清單)")
    
    if not portfolio.empty:
        # 使用 st.data_editor 讓使用者直接編輯
        edited_df = st.data_editor(
            portfolio,
            column_config={
                "Symbol": st.column_config.TextColumn("代碼", disabled=True),
                "Name": st.column_config.TextColumn("名稱", disabled=True),
                "Cost": st.column_config.NumberColumn("平均成本", format="%.2f"),
                "Shares": st.column_config.NumberColumn("持有股數"),
                "Note": st.column_config.TextColumn("備註")
            },
            hide_index=True,
            use_container_width=True
        )
        
        if st.button("💾 儲存所有變更"):
            gc = get_gsheet_client()
            sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
            
            # 🚨 關鍵邏輯：過濾掉股數為 0 的標的
            final_df = edited_df[edited_df['Shares'] > 0]
            
            # 更新整個試算表 (先清空再填入)
            header = ["Symbol", "Name", "Cost", "Shares", "Note"]
            sh.clear()
            sh.update('A1', [header] + final_df.values.tolist())
            
            st.success("庫存已成功同步至 Google Sheets！")
            st.cache_data.clear()
            st.rerun()
    else:
        st.info("目前庫存清單為空。")

# --- 原有功能區 (省略重複細節，確保調用邏輯正確) ---
elif st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股監控")
    # (同 V6.9 代碼...)
    if not portfolio.empty:
        cols = st.columns(3)
        for i, r in portfolio.iterrows():
            d = fetch_data_v6(r['Symbol'])
            if d is not None:
                adv, col, sc = get_v6_strategy(d)
                info = STOCK_DF[STOCK_DF['代碼'] == r['Symbol']]
                group = info['產業'].values[0] if not info.empty else "未知"
                with cols[i % 3]:
                    st.markdown(f'<div class="stock-card" style="border-top:5px solid {col}"><b>{r["Name"]} ({r["Symbol"]})</b> <span class="group-tag">{group}</span><br><span style="font-size:1.6em;font-weight:bold;">${d["Close"].iloc[-1]:.2f}</span><br><span style="color:{col}">{adv}</span></div>', unsafe_allow_html=True)
                    if st.button(f"顯示報告", key=f"p_{r['Symbol']}"): st.session_state.current_plot = (d, r['Name'])

elif st.session_state.menu == "screening":
    st.subheader("💰 低基期快篩")
    # (同 V6.9 代碼，加入排序邏輯...)
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 上限", value=15.0)
    pb_lim = c2.number_input("PB 上限", value=1.2)
    if c3.button("掃描全部"):
        filtered = STOCK_DF[(STOCK_DF['PE']>0) & (STOCK_DF['PE']<=pe_lim) & (STOCK_DF['PB']>0) & (STOCK_DF['PB']<=pb_lim)].copy()
        st.session_state.scan_results = filtered.sort_values(by=['產業', 'PE', 'PB'])
    if 'scan_results' in st.session_state:
        # 顯示結果卡片... (略)
        pass

elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 免庫存診斷")
    selection = st.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
    if st.button("分析") and selection != "請選擇...":
        code = selection.split(" ")[0]
        q_df = fetch_data_v6(code)
        if q_df is not None:
            # 顯示診斷卡片... (略)
            st.session_state.current_plot = (q_df, selection.split(" ")[1])

# 底部圖表
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(p_df, p_name), use_container_width=True)

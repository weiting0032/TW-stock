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
st.set_page_config(page_title="台股戰情指揮中心 V8.0", layout="wide", page_icon="📈")

# 自訂 CSS：包含看板與卡片設計
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 20px; border-radius: 15px; background-color: white; box-shadow: 3px 3px 10px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-container { display: flex; justify-content: space-around; background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.03); }
    .metric-item { text-align: center; border-right: 1px solid #eee; flex: 1; }
    .metric-item:last-child { border-right: none; }
    .metric-label { font-size: 0.95em; color: #666; margin-bottom: 8px; font-weight: 500; }
    .metric-value { font-size: 2em; font-weight: 800; color: #1a2a6c; }
    .group-tag { background-color: #f0f2f6; color: #555; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; margin-left: 5px; }
    .profit-up { color: #eb093b; } /* 台灣紅漲 */
    .profit-down { color: #00a651; } /* 台灣綠跌 */
    </style>
""", unsafe_allow_html=True)

# --- 1. 核心數據處理 ---

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
        df['Hist'] = (df['EMA12'] - df['EMA26']) - (df['EMA12'] - df['EMA26']).ewm(span=9, adjust=False).mean()
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
                        subplot_titles=("■ 股價與均線分析", "■ RSI 指標", "■ MACD 動能"))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1.2), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='purple', width=1.8), name='年線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9370DB', width=2), name='RSI'), row=2, col=1)
    colors = ['#eb093b' if v >= 0 else '#00a651' for v in df['Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    fig.update_layout(title=dict(text=f"<b>{name} 技術分析</b>", x=0.5), height=800, xaxis_rangeslider_visible=False)
    return fig

# --- 2. 主畫面邏輯 ---

with st.sidebar:
    st.title("📈 投資監控系統")
    if 'menu' not in st.session_state: st.session_state.menu = "portfolio"
    if st.button("🚀 庫存個股監控"): st.session_state.menu = "portfolio"
    if st.button("💰 低基期快篩"): st.session_state.menu = "screening"
    if st.button("🔍 免庫存診斷"): st.session_state.menu = "diagnosis"
    if st.button("📝 庫存清單管理"): st.session_state.menu = "management"

portfolio = load_portfolio()

# --- A. 庫存監控主畫面 ---
if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股動態監控")
    if not portfolio.empty:
        total_mv, total_cost, summary_list = 0, 0, []
        
        with st.spinner('同步市場即時報價中...'):
            for _, r in portfolio.iterrows():
                d = fetch_data_v6(r['Symbol'])
                if d is not None:
                    curr_p = d['Close'].iloc[-1]
                    mv, cv = curr_p * r['Shares'], r['Cost'] * r['Shares']
                    total_mv += mv
                    total_cost += cv
                    
                    fund = STOCK_DF[STOCK_DF['代碼'] == r['Symbol']]
                    summary_list.append({
                        'info': r, 'df': d, 'price': curr_p, 'mv': mv,
                        'pe': fund['PE'].values[0] if not fund.empty else "-",
                        'pb': fund['PB'].values[0] if not fund.empty else "-",
                        'industry': fund['產業'].values[0] if not fund.empty else "未知",
                        'profit_pct': (curr_p - r['Cost']) / r['Cost'] * 100
                    })

        # --- 總資產看板 (對標圖片需求) ---
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

        # --- 個股卡片 ---
        cols = st.columns(3)
        for i, s in enumerate(summary_list):
            adv, col, _ = get_v6_strategy(s['df'])
            p_class = "profit-up" if s['profit_pct'] >= 0 else "profit-down"
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="stock-card" style="border-top: 5px solid {col}">
                        <div style="display:flex; justify-content:space-between;"><b>{s['info']['Name']} ({s['info']['Symbol']})</b><span class="group-tag">{s['industry']}</span></div>
                        <div style="margin:10px 0;">
                            <span style="font-size:1.7em; font-weight:bold;">${s['price']:.2f}</span>
                            <span class="{p_class}" style="margin-left:10px; font-weight:bold;">{'+' if s['profit_pct']>=0 else ''}{s['profit_pct']:.2f}%</span>
                        </div>
                        <div style="font-size:0.85em; color:#666; display:flex; justify-content:space-between; border-top:1px dashed #eee; padding-top:10px;">
                            <span>PE: <b>{s['pe']}</b> | PB: <b>{s['pb']}</b></span>
                            <span style="color:{col}; font-weight:bold;">{adv}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看診斷報告", key=f"btn_{s['info']['Symbol']}"): 
                    st.session_state.current_plot = (s['df'], s['info']['Name'])

# --- B. 低基期快篩 (維持原功能) ---
elif st.session_state.menu == "screening":
    st.subheader("💰 低基期價值快篩")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 本益比上限", value=15.0)
    pb_lim = c2.number_input("PB 淨值比上限", value=1.2)
    
    if c3.button("啟動掃描"):
        # 維持原過濾邏輯
        st.session_state.scan_results = STOCK_DF[
            (STOCK_DF['PE'] > 0) & (STOCK_DF['PE'] <= pe_lim) & 
            (STOCK_DF['PB'] > 0) & (STOCK_DF['PB'] <= pb_lim)
        ].copy().sort_values(by=['產業', 'PE'])
    
    if 'scan_results' in st.session_state:
        st.write(f"📊 篩選結果：共找到 {len(st.session_state.scan_results)} 檔標的")
        st.dataframe(st.session_state.scan_results, use_container_width=True, hide_index=True)

# --- C. 免庫存診斷 ---
elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 全市場個股診斷")
    selection = st.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
    if st.button("分析") and selection != "請選擇...":
        code, name = selection.split(" ")[0], selection.split(" ")[1]
        df = fetch_data_v6(code)
        if df is not None:
            adv, color, score = get_v6_strategy(df)
            fund = STOCK_DF[STOCK_DF['代碼'] == code]
            st.markdown(f"""<div class="stock-card" style="border-left:10px solid {color}"><h3>{name} ({code}) - {adv}</h3><p>PE: {fund['PE'].values[0]} | PB: {fund['PB'].values[0]} | 評分: {score}/4</p></div>""", unsafe_allow_html=True)
            st.session_state.current_plot = (df, name)

# --- D. 庫存清單管理 ---
elif st.session_state.menu == "management":
    st.subheader("📝 庫存清單管理")
    with st.expander("➕ 新增個股到庫存"):
        new_sel = st.selectbox("選擇標的", options=["請選擇..."] + STOCK_OPTIONS)
        c1, c2 = st.columns(2)
        cost = c1.number_input("成本", min_value=0.0)
        shares = c2.number_input("股數", min_value=0, step=1000)
        if st.button("確認新增"):
            if new_sel != "請選擇..." and shares > 0:
                code, name = new_sel.split(" ")[0], new_sel.split(" ")[1]
                gc = get_gsheet_client()
                sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                sh.append_row([code, name, cost, shares, "-"])
                st.cache_data.clear(); st.rerun()

    if not portfolio.empty:
        edited = st.data_editor(portfolio, hide_index=True, use_container_width=True)
        if st.button("💾 儲存並同步到 Google Sheets"):
            gc = get_gsheet_client()
            sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
            sh.clear()
            sh.update('A1', [portfolio.columns.tolist()] + edited[edited['Shares']>0].values.tolist())
            st.cache_data.clear(); st.rerun()

# 底部技術指標圖表
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    st.plotly_chart(plot_v6_chart(p_df, p_name), use_container_width=True)

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
st.set_page_config(page_title="台股戰情指揮中心 V10.0", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .stock-card { border: 1px solid #eee; padding: 18px; border-radius: 12px; background-color: white; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-container { display: flex; justify-content: space-around; background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.03); }
    .metric-item { text-align: center; border-right: 1px solid #eee; flex: 1; }
    .metric-item:last-child { border-right: none; }
    .metric-label { font-size: 0.95em; color: #666; margin-bottom: 8px; font-weight: 500; }
    .metric-value { font-size: 2em; font-weight: 800; color: #1a2a6c; }
    .profit-up { color: #eb093b; font-weight: bold; }
    .profit-down { color: #00a651; font-weight: bold; }
    .group-tag { background-color: #f0f2f6; color: #555; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 核心數據處理 ---

def get_gsheet_client():
    credentials = st.secrets["gcp_service_account"]
    return gspread.service_account_from_dict(credentials)

@st.cache_data(ttl=300)
def load_portfolio():
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        df = pd.DataFrame(sh.sheet1.get_all_records())
        df['Symbol'] = df['Symbol'].astype(str).str.zfill(4)
        return df
    except:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

@st.cache_data(ttl=3600)
def get_market_data():
    url = "https://stock.wespai.com/lists"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        df = pd.read_html(res.text)[0]
        data = df.iloc[:, [0, 1, 2, 3, 14, 15]].copy()
        data.columns = ['代碼', '名稱', '產業', '現價', 'PE', 'PB']
        data['代碼'] = data['代碼'].astype(str).str.zfill(4)
        data['現價'] = pd.to_numeric(data['現價'], errors='coerce')
        data['PE'] = pd.to_numeric(data['PE'], errors='coerce').fillna(999.0)
        data['PB'] = pd.to_numeric(data['PB'], errors='coerce').fillna(999.0)
        return data.set_index('代碼').to_dict('index')
    except Exception as e:
        st.error(f"市場數據抓取失敗: {e}")
        return {}

MARKET_MAP = get_market_data()
STOCK_OPTIONS = [f"{k} {v['名稱']} ({v['產業']})" for k, v in MARKET_MAP.items()]

@st.cache_data(ttl=600)
def fetch_yf_history(symbol):
    time.sleep(random.uniform(0.5, 1.0))
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty:
            df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
        
        # 指標計算：MA
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA60'] = df['Close'].rolling(60).mean()
        
        # 指標計算：RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
        
        # 指標計算：MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
        return df
    except: return None

# --- 2. 側邊導覽 ---
with st.sidebar:
    st.title("🛡️ 數據戰情室")
    if 'menu' not in st.session_state: st.session_state.menu = "portfolio"
    if st.button("🚀 庫存個股監控"): st.session_state.menu = "portfolio"
    if st.button("💰 低基期快篩"): st.session_state.menu = "screening"
    if st.button("🔍 免庫存診斷"): st.session_state.menu = "diagnosis"
    if st.button("📝 庫存清單管理"): st.session_state.menu = "management"

portfolio = load_portfolio()

# --- 各功能區塊 ---
if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存動態監控")
    if not portfolio.empty:
        total_mv, total_cost = 0.0, 0.0
        details = []
        for _, r in portfolio.iterrows():
            m_data = MARKET_MAP.get(r['Symbol'])
            if m_data:
                curr_p = m_data['現價']
                mv = curr_p * r['Shares']
                cv = r['Cost'] * r['Shares']
                total_mv += mv
                total_cost += cv
                details.append({'r': r, 'm': m_data, 'cp': curr_p})

        diff = total_mv - total_cost
        p_ratio = (diff / total_cost * 100) if total_cost > 0 else 0
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item"><div class="metric-label">總資產市值</div><div class="metric-value">${total_mv:,.0f}</div></div>
                <div class="metric-item"><div class="metric-label">未實現損益</div>
                    <div class="metric-value {'profit-up' if diff>=0 else 'profit-down'}">{'+' if diff>=0 else ''}${diff:,.0f}</div>
                    <div class="{'profit-up' if diff>=0 else 'profit-down'}" style="font-weight:bold;">{'+' if diff>=0 else ''}{p_ratio:.2f}%</div>
                </div>
                <div class="metric-item"><div class="metric-label">總投入成本</div><div class="metric-value" style="color:#444;">${total_cost:,.0f}</div></div>
            </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        for i, item in enumerate(details):
            r, m, cp = item['r'], item['m'], item['cp']
            p_pct = (cp - r['Cost']) / r['Cost'] * 100 if r['Cost'] > 0 else 0
            with cols[i % 3]:
                st.markdown(f"""<div class="stock-card">
                    <b>{r['Name']} ({r['Symbol']})</b> <span class="group-tag">{m['產業']}</span><br>
                    <span style="font-size:1.6em;font-weight:bold;">${cp:.2f}</span>
                    <span class="{'profit-up' if p_pct>=0 else 'profit-down'}">{'+' if p_pct>=0 else ''}{p_pct:.2f}%</span>
                </div>""", unsafe_allow_html=True)
                if st.button(f"查看技術分析 {r['Symbol']}", key=f"btn_{r['Symbol']}"):
                    df = fetch_yf_history(r['Symbol'])
                    if df is not None: st.session_state.current_plot = (df, r['Name'])

elif st.session_state.menu == "screening":
    st.subheader("💰 低基期快篩")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 本益比上限", value=15.0)
    pb_lim = c2.number_input("PB 淨值比上限", value=1.2)
    if c3.button("啟動掃描"):
        results = [{'代碼': k, '名稱': v['名稱'], '產業': v['產業'], '現價': v['現價'], 'PE': v['PE'], 'PB': v['PB']} 
                   for k, v in MARKET_MAP.items() if 0 < v['PE'] <= pe_lim and 0 < v['PB'] <= pb_lim]
        st.session_state.scan_results_df = pd.DataFrame(results).sort_values(by=['產業', 'PE'])
    
    if 'scan_results_df' in st.session_state:
        df_display = st.session_state.scan_results_df
        sc_cols = st.columns(3)
        for i, (idx, row) in enumerate(df_display.iterrows()):
            with sc_cols[i % 3]:
                st.markdown(f"""<div class="stock-card"><b>{row['代碼']} {row['名稱']}</b><br>現價: ${row['現價']} | PE: {row['PE']}</div>""", unsafe_allow_html=True)
                if st.button(f"技術診斷 {row['代碼']}", key=f"sc_{row['代碼']}"):
                    df_hist = fetch_yf_history(row['代碼'])
                    if df_hist is not None: st.session_state.current_plot = (df_hist, row['名稱'])

elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 免庫存診斷")
    selection = st.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
    if st.button("執行診斷") and selection != "請選擇...":
        code, name = selection.split(" ")[0], selection.split(" ")[1]
        df = fetch_yf_history(code)
        if df is not None: st.session_state.current_plot = (df, name)

elif st.session_state.menu == "management":
    st.subheader("📝 庫存管理")
    edited = st.data_editor(portfolio, hide_index=True, use_container_width=True)
    if st.button("💾 儲存"):
        gc = get_gsheet_client(); sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
        sh.clear(); sh.update('A1', [portfolio.columns.tolist()] + edited.values.tolist())
        st.cache_data.clear(); st.rerun()

# --- 底部圖表顯示 (新增 MACD 與 功能名稱標籤) ---
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    
    # 建立三層子圖：K線(50%)、RSI(25%)、MACD(25%)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("【 K線與均線分析 】", "【 RSI 強弱指標 】", "【 MACD 趨勢指標 】")
    )

    # 1. K線圖
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color='orange', width=1), name='20MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA60'], line=dict(color='blue', width=1), name='60MA'), row=1, col=1)

    # 2. RSI
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_shape(type="line", x0=p_df.index[0], x1=p_df.index[-1], y0=70, y1=70, line=dict(color="red", dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=p_df.index[0], x1=p_df.index[-1], y0=30, y1=30, line=dict(color="green", dash="dash"), row=2, col=1)

    # 3. MACD
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], line=dict(color='blue'), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], line=dict(color='orange'), name='DEA'), row=3, col=1)
    # MACD 柱狀圖
    colors = ['red' if val >= 0 else 'green' for val in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, title=f"📊 {p_name} 深度技術分析報告")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("關閉圖表"):
        del st.session_state.current_plot
        st.rerun()

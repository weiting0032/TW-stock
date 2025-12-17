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
st.set_page_config(page_title="台股戰情指揮中心 V9.0", layout="wide", page_icon="📈")

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

# --- 1. 核心數據處理 (優先從 Wespai 獲取) ---

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

@st.cache_data(ttl=3600) # 每小時更新一次 Wespai 數據
def get_market_data():
    url = "https://stock.wespai.com/lists"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        df = pd.read_html(res.text)[0]
        # 欄位對應：0:代碼, 1:名稱, 2:產業, 3:現價, 14:PE, 15:PB
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
    """僅在點擊診斷時調用，降低 yfinance 負擔"""
    time.sleep(random.uniform(0.5, 1.0))
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty:
            df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
        
        # 計算診斷所需指標
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA60'] = df['Close'].rolling(60).mean()
        df['SMA240'] = df['Close'].rolling(240).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
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

# --- 功能 A: 庫存個股監控 (利用 Wespai 現價算損益) ---
if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存動態監控 (數據來源: Wespai)")
    if not portfolio.empty:
        total_mv, total_cost = 0.0, 0.0
        details = []

        # 這裡不使用 yfinance，直接從 MARKET_MAP 拿數據
        for _, r in portfolio.iterrows():
            m_data = MARKET_MAP.get(r['Symbol'])
            if m_data:
                curr_p = m_data['現價']
                mv = curr_p * r['Shares']
                cv = r['Cost'] * r['Shares']
                total_mv += mv
                total_cost += cv
                details.append({'r': r, 'm': m_data, 'cp': curr_p})

        # 總資產看板
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
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display:flex; justify-content:space-between;"><b>{r['Name']} ({r['Symbol']})</b> <span class="group-tag">{m['產業']}</span></div>
                    <div style="margin:10px 0;">
                        <span style="font-size:1.6em;font-weight:bold;">${cp:.2f}</span>
                        <span class="{'profit-up' if p_pct>=0 else 'profit-down'}" style="margin-left:10px;">{'+' if p_pct>=0 else ''}{p_pct:.2f}%</span>
                    </div>
                    <div style="font-size:0.85em; color:#666; border-top:1px dashed #eee; padding-top:8px;">
                        PE: {m['PE']} | PB: {m['PB']} | 成本: {r['Cost']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看技術分析 {r['Symbol']}", key=f"btn_{r['Symbol']}"):
                    with st.spinner('抓取歷史 K 線中...'):
                        df = fetch_yf_history(r['Symbol'])
                        if df is not None: st.session_state.current_plot = (df, r['Name'])

# --- 功能 B: 低基期快篩 (維持 V6.7) ---
elif st.session_state.menu == "screening":
    st.subheader("💰 低基期潛力標的快篩 (依族群/PE/PB排序)")
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 本益比上限", value=15.0)
    pb_lim = c2.number_input("PB 淨值比上限", value=1.2)
    
    if c3.button("啟動掃描"):
        # 1. 執行篩選並建立 DataFrame
        results = []
        for k, v in MARKET_MAP.items():
            if 0 < v['PE'] <= pe_lim and 0 < v['PB'] <= pb_lim:
                results.append({
                    '代碼': k,
                    '名稱': v['名稱'],
                    '產業': v['產業'],
                    '現價': v['現價'],
                    'PE': v['PE'],
                    'PB': v['PB']
                })
        
        df_res = pd.DataFrame(results)
        
        if not df_res.empty:
            # 2. 多層次排序：產業(族群) -> PE(低到高) -> PB(低到高)
            df_res = df_res.sort_values(by=['產業', 'PE', 'PB'], ascending=[True, True, True])
            st.session_state.scan_results_df = df_res
        else:
            st.session_state.scan_results_df = pd.DataFrame()

    if 'scan_results_df' in st.session_state:
        df_display = st.session_state.scan_results_df
        if not df_display.empty:
            st.info(f"符合標的共 {len(df_display)} 筆（排序順序：產業 > 本益比 > 淨值比）")
            
            sc_cols = st.columns(3)
            # 使用 iterrows 遍歷排序後的資料
            for i, (idx, row) in enumerate(df_display.iterrows()):
                with sc_cols[i % 3]:
                    st.markdown(f"""
                    <div class="stock-card">
                        <div style="display:flex; justify-content:space-between;">
                            <b>{row['代碼']} {row['名稱']}</b> 
                            <span class="group-tag">{row['產業']}</span>
                        </div>
                        <hr style="margin:8px 0; border:0; border-top:1px solid #eee;">
                        <div style="font-size:1.1em; margin-bottom:5px;">現價: <b>${row['現價']}</b></div>
                        <div style="font-size:0.85em; color:#666;">
                            PE: <span style="color:{'#eb093b' if row['PE'] < 10 else '#444'}">{row['PE']}</span> | 
                            PB: <span style="color:{'#eb093b' if row['PB'] < 1 else '#444'}">{row['PB']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"技術診斷 {row['代碼']}", key=f"sc_{row['代碼']}"):
                        with st.spinner('圖表生成中...'):
                            df_hist = fetch_yf_history(row['代碼'])
                            if df_hist is not None: 
                                st.session_state.current_plot = (df_hist, row['名稱'])
        else:
            st.warning("查無符合條件之標的，請放寬篩選標準。")

# --- 功能 C: 免庫存診斷 ---
elif st.session_state.menu == "diagnosis":
    st.subheader("🔍 全市場技術分析")
    selection = st.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
    if st.button("執行診斷") and selection != "請選擇...":
        code, name = selection.split(" ")[0], selection.split(" ")[1]
        df = fetch_yf_history(code)
        if df is not None: st.session_state.current_plot = (df, name)

# --- 功能 D: 庫存管理 ---
elif st.session_state.menu == "management":
    st.subheader("📝 庫存清單管理")
    edited = st.data_editor(portfolio, hide_index=True, use_container_width=True)
    if st.button("💾 儲存所有變更"):
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
        sh.clear()
        sh.update('A1', [portfolio.columns.tolist()] + edited.values.tolist())
        st.cache_data.clear(); st.rerun()

# 底部圖表顯示
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color='orange'), name='20MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{p_name} 分析報告")
    st.plotly_chart(fig, use_container_width=True)


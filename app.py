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
st.set_page_config(page_title="台股戰情指揮中心 V7.5", layout="wide", page_icon="📈")

# 自訂 CSS：強化資產看板與個股卡片視覺
st.markdown("""
    <style>
    .stock-card { border: 1px solid #eee; padding: 18px; border-radius: 12px; background-color: white; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-container { display: flex; justify-content: space-around; background-color: #ffffff; padding: 25px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.03); }
    .metric-item { text-align: center; border-right: 1px solid #eee; flex: 1; }
    .metric-item:last-child { border-right: none; }
    .metric-label { font-size: 0.95em; color: #666; margin-bottom: 8px; font-weight: 500; }
    .metric-value { font-size: 2em; font-weight: 800; color: #1a2a6c; font-family: 'Inter', sans-serif; }
    .group-tag { background-color: #f0f2f6; color: #555; padding: 2px 10px; border-radius: 6px; font-size: 0.8em; }
    .profit-up { color: #eb093b; } /* 台灣習慣：紅漲 */
    .profit-down { color: #00a651; } /* 台灣習慣：綠跌 */
    </style>
""", unsafe_allow_html=True)

# --- 1. 數據處理函數 ---

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
    except Exception as e:
        st.error(f"讀取 Google Sheets 失敗: {e}")
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
        return data
    except:
        return pd.DataFrame()

STOCK_DF = get_tw_map()
STOCK_OPTIONS = [f"{r['代碼']} {r['名稱']} ({r['產業']})" for _, r in STOCK_DF.iterrows()]

def fetch_stock_info(symbol):
    """獲取現價與技術指標數據"""
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        hist = ticker.history(period="1y")
        if hist.empty:
            ticker = yf.Ticker(f"{symbol}.TWO")
            hist = ticker.history(period="1y")
        
        if hist.empty: return None, None
        
        current_price = hist['Close'].iloc[-1]
        
        # 簡單計算技術指標用於策略判斷 (簡化版)
        hist['SMA20'] = hist['Close'].rolling(20).mean()
        hist['SMA60'] = hist['Close'].rolling(60).mean()
        return current_price, hist
    except:
        return None, None

# --- 2. 主邏輯控制 ---

if 'menu' not in st.session_state: st.session_state.menu = "portfolio"

with st.sidebar:
    st.title("🛡️ 投資指揮部")
    if st.button("🚀 庫存個股監控"): st.session_state.menu = "portfolio"
    if st.button("📝 庫存清單管理"): st.session_state.menu = "management"
    st.divider()
    st.caption("數據來源: Yahoo Finance / Wespai")

# --- A. 庫存監控主畫面 ---
if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股動態監控")
    portfolio = load_portfolio()

    if not portfolio.empty:
        total_market_value = 0
        total_cost = 0
        summary_data = []

        # 顯示載入進度
        with st.spinner('同步市場報價中...'):
            for _, row in portfolio.iterrows():
                price, hist = fetch_stock_info(row['Symbol'])
                if price:
                    mkt_val = price * row['Shares']
                    cost_val = row['Cost'] * row['Shares']
                    total_market_value += mkt_val
                    total_cost += cost_val
                    
                    # 匹配基本面
                    fund = STOCK_DF[STOCK_DF['代碼'] == row['Symbol']]
                    pe = fund['PE'].values[0] if not fund.empty else "-"
                    pb = fund['PB'].values[0] if not fund.empty else "-"
                    industry = fund['產業'].values[0] if not fund.empty else "未知"
                    
                    profit_pct = (price - row['Cost']) / row['Cost'] * 100
                    summary_data.append({
                        **row, 'Price': price, 'MktVal': mkt_val, 
                        'PE': pe, 'PB': pb, 'Industry': industry, 
                        'ProfitPct': profit_pct, 'Hist': hist
                    })

        # --- 1. 總資產看板 (對標圖片需求) ---
        unrealized_profit = total_market_value - total_cost
        profit_ratio = (unrealized_profit / total_cost * 100) if total_cost > 0 else 0
        p_class = "profit-up" if unrealized_profit >= 0 else "profit-down"
        p_sign = "+" if unrealized_profit >= 0 else ""

        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item">
                    <div class="metric-label">總資產市值</div>
                    <div class="metric-value">${total_market_value:,.0f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">總未實現損益</div>
                    <div class="metric-value {p_class}">{p_sign}${unrealized_profit:,.0f}</div>
                    <div class="{p_class}" style="font-weight:bold;">{p_sign}{profit_ratio:.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">總投入成本</div>
                    <div class="metric-value" style="color:#444;">${total_cost:,.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- 2. 個股監控卡片 ---
        cols = st.columns(3)
        for i, s in enumerate(summary_data):
            s_class = "profit-up" if s['ProfitPct'] >= 0 else "profit-down"
            s_sign = "+" if s['ProfitPct'] >= 0 else ""
            
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="stock-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <span style="font-size:1.2em; font-weight:bold;">{s['Name']} ({s['Symbol']})</span>
                            <span class="group-tag">{s['Industry']}</span>
                        </div>
                        <div style="margin-bottom:15px;">
                            <span style="font-size:1.8em; font-weight:800;">${s['Price']:.2f}</span>
                            <span class="{s_class}" style="font-size:1.1em; font-weight:bold; margin-left:10px;">
                                {s_sign}{s['ProfitPct']:.2f}%
                            </span>
                        </div>
                        <div style="display:flex; justify-content:space-between; border-top:1px dashed #eee; pt:10px; font-size:0.9em; color:#666;">
                            <div>PE: <b style="color:#333;">{s['PE']}</b></div>
                            <div>PB: <b style="color:#333;">{s['PB']}</b></div>
                            <div>成本: <b>{s['Cost']}</b></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button(f"🔍 技術分析診斷", key=f"btn_{s['Symbol']}"):
                    st.session_state.current_plot = (s['Hist'], s['Name'])

    else:
        st.info("目前庫存清單為空，請前往「管理」頁面新增。")

# --- B. 庫存管理畫面 ---
elif st.session_state.menu == "management":
    st.subheader("📝 庫存清單管理")
    # (此部分保留您原本的 data_editor 修改邏輯)
    portfolio = load_portfolio()
    
    with st.expander("➕ 新增個股"):
        selection = st.selectbox("選擇標的", options=["請選擇..."] + STOCK_OPTIONS)
        c1, c2 = st.columns(2)
        cost = c1.number_input("平均成本", min_value=0.0)
        shares = c2.number_input("持有股數", min_value=0, step=1000)
        
        if st.button("確認新增"):
            if selection != "請選擇..." and shares > 0:
                code, name = selection.split(" ")[0], selection.split(" ")[1]
                gc = get_gsheet_client()
                sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                sh.append_row([code, name, cost, shares, "-"])
                st.success(f"已新增 {name}")
                st.cache_data.clear()
                st.rerun()

    if not portfolio.empty:
        edited_df = st.data_editor(portfolio, use_container_width=True, hide_index=True)
        if st.button("💾 儲存並同步變更"):
            gc = get_gsheet_client()
            sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
            final_df = edited_df[edited_df['Shares'] > 0]
            sh.clear()
            sh.update('A1', [portfolio.columns.tolist()] + final_df.values.tolist())
            st.success("同步成功！")
            st.cache_data.clear()
            st.rerun()

# --- 底部圖表渲染 ---
if 'current_plot' in st.session_state:
    st.divider()
    df_plot, name_plot = st.session_state.current_plot
    # 這裡可以放置您原本的 plot_v6_chart 函數
    fig = go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['Open'], 
                    high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'])])
    fig.update_layout(title=f"{name_plot} 歷史走勢", height=500)
    st.plotly_chart(fig, use_container_width=True)

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
st.set_page_config(page_title="台股戰情指揮中心 V7.1", layout="wide", page_icon="📈")

# 自訂 CSS (加入損益顏色邏輯)
st.markdown("""
    <style>
    .stock-card { border: 1px solid #ddd; padding: 20px; border-radius: 15px; background-color: white; box-shadow: 3px 3px 10px rgba(0,0,0,0.05); margin-bottom: 15px; }
    .metric-container { display: flex; justify-content: space-between; background-color: #f8f9fa; padding: 20px; border-radius: 15px; margin-bottom: 25px; border: 1px solid #eee; }
    .metric-item { text-align: center; flex: 1; }
    .metric-label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
    .metric-value { font-size: 1.8em; font-weight: bold; color: #1e3c72; }
    .group-tag { background-color: #e1e4e8; color: #444; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; margin-left: 5px; }
    .profit { color: #d32f2f; } /* 台灣習慣紅色為漲/盈 */
    .loss { color: #2e7d32; }   /* 綠色為跌/虧 */
    </style>
""", unsafe_allow_html=True)

# --- 1. 數據獲取 ---
# 
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
    except: return pd.DataFrame()

STOCK_DF = get_tw_map()

def fetch_realtime_price(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        price = ticker.fast_info['last_price']
        if price is None: 
            price = yf.Ticker(f"{symbol}.TWO").fast_info['last_price']
        return price
    except: return None

# --- 2. 核心畫面邏輯 ---

if 'menu' not in st.session_state: st.session_state.menu = "portfolio"

# 側邊欄 (略，同原代碼)

if st.session_state.menu == "portfolio":
    st.subheader("🚀 庫存個股監控")
    portfolio = load_portfolio() # 呼叫您的讀取函數

    if not portfolio.empty:
        # --- A. 計算總體數據 ---
        total_cost = 0
        total_market_value = 0
        
        # 預先抓取所有庫存價格
        stock_data_list = []
        for _, row in portfolio.iterrows():
            current_price = fetch_realtime_price(row['Symbol'])
            if current_price:
                mkt_val = current_price * row['Shares']
                cost_val = row['Cost'] * row['Shares']
                total_cost += cost_val
                total_market_value += mkt_val
                
                # 合併基本面資訊
                fundamental = STOCK_DF[STOCK_DF['代碼'] == row['Symbol']]
                pe = fundamental['PE'].values[0] if not fundamental.empty else "-"
                pb = fundamental['PB'].values[0] if not fundamental.empty else "-"
                industry = fundamental['產業'].values[0] if not fundamental.empty else "未知"
                
                stock_data_list.append({
                    **row, 
                    'Price': current_price, 
                    'MktVal': mkt_val, 
                    'PE': pe, 'PB': pb, 
                    'Industry': industry,
                    'ProfitPct': (current_price - row['Cost']) / row['Cost'] * 100
                })

        # --- B. 總資產看板 ---
        total_profit = total_market_value - total_cost
        profit_color = "#d32f2f" if total_profit >= 0 else "#2e7d32"
        profit_sign = "+" if total_profit >= 0 else ""

        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item">
                    <div class="metric-label">總資產市值</div>
                    <div class="metric-value">${total_market_value:,.0f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">總未實現損益</div>
                    <div class="metric-value" style="color:{profit_color}">{profit_sign}${total_profit:,.0f}</div>
                    <div style="color:{profit_color}">{profit_sign}{total_profit/total_cost*100:.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">總投入成本</div>
                    <div class="metric-value">${total_cost:,.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- C. 個股監控卡片 ---
        cols = st.columns(3)
        for i, s in enumerate(stock_data_list):
            p_color = "#d32f2f" if s['ProfitPct'] >= 0 else "#2e7d32"
            p_sign = "+" if s['ProfitPct'] >= 0 else ""
            
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="stock-card">
                        <div style="display:flex; justify-content:space-between;">
                            <b>{s['Name']} ({s['Symbol']})</b>
                            <span class="group-tag">{s['Industry']}</span>
                        </div>
                        <div style="margin: 10px 0;">
                            <span style="font-size:1.6em; font-weight:bold;">${s['Price']:.2f}</span>
                            <span style="color:{p_color}; margin-left:10px; font-weight:bold;">{p_sign}{s['ProfitPct']:.2f}%</span>
                        </div>
                        <div style="font-size:0.85em; color:#666; display:flex; gap:15px;">
                            <span>PE: <b>{s['PE']}</b></span>
                            <span>PB: <b>{s['PB']}</b></span>
                            <span>成本: {s['Cost']}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看分析", key=f"btn_{s['Symbol']}"):
                    # 這裡調用您原有的 fetch_data_v6 來繪圖
                    d = fetch_data_v6(s['Symbol'])
                    st.session_state.current_plot = (d, s['Name'])

# (其餘 Screening, Diagnosis 管理邏輯維持不變...)

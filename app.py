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
st.set_page_config(page_title="台股戰情指揮中心 V13.1", layout="wide", page_icon="📈")

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
    .function-title { background-color: #1a2a6c; color: white; padding: 10px 20px; border-radius: 5px; margin-bottom: 20px; font-weight: bold; }
    .strategy-tag { font-size: 0.85em; padding: 2px 8px; border-radius: 4px; color: white; font-weight: bold; margin-top: 5px; display: inline-block; }
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
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').fillna(0.0)
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce').fillna(0)
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

def get_strategy_suggestion(df):
    if df is None or df.empty or len(df) < 26: 
        return ("資料不足", "#9e9e9e", "<span>資料不足以產生訊號</span>", "")
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    curr_price = last_row['Close']
    rsi = last_row['RSI']
    macd_hist = last_row['Hist']
    prev_macd_hist = prev_row['Hist']
    bb_lower = last_row['Lower']
    sma20 = last_row['SMA20']
    sma60 = last_row['SMA60']
    
    is_panic = rsi < 25
    is_oversold = rsi < 35
    is_buy_zone = curr_price < bb_lower * 1.02
    macd_turn_up = macd_hist < 0 and macd_hist > prev_macd_hist
    is_bullish_trend = curr_price > sma20 and sma20 > sma60
    
    if is_panic:
        return ("極度恐慌", "#d32f2f", f"<div style='background:#ffebee; padding:10px; border-left:5px solid #d32f2f; border-radius:5px;'><b style='color:#d32f2f'>⚠️ 極度恐慌 (RSI < 25)</b><br>RSI: {rsi:.1f}，市場情緒悲觀。</div>", f"RSI: {rsi:.1f}")
    elif is_oversold and is_buy_zone and macd_turn_up:
        return ("黃金買訊", "#2e7d32", f"<div style='background:#e8f5e9; padding:10px; border-left:5px solid #2e7d32; border-radius:5px;'><b style='color:#2e7d32'>🔥 強力買進訊號</b><br>RSI低檔 + 布林下軌 + MACD轉折。</div>", "技術面買訊")
    elif rsi > 75:
        return ("高檔過熱", "#ef6c00", f"<div style='background:#fff3e0; padding:10px; border-left:5px solid #ef6c00; border-radius:5px;'><b style='color:#ef6c00'>⛔ 高檔過熱 (RSI > 75)</b><br>RSI: {rsi:.1f}，建議減碼。</div>", f"RSI: {rsi:.1f}")
    elif is_bullish_trend and macd_hist > 0:
        return ("多頭續抱", "#1976d2", f"<div style='background:#e3f2fd; padding:10px; border-left:5px solid #1976d2; border-radius:5px;'><b style='color:#1976d2'>📈 多頭排列</b><br>股價動能強勁。</div>", "動能強勁")
    else:
        return ("觀望整理", "#757575", f"<div style='background:#f5f5f5; padding:10px; border-left:5px solid #757575; border-radius:5px;'><b style='color:#616161'>☕ 盤整中</b><br>等待趨勢確立。</div>", f"RSI: {rsi:.1f}")

@st.cache_data(ttl=600)
def fetch_yf_history(symbol):
    time.sleep(random.uniform(0.5, 1.0))
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        df = ticker.history(period="2y", auto_adjust=False)
        if df.empty:
            df = yf.Ticker(f"{symbol}.TWO").history(period="2y", auto_adjust=False)
        
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA60'] = df['Close'].rolling(60).mean()
        std20 = df['Close'].rolling(20).std()
        df['Lower'] = df['SMA20'] - (std20 * 2)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
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

# 載入庫存
if 'df_portfolio' not in st.session_state:
    st.session_state.df_portfolio = load_portfolio()

# --- 各項功能邏輯 ---

if st.session_state.menu == "portfolio":
    st.markdown('<div class="function-title">功能：🚀 庫存動態監控</div>', unsafe_allow_html=True)
    portfolio = st.session_state.df_portfolio
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
                hist_df = fetch_yf_history(r['Symbol'])
                strat_name, strat_color, _, _ = get_strategy_suggestion(hist_df)
                details.append({'r': r, 'm': m_data, 'cp': curr_p, 'strat': (strat_name, strat_color), 'df': hist_df})

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
            r, m, cp, strat, h_df = item['r'], item['m'], item['cp'], item['strat'], item['df']
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
                    <div class="strategy-tag" style="background-color:{strat[1]};">策略建議: {strat[0]}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"查看技術分析 {r['Symbol']}", key=f"btn_{r['Symbol']}"):
                    if h_df is not None: st.session_state.current_plot = (h_df, r['Name'])

elif st.session_state.menu == "screening":
    st.markdown('<div class="function-title">功能：💰 低基期潛力標的快篩</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 本益比上限", value=15.0)
    pb_lim = c2.number_input("PB 淨值比上限", value=1.2)
    
    if c3.button("啟動掃描"):
        results = []
        for k, v in MARKET_MAP.items():
            if 0 < v['PE'] <= pe_lim and 0 < v['PB'] <= pb_lim:
                results.append({'代碼': k, '名稱': v['名稱'], '產業': v['產業'], '現價': v['現價'], 'PE': v['PE'], 'PB': v['PB']})
        df_res = pd.DataFrame(results)
        if not df_res.empty:
            df_res = df_res.sort_values(by=['產業', 'PE', 'PB'], ascending=[True, True, True])
            st.session_state.scan_results_df = df_res
        else:
            st.session_state.scan_results_df = pd.DataFrame()

    if 'scan_results_df' in st.session_state:
        df_display = st.session_state.scan_results_df
        if not df_display.empty:
            st.info(f"符合標的共 {len(df_display)} 筆")
            sc_cols = st.columns(3)
            for i, (idx, row) in enumerate(df_display.iterrows()):
                with sc_cols[i % 3]:
                    h_df = fetch_yf_history(row['代碼'])
                    strat_name, strat_color, _, _ = get_strategy_suggestion(h_df)
                    st.markdown(f"""
                    <div class="stock-card">
                        <div style="display:flex; justify-content:space-between;"><b>{row['代碼']} {row['名稱']}</b><span class="group-tag">{row['產業']}</span></div>
                        <hr style="margin:8px 0; border:0; border-top:1px solid #eee;">
                        <div style="font-size:1.1em; margin-bottom:5px;">現價: <b>${row['現價']}</b></div>
                        <div style="font-size:0.85em; color:#666;">PE: {row['PE']} | PB: {row['PB']}</div>
                        <div class="strategy-tag" style="background-color:{strat_color};">策略建議: {strat_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"技術診斷 {row['代碼']}", key=f"sc_{row['代碼']}"):
                        if h_df is not None: st.session_state.current_plot = (h_df, row['名稱'])

elif st.session_state.menu == "diagnosis":
    st.markdown('<div class="function-title">功能：🔍 全市場技術分析診斷</div>', unsafe_allow_html=True)
    selection = st.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
    if st.button("執行診斷") and selection != "請選擇...":
        code, name = selection.split(" ")[0], selection.split(" ")[1]
        df = fetch_yf_history(code)
        if df is not None: st.session_state.current_plot = (df, name)

elif st.session_state.menu == "management":
    st.markdown('<div class="function-title">功能：📝 庫存清單管理系統</div>', unsafe_allow_html=True)
    
    # 1. 新增標的區塊
    with st.expander("➕ 新增標的至庫存", expanded=True):
        c1, c2, c3 = st.columns(3)
        new_sel = c1.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
        new_cost = c2.number_input("買入單價", min_value=0.0, step=0.1, key="add_cost")
        new_shares = c3.number_input("買入股數", min_value=1, step=100, key="add_shares")
        
        if st.button("確認新增"):
            if new_sel != "請選擇...":
                n_code, n_name = new_sel.split(" ")[0], new_sel.split(" ")[1]
                new_data = {'Symbol': n_code, 'Name': n_name, 'Cost': new_cost, 'Shares': new_shares, 'Note': ''}
                # 更新 session_state 中的 df
                st.session_state.df_portfolio = pd.concat([st.session_state.df_portfolio, pd.DataFrame([new_data])], ignore_index=True)
                st.success(f"✅ 已新增 {n_name}。注意：請務必點擊下方「儲存所有變更」才會寫入 Excel！")
            else:
                st.warning("請先選擇標的。")

    # 2. 顯示與編輯
    st.write("### 庫存列表編輯")
    # 使用 session_state 作為數據源
    edited_df = st.data_editor(st.session_state.df_portfolio, hide_index=True, use_container_width=True, key="portfolio_editor")
    
    if st.button("💾 儲存所有變更"):
        # 股數為 0 自動刪除邏輯
        final_df = edited_df[edited_df['Shares'] > 0].copy()
        with st.spinner('正在同步至 Google Sheets...'):
            try:
                gc = get_gsheet_client()
                sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                sh.clear()
                # 寫入包含標頭的完整資料
                sh.update('A1', [final_df.columns.tolist()] + final_df.values.tolist())
                # 重新載入並清除緩存
                st.session_state.df_portfolio = final_df
                st.cache_data.clear()
                st.success("🎉 資料已成功寫入 Excel！")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 寫入失敗，請檢查權限或網路: {e}")

# --- 底部圖表 ---
if 'current_plot' in st.session_state:
    st.divider()
    p_df, p_name = st.session_state.current_plot
    status, color, html, note = get_strategy_suggestion(p_df)
    st.markdown(f"### 💡 AI 策略詳細分析：{p_name}")
    st.markdown(html, unsafe_allow_html=True)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3],
                        subplot_titles=("股價 K 線與均線", "RSI 強弱指標", "MACD 指標"))
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color='orange', width=1), name='20MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA60'], line=dict(color='blue', width=1), name='60MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Lower'], line=dict(color='rgba(200,200,200,0.5)', dash='dot'), name='BB下軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['RSI'], line=dict(color='purple'), name='RSI(14)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], line=dict(color='blue'), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], line=dict(color='orange'), name='MACD'), row=3, col=1)
    bar_colors = ['#eb093b' if val >= 0 else '#00a651' for val in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], marker_color=bar_colors, name='OSC柱狀圖'), row=3, col=1)
    fig.update_layout(height=850, xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

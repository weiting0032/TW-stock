import streamlit as st
import gspread
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import random
import math
from datetime import datetime, timedelta
import yfinance as yf  # 引入 yfinance 作為穩定備援

# --- 0. 基礎設定 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' 
st.set_page_config(page_title="台股戰情指揮中心 V14 (Pro)", layout="wide", page_icon="📈", initial_sidebar_state="collapsed")

# --- RWD UI 與台股專屬配色 (紅漲綠跌) ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 800; white-space: nowrap !important; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem !important; color: #757575; font-weight: 600; }
    .stMetric { border: 1px solid #e0e0e0; padding: 12px !important; border-radius: 12px; background: #ffffff; box-shadow: 0 2px 6px rgba(0,0,0,0.02); }
    
    .mobile-card { border: 1px solid #eeeeee; border-radius: 16px; padding: 18px; margin-bottom: 15px; background: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.04); transition: transform 0.2s ease; }
    .mobile-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.08); }
    .mobile-card-title { font-size: 1.2rem; font-weight: 800; margin-bottom: 10px; color: #212121; border-bottom: 2px solid #f5f5f5; padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center;}
    .mobile-card-text { font-size: 0.95rem; line-height: 1.6; color: #424242; }
    
    .action-box { margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 10px; font-size: 0.95rem; border-left: 5px solid #1a2a6c; }
    
    .profit-up { color: #eb093b; font-weight: 800; }
    .profit-down { color: #00a651; font-weight: 800; }
    .profit-flat { color: #757575; font-weight: 800; }
    
    .group-tag { background-color: #e3f2fd; color: #1565c0; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; }
    .function-title { background-color: #1a2a6c; color: white; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; font-weight: 800; font-size: 1.2rem; box-shadow: 0 4px 10px rgba(26,42,108,0.2); }
    
    @media (max-width: 600px) {
        [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        .stMetric { padding: 10px !important; }
        .mobile-card { padding: 15px; }
        .mobile-card-title { font-size: 1.1rem; }
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session States ---
if 'menu' not in st.session_state:
    st.session_state.menu = "portfolio"
if 'current_plot' not in st.session_state:
    st.session_state.current_plot = None

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

def calculate_indicators(df):
    """計算所有技術指標"""
    if df is None or len(df) < 60:
        return None
        
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['SMA20'] + (std20 * 2)
    df['BB_Lower'] = df['SMA20'] - (std20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['SMA20'] + 1e-9)
    
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain/(loss+1e-9))))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    
    df['VOL_SMA20'] = df['Volume'].rolling(20).mean()
    return df.dropna().copy()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_history(symbol):
    """資料抓取引擎 (FinMind 優先，YFinance 完美備援)"""
    # 1. 嘗試 FinMind
    try:
        time.sleep(random.uniform(0.1, 0.3))
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockPrice", "data_id": symbol, "start_date": start_date, "end_date": end_date}
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get('msg') == 'success' and data.get('data'):
                df = pd.DataFrame(data['data'])
                df = df.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'trading_volume': 'Volume'})
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df_calc = calculate_indicators(df)
                if df_calc is not None: return df_calc
    except Exception:
        pass # FinMind 失敗，往下走備援
        
    # 2. 備援 YFinance (涵蓋上市 .TW 與上櫃 .TWO)
    try:
        tk = yf.Ticker(f"{symbol}.TW")
        df = tk.history(period="2y", auto_adjust=True)
        if df.empty:
            tk = yf.Ticker(f"{symbol}.TWO")
            df = tk.history(period="2y", auto_adjust=True)
            
        if not df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None) # 移除時區避免繪圖報錯
            df_calc = calculate_indicators(df)
            if df_calc is not None: return df_calc
    except Exception:
        pass
        
    return None

def get_strategy_suggestion(df, held_shares=0):
    """策略核心：回傳結構化訊號、具體價格與建議張數"""
    if df is None or df.empty or len(df) < 20: 
        return {"action": "WATCH", "name": "資料不足", "color": "#9e9e9e", "html": "歷史資料不足以產生技術訊號", "tp": None, "sl": None, "suggest_qty": 0}
        
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    close, rsi, macd_hist = float(last['Close']), float(last['RSI']), float(last['Hist'])
    atr, sma20, sma60 = float(last.get('ATR', 0)), float(last['SMA20']), float(last['SMA60'])
    bb_width, vol, vol_sma20 = float(last['BB_Width']), float(last['Volume']), float(last['VOL_SMA20'])
    
    # 預設防守與獲利點 (ATR倍數法)
    stop_loss = max(close - 1.5 * atr, close * 0.9) if atr else close * 0.9
    take_profit = close + 3 * atr if atr else close * 1.1
    
    # 台股一張為 1000 股，以此為基礎單位建議
    base_qty_lots = 1 if close > 100 else 2 
    
    is_squeeze = bb_width < 0.08
    is_breakout = close > last['BB_Upper'] and vol > vol_sma20 * 1.5
    is_bullish = close > sma20 and sma20 > sma60
    
    res = {"action": "WATCH", "tp": take_profit, "sl": stop_loss, "suggest_qty": 0}

    # 1. 出場訊號
    if held_shares > 0 and (close < sma60 or rsi > 80):
        res.update({
            "action": "SELL", "name": "獲利了結/停損", "color": "#00a651",
            "html": f"<b style='color:#00a651'>📉 跌破支撐或過熱 (RSI: {rsi:.1f})</b><br>動能減弱，建議收回資金。",
            "suggest_qty": math.ceil(held_shares / 1000)
        })
    # 2. 強勢擠壓突破
    elif is_breakout and prev['BB_Width'] < 0.1:
        res.update({
            "action": "BUY", "name": "壓縮放量突破", "color": "#eb093b",
            "html": f"<b style='color:#eb093b'>🚀 布林壓縮後放量突破</b><br>動能爆發，建議順勢進場。",
            "suggest_qty": base_qty_lots
        })
    # 3. 超賣反彈
    elif rsi < 30 and macd_hist > prev['Hist']:
        res.update({
            "action": "BUY", "name": "超賣轉折", "color": "#eb093b",
            "html": f"<b style='color:#eb093b'>🔥 底部背離/轉折 (RSI: {rsi:.1f})</b><br>指標低檔翻揚，具備搶反彈空間。",
            "suggest_qty": base_qty_lots
        })
    # 4. 多頭續抱
    elif is_bullish:
        res.update({
            "action": "HOLD", "name": "多頭續抱", "color": "#f57c00",
            "html": f"<b style='color:#f57c00'>📈 均線多頭排列</b><br>趨勢向上，沿 20MA 操作。",
            "suggest_qty": 0
        })
    # 5. 觀望
    else:
        res.update({
            "action": "WATCH", "name": "觀望整理", "color": "#757575",
            "html": f"<b style='color:#757575'>☕ 盤整無明顯方向</b><br>等待均線糾結或突破表態。",
            "suggest_qty": 0
        })
        
    return res

def format_color(val):
    if val > 0: return f'<span class="profit-up">+{val:,.2f}</span>'
    elif val < 0: return f'<span class="profit-down">{val:,.2f}</span>'
    return f'<span class="profit-flat">{val:,.2f}</span>'

# --- 2. 側邊導覽 ---
with st.sidebar:
    st.title("🛡️ 戰情控制台")
    mobile_mode = st.toggle("📱 手機卡片模式", value=True)
    
    st.markdown("---")
    if st.button("🚀 庫存動態", use_container_width=True): 
        st.session_state.menu = "portfolio"
    if st.button("💰 潛力快篩", use_container_width=True): 
        st.session_state.menu = "screening"
    if st.button("🔍 個股診斷", use_container_width=True): 
        st.session_state.menu = "diagnosis"
    if st.button("📝 庫存管理", use_container_width=True): 
        st.session_state.menu = "management"
    
    if st.button("🔄 強制刷新資料", use_container_width=True):
        st.cache_data.clear()
        st.session_state.current_plot = None
        st.rerun()

# 載入庫存
if 'df_portfolio' not in st.session_state:
    st.session_state.df_portfolio = load_portfolio()

# --- 各項功能邏輯 ---
if st.session_state.menu == "portfolio":
    st.markdown('<div class="function-title">🚀 庫存動態監控 (自動校正 RWD)</div>', unsafe_allow_html=True)
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
                
                h_df = fetch_stock_history(r['Symbol'])
                if h_df is not None:
                    strat = get_strategy_suggestion(h_df, held_shares=r['Shares'])
                    details.append({'r': r, 'm': m_data, 'cp': curr_p, 'strat': strat, 'df': h_df})
                else:
                    # 避免 API 掛掉時整頁崩潰
                    strat_err = {"action": "WATCH", "name": "連線失敗", "color": "#9e9e9e", "html": "無法取得歷史價格", "tp": None, "sl": None, "suggest_qty": 0}
                    details.append({'r': r, 'm': m_data, 'cp': curr_p, 'strat': strat_err, 'df': None})

        diff = total_mv - total_cost
        p_ratio = (diff / total_cost * 100) if total_cost > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("總市值 (TWD)", f"${total_mv:,.0f}")
        m2.metric("未實現損益", f"${diff:,.0f}", f"{p_ratio:.2f}%")
        m3.metric("總投入成本", f"${total_cost:,.0f}")
        
        st.divider()

        cols = st.columns(1 if mobile_mode else 3)
        for i, item in enumerate(details):
            r, m, cp, strat = item['r'], item['m'], item['cp'], item['strat']
            p_pct = (cp - r['Cost']) / r['Cost'] * 100 if r['Cost'] > 0 else 0
            diff_val = (cp - r['Cost']) * r['Shares']
            
            # 強制數值格式化與防空值
            tp_str = f"${strat['tp']:.2f}" if strat.get('tp') is not None else "-"
            sl_str = f"${strat['sl']:.2f}" if strat.get('sl') is not None else "-"
            
            if strat['action'] == "BUY":
                action_html = f"<div class='action-box'>🛒 <b>策略建議：</b><span class='profit-up'>建議加碼 {strat['suggest_qty']} 張 (停損設 {sl_str})</span><br>{strat['html']}</div>"
            elif strat['action'] == "SELL":
                action_html = f"<div class='action-box'>📉 <b>策略建議：</b><span class='profit-down'>建議減碼 {strat['suggest_qty']} 張 (獲利入袋/停損)</span><br>{strat['html']}</div>"
            elif strat['action'] == "HOLD":
                action_html = f"<div class='action-box'>🛡️ <b>策略建議：</b><span style='color:#f57c00;'>持股續抱，跌破 {sl_str} 出場。</span><br>{strat['html']}</div>"
            else:
                action_html = f"<div class='action-box'>👀 <b>策略建議：</b>觀望中，無明確動作。</div>"

            with cols[i % (1 if mobile_mode else 3)]:
                st.markdown(f"""
                <div class="mobile-card">
                    <div class="mobile-card-title">
                        <span>{r['Name']} ({r['Symbol']})</span>
                        <span class="group-tag">{m['產業']}</span>
                    </div>
                    <div class="mobile-card-text">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin:10px 0;">
                            <span style="font-size:2rem; font-weight:800;">${cp:.2f}</span>
                            <span style="font-size:1.2rem;">{format_color(p_pct)}%</span>
                        </div>
                        持有股數：<b>{r['Shares']:,.0f} 股</b> | 損益：<b>{format_color(diff_val)}</b><br>
                        平均成本：${r['Cost']:.2f} | PE: {m['PE']} | PB: {m['PB']}<br>
                        停損防守：<b>{sl_str}</b> | 停利目標：<b>{tp_str}</b>
                    </div>
                    {action_html}
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"📊 看圖 {r['Symbol']}", key=f"btn_{r['Symbol']}", use_container_width=True):
                    if item['df'] is not None: 
                        st.session_state.current_plot = (item['df'], r['Name'], strat)
                    else:
                        st.warning("⚠️ 此標的目前 API 無法取得資料，無法繪圖。")

elif st.session_state.menu == "screening":
    st.markdown('<div class="function-title">💰 價值與動能潛力快篩</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 1])
    pe_lim = c1.number_input("PE 本益比上限 (過濾昂貴股)", value=15.0)
    pb_lim = c2.number_input("PB 淨值比上限 (過濾高估值)", value=1.2)
    
    if c3.button("啟動大盤掃描", use_container_width=True):
        with st.spinner('掃描大盤中...'):
            results = []
            for k, v in MARKET_MAP.items():
                if 0 < v['PE'] <= pe_lim and 0 < v['PB'] <= pb_lim:
                    results.append({'代碼': k, '名稱': v['名稱'], '產業': v['產業'], '現價': v['現價'], 'PE': v['PE'], 'PB': v['PB']})
            df_res = pd.DataFrame(results)
            if not df_res.empty:
                df_res = df_res.sort_values(by=['產業', 'PE', 'PB'])
                st.session_state.scan_results_df = df_res
            else:
                st.session_state.scan_results_df = pd.DataFrame()

    if 'scan_results_df' in st.session_state and not st.session_state.scan_results_df.empty:
        df_display = st.session_state.scan_results_df
        st.success(f"🎯 發現 {len(df_display)} 檔符合條件標的")
        
        sc_cols = st.columns(1 if mobile_mode else 3)
        for i, (idx, row) in enumerate(df_display.head(30).iterrows()):
            with sc_cols[i % (1 if mobile_mode else 3)]:
                h_df = fetch_stock_history(row['代碼'])
                if h_df is not None:
                    strat = get_strategy_suggestion(h_df)
                    tp_str = f"${strat['tp']:.2f}" if strat.get('tp') is not None else "-"
                    sl_str = f"${strat['sl']:.2f}" if strat.get('sl') is not None else "-"
                    
                    st.markdown(f"""
                    <div class="mobile-card">
                        <div class="mobile-card-title">
                            <span>{row['名稱']} ({row['代碼']})</span>
                            <span class="group-tag">{row['產業']}</span>
                        </div>
                        <div class="mobile-card-text">
                            現價: <b style="font-size:1.3rem;">${row['現價']}</b><br>
                            PE: {row['PE']} | PB: {row['PB']}<br>
                            停損: {sl_str} | 目標: {tp_str}
                        </div>
                        <div style="margin-top:10px; padding:6px; background:{strat['color']}22; border-radius:6px; border-left:4px solid {strat['color']}; font-weight:700; color:{strat['color']};">
                            {strat['name']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"技術診斷 {row['代碼']}", key=f"sc_{row['代碼']}", use_container_width=True):
                        st.session_state.current_plot = (h_df, row['名稱'], strat)

elif st.session_state.menu == "diagnosis":
    st.markdown('<div class="function-title">🔍 全市場技術分析診斷</div>', unsafe_allow_html=True)
    selection = st.selectbox("搜尋台股標的", options=["請選擇..."] + STOCK_OPTIONS)
    
    # 核心修正：按鈕點擊後寫入 session_state 即會往下執行底部繪圖，不再呼叫 rerun 導致狀態洗掉
    if st.button("執行深度診斷", use_container_width=True) and selection != "請選擇...":
        code = selection.split(" ")[0]
        name = selection.split(" ")[1]
        with st.spinner("🚀 AI 技術指標運算中... (優先 FinMind，若失敗自動切換 YFinance)"):
            df = fetch_stock_history(code)
            if df is not None: 
                strat = get_strategy_suggestion(df)
                st.session_state.current_plot = (df, name, strat)
            else:
                st.error(f"❌ 取得 {name} ({code}) 歷史資料失敗！API 目前限流且備援線路無回應，請稍後再試。")

elif st.session_state.menu == "management":
    st.markdown('<div class="function-title">📝 雲端庫存清單管理系統</div>', unsafe_allow_html=True)
    
    with st.expander("➕ 新增交易紀錄", expanded=True):
        c1, c2, c3 = st.columns(3)
        new_sel = c1.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS)
        new_cost = c2.number_input("買入單價 (TWD)", min_value=0.0, step=0.1)
        new_shares = c3.number_input("買入股數 (1張=1000)", min_value=1, step=1000, value=1000)
        
        if st.button("暫存至表格", use_container_width=True):
            if new_sel != "請選擇...":
                n_code, n_name = new_sel.split(" ")[0], new_sel.split(" ")[1]
                new_data = {'Symbol': n_code, 'Name': n_name, 'Cost': new_cost, 'Shares': new_shares, 'Note': ''}
                st.session_state.df_portfolio = pd.concat([st.session_state.df_portfolio, pd.DataFrame([new_data])], ignore_index=True)
                st.success(f"✅ 已暫存 {n_name}。請記得點擊下方「儲存至 Google Sheets」！")
            else:
                st.warning("請先選擇標的。")

    st.write("### 庫存列表編輯器")
    edited_df = st.data_editor(st.session_state.df_portfolio, hide_index=True, use_container_width=True, key="portfolio_editor")
    
    if st.button("💾 儲存至 Google Sheets", use_container_width=True, type="primary"):
        final_df = edited_df[edited_df['Shares'] > 0].copy()
        with st.spinner('正在同步至雲端...'):
            try:
                gc = get_gsheet_client()
                sh = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                sh.clear()
                sh.update('A1', [final_df.columns.tolist()] + final_df.values.tolist())
                st.session_state.df_portfolio = final_df
                st.cache_data.clear()
                st.success("🎉 資料同步成功！")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 寫入失敗，請檢查權限或 API 配額: {e}")

# --- 3. 底部圖表渲染區 (全域解耦，確保任何分頁與按鈕皆可穩定觸發) ---
if st.session_state.current_plot is not None:
    st.divider()
    p_df, p_name, strat = st.session_state.current_plot
    
    st.markdown(f"### 💡 AI 策略解析：{p_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("參考現價", f"${p_df['Close'].iloc[-1]:.2f}")
    
    # 這裡的防護是確保圖表區也能正常格式化 TP/SL
    tp_val = f"${strat['tp']:.2f}" if strat.get('tp') is not None else "-"
    sl_val = f"${strat['sl']:.2f}" if strat.get('sl') is not None else "-"
    
    c2.metric("ATR 停損防守", sl_val)
    c3.metric("ATR 停利目標", tp_val)
    
    st.markdown(f"<div class='action-box'>{strat['html']}</div>", unsafe_allow_html=True)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3],
                        subplot_titles=("股價 K 線與布林通道", "RSI 強弱指標", "MACD 動能"))
    
    fig.add_trace(go.Candlestick(
        x=p_df.index, open=p_df['Open'], high=p_df['High'], low=p_df['Low'], close=p_df['Close'], 
        name='K線', increasing_line_color='#eb093b', decreasing_line_color='#00a651'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['SMA20'], line=dict(color='#ff9800', width=1.5), name='20MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['BB_Upper'], line=dict(color='rgba(100,100,100,0.3)', dash='dot'), name='BB上軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['BB_Lower'], fill='tonexty', fillcolor='rgba(23,190,207,0.05)', line=dict(color='rgba(100,100,100,0.3)', dash='dot'), name='BB下軌'), row=1, col=1)
    
    if strat.get('sl'): 
        fig.add_hline(y=strat['sl'], line_dash="dash", line_color="#00a651", row=1, col=1, annotation_text="停損")
    if strat.get('tp'): 
        fig.add_hline(y=strat['tp'], line_dash="dash", line_color="#eb093b", row=1, col=1, annotation_text="停利")

    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['RSI'], line=dict(color='#9c27b0'), name='RSI(14)'), row=2, col=1)
    fig.add_hline(y=75, line_dash="dash", line_color="#eb093b", row=2, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="#00a651", row=2, col=1)
    
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['MACD'], line=dict(color='#1976d2'), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['Signal'], line=dict(color='#ff9800'), name='MACD'), row=3, col=1)
    
    bar_colors = ['#eb093b' if val >= 0 else '#00a651' for val in p_df['Hist']]
    fig.add_trace(go.Bar(x=p_df.index, y=p_df['Hist'], marker_color=bar_colors, name='OSC'), row=3, col=1)
    
    fig.update_layout(height=650 if mobile_mode else 850, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import numpy as np
import os # <--- 修正：確保 os 模組在全域範圍被導入
# 移除所有舊的 gsheets 連線和環境修正代碼

# --- 0. 基礎設定 ---
# 🚨 請將此處替換為您的 Google Sheet 試算表名稱 (例如: Streamlit TW Stock)
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' # <--- 請務必在這裡替換成您的試算表名稱！
STOCK_MAP_FILE = 'tw_stock_map.csv' # 仍保留本地快取

# 版本說明修改
st.set_page_config(page_title="台股戰情指揮中心 V3.4 (Google Sheet 持久化)", layout="wide", page_icon="📈")

# 自訂 CSS
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .stDataFrame { font-size: 16px; }
    /* 調整按鈕大小與間距 */
    .stButton>button { height: 2em; margin: 2px; }
    
    /* 自訂表格樣式 */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .custom-table th, .custom-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    .custom-table th {
        background-color: #f2f2f2;
        color: #333;
    }
    /* 讓名稱看起來像可點擊的連結 */
    .clickable-name {
        color: #1976D2; /* Streamlit Blue */
        cursor: pointer;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Google Sheets 連線核心函數 (使用 gspread) ---

def get_gsheets_client():
    """初始化並返回 gspread 客戶端和試算表對象"""
    try:
        # 1. 驗證連線：使用 st.secrets 自動讀取 .streamlit/secrets.toml
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        
        # 2. 開啟您的試算表 (使用全域變數 PORTFOLIO_SHEET_TITLE)
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        return gc, sh
    except KeyError:
        st.error("⚠️ 錯誤：找不到 st.secrets['gcp_service_account'] 憑證。請檢查 secrets.toml 檔案。")
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"⚠️ 錯誤：找不到名為 '{PORTFOLIO_SHEET_TITLE}' 的試算表。請確認名稱是否正確。")
    except Exception as e:
        if "forbidden" in str(e).lower():
             st.error(f"⚠️ 錯誤：權限不足。請確認已將 Service Account Email 分享給此試算表。")
        else:
            st.error(f"⚠️ 連線 Google Sheets 失敗。錯誤: {e}")
            st.exception(e)
    return None, None

@st.cache_data(ttl=600) # 緩存 600 秒
def load_portfolio():
    """從 Google Sheet 載入投資組合數據 (使用 gspread)"""
    gc, sh = get_gsheets_client()
    if sh is None:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

    try:
        # 讀取整個工作表 (Sheet 1)
        worksheet = sh.sheet1
        # 讀取所有記錄並轉換為 DataFrame
        df = pd.DataFrame(worksheet.get_all_records())
        
        # 清理和確保欄位存在 (假設 Sheets 的第一行是 ['Symbol', 'Name', 'Cost', 'Shares', 'Note'])
        if df.empty or len(df.columns) < 5:
            # 如果讀取是空的或欄位不對，創建一個空的標準結構
            return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

        # 確保欄位名稱正確
        df.columns = ['Symbol', 'Name', 'Cost', 'Shares', 'Note']
        df['Symbol'] = df['Symbol'].astype(str).str.zfill(4)
        
        # 確保數字欄位格式正確
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').fillna(0.0)
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce').fillna(0).astype(int)
        df['Note'] = df['Note'].astype(str).fillna('')
        
        # 僅保留 Symbol 不為空且 Shares >= 0 的行
        df = df[(df['Symbol'] != '') & (df['Symbol'].str.len() >= 4)].copy().reset_index(drop=True) 
        
        return df
    
    except Exception as e:
        st.error(f"⚠️ Google Sheet 讀取工作表內容失敗。錯誤: {e}")
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

def save_portfolio(df):
    """將投資組合數據寫入 Google Sheet (使用 gspread)"""
    # 確保 Name 和 Note 欄位是最新的
    df['Name'] = df['Symbol'].apply(get_stock_name)
    df['Note'] = df['Note'].fillna('')
    
    # 過濾掉 Shares < 0 的錯誤數據
    df_to_save = df[df['Shares'] >= 0].copy()
    
    gc, sh = get_gsheets_client()
    if sh is None:
        return False
        
    try:
        # 寫入 Google Sheet (使用 '工作表1')
        worksheet = sh.sheet1
        
        # 為了覆蓋，先清空工作表內容（保留標題行）
        worksheet.clear()
        
        # 轉換 DataFrame 為列表，包含標題行
        data_list = [df_to_save.columns.values.tolist()] + df_to_save.values.tolist()
        
        # 將資料寫入
        worksheet.update(data_list)
        st.toast("✅ 投資組合已成功儲存至 Google Sheets！", icon='💾')
        return True
    
    except Exception as e:
        st.error(f"⚠️ Google Sheet 儲存失敗。請檢查您的 Sheets 權限是否足夠。錯誤: {e}")
        return False

# --- 1. 股票代碼清單爬蟲與管理 (保持不變) ---
@st.cache_data(ttl=86400)
def get_tw_stock_map():
    """
    從 stock.wespai.com 抓取股票代碼、名稱、產業、P/E、P/B 對照表。
    採用本地 CSV 進行快取，如果網路抓取失敗，則使用本地備份。
    """
    url = "https://stock.wespai.com/lists"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    # 檢查是否在 Streamlit Cloud 環境
    is_cloud_env = 'STREAMLIT_CLOUD' in os.environ # <--- 修正：os 在這裡已經被全域引入
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            dfs = pd.read_html(response.text)
            target_df = None
            for df in dfs:
                if len(df) > 100 and df.shape[1] >= 16: 
                    target_df = df
                    break
            
            if target_df is not None:
                data = target_df.iloc[:, [0, 1, 2, 14, 15]].copy()
                data.columns = ['代碼', '名稱', '產業類別', 'PE', 'PB']
                
                data['代碼'] = data['代碼'].astype(str).str.zfill(4)
                data['名稱'] = data['名稱'].astype(str)
                data['產業類別'] = data['產業類別'].astype(str).replace('nan', 'N/A')
                
                data['PE'] = pd.to_numeric(data['PE'], errors='coerce').round(2)
                data['PB'] = pd.to_numeric(data['PB'], errors='coerce').round(2)
                
                # 在 Streamlit Cloud 上不保證可以寫入本地檔案系統
                if not is_cloud_env: 
                    # 只有在本地環境才嘗試保存快取
                    data.to_csv(STOCK_MAP_FILE, index=False)
                
                return data.set_index('代碼').apply(lambda x: x.to_dict(), axis=1).to_dict()

    except Exception as e:
        st.sidebar.warning(f"網路抓取失敗，嘗試讀取離線檔案。")

    # 讀取本地備份 (僅在本地環境運行時有效)
    if not is_cloud_env and os.path.exists(STOCK_MAP_FILE):
        df = pd.read_csv(STOCK_MAP_FILE, dtype={'代碼': str})
        df['代碼'] = df['代碼'].str.zfill(4)
        df['PE'] = pd.to_numeric(df['PE'], errors='coerce').round(2)
        df['PB'] = pd.to_numeric(df['PB'], errors='coerce').round(2)
        return df.set_index('代碼').apply(lambda x: x.to_dict(), axis=1).to_dict()
    
    # 失敗回退清單
    return {
        "2330": {"名稱": "台積電", "產業類別": "半導體", "PE": np.nan, "PB": np.nan}, 
        "0050": {"名稱": "元大台灣50", "產業類別": "ETF", "PE": np.nan, "PB": np.nan},
    }

# 重新執行抓取代碼
TW_STOCKS = get_tw_stock_map()
STOCK_SEARCH_LIST = [f"{code} {info['名稱']}" for code, info in TW_STOCKS.items()]

def get_stock_name(symbol):
    base_symbol = symbol.split('.')[0]
    return TW_STOCKS.get(base_symbol, {}).get('名稱', symbol)

def get_stock_fundamentals(symbol):
    base_symbol = symbol.split('.')[0]
    info = TW_STOCKS.get(base_symbol, {})
    
    industry = info.get('產業類別', 'N/A')
    pe = info.get('PE')
    pb = info.get('PB')
    
    pe_str = f"{pe:.2f}" if pd.notna(pe) else 'N/A'
    pb_str = f"{pb:.2f}" if pd.notna(pb) else 'N/A'
    
    return industry, pe_str, pb_str

# --- 2. 資料存取函數 (已使用新的 gspread 函數替換舊的 load_portfolio/save_portfolio) ---
# 舊的 load_portfolio/save_portfolio 函數已被上面的新函數覆蓋和替換。


# --- 3. Session State 初始化 (保持不變) ---
if 'input_cost' not in st.session_state: st.session_state.input_cost = 0.0
if 'input_shares' not in st.session_state: st.session_state.input_shares = 0
if 'input_note' not in st.session_state: st.session_state.input_note = ''
if 'search_symbol_key' not in st.session_state: st.session_state.search_symbol_key = ""

if 'portfolio_df' not in st.session_state:
    # 首次載入時呼叫新的 load_portfolio
    st.session_state.portfolio_df = load_portfolio()
    
if 'quick_search_result' not in st.session_state:
    st.session_state.quick_search_result = None
if 'low_base_filter' not in st.session_state: 
    st.session_state.low_base_filter = False 
if 'low_base_df' not in st.session_state: 
    st.session_state.low_base_df = None
if 'detail_symbol' not in st.session_state: 
    st.session_state.detail_symbol = None 
    
if 'max_pe' not in st.session_state:
    st.session_state.max_pe = 15.0
if 'max_pb' not in st.session_state:
    st.session_state.max_pb = 2.0
if 'detail_button_clicked' not in st.session_state:
    st.session_state.detail_button_clicked = None


# --- 4. 指標計算與策略函數 (保持不變) ---

@st.cache_data(ttl=3600)
def get_stock_data(symbol_input, period="1y"):
    symbol = symbol_input.split(' ')[0] if ' ' in symbol_input else symbol_input
    stock_name = get_stock_name(symbol)

    full_symbol = symbol if '.' in symbol else f"{symbol}.TW"
    stock = yf.Ticker(full_symbol)
    
    df = stock.history(period=period)
    if df.empty and '.' not in symbol:
        full_symbol = f"{symbol}.TWO"
        stock = yf.Ticker(full_symbol)
        df = stock.history(period=period)
        
    return df, full_symbol, stock_name

def calculate_indicators(df):
    if df.empty or len(df) < 20: return df
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    df['SMA240'] = df['Close'].rolling(window=240).mean() 
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (df['STD20'] * 2)
    df['Lower'] = df['SMA20'] - (df['STD20'] * 2)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

def get_strategy_suggestion(df):
    if df.empty or len(df) < 26: 
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
    
    short_status = "觀望整理"
    color_code = "#757575" 
    html_msg = ""
    comment = ""

    if is_panic:
        short_status = "極度恐慌"
        color_code = "#d32f2f" 
        comment = f"RSI: {rsi:.1f}，市場情緒悲觀，留意超跌反彈機會。"
        html_msg = f"""<div style='background:#ffebee; padding:10px; border-left:5px solid {color_code}'>
        <b style='color:{color_code}'>⚠️ 極度恐慌 (RSI < 25)</b><br>{comment}</div>"""
        
    elif is_oversold and is_buy_zone and macd_turn_up:
        short_status = "黃金買訊"
        color_code = "#2e7d32" 
        comment = "RSI低檔 + 布林下軌 + MACD轉折，多重訊號支撐。"
        html_msg = f"""<div style='background:#e8f5e9; padding:10px; border-left:5px solid {color_code}'>
        <b style='color:{color_code}'>🔥 強力買進訊號</b><br>{comment}</div>"""
        
    elif rsi > 75:
        short_status = "高檔過熱"
        color_code = "#ef6c00" 
        comment = f"RSI: {rsi:.1f}，短線過熱，建議減碼或觀望。"
        html_msg = f"""<div style='background:#fff3e0; padding:10px; border-left:5px solid {color_code}'>
        <b style='color:{color_code}'>⛔ 高檔過熱 (RSI > 75)</b><br>{comment}</div>"""
        
    elif is_bullish_trend and macd_hist > 0:
        short_status = "多頭續抱"
        color_code = "#1976d2" 
        comment = "股價沿月線上漲，動能強勁，宜順勢操作。"
        html_msg = f"""<div style='background:#e3f2fd; padding:10px; border-left:5px solid {color_code}'>
        <b style='color:{color_code}'>📈 多頭排列</b><br>{comment}</div>"""
        
    else:
        comment = f"RSI: {rsi:.1f}，無明確方向，等待趨勢確立。"
        html_msg = f"""<div style='background:#f5f5f5; padding:10px; border-left:5px solid {color_code}'>
        <b style='color:#616161'>☕ 盤整中</b><br>{comment}</div>"""
        
    return short_status, color_code, html_msg, comment

def plot_stock_chart(df_an, stock_name, selected_symbol):
    """繪製K線圖、RSI和MACD圖表"""
    
    chart_data = df_an.tail(150)
    
    # 根據是否在詳情頁調整高度
    chart_height = 500 if st.session_state.detail_symbol else 700 
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05,
                        subplot_titles=(f"{stock_name} 股價 & 均線/布林", "RSI 強弱指標", "MACD 動能"))
    
    # 1. 股價與均線 (Row 1)
    fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data['Open'], high=chart_data['High'], low=chart_data['Low'], close=chart_data['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA20'], line=dict(color='orange', width=1), name='月線(20MA)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA60'], line=dict(color='cyan', width=1), name='季線(60MA)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['SMA240'], line=dict(color='purple', width=1.5), name='年線(240MA)'), row=1, col=1)
    
    # 布林帶
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Upper'], line=dict(color='rgba(150,150,150,0.3)', dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Lower'], line=dict(color='rgba(150,150,150,0.3)', dash='dot'), fill='tonexty', fillcolor='rgba(150,150,150,0.05)', showlegend=False), row=1, col=1)

    # 2. RSI (Row 2)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="超買")
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="超賣")
    
    # 3. MACD (Row 3)
    colors = ['#ef5350' if v < 0 else '#66bb6a' for v in chart_data['Hist']]
    fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['MACD'], line=dict(color='orange'), name='DIF'), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Signal'], line=dict(color='blue'), name='DEA'), row=3, col=1)
    
    fig.update_layout(height=chart_height, xaxis_rangeslider_visible=False, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)


# --- 5. 側邊欄控制台函數 (保持不變) ---
def autofill_stock_info_fix():
    selected_option = st.session_state.search_symbol_key
    selected_symbol = selected_option.split(' ')[0] if selected_option else None
    
    df = st.session_state.portfolio_df
    
    if selected_symbol and selected_symbol in df['Symbol'].values:
        record = df[df['Symbol'] == selected_symbol].iloc[0]
        st.session_state.input_cost = record['Cost']
        st.session_state.input_shares = record['Shares']
        st.session_state.input_note = record['Note'] if record['Note'] else ''
    else:
        st.session_state.input_cost = 0.0
        st.session_state.input_shares = 0
        st.session_state.input_note = ''

def low_base_screening(max_pe, max_pb):
    """根據 P/E > 0, P/E ≤ max_pe 和 P/B ≤ max_pb 篩選標的，並進行排序。"""
    
    data_list = []
    
    for code, info in TW_STOCKS.items():
        if pd.notna(info.get('PE')) and pd.notna(info.get('PB')):
            data_list.append({
                "代碼": code,
                "名稱": info['名稱'],
                "產業類別": info['產業類別'],
                "PE": info['PE'],
                "PB": info['PB'],
            })
    
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    
    # 2. 篩選條件 (確保 PE > 0)
    df_filtered = df[
        (df['PE'] > 0) &
        (df['PE'] <= max_pe) & 
        (df['PB'] <= max_pb)
    ].copy()
    
    # 3. 排序 (產業類別 -> 低 PE -> 低 PB)
    df_sorted = df_filtered.sort_values(
        by=['產業類別', 'PE', 'PB'],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return df_sorted

# --- 6. 側邊欄渲染 (保持不變) ---
st.sidebar.title("🎛️ 指揮控制台")

# A. 新增/更新 庫存
with st.sidebar.expander("➕ 新增/更新 監控標的", expanded=False):
    
    search_symbol = st.selectbox(
        "搜尋股票", 
        options=[""] + STOCK_SEARCH_LIST,
        key="search_symbol_key", 
    )
    
    # 執行 autofill_stock_info_fix
    autofill_stock_info_fix()

    in_cost = st.number_input("平均成本", min_value=0.0, step=0.1, key="input_cost")
    in_shares = st.number_input("持有股數", min_value=0, step=1000, key="input_shares")
    in_note = st.text_input("備註", key="input_note")
    
    c1, c2 = st.columns(2)
    
    if c1.button("💾 儲存/更新", key="save_button"):
        if search_symbol:
            in_symbol = search_symbol.split(' ')[0]
            in_name = get_stock_name(in_symbol)
            df = st.session_state.portfolio_df
            new_cost = st.session_state.input_cost
            new_shares = st.session_state.input_shares
            new_note = st.session_state.input_note
            
            if in_symbol in df['Symbol'].values:
                df.loc[df['Symbol'] == in_symbol, ['Cost', 'Shares', 'Note', 'Name']] = [new_cost, new_shares, new_note, in_name]
            else:
                new_row = pd.DataFrame({'Symbol':[in_symbol], 'Name':[in_name], 'Cost':[new_cost], 'Shares':[new_shares], 'Note':[new_note]})
                df = pd.concat([df, new_row], ignore_index=True)
            
            st.session_state.portfolio_df = df
            # 呼叫新的 save_portfolio
            save_portfolio(df)
            st.success(f"已更新 {in_name} (股數: {new_shares})")
            st.cache_data.clear() # 清除快取，確保下次 load_portfolio 讀取新數據
            st.rerun()

    if c2.button("🗑️ 刪除", key="delete_button"):
            if search_symbol:
                in_symbol = search_symbol.split(' ')[0]
                df = st.session_state.portfolio_df
                st.session_state.portfolio_df = df[df['Symbol'] != in_symbol]
                # 呼叫新的 save_portfolio
                save_portfolio(st.session_state.portfolio_df)
                st.warning("已刪除該監控標的")
                st.session_state.search_symbol_key = "" 
                st.session_state.input_cost = 0.0
                st.session_state.input_shares = 0
                st.session_state.input_note = ''
                st.cache_data.clear() # 清除快取，確保下次 load_portfolio 讀取新數據
                st.rerun()

# B. 低基期標的快篩
st.sidebar.markdown("---")
with st.sidebar.expander("💰 低基期標的快篩", expanded=True):
    
    with st.form("low_base_form"):
        st.caption("設定篩選條件：(PE需 > 0)")
        pe_col, pb_col = st.columns(2)
        
        max_pe_input = pe_col.number_input("本益比上限 (PE ≤)", min_value=1.0, max_value=50.0, value=st.session_state.max_pe, step=1.0, key='max_pe_input')
        max_pb_input = pb_col.number_input("股價淨值比上限 (PB ≤)", min_value=0.5, max_value=10.0, value=st.session_state.max_pb, step=0.1, key='max_pb_input')
        
        submit_button = st.form_submit_button("執行快篩", type="primary")

    if submit_button:
        st.session_state.quick_search_result = None 
        st.session_state.detail_symbol = None 
        
        st.session_state.max_pe = max_pe_input
        st.session_state.max_pb = max_pb_input
        
        df_result = low_base_screening(st.session_state.max_pe, st.session_state.max_pb)
        
        if df_result.empty:
            st.error("查無符合條件的標的。請放寬條件。")
            st.session_state.low_base_filter = False
            st.session_state.low_base_df = None
        else:
            st.session_state.low_base_filter = True
            st.session_state.low_base_df = df_result
        st.rerun()


# C. 個股快篩 (非庫存)
st.sidebar.markdown("---")
with st.sidebar.expander("🔍 個股快篩 (免庫存)", expanded=False):
    qs_input = st.selectbox("輸入代號或名稱查詢動向", options=[""] + STOCK_SEARCH_LIST, key="qs")
    if st.button("分析動向"):
        if qs_input:
            q_sym = qs_input.split(' ')[0]
            st.session_state.quick_search_result = q_sym
            st.session_state.selected_symbol_main = None 
            st.session_state.low_base_filter = False 
            st.session_state.detail_symbol = None 
            st.rerun() 

# --- 7. 主畫面邏輯與渲染 ---

# ----------------------------------------------------------------------
# 核心修正: 詳情頁面渲染邏輯 (取代彈窗)
# ----------------------------------------------------------------------

if st.session_state.detail_symbol:
    detail_sym = st.session_state.detail_symbol
    
    # --- 詳情內容開始 ---
    
    st.title(f"🎯 {get_stock_name(detail_sym)} ({detail_sym}) 詳細戰情分析")
    
    # 關閉按鈕：返回篩選列表
    if st.button("⬅️ 返回低基期標的快篩結果", key="return_to_filter"):
        st.session_state.detail_symbol = None
        # 確保回到快篩結果列表 (低基期篩選狀態保持 True)
        st.session_state.low_base_filter = True 
        st.rerun()
             
    st.markdown("---")

    # 載入數據與分析
    with st.spinner(f"正在抓取並分析 **{get_stock_name(detail_sym)} ({detail_sym})** 的詳細數據..."):
        # 抓取較長數據以確保圖表完整性
        data_df, _, name = get_stock_data(detail_sym, period="1y") 
    
    if not data_df.empty and len(data_df) >= 26:
        data_df_an = calculate_indicators(data_df)
        last_row = data_df_an.iloc[-1]
        
        _, _, strat_html, _ = get_strategy_suggestion(data_df_an)
        industry, pe_str, pb_str = get_stock_fundamentals(detail_sym)

        # 股價/指標資訊
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        p_c1.metric("現價", f"{last_row['Close']:.2f}")
        p_c2.metric("RSI (14)", f"{last_row['RSI']:.1f}")
        p_c3.metric("本益比 (PE)", pe_str)
        p_c4.metric("股價淨值比 (PB)", pb_str)
        st.info(f"**產業類別:** {industry}")

        # 建議
        st.markdown("---")
        st.markdown("🧠 **戰情分析官建議**")
        st.markdown(strat_html, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("📈 **技術趨勢圖** (近一年)")
        # 使用原來的 plot_stock_chart 函數
        plot_stock_chart(data_df_an, name, detail_sym)
            
    else:
        st.warning(f"查無 {detail_sym} 足夠的技術分析數據。")
    
    # 底部關閉按鈕，確保能返回
    st.markdown("---")
    if st.button("⬅️ 返回低基期標的快篩結果", key="return_to_filter_bottom"): 
        st.session_state.detail_symbol = None
        st.session_state.low_base_filter = True
        st.rerun()
        
    st.stop() # 詳情頁面渲染結束，停止後續渲染


# 優先處理低基期快篩結果 (主頁面)
if st.session_state.low_base_filter and st.session_state.low_base_df is not None:
    
    df_result = st.session_state.low_base_df.copy()
    
    st.title("💰 低基期標的快篩結果")
    st.subheader(f"篩選條件: PE > 0, PE ≤ {st.session_state.max_pe:.1f}, PB ≤ {st.session_state.max_pb:.1f}")
    st.info(f"共篩選出 **{len(df_result)}** 檔符合條件的標的，點擊 **名稱** 查看詳細戰情分析。")
    
    # --- 模擬表格 ---
    
    # 1. 渲染表格標題
    header_cols = st.columns([1, 2, 2, 1, 1])
    headers = ["代碼", "名稱 (點擊查看詳情)", "產業類別", "PE", "PB"]
    for col, header_text in zip(header_cols, headers):
        col.markdown(f"**{header_text}**", unsafe_allow_html=True)
    st.markdown("---") # 分隔線
    
    # 2. 渲染每一行數據
    for i, row in df_result.iterrows():
        sym = row['代碼']
        name = row['名稱']
        
        # 使用 st.columns 模擬表格行
        data_cols = st.columns([1, 2, 2, 1, 1]) 
        
        # 欄位 1: 代碼 (普通文字)
        data_cols[0].write(sym)
        
        # 欄位 2: 名稱 (可點擊按鈕，模擬連結)
        # 關鍵修正：點擊後設定 detail_symbol
        if data_cols[1].button(
            name, 
            key=f"detail_name_{sym}",
            help="點擊此處查看詳細技術分析" 
        ):
            st.session_state.detail_symbol = sym
            st.session_state.low_base_filter = True # 保持篩選狀態為 True
            st.rerun() # 觸發詳情頁面邏輯
        
        # 欄位 3, 4, 5: 產業類別, PE, PB
        data_cols[2].write(row['產業類別'])
        data_cols[3].write(f"{row['PE']:.2f}")
        data_cols[4].write(f"{row['PB']:.2f}")
        
    st.markdown("---")
    
    st.stop()
    

# 優先處理個股快篩結果 (非庫存)
if st.session_state.quick_search_result:
    qs_sym = st.session_state.quick_search_result
    
    with st.spinner(f"正在分析 {qs_sym} 的最新戰情..."):
        time.sleep(1) 
        q_df, _, q_name = get_stock_data(qs_sym, period="2y")
        
        if not q_df.empty and len(q_df) >= 26:
            q_df_an = calculate_indicators(q_df)
            last = q_df_an.iloc[-1]
            
            industry, pe, pb = get_stock_fundamentals(qs_sym)
            
            st.title("🔎 個股戰情快篩")
            st.subheader(f"{q_name} ({qs_sym}) 即時戰情")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", f"{last['Close']:.2f}")
            c2.metric("RSI (14)", f"{last['RSI']:.1f}")
            c3.metric("本益比 (PE)", f"{pe}")
            c4.metric("股價淨值比 (PB)", f"{pb}")
            st.info(f"**產業類別:** {industry}")

            st.markdown("---")
            st.subheader("🧠 戰情分析官建議")
            _, _, strat_html, _ = get_strategy_suggestion(q_df_an)
            st.markdown(strat_html, unsafe_allow_html=True)
            
            st.subheader("📈 技術趨勢圖")
            plot_stock_chart(q_df_an, q_name, qs_sym)
            
        else:
            st.error(f"❌ 查無 {qs_sym} 資料或資料不足，無法進行分析。")
            st.session_state.quick_search_result = None
    
    st.stop() 

# 正常庫存模式 (預設畫面)
portfolio = st.session_state.portfolio_df
if portfolio.empty:
    st.title("🚀 台股戰情分析室 V3.4 (Google Sheet 持久化)")
    st.info("⚠️ 警告：Google Sheet 載入失敗或庫存為空。\n\n請在側邊欄 **「新增/更新 監控標的」** 中加入您的股票，或使用 **「低基期標的快篩」** 尋找潛力標的。")
    st.stop()

# 庫存選擇邏輯
options = [f"{r['Symbol']} {r['Name']}" for i, r in portfolio.iterrows()]
if 'selected_symbol_main' not in st.session_state or st.session_state.selected_symbol_main not in [r['Symbol'] for i, r in portfolio.iterrows()]:
    # 檢查 options 是否為空，以防讀取 Sheets 失敗
    if options:
        st.session_state.selected_symbol_main = options[0].split(' ')[0]
    else:
        # 如果選項仍為空，則停止
        st.stop()

col1, col2 = st.columns([3, 2])
with col1: st.title("🚀 台股戰情分析室 V3.4 (Google Sheet 持久化)")
with col2: 
    # 確保 sel_opt 存在於 options 中，防止讀取失敗導致錯誤
    try:
        index = options.index(f"{st.session_state.selected_symbol_main} {get_stock_name(st.session_state.selected_symbol_main)}")
    except ValueError:
        index = 0
        
    sel_opt = st.selectbox("切換庫存戰情視角", options, index=index)
    sel_sym = sel_opt.split(' ')[0]
    st.session_state.selected_symbol_main = sel_sym

# 抓取並分析資料
raw_df, yf_sym, stock_name = get_stock_data(sel_sym, period="2y")
if raw_df.empty or len(raw_df) < 2: st.error("資料讀取失敗"); st.stop()
df_an = calculate_indicators(raw_df)
last = df_an.iloc[-1]
prev = df_an.iloc[-2]

# 取得基本面數據
industry, pe, pb = get_stock_fundamentals(sel_sym)

# 計算基本數值 (庫存模式)
curr_rec = portfolio[portfolio['Symbol'] == sel_sym].iloc[0]
my_shares = curr_rec['Shares']
my_cost = curr_rec['Cost']
mkt_val = last['Close'] * my_shares
cost_val = my_cost * my_shares
profit = mkt_val - cost_val
# 避免除以零
profit_pct = (profit / cost_val * 100) if cost_val > 0 and my_shares > 0 else 0
diff_pct = (last['Close'] - prev['Close']) / prev['Close'] * 100

# --- 8. 渲染 Tab 內容 ---
tab1, tab2, tab3 = st.tabs(["📊 個股戰情", "🏦 資產與建議總覽", "📋 原始數據"])

with tab1:
    st.subheader(f"{stock_name} ({sel_sym}) 最新數據")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("現價", f"{last['Close']:.2f}", f"{diff_pct:.2f}%")
    c2.metric("庫存損益", f"${int(profit):,}", f"{profit_pct:.2f}%")
    c3.metric("本益比 (PE)", f"{pe}")
    c4.metric("股價淨值比 (PB)", f"{pb}")
    st.info(f"**產業類別:** {industry}")

    st.markdown("---")
    st.subheader("🧠 戰情分析官建議")
    _, _, strat_html, _ = get_strategy_suggestion(df_an)
    st.markdown(strat_html, unsafe_allow_html=True)
    
    st.subheader("📈 技術趨勢圖")
    plot_stock_chart(df_an, stock_name, sel_sym)

with tab2:
    st.subheader("🏦 投資組合總覽 & 操作建議")
    st.caption("✨ 系統會抓取最新數據並計算操作建議，請耐心等候。")
    
    total_mkt = 0
    total_cost = 0
    rows = []
    
    # 確保 portfolio_df 不為空，以防 load_portfolio 失敗返回空 DF
    if not portfolio.empty:
        progress = st.progress(0, text="載入中...")
        
        for i, r in portfolio.iterrows():
            d, _, _ = get_stock_data(r['Symbol'], period="6mo") 
            
            advice = "N/A"
            cp = 0.0
            pl = 0
            pl_pct = 0
            
            curr_industry, curr_pe, curr_pb = get_stock_fundamentals(r['Symbol'])
            
            if not d.empty and len(d) >= 26: 
                d = calculate_indicators(d)
                cp = d['Close'].iloc[-1]
                s_txt, s_col, _, _ = get_strategy_suggestion(d)
                advice = f"<span style='color:{s_col}; font-weight:bold'>{s_txt}</span>"
                
                mv = cp * r['Shares']
                cv = r['Cost'] * r['Shares']
                total_mkt += mv
                total_cost += cv
                pl = mv - cv
                pl_pct = (pl / cv * 100) if cv > 0 else 0
                
            note_display = r['Note'] if r['Note'] else ''
            
            rows.append({
                "代碼": r['Symbol'],
                "名稱": r['Name'],
                "產業類別": curr_industry, 
                "本益比 (PE)": curr_pe,   
                "股價淨值比 (PB)": curr_pb, 
                "操作建議": advice,
                "現價": f"{cp:.2f}",
                "損益": int(pl),
                "損益%": f"{pl_pct:.2f}%", 
                "市值": int(cp * r['Shares']),
                "股數": int(r['Shares']),
                "平均成本": f"{r['Cost']:.2f}",
                "備註": note_display
            })
            progress.progress((i+1)/len(portfolio), text=f"正在計算 {r['Name']} ({r['Symbol']})...")
        
        progress.empty()
    
    total_pl = total_mkt - total_cost
    pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("總資產市值", f"${int(total_mkt):,}")
    m2.metric("總未實現損益", f"${int(total_pl):,}", f"{pct:.2f}%")
    m3.metric("總投入成本", f"${int(total_cost):,}")
    
    st.divider()
    
    if rows:
        df_show = pd.DataFrame(rows)
        # 增加損益、市值和股數的格式化
        df_show['損益'] = df_show['損益'].apply(lambda x: f'${x:,}')
        df_show['市值'] = df_show['市值'].apply(lambda x: f'${x:,}')
        df_show['股數'] = df_show['股數'].apply(lambda x: f'{x:,}')
        
        # 使用 markdown 模擬表格並允許 HTML 渲染建議顏色
        st.markdown(
            df_show.to_html(escape=False, index=False), 
            unsafe_allow_html=True
        )
    else:
        st.info("目前投資組合為空，或 Google Sheet 載入失敗。")


with tab3:
    st.subheader(f"📋 {stock_name} 原始數據檢視")
    st.dataframe(df_an.sort_index(ascending=False), use_container_width=True)

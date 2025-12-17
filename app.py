import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import os
from datetime import datetime, date

# --- 0. 基礎設定 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' # 請確保與您的 Google Sheet 名稱一致
st.set_page_config(page_title="台股戰情中心 V6 回測版", layout="wide", page_icon="📈")
st.title("🚀 台股戰情室 V6 策略整合版")

# --- 1. Google Sheets 連線函數 ---
def get_gsheets_client():
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        return gc, sh
    except Exception as e:
        st.error(f"❌ Google Sheets 連線失敗: {e}")
        return None, None

def load_portfolio():
    gc, sh = get_gsheets_client()
    if sh is None: return pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Price', 'Shares'])
    try:
        worksheet = sh.sheet1
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame(columns=['Date', 'Symbol', 'Type', 'Price', 'Shares'])

def save_trade_to_sheets(date_val, symbol, trans_type, price, shares):
    gc, sh = get_gsheets_client()
    if sh:
        try:
            worksheet = sh.sheet1
            worksheet.append_row([str(date_val), symbol, trans_type, price, shares])
            return True
        except Exception as e:
            st.error(f"儲存失敗: {e}")
    return False

# --- 2. 側邊欄控制台 ---
st.sidebar.header("🕹️ 控制台")
initial_capital = st.sidebar.number_input("初始資金 (TWD)", value=1000000, step=10000)

with st.sidebar.form("trade_form"):
    st.markdown("### 📝 交易記帳")
    d = st.date_input("日期", date.today())
    sym = st.text_input("股票代碼", value="2330")
    t = st.selectbox("類別", ["買入", "賣出"])
    p = st.number_input("成交價格", min_value=0.0)
    s = st.number_input("成交股數", min_value=0.0)
    if st.form_submit_button("同步至雲端"):
        if save_trade_to_sheets(d, sym, t, p, s):
            st.sidebar.success("已同步至 Google Sheets")
            st.rerun()

# --- 3. 核心指標運算 ---
target_stock = st.text_input("🔍 輸入要分析的台股代碼", value="2330")
full_symbol = f"{target_stock}.TW" if ".TW" not in target_stock else target_stock

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="2y", auto_adjust=False)
    return df

hist = fetch_data(full_symbol)

if not hist.empty:
    # --- 技術指標計算 (NVDA V6 邏輯) ---
    hist['SMA20'] = hist['Close'].rolling(20).mean()
    hist['SMA60'] = hist['Close'].rolling(60).mean()
    hist['SMA200'] = hist['Close'].rolling(200).mean()
    
    std = hist['Close'].rolling(20).std()
    hist['BB_upper'] = hist['SMA20'] + 2 * std
    hist['BB_lower'] = hist['SMA20'] - 2 * std
    hist['BB_pos'] = (hist['Close'] - hist['BB_lower']) / (hist['BB_upper'] - hist['BB_lower']) * 100
    
    delta = hist['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    hist['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = ema12 - ema26
    hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
    hist['Hist'] = hist['MACD'] - hist['Signal']

    # --- 4. 策略邏輯分析 ---
    row = hist.iloc[-1]
    prev_row = hist.iloc[-2]
    curr_price = float(row['Close'])
    
    # 趨勢與分數判斷
    bull_trend = curr_price > row['SMA200']
    oversold_rsi = 40 if bull_trend else 30
    overbought_rsi = 78 if bull_trend else 70
    
    score = 0
    score += 1 if row['RSI'] < oversold_rsi else 0
    score += 1 if row['BB_pos'] < 15 else 0
    score += 1 if (row['Hist'] > prev_row['Hist'] and row['MACD'] > 0) else 0
    score += 1 if bull_trend else 0

    # 決策動作
    action = "觀望 (HOLD)"
    shares_advice = 0
    
    # 防守賣出邏輯
    trend_break = curr_price < row['SMA60'] and row['SMA20'] < row['SMA60']
    bull_protect = bull_trend and (curr_price > row['SMA60']) # 多頭保護條件
    
    if not bull_protect and (row['RSI'] > overbought_rsi or row['BB_pos'] > 85 or trend_break):
        action = "⚠️ 減碼/賣出 (SELL)"
    elif score >= 3:
        action = "🔥 強烈買進 (STRONG BUY)"
    elif score == 2:
        action = "分批買進 (BUY)"

    # --- 5. 畫面渲染 ---
    st.subheader(f"📊 {full_symbol} 策略訊號")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("建議行動", action)
    c2.metric("RSI (14)", f"{row['RSI']:.1f}")
    c3.metric("布林位置", f"{row['BB_pos']:.1f}%")
    c4.metric("策略評分", f"{score} / 4")

    # --- 6. 技術圖表 ---
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
    
    # 主圖
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA60'], line=dict(color='cyan', width=1.5), name="60MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='magenta', width=2), name="200MA"), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='orange'), name="RSI"), row=2, col=1)
    fig.add_hline(y=oversold_rsi, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=overbought_rsi, line_dash="dash", line_color="red", row=2, col=1)
    
    # MACD
    colors = ['#2E8B57' if v >= 0 else '#CD5C5C' for v in hist['Hist']]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Hist'], marker_color=colors, name="MACD柱"), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- 7. 交易紀錄顯示 ---
    st.subheader("📋 雲端交易紀錄 (Google Sheets)")
    trades_df = load_portfolio()
    if not trades_df.empty:
        st.dataframe(trades_df.sort_index(ascending=False), use_container_width=True)
    else:
        st.info("目前尚無交易紀錄，請由左側側邊欄輸入。")

else:
    st.error("找不到股票數據，請確認代碼是否正確（例如台積電請輸入 2330）。")

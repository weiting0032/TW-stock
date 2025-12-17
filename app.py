import streamlit as st
import gspread
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import math
from datetime import datetime, date

# --- 0. 基礎設定 ---
PORTFOLIO_SHEET_TITLE = 'Streamlit TW Stock' 
st.set_page_config(page_title="台股戰情指揮中心 V6 策略版", layout="wide", page_icon="📈")

# --- 1. Google Sheets 連線與資料存取 ---
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
    if sh is None: return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])
    try:
        worksheet = sh.sheet1
        df = pd.DataFrame(worksheet.get_all_records())
        if df.empty: return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])
        return df
    except:
        return pd.DataFrame(columns=['Symbol', 'Name', 'Cost', 'Shares', 'Note'])

def save_to_sheets(symbol, name, cost, shares, note):
    gc, sh = get_gsheets_client()
    if sh:
        try:
            worksheet = sh.sheet1
            worksheet.append_row([symbol, name, float(cost), float(shares), note])
            st.sidebar.success(f"✅ {symbol} 紀錄已同步至雲端")
        except Exception as e:
            st.sidebar.error(f"儲存失敗: {e}")

# --- 2. 側邊欄控制台 ---
st.sidebar.header("🕹️ 控制台")
initial_capital = st.sidebar.number_input("初始資金", value=1000000)

with st.sidebar.form("trade_form"):
    st.markdown("### 📝 手動記帳")
    in_sym = st.text_input("股票代碼 (例: 2330)")
    in_type = st.selectbox("類型", ["買入", "賣出"])
    in_price = st.number_input("成交價", min_value=0.0)
    in_shares = st.number_input("股數", min_value=0.0)
    in_note = st.text_input("備註", value="策略執行")
    if st.form_submit_button("送出並同步雲端"):
        save_to_sheets(in_sym, in_sym, in_price, in_shares, in_type + ":" + in_note)
        st.rerun()

# --- 3. 核心指標運算函數 (NVDA 策略邏輯) ---
def calculate_v6_metrics(df):
    # 均線
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # 布林通道
    std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['SMA20'] + 2 * std
    df['BB_lower'] = df['SMA20'] - 2 * std
    df['BB_pos'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower']) * 100
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

# --- 4. 主畫面邏輯 ---
target_stock = st.text_input("🔍 輸入標的代碼", value="2330")

if target_stock:
    # 這裡保留彈性，不強制加 .TW，由用戶輸入或代碼邏輯決定
    stock_df = yf.Ticker(target_stock).history(period="2y", auto_adjust=False)
    
    if not stock_df.empty:
        df = calculate_v6_metrics(stock_df)
        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # --- 策略核心判斷 (NVDA 邏輯) ---
        price = row['Close']
        bull_trend = price > row['SMA200']
        oversold_rsi = 40 if bull_trend else 30
        overbought_rsi = 78 if bull_trend else 70
        
        # 評分系統
        score = 0
        score += 1 if row['RSI'] < oversold_rsi else 0
        score += 1 if row['BB_pos'] < 15 else 0
        score += 1 if (row['Hist'] > prev_row['Hist'] and row['MACD'] > 0) else 0
        score += 1 if bull_trend else 0
        
        # 決策動作
        action = "觀望 (HOLD)"
        trend_break = price < row['SMA60'] and row['SMA20'] < row['SMA60']
        bull_protect = bull_trend and (price > row['SMA60'])
        
        if not bull_protect and (row['RSI'] > overbought_rsi or row['BB_pos'] > 85 or trend_break):
            action = "⚠️ 賣出/減碼 (SELL)"
        elif score >= 3:
            action = "🔥 強烈買進 (STRONG BUY)"
        elif score == 2:
            action = "買進 (BUY)"
            
        # --- 顯示結果 ---
        st.subheader(f"📊 {target_stock} 戰情分析")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("現價", f"{price:.2f}")
        m2.metric("RSI", f"{row['RSI']:.1f}")
        m3.metric("操作建議", action)
        m4.metric("策略評分", f"{score}/4")

        # 圖表展示
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='cyan', width=1.5), name="60MA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='magenta', width=2), name="200MA"), row=1, col=1)
        
        colors = ['#00ff00' if v >= 0 else '#ff4b4b' for v in df['Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], marker_color=colors, name="MACD柱"), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 5. 資產概況 (從雲端讀取) ---
        st.subheader("🏦 雲端資產概況")
        portfolio = load_portfolio()
        if not portfolio.empty:
            # 這裡可以加入您原本的損益計算邏輯
            st.dataframe(portfolio, use_container_width=True)
        else:
            st.info("雲端目前無紀錄。")
    else:
        st.error("找不到股票數據，請確認代碼。")

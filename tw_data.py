"""資料抓取層（Google Sheets、Wespai、FinMind、YFinance）"""
import random
import time
from datetime import datetime, timedelta

import gspread
import pandas as pd
import pytz
import requests
import streamlit as st
import yfinance as yf

from tw_indicators import calculate_indicators

PORTFOLIO_SHEET_TITLE = "Streamlit TW Stock"
_TW = pytz.timezone("Asia/Taipei")


def get_gsheet_client():
    return gspread.service_account_from_dict(st.secrets["gcp_service_account"])


@st.cache_data(ttl=300, show_spinner=False)
def load_portfolio() -> pd.DataFrame:
    try:
        gc = get_gsheet_client()
        df = pd.DataFrame(gc.open(PORTFOLIO_SHEET_TITLE).sheet1.get_all_records())
        df["Symbol"] = df["Symbol"].astype(str).str.zfill(4)
        df["Cost"]   = pd.to_numeric(df["Cost"],   errors="coerce").fillna(0.0)
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Name", "Cost", "Shares", "Note"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_watchlist() -> pd.DataFrame:
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        try:
            ws = sh.worksheet("Watchlist")
        except Exception:
            ws = sh.add_worksheet("Watchlist", rows=200, cols=4)
            ws.append_row(["Symbol", "Name", "Note", "AddedAt"])
            return pd.DataFrame(columns=["Symbol", "Name", "Note", "AddedAt"])
        df = pd.DataFrame(ws.get_all_records())
        if df.empty:
            return pd.DataFrame(columns=["Symbol", "Name", "Note", "AddedAt"])
        df["Symbol"] = df["Symbol"].astype(str).str.zfill(4)
        return df
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Name", "Note", "AddedAt"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data() -> dict:
    url = "https://stock.wespai.com/lists"
    try:
        res  = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        data = pd.read_html(res.text)[0].iloc[:, [0, 1, 2, 3, 14, 15]].copy()
        data.columns = ["代碼", "名稱", "產業", "現價", "PE", "PB"]
        data["代碼"] = data["代碼"].astype(str).str.zfill(4)
        data["現價"] = pd.to_numeric(data["現價"], errors="coerce")
        data["PE"]   = pd.to_numeric(data["PE"],   errors="coerce").fillna(999.0)
        data["PB"]   = pd.to_numeric(data["PB"],   errors="coerce").fillna(999.0)
        return data.set_index("代碼").to_dict("index")
    except Exception as e:
        st.error(f"市場報價抓取失敗: {e}")
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_history(symbol: str) -> pd.DataFrame | None:
    """FinMind 優先 → YFinance (.TW / .TWO) 備援"""
    # 1. FinMind
    try:
        time.sleep(random.uniform(0.05, 0.2))
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=760)).strftime("%Y-%m-%d")
        res   = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockPrice", "data_id": symbol,
                    "start_date": start, "end_date": end},
            timeout=5,
        )
        if res.status_code == 200:
            data = res.json()
            if data.get("msg") == "success" and data.get("data"):
                df = pd.DataFrame(data["data"]).rename(columns={
                    "date": "Date", "open": "Open", "max": "High",
                    "min": "Low", "close": "Close", "trading_volume": "Volume",
                })
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                result = calculate_indicators(df)
                if result is not None:
                    return result
    except Exception:
        pass

    # 2. YFinance
    for suffix in [".TW", ".TWO"]:
        try:
            df = yf.Ticker(f"{symbol}{suffix}").history(period="3y", auto_adjust=True)
            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                result = calculate_indicators(df)
                if result is not None:
                    return result
        except Exception:
            continue
    return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weekly_history(symbol: str) -> pd.DataFrame | None:
    """日線重採樣為週線並重新計算指標"""
    daily = fetch_stock_history(symbol)
    if daily is None or daily.empty:
        return None
    try:
        weekly = daily[["Open", "High", "Low", "Close", "Volume"]].resample("W").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()
        return calculate_indicators(weekly) if len(weekly) >= 20 else None
    except Exception:
        return None


# ── 資產歷史快照 ──────────────────────────────────────────────────────────────

def load_portfolio_history() -> pd.DataFrame:
    empty = pd.DataFrame(columns=["Date", "MarketValue", "Cost", "PnL"])
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        try:
            ws = sh.worksheet("PortfolioHistory")
        except Exception:
            ws = sh.add_worksheet("PortfolioHistory", rows=1000, cols=4)
            ws.append_row(["Date", "MarketValue", "Cost", "PnL"])
            return empty
        rows = ws.get_all_values()
        if len(rows) <= 1:
            return empty
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df["Date"] = pd.to_datetime(df["Date"])
        for col in ["MarketValue", "Cost", "PnL"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return empty


def save_portfolio_snapshot(market_value: float, cost: float, pnl: float):
    """每交易日收盤後儲存市值快照（同日更新而非新增）"""
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        try:
            ws = sh.worksheet("PortfolioHistory")
        except Exception:
            ws = sh.add_worksheet("PortfolioHistory", rows=1000, cols=4)
            ws.append_row(["Date", "MarketValue", "Cost", "PnL"])
        rows = ws.get_all_values()
        for idx, row in enumerate(rows[1:], 2):
            if row and row[0] == today:
                ws.update(f"A{idx}:D{idx}", [[today, round(market_value), round(cost), round(pnl)]])
                return
        ws.append_row([today, round(market_value), round(cost), round(pnl)])
    except Exception:
        pass


# ── Watchlist 寫入 ────────────────────────────────────────────────────────────

def add_to_watchlist(symbol: str, name: str, note: str = ""):
    today = datetime.now(_TW).strftime("%Y-%m-%d %H:%M")
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        try:
            ws = sh.worksheet("Watchlist")
        except Exception:
            ws = sh.add_worksheet("Watchlist", rows=200, cols=4)
            ws.append_row(["Symbol", "Name", "Note", "AddedAt"])
        # avoid duplicates
        rows = ws.get_all_values()
        for row in rows[1:]:
            if row and row[0] == symbol:
                return False  # already exists
        ws.append_row([symbol, name, note, today])
        load_watchlist.clear()
        return True
    except Exception:
        return False


def remove_from_watchlist(symbol: str):
    try:
        gc = get_gsheet_client()
        ws = gc.open(PORTFOLIO_SHEET_TITLE).worksheet("Watchlist")
        rows = ws.get_all_values()
        for idx, row in enumerate(rows[1:], 2):
            if row and row[0] == symbol:
                ws.delete_rows(idx)
                load_watchlist.clear()
                return True
    except Exception:
        pass
    return False

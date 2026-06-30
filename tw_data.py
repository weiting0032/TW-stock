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
        data = pd.read_html(res.text)[0].iloc[:, [0, 1, 2, 3, 4, 8, 9, 10, 11, 14, 15, 17]].copy()
        data.columns = ["代碼", "名稱", "產業", "現價", "漲跌幅", "投信", "外資", "自營", "三大合計", "PE", "PB", "融資率"]
        data["代碼"] = data["代碼"].astype(str).str.zfill(4)
        data["現價"] = pd.to_numeric(data["現價"], errors="coerce")
        data["PE"]   = pd.to_numeric(data["PE"],   errors="coerce").fillna(999.0)
        data["PB"]   = pd.to_numeric(data["PB"],   errors="coerce").fillna(999.0)
        data["漲跌幅"] = pd.to_numeric(
            data["漲跌幅"].astype(str).str.replace("%", "", regex=False), errors="coerce"
        ).fillna(0.0)
        for col in ["投信", "外資", "自營", "三大合計"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        data["融資率"] = pd.to_numeric(
            data["融資率"].astype(str).str.replace("%", "", regex=False), errors="coerce"
        ).fillna(0.0)
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



@st.cache_data(ttl=3600, show_spinner=False)
def fetch_taiex_cycle() -> dict | None:
    """
    下載加權指數 (^TWII) 3年日線，計算目前牛熊週期狀態。
    週期定義：收盤 > SMA60 → 上漲週期；收盤 < SMA60 → 下跌週期。
    翻轉風險：距 SMA60 距離 + MACD Histogram 方向。
    """
    try:
        df = yf.Ticker("^TWII").history(period="3y", auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df[["Open", "High", "Low", "Close"]].copy()
        df["SMA20"]  = df["Close"].rolling(20).mean()
        df["SMA60"]  = df["Close"].rolling(60).mean()
        df["SMA240"] = df["Close"].rolling(240, min_periods=60).mean()

        ema12        = df["Close"].ewm(span=12, adjust=False).mean()
        ema26        = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]   = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["Hist"]   = df["MACD"] - df["Signal"]

        df = df.dropna(subset=["SMA60"])
        if len(df) < 5:
            return None

        # 1 = 上漲週期，0 = 下跌週期
        df["is_up"] = (df["Close"] > df["SMA60"]).astype(int)
        phase_arr   = df["is_up"].values
        last        = df.iloc[-1]

        current_phase = int(last["is_up"])

        # 計算連續天數
        days_in_cycle = 0
        for i in range(len(phase_arr) - 1, -1, -1):
            if phase_arr[i] == current_phase:
                days_in_cycle += 1
            else:
                break

        cycle_start = df.index[len(df) - days_in_cycle]

        # 歷史週期長度統計（排除目前進行中的）
        cycle_lengths = []
        curr_len, curr_p = 1, phase_arr[0]
        for i in range(1, len(phase_arr)):
            if phase_arr[i] == curr_p:
                curr_len += 1
            else:
                cycle_lengths.append(curr_len)
                curr_len, curr_p = 1, phase_arr[i]
        avg_cycle = int(sum(cycle_lengths) / len(cycle_lengths)) if cycle_lengths else 30
        # 同方向週期均值（每隔一個取，因上/下交替）
        same_dir = cycle_lengths[::2] if len(cycle_lengths) >= 2 else cycle_lengths
        avg_same = int(sum(same_dir) / len(same_dir)) if same_dir else avg_cycle

        close   = float(last["Close"])
        sma20   = float(last["SMA20"])
        sma60   = float(last["SMA60"])
        sma240_v = last.get("SMA240")
        sma240  = float(sma240_v) if sma240_v is not None and not pd.isna(sma240_v) else None
        macd_v  = float(last["MACD"])
        hist_v  = float(last["Hist"])

        dist_pct = (close - sma60) / sma60 * 100  # 正=在SMA60上方, 負=下方

        # 翻轉風險判斷
        if current_phase == 1:  # 上漲週期，留意跌破
            if abs(dist_pct) < 1.5 and hist_v < 0:
                flip_risk, flip_msg = "high",   f"⚠️ 距SMA60僅 {dist_pct:+.1f}%，MACD Hist 轉負，翻轉風險高"
            elif abs(dist_pct) < 3.0:
                flip_risk, flip_msg = "medium",  f"注意：距SMA60 {dist_pct:+.1f}%，接近警戒線"
            else:
                flip_risk, flip_msg = "safe",    f"距SMA60 {dist_pct:+.1f}%，上漲格局穩定"
        else:  # 下跌週期，留意突破
            if abs(dist_pct) < 1.5 and hist_v > 0:
                flip_risk, flip_msg = "high",   f"⚠️ 距SMA60僅 {dist_pct:+.1f}%，MACD Hist 轉正，回升機率高"
            elif abs(dist_pct) < 3.0:
                flip_risk, flip_msg = "medium",  f"注意：距SMA60 {dist_pct:+.1f}%，可能蓄勢突破"
            else:
                flip_risk, flip_msg = "safe",    f"距SMA60 {dist_pct:.1f}%，下跌格局延續"

        high52 = float(df["Close"].rolling(252, min_periods=60).max().iloc[-1])
        low52  = float(df["Close"].rolling(252, min_periods=60).min().iloc[-1])

        return {
            "phase":          current_phase,
            "phase_label":    "上漲週期" if current_phase == 1 else "下跌週期",
            "days_in_cycle":  days_in_cycle,
            "cycle_start":    cycle_start.strftime("%Y-%m-%d"),
            "close":          round(close, 0),
            "sma60":          round(sma60, 0),
            "sma240":         round(sma240, 0) if sma240 else None,
            "dist_pct":       round(dist_pct, 1),
            "flip_risk":      flip_risk,
            "flip_msg":       flip_msg,
            "macd":           round(macd_v, 1),
            "hist":           round(hist_v, 1),
            "avg_cycle_days": avg_cycle,
            "avg_same_days":  avg_same,
            "high52w":        round(high52, 0),
            "low52w":         round(low52, 0),
        }
    except Exception:
        return None


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

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
        import io as _io
        res  = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        # pandas>=3 不再接受字面 HTML 字串（會當成檔案路徑），必須包 StringIO
        data = pd.read_html(_io.StringIO(res.text))[0].iloc[:, [0, 1, 2, 3, 4, 8, 9, 10, 11, 14, 15, 17]].copy()
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
        records = data.set_index("代碼").to_dict("index")

        # DB 籌碼強化欄位（法人因子新版輸入；A/B 回放驗證 2026-07-05 採用）。
        # DB 缺失/查詢失敗時安靜跳過 → get_strategy 自動走 wespai 舊版 fallback。
        try:
            from tw_db import get_conn
            from tw_institutional import get_inst_summary, get_trust_streak
            _conn = get_conn()
            _sum = get_inst_summary(_conn, days=5)
            _stk = get_trust_streak(_conn, lookback=15)
            _f = dict(zip(_sum["stock_id"], _sum["f_net"] / 1000))
            _t = dict(zip(_sum["stock_id"], _sum["t_net"] / 1000))
            _s = dict(zip(_stk["stock_id"], _stk["streak"]))
            _mu = dict(_conn.execute(
                "SELECT stock_id, CAST(margin_balance AS REAL)/NULLIF(margin_quota,0)*100 "
                "FROM margin_trading WHERE trade_date="
                "(SELECT MAX(trade_date) FROM margin_trading)"))
            for _sid, _rec in records.items():
                if _sid in _f:
                    _rec["f_net_5d"] = int(round(_f[_sid]))
                    _rec["t_net_5d"] = int(round(_t.get(_sid, 0)))
                    _rec["trust_streak"] = int(_s.get(_sid, 0))
                _u = _mu.get(_sid)
                if _u is not None:
                    _rec["margin_util"] = round(float(_u), 1)
        except Exception:
            pass
        return records
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

        # 歷史週期長度統計（排除目前進行中的，並過濾 < 5 天的雜訊假突破）
        MIN_CYCLE = 5  # 小於此天數視為雜訊不納入統計
        up_lens, dn_lens = [], []
        curr_len, curr_p = 1, phase_arr[0]
        for i in range(1, len(phase_arr) - days_in_cycle):  # 排除目前進行中
            if phase_arr[i] == curr_p:
                curr_len += 1
            else:
                if curr_len >= MIN_CYCLE:
                    (up_lens if curr_p == 1 else dn_lens).append(curr_len)
                curr_len, curr_p = 1, phase_arr[i]
        if curr_len >= MIN_CYCLE:
            (up_lens if curr_p == 1 else dn_lens).append(curr_len)

        avg_up = int(sum(up_lens) / len(up_lens)) if up_lens else 50
        avg_dn = int(sum(dn_lens) / len(dn_lens)) if dn_lens else 20
        avg_same = avg_up if current_phase == 1 else avg_dn
        est_remaining = max(0, avg_same - days_in_cycle)

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
            "avg_up_days":    avg_up,
            "avg_dn_days":    avg_dn,
            "avg_same_days":  avg_same,
            "est_remaining":  est_remaining,
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


# ── 本地 DB 快取封裝（法人籌碼 + 月營收）────────────────────────────────────
# 注意：sqlite3.Connection 不可雜湊，cached 函式內部自行呼叫 get_conn()

@st.cache_data(ttl=900, show_spinner=False)
def cached_get_inst_flow(stock_id: str, days: int = 60):
    """get_inst_flow 快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_institutional import get_inst_flow
        conn = get_conn()
        return get_inst_flow(conn, stock_id, days)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_inst_summary(days: int = 5):
    """get_inst_summary 快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_institutional import get_inst_summary
        conn = get_conn()
        return get_inst_summary(conn, days)
    except Exception:
        import pandas as pd
        return pd.DataFrame(columns=["stock_id", "f_net", "t_net", "all_net"])


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_trust_streak(lookback: int = 15):
    """get_trust_streak 快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_institutional import get_trust_streak
        conn = get_conn()
        return get_trust_streak(conn, lookback)
    except Exception:
        import pandas as pd
        return pd.DataFrame(columns=["stock_id", "streak"])


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_revenue_history(stock_id: str, months: int = 24):
    """get_revenue_history 快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_revenue import get_revenue_history
        conn = get_conn()
        return get_revenue_history(conn, stock_id, months)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_revenue_signals():
    """get_revenue_signals 快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_revenue import get_revenue_signals
        conn = get_conn()
        return get_revenue_signals(conn)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_industry_rotation(days: int = 5):
    """
    產業輪動視圖（TTL=15分鐘）。
    回傳 (個股明細 df, 產業聚合 df)：
    - 明細：stock_id/名稱/產業/close/f_net/t_net(股)/f_amt/t_amt(億,估)/ret20/yoy_pct
    - 聚合：產業/檔數/外資額(億)/投信額(億)/20日報酬中位/YoY中位/YoY>20比率
    金額為估算：淨買賣超股數 × 最新收盤價。產業別取 wespai（涵蓋上市櫃）。
    """
    import pandas as pd
    try:
        from tw_db import get_conn
        from tw_institutional import get_inst_summary
        from tw_prices import get_recent_returns
        from tw_revenue import get_revenue_signals

        mm = get_market_data()
        if not mm:
            return pd.DataFrame(), pd.DataFrame()
        conn = get_conn()
        s = get_inst_summary(conn, days)
        r20 = get_recent_returns(conn, 20)
        rev = get_revenue_signals(conn)[["stock_id", "yoy_pct"]]

        base = pd.DataFrame([
            {"stock_id": sid, "名稱": v.get("名稱", ""), "產業": v.get("產業", "")}
            for sid, v in mm.items()
        ])
        base = base[base["產業"].astype(str).str.len() > 0]
        df = (base.merge(s, on="stock_id", how="inner")
                  .merge(r20, on="stock_id", how="left")
                  .merge(rev, on="stock_id", how="left"))
        df["f_amt"] = (df["f_net"] * df["close"] / 1e8).round(2)   # 億
        df["t_amt"] = (df["t_net"] * df["close"] / 1e8).round(2)

        agg = df.groupby("產業").agg(
            檔數=("stock_id", "size"),
            外資額=("f_amt", "sum"),
            投信額=("t_amt", "sum"),
            報酬中位=("ret_pct", "median"),
            YoY中位=("yoy_pct", "median"),
            強營收比=("yoy_pct", lambda x: (x > 20).mean() * 100),
        ).reset_index()
        agg = agg[agg["檔數"] >= 3]
        for c in ["外資額", "投信額", "報酬中位", "YoY中位", "強營收比"]:
            agg[c] = agg[c].round(2)
        agg = agg.sort_values("投信額", ascending=False).reset_index(drop=True)
        return df, agg
    except Exception:
        import pandas as pd
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_margin_history(stock_id: str, days: int = 60):
    """get_margin_history 快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_margin import get_margin_history
        conn = get_conn()
        return get_margin_history(conn, stock_id, days)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_get_tdcc_trend(stock_id: str):
    """get_tdcc_trend 快取封裝（週頻資料，TTL=1小時）"""
    try:
        from tw_db import get_conn
        from tw_tdcc import get_tdcc_trend
        conn = get_conn()
        return get_tdcc_trend(conn, stock_id)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_signal_journal():
    """訊號日誌（樣本外追蹤）快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_signal_log import get_journal
        conn = get_conn()
        return get_journal(conn)
    except Exception:
        import pandas as pd
        return ({"n_total": 0, "n20": 0, "avg20": None, "win20": None,
                 "avg60": None}, pd.DataFrame())


@st.cache_data(ttl=900, show_spinner=False)
def cached_get_data_status():
    """get_data_status 快取封裝（TTL=15分鐘）"""
    try:
        from tw_db import get_conn
        from tw_institutional import get_data_status
        conn = get_conn()
        return get_data_status(conn)
    except Exception:
        return {
            "inst_flow": {"latest_date": None, "total_rows": 0, "stock_count": 0},
            "monthly_revenue": {"latest_month": None, "stock_count": 0},
        }

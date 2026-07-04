"""
tw_prices.py -- 全市場每日行情（開高低收/量/值）資料層

來源（2026-07-04 實測）：
- 上市 TWSE MI_INDEX：https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX
  ?date=YYYYMMDD&type=ALLBUT0999&response=json，一次回全部上市個股，
  股票表以 fields 含「證券代號」辨識（tables 索引會浮動，不可寫死）。
- 上櫃 TPEx dailyQuotes：https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes
  ?type=EW&date=YYYY/MM/DD&response=json（新站模式，同 tw_institutional；
  需瀏覽器 UA 且 verify=False——TPEx 憑證缺 SubjectKeyIdentifier）。

同步採與 inst_flow 相同的「缺漏回填」語義，可回補歷史、冪等、可中斷續跑。
單位：volume=股、turnover=元。無成交之價格欄（'--'）存 NULL。
"""

import random
import sqlite3
import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import requests
import urllib3

from tw_db import is_trading_day, upsert

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

_TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
_TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes"


def _num(v):
    """'1,234.56' -> float；'--'/空 -> None。"""
    s = str(v).replace(",", "").strip()
    if s in ("", "--", "---", "None", "X"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int0(v):
    f = _num(v)
    return int(f) if f is not None else 0


def fetch_twse_prices(d: date) -> Optional[pd.DataFrame]:
    """上市全部個股當日行情；非交易日/未發布回 None。"""
    r = requests.get(
        _TWSE_URL,
        params={"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"},
        headers=_HEADERS, timeout=30,
    )
    j = r.json()
    if j.get("stat") != "OK":
        return None
    stock_tbl = None
    for t in j.get("tables", []):
        if "證券代號" in "".join(t.get("fields", [])):
            stock_tbl = t
            break
    if stock_tbl is None or not stock_tbl.get("data"):
        return None
    df = pd.DataFrame(stock_tbl["data"], columns=stock_tbl["fields"])
    out = pd.DataFrame({
        "trade_date": d.isoformat(),
        "stock_id":   df["證券代號"].astype(str).str.strip(),
        "market":     "TWSE",
        "open":   df["開盤價"].map(_num),
        "high":   df["最高價"].map(_num),
        "low":    df["最低價"].map(_num),
        "close":  df["收盤價"].map(_num),
        "volume": df["成交股數"].map(_int0),
        "turnover": df["成交金額"].map(_int0),
        "name":   df["證券名稱"].astype(str).str.strip(),
    })
    return out[out["stock_id"].str.fullmatch(r"\d{4}")].reset_index(drop=True)


def fetch_tpex_prices(d: date) -> Optional[pd.DataFrame]:
    """上櫃全部個股當日行情；非交易日/未發布/被擋回 None。"""
    r = requests.get(
        _TPEX_URL,
        params={"type": "EW", "date": d.strftime("%Y/%m/%d"), "response": "json"},
        headers=_HEADERS, timeout=30, verify=False,
    )
    if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
        return None
    j = r.json()
    tbl = None
    for t in j.get("tables", []):
        if "代號" in "".join(t.get("fields", [])):
            tbl = t
            break
    if tbl is None or not tbl.get("data"):
        return None
    df = pd.DataFrame(tbl["data"], columns=tbl["fields"])

    def col(kw):
        return next((c for c in df.columns if kw in c), None)

    c_vol, c_amt = col("成交股數"), col("成交金額")
    if col("代號") is None or col("收盤") is None:
        return None
    out = pd.DataFrame({
        "trade_date": d.isoformat(),
        "stock_id":   df[col("代號")].astype(str).str.strip(),
        "market":     "TPEX",
        "open":   df[col("開盤")].map(_num),
        "high":   df[col("最高")].map(_num),
        "low":    df[col("最低")].map(_num),
        "close":  df[col("收盤")].map(_num),
        "volume": df[c_vol].map(_int0) if c_vol else 0,
        "turnover": df[c_amt].map(_int0) if c_amt else 0,
        "name":   df[col("名稱")].astype(str).str.strip() if col("名稱") else "",
    })
    return out[out["stock_id"].str.fullmatch(r"\d{4}")].reset_index(drop=True)


def sync_daily_prices(
    conn: sqlite3.Connection,
    backfill_days: int = 0,
    progress_cb: Optional[Callable] = None,
) -> None:
    """缺漏回填語義，同 tw_institutional.sync_inst_flow。"""
    from pytz import timezone

    tw_now = datetime.now(timezone("Asia/Taipei"))
    today = tw_now.date()

    if backfill_days > 0:
        window_start = today - timedelta(days=backfill_days)
    else:
        row = conn.execute("SELECT MAX(trade_date) FROM daily_price").fetchone()
        window_start = (
            date.fromisoformat(row[0]) + timedelta(days=1) if row and row[0] else today
        )

    def _done_dates(market: str, source: str) -> set:
        done = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT trade_date FROM daily_price "
                "WHERE market=? AND trade_date>=?",
                (market, window_start.isoformat()),
            )
        }
        done |= {
            r[0]
            for r in conn.execute(
                "SELECT sync_date FROM sync_log "
                "WHERE source=? AND status IN ('ok','empty')",
                (source,),
            )
        }
        return done

    twse_done = _done_dates("TWSE", "TWSE_PRICE")
    tpex_done = _done_dates("TPEX", "TPEX_PRICE")

    def _log(source: str, dd: date, status: str, n: int, msg: str = None):
        conn.execute(
            "INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
            (source, dd.isoformat(), status, n, msg),
        )
        conn.commit()

    def _store(df: pd.DataFrame) -> int:
        """行情寫 daily_price；名稱另存 stock_names（給 UI/推播顯示用）。"""
        names = df[["stock_id", "name"]].drop_duplicates("stock_id")
        names = names[names["name"].astype(str).str.len() > 0]
        if not names.empty:
            upsert(conn, "stock_names", names)
        return upsert(conn, "daily_price", df.drop(columns=["name"]))

    d = window_start
    while d <= today:
        if d == today and tw_now.hour < 18:
            print(f"[price] {d} Taipei {tw_now.strftime('%H:%M')} < 18:00, skip today")
            break
        if not is_trading_day(conn, d):
            d += timedelta(days=1)
            continue

        iso = d.isoformat()
        need_twse = iso not in twse_done
        need_tpex = iso not in tpex_done
        if not (need_twse or need_tpex):
            d += timedelta(days=1)
            continue

        total_rows = 0
        if need_twse:
            try:
                df = fetch_twse_prices(d)
                if df is not None and not df.empty:
                    total_rows += _store(df)
                    _log("TWSE_PRICE", d, "ok", total_rows)
                elif d != today:
                    _log("TWSE_PRICE", d, "empty", 0, "stat!=OK")
            except Exception as e:
                _log("TWSE_PRICE", d, "error", 0, str(e))

        if need_tpex:
            try:
                df = fetch_tpex_prices(d)
                if df is not None and not df.empty:
                    n = _store(df)
                    total_rows += n
                    _log("TPEX_PRICE", d, "ok", n)
                elif df is None:
                    _log("TPEX_PRICE", d, "blocked", 0, "無回應/被擋")
                elif d != today:
                    _log("TPEX_PRICE", d, "empty", 0, "回應無資料")
            except Exception as e:
                _log("TPEX_PRICE", d, "error", 0, str(e))

        if progress_cb:
            progress_cb(d, total_rows)
        else:
            print(f"[price] {d} rows={total_rows}")

        time.sleep(5 + random.random())
        d += timedelta(days=1)


# ── 查詢 ──────────────────────────────────────────────────────────────────────

def get_price_history(
    conn: sqlite3.Connection, stock_id: str, days: int = 250
) -> pd.DataFrame:
    sql = """
        SELECT * FROM daily_price WHERE stock_id = ?
        ORDER BY trade_date DESC LIMIT ?
    """
    df = pd.read_sql_query(sql, conn, params=(stock_id, days))
    return df.sort_values("trade_date").reset_index(drop=True)


def get_price_panel(conn: sqlite3.Connection, start: str = None) -> pd.DataFrame:
    """回測用：整段期間全市場長表（trade_date, stock_id, open, close, turnover）。"""
    sql = "SELECT trade_date, stock_id, open, close, turnover FROM daily_price"
    params = ()
    if start:
        sql += " WHERE trade_date >= ?"
        params = (start,)
    return pd.read_sql_query(sql, conn, params=params)

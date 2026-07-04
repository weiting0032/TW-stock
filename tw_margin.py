"""
tw_margin.py -- 融資融券餘額資料層（單位：張）

來源（2026-07-04 實測）：
- 上市 TWSE MI_MARGN：https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN
  ?date=YYYYMMDD&selectType=ALL&response=json
  ⚠️ 欄名重複（融資/融券兩段都叫 買進/賣出）→ 必須用「位置」對映，不可用名稱。
- 上櫃 TPEx：https://www.tpex.org.tw/www/zh-tw/margin/balance
  ?date=YYYY/MM/DD&response=json（新站，日期參數有效可回補歷史；
  欄名唯一 → 名稱對映；需 UA + verify=False）。

同步採缺漏回填語義（同 tw_prices/tw_institutional），可回補、冪等、可中斷。
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
_TWSE_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
_TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/margin/balance"

_COLS = ["trade_date", "stock_id", "market",
         "margin_buy", "margin_sell", "margin_redeem",
         "margin_balance", "margin_prev", "margin_quota",
         "short_buy", "short_sell", "short_redeem",
         "short_balance", "short_prev", "short_quota", "offset_lots"]


def _n(v) -> int:
    s = str(v).replace(",", "").strip()
    if s in ("", "--", "---", "None"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def fetch_twse_margin(d: date) -> Optional[pd.DataFrame]:
    r = requests.get(
        _TWSE_URL,
        params={"date": d.strftime("%Y%m%d"), "selectType": "ALL",
                "response": "json"},
        headers=_HEADERS, timeout=30,
    )
    j = r.json()
    if j.get("stat") != "OK":
        return None
    tbl = None
    for t in j.get("tables", []):
        if "代號" in "".join(t.get("fields", [])) and len(t.get("fields", [])) >= 16:
            tbl = t
            break
    if tbl is None or not tbl.get("data"):
        return None
    rows = []
    for x in tbl["data"]:
        sid = str(x[0]).strip()
        # 位置對映（欄名重複不可靠）：
        # [2]資買 [3]資賣 [4]現償 [5]資前日 [6]資今日 [7]資限額
        # [8]券買 [9]券賣 [10]券償 [11]券前日 [12]券今日 [13]券限額 [14]資券互抵
        rows.append((d.isoformat(), sid, "TWSE",
                     _n(x[2]), _n(x[3]), _n(x[4]),
                     _n(x[6]), _n(x[5]), _n(x[7]),
                     _n(x[8]), _n(x[9]), _n(x[10]),
                     _n(x[12]), _n(x[11]), _n(x[13]), _n(x[14])))
    out = pd.DataFrame(rows, columns=_COLS)
    return out[out["stock_id"].str.fullmatch(r"\d{4}")].reset_index(drop=True)


def fetch_tpex_margin(d: date) -> Optional[pd.DataFrame]:
    r = requests.get(
        _TPEX_URL,
        params={"date": d.strftime("%Y/%m/%d"), "response": "json"},
        headers=_HEADERS, timeout=30, verify=False,
    )
    if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
        return None
    j = r.json()
    tbl = (j.get("tables") or [{}])[0]
    fields, data = tbl.get("fields") or [], tbl.get("data") or []
    if not fields or not data:
        return None
    df = pd.DataFrame(data, columns=fields)

    def col(kw):
        return next((c for c in df.columns if kw in c), None)

    need = {"margin_prev": "前資餘額", "margin_buy": "資買", "margin_sell": "資賣",
            "margin_redeem": "現償", "margin_balance": "資餘額",
            "margin_quota": "資限額",
            "short_prev": "前券餘額", "short_sell": "券賣", "short_buy": "券買",
            "short_redeem": "券償", "short_balance": "券餘額",
            "short_quota": "券限額", "offset_lots": "資券相抵"}
    mapped = {k: col(kw) for k, kw in need.items()}
    if mapped["margin_balance"] is None or col("代號") is None:
        return None
    out = pd.DataFrame({"trade_date": d.isoformat(),
                        "stock_id": df[col("代號")].astype(str).str.strip(),
                        "market": "TPEX"})
    for k, c in mapped.items():
        out[k] = df[c].map(_n) if c else 0
    out = out[_COLS[:3] + [c for c in _COLS[3:]]]  # 欄序對齊 schema
    return out[out["stock_id"].str.fullmatch(r"\d{4}")].reset_index(drop=True)


def sync_margin(
    conn: sqlite3.Connection,
    backfill_days: int = 0,
    progress_cb: Optional[Callable] = None,
) -> None:
    """缺漏回填語義（同 tw_prices.sync_daily_prices）。"""
    from pytz import timezone

    tw_now = datetime.now(timezone("Asia/Taipei"))
    today = tw_now.date()

    if backfill_days > 0:
        window_start = today - timedelta(days=backfill_days)
    else:
        row = conn.execute("SELECT MAX(trade_date) FROM margin_trading").fetchone()
        window_start = (
            date.fromisoformat(row[0]) + timedelta(days=1) if row and row[0] else today
        )

    def _done(market, source):
        done = {r[0] for r in conn.execute(
            "SELECT DISTINCT trade_date FROM margin_trading "
            "WHERE market=? AND trade_date>=?",
            (market, window_start.isoformat()))}
        done |= {r[0] for r in conn.execute(
            "SELECT sync_date FROM sync_log "
            "WHERE source=? AND status IN ('ok','empty')", (source,))}
        return done

    twse_done = _done("TWSE", "TWSE_MARGIN")
    tpex_done = _done("TPEX", "TPEX_MARGIN")

    def _log(source, dd, status, n, msg=None):
        conn.execute("INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                     (source, dd.isoformat(), status, n, msg))
        conn.commit()

    d = window_start
    while d <= today:
        if d == today and tw_now.hour < 18:
            break
        if not is_trading_day(conn, d):
            d += timedelta(days=1)
            continue
        iso = d.isoformat()
        need_twse, need_tpex = iso not in twse_done, iso not in tpex_done
        if not (need_twse or need_tpex):
            d += timedelta(days=1)
            continue
        total = 0
        if need_twse:
            try:
                df = fetch_twse_margin(d)
                if df is not None and not df.empty:
                    total += upsert(conn, "margin_trading", df)
                    _log("TWSE_MARGIN", d, "ok", total)
                elif d != today:
                    _log("TWSE_MARGIN", d, "empty", 0, "stat!=OK")
            except Exception as e:
                _log("TWSE_MARGIN", d, "error", 0, str(e))
        if need_tpex:
            try:
                df = fetch_tpex_margin(d)
                if df is not None and not df.empty:
                    n = upsert(conn, "margin_trading", df)
                    total += n
                    _log("TPEX_MARGIN", d, "ok", n)
                elif df is None:
                    _log("TPEX_MARGIN", d, "blocked", 0, "無回應/被擋")
                elif d != today:
                    _log("TPEX_MARGIN", d, "empty", 0, "回應無資料")
            except Exception as e:
                _log("TPEX_MARGIN", d, "error", 0, str(e))
        if progress_cb:
            progress_cb(d, total)
        else:
            print(f"[margin] {d} rows={total}")
        time.sleep(5 + random.random())
        d += timedelta(days=1)


def get_margin_history(conn: sqlite3.Connection, stock_id: str,
                       days: int = 60) -> pd.DataFrame:
    """近 N 筆 + 衍生欄位：券資比%、資使用率%。"""
    df = pd.read_sql_query(
        "SELECT * FROM margin_trading WHERE stock_id=? "
        "ORDER BY trade_date DESC LIMIT ?", conn, params=(stock_id, days))
    df = df.sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        return df
    df["short_margin_ratio"] = (
        df["short_balance"] / df["margin_balance"].replace(0, pd.NA) * 100
    ).astype(float).round(2)
    df["margin_util"] = (
        df["margin_balance"] / df["margin_quota"].replace(0, pd.NA) * 100
    ).astype(float).round(2)
    return df

"""
tw_dividends.py -- 除權息事件與股價還原

來源與涵蓋（2026-07-04 實測）：
- 上市（含 ETF/0050）：TWSE TWT49U 除權息計算結果表，支援日期區間查歷史 → 完整回補。
- 上櫃：TPEx 改版後歷史查詢已死（舊站/新站/OpenAPI 皆只回近期快照，FinMind 該表付費）。
  策略：每日同步 OpenAPI tpex_exright_daily 快照「往前累積」；歷史部分不還原並於
  回測揭露（影響＝低估上櫃含息報酬，屬保守偏誤）。
- 拆分偵測：無除權息事件卻出現 >38% 隔夜跳空（台股漲跌幅 10%，正常日不可能）
  → 以 open/prev_close 推定還原因子（例：0050 於 2025 年中 1 拆 4）。

還原語義：factor = 參考價/前收盤，乘在 ex_date「之前」的所有價格（back-adjust）。
"""

import random
import re
import sqlite3
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import urllib3

from tw_db import upsert

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_TWT49U = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"
_TPEX_DAILY = "https://www.tpex.org.tw/openapi/v1/tpex_exright_daily"


def _num(v) -> Optional[float]:
    s = str(v).replace(",", "").strip()
    if s in ("", "--", "---", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _roc_to_iso(s: str) -> Optional[str]:
    """'115年06月01日' / '115/07/06' / '1150706' → '2026-07-06'"""
    s = str(s).strip().replace("年", "/").replace("月", "/").replace("日", "")
    if "/" in s:
        parts = s.split("/")
    elif len(s) == 7 and s.isdigit():
        parts = [s[:3], s[3:5], s[5:7]]
    else:
        return None
    try:
        return f"{int(parts[0]) + 1911}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    except (ValueError, IndexError):
        return None


def fetch_twse_dividends(start: date, end: date) -> Optional[pd.DataFrame]:
    """TWT49U 日期區間查詢（實測單月 300+ 筆正常回傳）。"""
    r = requests.get(
        _TWT49U,
        params={"startDate": start.strftime("%Y%m%d"),
                "endDate": end.strftime("%Y%m%d"), "response": "json"},
        headers=_HEADERS, timeout=30,
    )
    j = r.json()
    if j.get("stat") != "OK" or not j.get("data"):
        return None
    df = pd.DataFrame(j["data"], columns=j["fields"])
    out = pd.DataFrame({
        "ex_date":      df["資料日期"].map(_roc_to_iso),
        "stock_id":     df["股票代號"].astype(str).str.strip(),
        "market":       "TWSE",
        "before_close": df["除權息前收盤價"].map(_num),
        "ref_price":    df["除權息參考價"].map(_num),
        "div_value":    df["權值+息值"].map(_num),
        "kind":         df["權/息"].astype(str).str.strip(),
        "src":          "TWT49U",
    })
    out = out[out["stock_id"].str.fullmatch(r"\d{4}") & out["ex_date"].notna()]
    out["factor"] = out["ref_price"] / out["before_close"]
    out = out[(out["factor"] > 0) & (out["factor"] <= 1.5) & out["factor"].notna()]
    return out.reset_index(drop=True)


def fetch_tpex_upcoming() -> Optional[pd.DataFrame]:
    """TPEx OpenAPI 快照（下一個除權息日的事件，通常 1~30 筆）。"""
    r = requests.get(_TPEX_DAILY, headers=_HEADERS, timeout=30, verify=False)
    if r.status_code != 200:
        return None
    rows = r.json()
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "ex_date":      df["Date"].map(_roc_to_iso),
        "stock_id":     df["SecuritiesCompanyCode"].astype(str).str.strip(),
        "market":       "TPEX",
        "before_close": df["ClosePriceBeforeExRightsDiviend"].map(_num),
        "ref_price":    df["ExRightsDiviendQuote"].map(_num),
        "div_value":    df["StockDividendPlusCashDividend"].map(_num),
        "kind":         df["ExRightsDiviend"].astype(str).str.strip(),
        "src":          "TPEX_API",
    })
    out = out[out["stock_id"].str.fullmatch(r"\d{4}") & out["ex_date"].notna()]
    out["factor"] = out["ref_price"] / out["before_close"]
    out = out[(out["factor"] > 0) & (out["factor"] <= 1.5) & out["factor"].notna()]
    return out.reset_index(drop=True)


def backfill_twse_dividends(conn: sqlite3.Connection, months: int = 25) -> int:
    """按月回補 TWT49U 歷史（事件量小，25 個月 ≈ 25 次請求）。"""
    total = 0
    today = date.today()
    for i in range(months, -1, -1):
        anchor = (today.replace(day=1) - timedelta(days=i * 28)).replace(day=1)
        month_end = (anchor + timedelta(days=40)).replace(day=1) - timedelta(days=1)
        try:
            df = fetch_twse_dividends(anchor, min(month_end, today))
            if df is not None and not df.empty:
                total += upsert(conn, "dividend_events", df)
        except Exception as e:
            print(f"[div] {anchor:%Y-%m} error: {e}")
        time.sleep(4 + random.random())
    return total


def sync_dividends(conn: sqlite3.Connection) -> None:
    """每日增量：TWSE 近 45 天區間重抓（冪等）＋ TPEx 快照累積。"""
    today = date.today()
    try:
        df = fetch_twse_dividends(today - timedelta(days=45), today)
        n = upsert(conn, "dividend_events", df) if df is not None else 0
        print(f"[div] TWSE recent45d rows={n}")
    except Exception as e:
        print(f"[div] TWSE error: {e}")
    time.sleep(3)
    try:
        df = fetch_tpex_upcoming()
        n = upsert(conn, "dividend_events", df) if df is not None else 0
        print(f"[div] TPEX snapshot rows={n}")
    except Exception as e:
        print(f"[div] TPEX error: {e}")


def detect_splits(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    掃描 daily_price：無除權息事件卻有 |隔夜跳空| 超過 ±38% → 推定拆分/併股。
    台股單日漲跌幅 ±10%，除權最深也極少 >35%，此閾值下誤判率極低。
    回傳偵測結果（同時 upsert 入 dividend_events, kind='拆分(推定)'）。
    """
    px = pd.read_sql_query(
        "SELECT trade_date, stock_id, market, open, close FROM daily_price "
        "WHERE open IS NOT NULL AND close IS NOT NULL ORDER BY stock_id, trade_date",
        conn,
    )
    px["prev_close"] = px.groupby("stock_id")["close"].shift(1)
    px["ratio"] = px["open"] / px["prev_close"]
    cand = px[(px["ratio"] < 0.62) | (px["ratio"] > 1.6)].copy()
    if cand.empty:
        return pd.DataFrame()

    known = {
        (r[0], r[1])
        for r in conn.execute("SELECT ex_date, stock_id FROM dividend_events")
    }
    cand = cand[~cand.apply(lambda r: (r["trade_date"], r["stock_id"]) in known, axis=1)]
    if cand.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "ex_date":      cand["trade_date"],
        "stock_id":     cand["stock_id"],
        "market":       cand["market"],
        "before_close": cand["prev_close"],
        "ref_price":    cand["open"],
        "div_value":    0.0,
        "kind":         "拆分(推定)",
        "factor":       cand["ratio"],
        "src":          "detector",
    }).reset_index(drop=True)
    upsert(conn, "dividend_events", out)
    return out


# ── 還原 ──────────────────────────────────────────────────────────────────────

def get_adjustment_factors(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ex_date, stock_id, factor FROM dividend_events "
        "WHERE factor IS NOT NULL AND factor > 0", conn,
    )


def apply_adjustment(panel: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    back-adjust：adj(t) = raw(t) × ∏_{ex_date > t} factor。
    panel: index=trade_date(str), columns=stock_id。ex_date 不在 index 的事件忽略。
    """
    if factors.empty:
        return panel
    F = pd.DataFrame(1.0, index=panel.index, columns=panel.columns)
    f = factors[factors["ex_date"].isin(panel.index)
                & factors["stock_id"].isin(panel.columns)]
    for _, row in f.iterrows():
        F.at[row["ex_date"], row["stock_id"]] *= row["factor"]
    G_incl = F.iloc[::-1].cumprod().iloc[::-1]   # ∏_{s>=t}
    G = G_incl / F                               # ∏_{s>t}（ex_date 當天已是新價，不乘自身）
    return panel * G

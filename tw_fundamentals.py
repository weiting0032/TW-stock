"""
tw_fundamentals.py -- 季度財報層（EPS + 三率，累計制）

來源（2026-07-05 實測）：MOPS 彙總表 POST（上市 sii / 上櫃 otc 同端點）：
- ajax_t163sb19 綜合損益彙總：基本每股盈餘(元)、營業收入、營業利益、稅後淨利
  （依產業分 30+ 張表，含金融業——金融業無毛利概念，該欄留 NULL）
- ajax_t163sb04 營益分析：補「營業毛利（毛損）淨額」算毛利率（一般業）
數值為財報「累計制」（season=02 即上半年累計）。單季/TTM/同季YoY 在查詢層換算：
- 單季 EPS = 本季累計 − 前一季累計（Q1 即累計值）
- TTM EPS = 最新累計 + 去年 Q4 累計 − 去年同季累計
- 毛利率轉折 = 本期累計毛利率 − 去年「同季」累計毛利率（同季相比避開季節性）

公告期限：Q1→5/15、Q2→8/14、Q3→11/14、年報(Q4)→隔年 3/31。
"""

import io
import random
import sqlite3
import time
from datetime import date, datetime
from typing import Optional, Tuple

import pandas as pd
import requests

from tw_db import upsert

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_SB19 = "https://mopsov.twse.com.tw/mops/web/ajax_t163sb19"
_SB04 = "https://mopsov.twse.com.tw/mops/web/ajax_t163sb04"


def _post_tables(url: str, year_roc: int, season: int, typek: str):
    r = requests.post(url, data={
        "encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1",
        "isQuery": "Y", "TYPEK": typek,
        "year": str(year_roc), "season": f"{season:02d}",
    }, headers=_HEADERS, timeout=90)
    if r.status_code != 200 or len(r.content) < 5000:
        return None
    try:
        return pd.read_html(io.StringIO(r.text))
    except ValueError:
        return None


def _flat_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["".join(map(str, c)).replace(" ", "") if isinstance(c, tuple)
                  else str(c).replace(" ", "") for c in df.columns]
    return df


def _pick(df: pd.DataFrame, kw: str):
    return next((c for c in df.columns if kw in c), None)


def _num(v) -> Optional[float]:
    s = str(v).replace(",", "").strip()
    if s in ("", "--", "---", "None", "nan", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_quarter(year: int, season: int, typek: str) -> Optional[pd.DataFrame]:
    """抓一個市場一季（累計值）。year 西元。回傳 schema 對齊的 DataFrame。"""
    y_roc = year - 1911
    tabs19 = _post_tables(_SB19, y_roc, season, typek)
    if not tabs19:
        return None
    frames = []
    for t in tabs19:
        t = _flat_cols(t)
        if _pick(t, "公司代號") and _pick(t, "每股盈餘"):
            frames.append(t)
    if not frames:
        return None
    rows = {}
    for t in frames:
        c_id, c_eps = _pick(t, "公司代號"), _pick(t, "每股盈餘")
        c_rev, c_op = _pick(t, "營業收入"), _pick(t, "營業利益")
        c_net = _pick(t, "稅後淨利")
        for _, r in t.iterrows():
            sid = str(r[c_id]).strip().split(".")[0].zfill(4)
            if not sid.isdigit() or len(sid) != 4:
                continue
            rev = _num(r[c_rev]) if c_rev else None
            op = _num(r[c_op]) if c_op else None
            net = _num(r[c_net]) if c_net else None
            rows[sid] = {
                "stock_id": sid,
                "quarter": f"{year}Q{season}",
                "revenue_m": round(rev / 1000, 1) if rev else None,   # 千元→百萬
                "gross_margin": None,
                "op_margin": round(op / rev * 100, 2) if op is not None and rev else None,
                "net_margin": round(net / rev * 100, 2) if net is not None and rev else None,
                "eps_cum": _num(r[c_eps]),
                "fetched_at": datetime.now().strftime("%Y-%m-%d"),
            }
    time.sleep(4 + random.random())

    tabs04 = _post_tables(_SB04, y_roc, season, typek)
    if tabs04:
        for t in tabs04:
            t = _flat_cols(t)
            c_id = _pick(t, "公司代號") or _pick(t, "公司代號")
            c_gp = _pick(t, "營業毛利（毛損）淨額") or _pick(t, "營業毛利（毛損）")
            c_rev = _pick(t, "營業收入")
            if not (c_id and c_gp and c_rev):
                continue
            for _, r in t.iterrows():
                sid = str(r[c_id]).strip().split(".")[0].zfill(4)
                if sid in rows:
                    gp, rev = _num(r[c_gp]), _num(r[c_rev])
                    if gp is not None and rev:
                        rows[sid]["gross_margin"] = round(gp / rev * 100, 2)
    return pd.DataFrame(list(rows.values()))


def expected_quarter(today: date = None) -> Tuple[int, int]:
    """依公告期限推算「現在應該已公告完成」的最新季度 (西元年, 季)。"""
    d = today or date.today()
    y = d.year
    if d >= date(y, 11, 15):
        return y, 3
    if d >= date(y, 8, 15):
        return y, 2
    if d >= date(y, 5, 16):
        return y, 1
    if d >= date(y, 4, 1):
        return y - 1, 4
    return y - 1, 3


def sync_quarterly_fin(conn: sqlite3.Connection) -> int:
    """該有的最新季不在庫（或列數過少）才抓，抓過就跳過（季頻資料）。"""
    y, s = expected_quarter()
    q = f"{y}Q{s}"
    n_have = conn.execute(
        "SELECT COUNT(*) FROM quarterly_fin WHERE quarter=?", (q,)).fetchone()[0]
    if n_have >= 1200:
        return 0
    total = 0
    for typek in ("sii", "otc"):
        try:
            df = fetch_quarter(y, s, typek)
            if df is not None and not df.empty:
                total += upsert(conn, "quarterly_fin", df)
        except Exception as e:
            print(f"[fin] {q} {typek} error: {e}")
        time.sleep(4 + random.random())
    conn.execute("INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                 ("QFIN", q, "ok" if total else "error", total, None))
    conn.commit()
    print(f"[fin] {q} rows={total}")
    return total


def backfill_quarterly_fin(conn: sqlite3.Connection, quarters: int = 13) -> int:
    """從最新應公告季往回補 N 季（含上市+上櫃）。"""
    y, s = expected_quarter()
    total = 0
    for _ in range(quarters):
        q = f"{y}Q{s}"
        n_have = conn.execute(
            "SELECT COUNT(*) FROM quarterly_fin WHERE quarter=?", (q,)).fetchone()[0]
        if n_have < 1200:
            for typek in ("sii", "otc"):
                try:
                    df = fetch_quarter(y, s, typek)
                    if df is not None and not df.empty:
                        n = upsert(conn, "quarterly_fin", df)
                        total += n
                        print(f"[fin] {q} {typek} rows={n}")
                except Exception as e:
                    print(f"[fin] {q} {typek} error: {e}")
                time.sleep(4 + random.random())
        else:
            print(f"[fin] {q} already have {n_have}, skip")
        s -= 1
        if s == 0:
            y, s = y - 1, 4
    return total


# ── 查詢層（累計制 → 單季 / TTM / 同季YoY）────────────────────────────────────

def get_fin_history(conn: sqlite3.Connection, stock_id: str,
                    quarters: int = 12) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM quarterly_fin WHERE stock_id=? ORDER BY quarter",
        conn, params=(stock_id,))
    if df.empty:
        return df
    df["yr"] = df["quarter"].str[:4].astype(int)
    df["sn"] = df["quarter"].str[-1].astype(int)
    # 單季 EPS：Q1=累計；Q2~Q4 = 本季累計 − 上一季累計（同年）
    prev = df.set_index(["yr", "sn"])["eps_cum"]
    df["eps_q"] = df.apply(
        lambda r: r["eps_cum"] if r["sn"] == 1 or pd.isna(r["eps_cum"])
        else (r["eps_cum"] - prev.get((r["yr"], r["sn"] - 1))
              if prev.get((r["yr"], r["sn"] - 1)) is not None else None),
        axis=1)
    # 同季 YoY 毛利率差（pp）
    gm_prev = df.set_index(["yr", "sn"])["gross_margin"]
    df["gm_yoy_delta"] = df.apply(
        lambda r: (r["gross_margin"] - gm_prev.get((r["yr"] - 1, r["sn"])))
        if pd.notna(r["gross_margin"]) and gm_prev.get((r["yr"] - 1, r["sn"])) is not None
        else None, axis=1)
    return df.tail(quarters).reset_index(drop=True)


def get_fin_signals(conn: sqlite3.Connection) -> pd.DataFrame:
    """每股最新季：eps_q、ttm_eps、gross_margin、gm_yoy_delta、margin_up 旗標。"""
    df = pd.read_sql_query("SELECT * FROM quarterly_fin", conn)
    if df.empty:
        return df
    df["yr"] = df["quarter"].str[:4].astype(int)
    df["sn"] = df["quarter"].str[-1].astype(int)
    idx = df.set_index(["stock_id", "yr", "sn"])
    eps = idx["eps_cum"]
    gm = idx["gross_margin"]

    latest = df.sort_values(["yr", "sn"]).groupby("stock_id").tail(1)
    out = []
    for r in latest.itertuples():
        k = (r.stock_id, r.yr, r.sn)
        eps_q = None
        if pd.notna(r.eps_cum):
            if r.sn == 1:
                eps_q = r.eps_cum
            else:
                p = eps.get((r.stock_id, r.yr, r.sn - 1))
                eps_q = r.eps_cum - p if p is not None and pd.notna(p) else None
        # TTM = 最新累計 + 去年Q4累計 − 去年同季累計
        ttm = None
        if pd.notna(r.eps_cum):
            q4 = eps.get((r.stock_id, r.yr - 1, 4))
            same = eps.get((r.stock_id, r.yr - 1, r.sn))
            if r.sn == 4:
                ttm = r.eps_cum
            elif q4 is not None and same is not None and pd.notna(q4) and pd.notna(same):
                ttm = r.eps_cum + q4 - same
        gm_d = None
        if pd.notna(r.gross_margin):
            g0 = gm.get((r.stock_id, r.yr - 1, r.sn))
            if g0 is not None and pd.notna(g0):
                gm_d = round(r.gross_margin - g0, 2)
        out.append({
            "stock_id": r.stock_id, "quarter": r.quarter,
            "eps_q": round(eps_q, 2) if eps_q is not None else None,
            "ttm_eps": round(ttm, 2) if ttm is not None else None,
            "gross_margin": r.gross_margin,
            "gm_yoy_delta": gm_d,
            "margin_up": bool(gm_d is not None and gm_d > 0),
        })
    return pd.DataFrame(out)

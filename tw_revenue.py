"""
tw_revenue.py — 月營收資料抓取與同步
來源：mopsov.twse.com.tw HTML 表格（上市 sii、上櫃 otc）。
無 streamlit 依賴。
"""

import io
import sqlite3
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

from tw_db import get_conn, upsert

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _BROWSER_UA}

_MOPS_BASE = "https://mopsov.twse.com.tw/nas/t21/{mkt}/t21sc03_{roc_year}_{month}_0.html"


def _roc_year(western_year: int) -> int:
    return western_year - 1911


def _find_col(cols, *keywords) -> Optional[str]:
    """在欄名列表中找第一個包含所有 keywords（不分大小寫）的欄。"""
    for c in cols:
        c_l = str(c).lower()
        if all(k.lower() in c_l for k in keywords):
            return c
    return None


def fetch_mops_revenue(year: int, month: int, mkt: str) -> Optional[pd.DataFrame]:
    """
    抓 MOPS 月營收頁（mkt='sii' 上市 / 'otc' 上櫃）。
    回傳 DataFrame 或 None（404/無表）。
    單位：千元（原始值，不換算）。
    """
    roc_y = _roc_year(year)
    url = _MOPS_BASE.format(mkt=mkt, roc_year=roc_y, month=month)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        if resp.status_code == 404:
            print(f"[MOPS] 404 {url}")
            return None
        resp.raise_for_status()
    except requests.HTTPError:
        return None
    except Exception as e:
        print(f"[MOPS] request failed {url}: {type(e).__name__}")
        return None

    html_content = resp.content.decode("utf-8", errors="replace")

    try:
        all_tables = pd.read_html(
            io.StringIO(html_content), flavor="lxml", encoding="utf-8"
        )
    except Exception as e:
        print(f"[MOPS] read_html failed {url}: {type(e).__name__}")
        return None

    frames = []
    for tbl in all_tables:
        # 攤平 MultiIndex 欄名（MOPS 頁面的每產業表都是 MultiIndex）
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [
                " ".join(str(c).strip() for c in col if "Unnamed" not in str(c) and str(c) != "nan")
                for col in tbl.columns
            ]

        # 目標表特徵：欄數 10~12、第一欄為 4 碼數字
        if len(tbl.columns) not in range(9, 14):
            continue

        # 第一欄必須是股票代號（4 碼數字）
        col0 = tbl.iloc[:, 0].astype(str).str.strip().str.zfill(4)
        valid_mask = col0.str.match(r"^\d{4}$")
        if valid_mask.sum() < 2:
            continue

        tbl = tbl[valid_mask].copy()
        tbl.reset_index(drop=True, inplace=True)
        n = len(tbl)

        def get_numeric(col_idx):
            if col_idx >= len(tbl.columns):
                return pd.Series([None] * n)
            return pd.to_numeric(
                tbl.iloc[:, col_idx].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

        # 位置映射（sii 與 otc 均相同）：
        # [0]=代號 [1]=名稱 [2]=當月營收 [3]=上月 [4]=去年同月
        # [5]=上月增減% [6]=去年同月增減%
        # [7]=當月累計 [8]=去年累計 [9]=累計增減%
        out = pd.DataFrame({
            "stock_id":    col0[valid_mask].values,
            "year_month":  f"{year}-{month:02d}",
            "revenue":     get_numeric(2).values,
            "mom_pct":     get_numeric(5).values,
            "yoy_pct":     get_numeric(6).values,
            "cum_revenue": get_numeric(7).values,
            "cum_yoy_pct": get_numeric(9).values,
            "industry":    None,
            "fetched_at":  datetime.utcnow().isoformat(),
        })
        frames.append(out)

    if not frames:
        print(f"[MOPS] no stock table found {url}")
        return None

    result = pd.concat(frames, ignore_index=True)
    # 去重（同代號保留第一筆，與頁面出現順序一致）
    result = result.drop_duplicates(subset=["stock_id"], keep="first")
    return result


def _update_industry(conn: sqlite3.Connection) -> None:
    """從 TWSE OpenAPI t187ap05_L 抓上市公司產業別，UPDATE 進 monthly_revenue。"""
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for rec in data:
            sid = str(rec.get("公司代號", "")).strip().zfill(4)
            ind = rec.get("產業別", "")
            if sid and ind:
                conn.execute(
                    "UPDATE monthly_revenue SET industry=? WHERE stock_id=?",
                    (ind, sid),
                )
        conn.commit()
        print(f"[industry] updated, source {len(data)} records")
    except Exception as e:
        print(f"[industry] update failed (non-blocking): {type(e).__name__}")


def sync_monthly_revenue(
    conn: sqlite3.Connection, months_back: int = 1
) -> None:
    """
    同步最近 months_back 個「營收月份」（從上月往前推）。
    規則：今日 < 次月 11 日 → 資料可能不完整，sync_log 加 partial 標記。
    """
    today = date.today()
    for i in range(months_back):
        # 計算「i 個月前」的年月
        total_month = (today.year * 12 + today.month - 1) - (i + 1)
        rev_year = total_month // 12
        rev_month = total_month % 12 + 1

        # 判斷是否 partial（今天 < 次月 11 日 → 當月資料仍陸續申報中）
        cutoff = date(rev_year + (1 if rev_month == 12 else 0),
                      (rev_month % 12) + 1, 11)
        is_partial = today < cutoff

        for mkt in ["sii", "otc"]:
            df = fetch_mops_revenue(rev_year, rev_month, mkt)
            if df is not None and not df.empty:
                n = upsert(conn, "monthly_revenue", df)
                status = "partial" if is_partial else "ok"
                msg = "partial: filing period not yet closed" if is_partial else None
                conn.execute(
                    "INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                    (f"revenue_{mkt}", f"{rev_year}-{rev_month:02d}", status, n, msg),
                )
                conn.commit()
                print(f"[revenue] {rev_year}-{rev_month:02d} {mkt}: {n} rows ({status})")
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                    (f"revenue_{mkt}", f"{rev_year}-{rev_month:02d}", "no_data", 0, None),
                )
                conn.commit()
            time.sleep(5)

    # 回補產業別
    _update_industry(conn)


def backfill_revenue(conn: sqlite3.Connection, months: int = 24) -> None:
    """
    回補歷史月營收，從最近一個完整月份往前 months 個月。
    已有資料的月份也會重抓（upsert 冪等）。
    """
    today = date.today()
    for i in range(months):
        total_month = (today.year * 12 + today.month - 1) - (i + 1)
        rev_year = total_month // 12
        rev_month = total_month % 12 + 1

        for mkt in ["sii", "otc"]:
            df = fetch_mops_revenue(rev_year, rev_month, mkt)
            if df is not None and not df.empty:
                n = upsert(conn, "monthly_revenue", df)
                conn.execute(
                    "INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                    (f"revenue_{mkt}", f"{rev_year}-{rev_month:02d}", "ok", n, None),
                )
                conn.commit()
                if i % 10 == 0:
                    print(f"[backfill] {rev_year}-{rev_month:02d} {mkt}: {n} rows")
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                    (f"revenue_{mkt}", f"{rev_year}-{rev_month:02d}", "no_data", 0, None),
                )
                conn.commit()
            time.sleep(5)

    _update_industry(conn)


# ── 查詢函式 ──────────────────────────────────────────────────────────────────

def get_revenue_history(
    conn: sqlite3.Connection, stock_id: str, months: int = 24
) -> pd.DataFrame:
    """回傳指定個股近 N 個月營收，按年月升冪。"""
    sql = """
        SELECT * FROM monthly_revenue
        WHERE stock_id = ?
        ORDER BY year_month DESC
        LIMIT ?
    """
    df = pd.read_sql_query(sql, conn, params=(stock_id, months))
    return df.sort_values("year_month").reset_index(drop=True)


def get_revenue_signals(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    每股最新月 YoY/MoM 與前一月 YoY；
    turnaround = 最新 YoY>0 且前月 YoY<=0；
    accelerating = YoY 連 2 月遞增且最新 >10。
    """
    sql = """
        WITH ranked AS (
            SELECT stock_id, year_month, yoy_pct, mom_pct,
                   ROW_NUMBER() OVER (PARTITION BY stock_id ORDER BY year_month DESC) AS rn
            FROM monthly_revenue
            WHERE yoy_pct IS NOT NULL
        )
        SELECT
            a.stock_id,
            a.yoy_pct,
            a.mom_pct,
            b.yoy_pct AS prev_yoy
        FROM ranked a
        LEFT JOIN ranked b ON a.stock_id = b.stock_id AND b.rn = 2
        WHERE a.rn = 1
    """
    df = pd.read_sql_query(sql, conn)
    if df.empty:
        return df

    df["turnaround"] = (df["yoy_pct"] > 0) & (df["prev_yoy"].fillna(0) <= 0)
    df["accelerating"] = (
        (df["yoy_pct"] > df["prev_yoy"].fillna(df["yoy_pct"])) & (df["yoy_pct"] > 10)
    )
    return df.reset_index(drop=True)

"""
tw_tdcc.py -- 集保股權分散表（每週五快照，大戶持股比）

來源（2026-07-04 實測）：https://opendata.tdcc.com.tw/getOD.ashx?id=1-5
- 每次僅回「最新一週」CSV（~2.3MB / 68k 列），欄位：
  資料日期,證券代號,持股分級,人數,股數,占集保庫存數比例%
- 歷史不提供 → 每週同步「往前累積」（與 TPEx 除權息同策略）。
- 憑證缺 SubjectKeyIdentifier（同 TPEx）→ verify=False。

持股分級：1=1-999股 … 11=200-400張 12=400-600張 13=600-800張
14=800-1000張 15=>1000張 16=差異數調整 17=合計。
大戶(>400張)比 = 12~15 級 pct 加總；千張大戶比 = 15 級。
"""

import io
import sqlite3
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import urllib3

from tw_db import upsert

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_tdcc() -> Optional[pd.DataFrame]:
    r = requests.get(_URL, headers=_HEADERS, timeout=120, verify=False)
    if r.status_code != 200 or len(r.content) < 1000:
        return None
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
    df.columns = ["data_date", "stock_id", "level", "holders", "shares", "pct"]
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df = df[df["stock_id"].str.fullmatch(r"\d{4}")]
    dd = df["data_date"].astype(str)
    df["data_date"] = dd.str[:4] + "-" + dd.str[4:6] + "-" + dd.str[6:8]
    for c in ["level", "holders", "shares"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
    return df.reset_index(drop=True)


def sync_tdcc(conn: sqlite3.Connection) -> int:
    """每週快照累積；DB 內最新日期距今 <6 天則跳過（避免每日白抓 2.3MB）。"""
    row = conn.execute("SELECT MAX(data_date) FROM tdcc_dispersion").fetchone()
    if row and row[0]:
        age = (date.today() - date.fromisoformat(row[0])).days
        if age < 6:
            return 0
    df = fetch_tdcc()
    if df is None or df.empty:
        conn.execute("INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                     ("TDCC", date.today().isoformat(), "error", 0, "抓取失敗"))
        conn.commit()
        return 0
    n = upsert(conn, "tdcc_dispersion", df)
    conn.execute("INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
                 ("TDCC", df["data_date"].iloc[0], "ok", n, None))
    conn.commit()
    return n


def get_tdcc_trend(conn: sqlite3.Connection, stock_id: str) -> pd.DataFrame:
    """每週：大戶(>400張)比%、千張大戶比%、總股東人數。"""
    df = pd.read_sql_query(
        "SELECT data_date, level, holders, pct FROM tdcc_dispersion "
        "WHERE stock_id=? ORDER BY data_date", conn, params=(stock_id,))
    if df.empty:
        return df
    out = []
    for d, g in df.groupby("data_date"):
        big = g[g["level"].between(12, 15)]["pct"].sum()
        big1000 = g[g["level"] == 15]["pct"].sum()
        total = g[g["level"] == 17]["holders"].sum()
        out.append({"data_date": d, "big400_pct": round(big, 2),
                    "big1000_pct": round(big1000, 2), "total_holders": int(total)})
    return pd.DataFrame(out)

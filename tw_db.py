"""
tw_db.py — 本地 SQLite 持久層基礎設施
提供連線、建表、upsert、假日曆、交易日判斷。
無 streamlit 依賴，可在 headless CLI 中使用。
"""

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import requests

DB_PATH = Path(__file__).parent / "data" / "tw_market.db"

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS inst_flow (
    trade_date      TEXT NOT NULL,
    stock_id        TEXT NOT NULL,
    market          TEXT NOT NULL,
    foreign_buy     INTEGER,
    foreign_sell    INTEGER,
    foreign_net     INTEGER,
    trust_buy       INTEGER,
    trust_sell      INTEGER,
    trust_net       INTEGER,
    dealer_self_net INTEGER,
    dealer_hedge_net INTEGER,
    dealer_net      INTEGER,
    total_net       INTEGER,
    PRIMARY KEY (trade_date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_inst_stock ON inst_flow (stock_id, trade_date);

CREATE TABLE IF NOT EXISTS monthly_revenue (
    stock_id    TEXT NOT NULL,
    year_month  TEXT NOT NULL,
    revenue     REAL,
    mom_pct     REAL,
    yoy_pct     REAL,
    cum_revenue REAL,
    cum_yoy_pct REAL,
    industry    TEXT,
    fetched_at  TEXT,
    PRIMARY KEY (stock_id, year_month)
);

CREATE TABLE IF NOT EXISTS sync_log (
    source      TEXT,
    sync_date   TEXT,
    status      TEXT,
    rows        INTEGER,
    msg         TEXT,
    PRIMARY KEY (source, sync_date)
);

CREATE TABLE IF NOT EXISTS holidays (
    date TEXT PRIMARY KEY,
    name TEXT
);

CREATE TABLE IF NOT EXISTS daily_price (
    trade_date TEXT NOT NULL,
    stock_id   TEXT NOT NULL,
    market     TEXT NOT NULL,
    open       REAL,
    high       REAL,
    low        REAL,
    close      REAL,
    volume     INTEGER,   -- 股
    turnover   INTEGER,   -- 元
    PRIMARY KEY (trade_date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_price_stock ON daily_price (stock_id, trade_date);
"""

# 快照（給雲端手機用）只帶這些表；daily_price 體積大且雲端用不到，
# 排除以維持手機冷啟動速度（見 tw_snapshot.make_snapshot）。
SNAPSHOT_TABLES = ["inst_flow", "monthly_revenue", "sync_log", "holidays"]


def get_conn() -> sqlite3.Connection:
    """建立（或開啟）資料庫連線，自動建目錄與表。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.executescript(_DDL)
    conn.commit()
    return conn


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE 整個 DataFrame 進指定表，回傳寫入列數。"""
    if df is None or df.empty:
        return 0
    rows = [tuple(r) for r in df.itertuples(index=False)]
    placeholders = ", ".join(["?"] * len(df.columns))
    cols = ", ".join(df.columns)
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def sync_holidays(conn: sqlite3.Connection) -> int:
    """抓 TWSE OpenAPI 全年休市日，民國日期轉西元，存入 holidays 表。"""
    url = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"
    resp = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for rec in data:
        roc_str = rec.get("Date", "")  # 7碼，如 "1150101"
        name = rec.get("Name", "")
        if len(roc_str) == 7:
            roc_year = int(roc_str[:3])
            western_year = roc_year + 1911
            iso_date = f"{western_year}-{roc_str[3:5]}-{roc_str[5:7]}"
            rows.append((iso_date, name))
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO holidays (date, name) VALUES (?, ?)", rows
        )
        conn.commit()
    return len(rows)


def is_trading_day(conn: sqlite3.Connection, d: date) -> bool:
    """回傳 d 是否為交易日（週一~五 且 不在 holidays 表）。"""
    # 若 holidays 表為空先同步
    cur = conn.execute("SELECT COUNT(*) FROM holidays")
    if cur.fetchone()[0] == 0:
        sync_holidays(conn)

    if d.weekday() >= 5:  # 週六=5, 週日=6
        return False
    iso = d.isoformat()
    cur = conn.execute("SELECT 1 FROM holidays WHERE date=?", (iso,))
    return cur.fetchone() is None

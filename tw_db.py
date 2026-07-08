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

CREATE TABLE IF NOT EXISTS stock_names (
    stock_id TEXT PRIMARY KEY,
    name     TEXT
);

CREATE TABLE IF NOT EXISTS signal_log (
    signal_date TEXT NOT NULL,
    stock_id    TEXT NOT NULL,
    strategy    TEXT NOT NULL,     -- 'composite_v1'
    params      TEXT,              -- 觸發當下的參數 json（追溯用）
    close_at_signal REAL,
    f_net INTEGER, t_net INTEGER, streak INTEGER, yoy_pct REAL,
    entry_open REAL,               -- T+1 開盤（次日由排程回填）
    ret_5 REAL, ret_20 REAL, ret_60 REAL,   -- 含成本、含息還原
    cycle_phase INTEGER,           -- 訊號日大盤相位 1=牛/0=熊（2026-07-08 實證閘門）
    PRIMARY KEY (signal_date, stock_id, strategy)
);

CREATE TABLE IF NOT EXISTS dividend_events (
    ex_date      TEXT NOT NULL,   -- 除權息/拆分生效日
    stock_id     TEXT NOT NULL,
    market       TEXT,
    before_close REAL,            -- 除權息前收盤價
    ref_price    REAL,            -- 除權息參考價（官方）
    div_value    REAL,            -- 權值+息值
    kind         TEXT,            -- 息/權/權息/拆分(推定)
    factor       REAL,            -- ref/before，還原乘數（乘在 ex_date 之前的價格上）
    src          TEXT,            -- TWT49U / TPEX_API / detector
    PRIMARY KEY (ex_date, stock_id)
);

CREATE TABLE IF NOT EXISTS quarterly_fin (
    stock_id     TEXT NOT NULL,
    quarter      TEXT NOT NULL,   -- '2026Q1'；數值為財報「累計制」原值
    revenue_m    REAL,            -- 累計營業收入（百萬元）
    gross_margin REAL,            -- 累計毛利率%
    op_margin    REAL,            -- 累計營益率%
    net_margin   REAL,            -- 累計稅後純益率%
    eps_cum      REAL,            -- 累計基本每股盈餘（元）
    fetched_at   TEXT,
    PRIMARY KEY (stock_id, quarter)
);

CREATE TABLE IF NOT EXISTS margin_trading (
    trade_date TEXT NOT NULL,
    stock_id   TEXT NOT NULL,
    market     TEXT NOT NULL,
    margin_buy INTEGER, margin_sell INTEGER, margin_redeem INTEGER,
    margin_balance INTEGER, margin_prev INTEGER, margin_quota INTEGER,
    short_buy INTEGER, short_sell INTEGER, short_redeem INTEGER,
    short_balance INTEGER, short_prev INTEGER, short_quota INTEGER,
    offset_lots INTEGER,               -- 資券互抵；全表單位：張
    PRIMARY KEY (trade_date, stock_id)
);
CREATE INDEX IF NOT EXISTS idx_margin_stock ON margin_trading (stock_id, trade_date);

CREATE TABLE IF NOT EXISTS tdcc_dispersion (
    data_date TEXT NOT NULL,           -- 每週五
    stock_id  TEXT NOT NULL,
    level     INTEGER NOT NULL,        -- 持股分級 1-15,16=調整,17=合計
    holders   INTEGER,
    shares    INTEGER,
    pct       REAL,
    PRIMARY KEY (data_date, stock_id, level)
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

# 快照（給雲端手機用）帶哪些表：{表名: WHERE 條件或 None}。
# daily_price 體積大且雲端不查 → 不帶；margin_trading 只帶近 130 天切片
#（診斷圖顯示 60 天，留緩衝）。見 tw_snapshot.make_snapshot。
SNAPSHOT_TABLES = {
    "inst_flow": None,
    "monthly_revenue": None,
    "sync_log": None,
    "holidays": None,
    "signal_log": None,
    "stock_names": None,
    "tdcc_dispersion": None,
    "quarterly_fin": None,
    "margin_trading": "trade_date >= date('now','-130 days')",
    # 近 45 天價格切片（~2MB）：產業輪動的 20 日報酬在雲端也能算
    "daily_price": "trade_date >= date('now','-45 days')",
}


def get_conn() -> sqlite3.Connection:
    """建立（或開啟）資料庫連線，自動建目錄與表。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.executescript(_DDL)
    # 輕量 migration：既有 DB 補新欄位（CREATE IF NOT EXISTS 不會改舊表）
    cols = {r[1] for r in conn.execute("PRAGMA table_info(signal_log)")}
    if "cycle_phase" not in cols:
        conn.execute("ALTER TABLE signal_log ADD COLUMN cycle_phase INTEGER")
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

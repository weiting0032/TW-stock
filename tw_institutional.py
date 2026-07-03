"""
tw_institutional.py — 三大法人買賣超資料抓取與同步
支援 TWSE T86、TPEx（新站/舊站）、FinMind 備援。
無 streamlit 依賴。
"""

import random
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import requests

from tw_db import get_conn, is_trading_day, upsert

# ── 常數 ──────────────────────────────────────────────────────────────────────

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _BROWSER_UA}

# T86 欄位到 schema 的映射（2026 實測 19 欄）
# 「外資」欄已改名為「外陸資(不含外資自營商)」+「外資自營商」兩組
# foreign_buy/sell/net 使用「外陸資(不含外資自營商)」層級，符合市場慣例
_T86_COL_MAP = {
    "證券代號":                          "stock_id",
    "外陸資買進股數(不含外資自營商)":      "foreign_buy",
    "外陸資賣出股數(不含外資自營商)":      "foreign_sell",
    "外陸資買賣超股數(不含外資自營商)":    "foreign_net",
    "投信買進股數":                       "trust_buy",
    "投信賣出股數":                       "trust_sell",
    "投信買賣超股數":                     "trust_net",
    "自營商買賣超股數":                   "dealer_net",
    "自營商買賣超股數(自行買賣)":          "dealer_self_net",
    "自營商買賣超股數(避險)":             "dealer_hedge_net",
    "三大法人買賣超股數":                 "total_net",
}

# 允許的股票代號格式（4碼數字）
_STOCK_ID_RE = re.compile(r"^\d{4}$")


def _to_int(val) -> int:
    """去千分位逗號，空值/'--'/nan 視為 0，轉 int。"""
    if val is None:
        return 0
    s = str(val).strip()
    if s in ("", "--", "-", "nan", "NaN", "None"):
        return 0
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return 0


# ── TWSE T86 ─────────────────────────────────────────────────────────────────

def fetch_t86(d: date) -> Optional[pd.DataFrame]:
    """
    抓 TWSE T86 上市三大法人買賣超日報。
    回傳 DataFrame（含 trade_date, stock_id, market 及各法人欄位）或 None。
    """
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    params = {
        "date": d.strftime("%Y%m%d"),
        "selectType": "ALLBUT0999",
        "response": "json",
    }
    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"[T86] request failed {d}: {type(e).__name__}")
        return None

    if payload.get("stat") != "OK":
        print(f"[T86] stat != OK ({d}): {payload.get('stat')}")
        return None

    fields = payload.get("fields", [])
    data = payload.get("data", [])
    if not fields or not data:
        return None

    raw_df = pd.DataFrame(data, columns=fields)

    # 防禦：確認所有預期欄位存在（_T86_COL_MAP 中的原文欄名，排除 stock_id 本身）
    required = [k for k in _T86_COL_MAP if k != "證券代號"]
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise ValueError(f"[T86] fields 缺少預期欄位: {missing}")

    # 過濾 4 碼數字代號
    raw_df = raw_df[raw_df["證券代號"].str.strip().str.match(r"^\d{4}$")].copy()
    raw_df = raw_df.reset_index(drop=True)

    int_cols = {
        "foreign_buy":     "外陸資買進股數(不含外資自營商)",
        "foreign_sell":    "外陸資賣出股數(不含外資自營商)",
        "foreign_net":     "外陸資買賣超股數(不含外資自營商)",
        "trust_buy":       "投信買進股數",
        "trust_sell":      "投信賣出股數",
        "trust_net":       "投信買賣超股數",
        "dealer_net":      "自營商買賣超股數",
        "dealer_self_net": "自營商買賣超股數(自行買賣)",
        "dealer_hedge_net":"自營商買賣超股數(避險)",
        "total_net":       "三大法人買賣超股數",
    }

    # 建構輸出 DataFrame（字典形式，避免空 DF 賦值問題）
    out_dict = {
        "trade_date": [d.isoformat()] * len(raw_df),
        "stock_id":   raw_df["證券代號"].str.strip().tolist(),
        "market":     ["TWSE"] * len(raw_df),
    }
    for out_col, src_col in int_cols.items():
        out_dict[out_col] = raw_df[src_col].apply(_to_int).tolist()

    return pd.DataFrame(out_dict)


# ── TPEx 上櫃三大法人 ─────────────────────────────────────────────────────────

def _parse_tpex_tables(payload: dict, d: date) -> Optional[pd.DataFrame]:
    """
    解析 TPEx JSON 回應（新站/舊站回應格式相同）。
    結構: {"tables": [{"fields": [...], "data": [[...],...]}]}
    24 欄位置映射（實測 2026-07-02 確認）：
      [0]=代號  [1]=名稱
      [2-4]  = 外資及陸資(不含外資自營商) buy/sell/net
      [5-7]  = 外資自營商 buy/sell/net
      [8-10] = 外資及陸資合計 buy/sell/net  ← foreign_* 使用此層
      [11-13]= 投信 buy/sell/net
      [14-16]= 自營商(自行買賣) buy/sell/net
      [17-19]= 自營商(避險) buy/sell/net
      [20-22]= 自營商合計 buy/sell/net
      [23]   = 三大法人買賣超股數合計
    """
    tables = payload.get("tables")
    if not tables:
        return None
    tbl = tables[0]
    data = tbl.get("data")
    if not data:
        return None

    rows = []
    for row in data:
        if len(row) < 24:
            continue
        sid = str(row[0]).strip()
        if not _STOCK_ID_RE.match(sid):
            continue
        rows.append({
            "trade_date":       d.isoformat(),
            "stock_id":         sid,
            "market":           "TPEX",
            "foreign_buy":      _to_int(row[8]),   # 外資及陸資合計買進
            "foreign_sell":     _to_int(row[9]),   # 外資及陸資合計賣出
            "foreign_net":      _to_int(row[10]),  # 外資及陸資合計買賣超
            "trust_buy":        _to_int(row[11]),
            "trust_sell":       _to_int(row[12]),
            "trust_net":        _to_int(row[13]),
            "dealer_self_net":  _to_int(row[16]),  # 自營商(自行買賣)買賣超
            "dealer_hedge_net": _to_int(row[19]),  # 自營商(避險)買賣超
            "dealer_net":       _to_int(row[22]),  # 自營商合計買賣超
            "total_net":        _to_int(row[23]),  # 三大法人合計
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def fetch_tpex_inst(d: date) -> Optional[pd.DataFrame]:
    """
    抓 TPEx 上櫃三大法人買賣超。
    依序嘗試新站→舊站，兩者都失敗回 None。
    """
    date_roc = f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"
    date_slash = f"{d.year}/{d.month:02d}/{d.day:02d}"

    # ── 端點 1：新站 ──────────────────────────────────────────────────────────
    # TPEx SSL 憑證遺漏 SubjectKeyIdentifier 擴充，Python 3.13 嚴格驗證會失敗
    # 使用 verify=False（不驗憑證）搭配 urllib3 suppress warning
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    new_url = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
    new_params = {
        "type": "Daily",
        "sect": "EW",
        "date": date_slash,
        "response": "json",
    }
    try:
        resp = requests.get(
            new_url, params=new_params, headers=_HEADERS, timeout=15, verify=False
        )
        if resp.status_code == 200:
            payload = resp.json()
            df = _parse_tpex_tables(payload, d)
            if df is not None and not df.empty:
                print(f"[TPEx] new-site OK {d}: {len(df)} rows")
                return df
            print(f"[TPEx] new-site responded but parse empty {d}")
        else:
            print(f"[TPEx] new-site HTTP {resp.status_code} {d}")
    except Exception as e:
        print(f"[TPEx] new-site exception {d}: {type(e).__name__}")

    # ── 端點 2：舊站 ──────────────────────────────────────────────────────────
    old_url = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
    old_params = {
        "l": "zh-tw",
        "o": "json",
        "se": "EW",
        "t": "D",
        "d": date_roc,
    }
    try:
        resp = requests.get(
            old_url, params=old_params, headers=_HEADERS, timeout=15, verify=False
        )
        if resp.status_code == 200:
            payload = resp.json()
            df = _parse_tpex_tables(payload, d)
            if df is not None and not df.empty:
                print(f"[TPEx] old-site OK {d}: {len(df)} rows")
                return df
            print(f"[TPEx] old-site responded but parse empty {d}")
        else:
            print(f"[TPEx] old-site HTTP {resp.status_code} {d}")
    except Exception as e:
        print(f"[TPEx] old-site exception {d}: {type(e).__name__}")

    print(f"[TPEx] both endpoints failed {d}")
    return None


# ── FinMind 備援（個股） ──────────────────────────────────────────────────────

def fetch_finmind_inst(
    stock_id: str, start_date: date, end_date: date
) -> pd.DataFrame:
    """
    FinMind TaiwanStockInstitutionalInvestorsBuySell（個股備援用）。
    回傳標準 schema DataFrame（可能為空）。
    """
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("msg") != "success" or not payload.get("data"):
            return pd.DataFrame()
    except Exception as e:
        print(f"[FinMind] request failed {stock_id}: {type(e).__name__}")
        return pd.DataFrame()

    raw = pd.DataFrame(payload["data"])
    # FinMind long format: date, stock_id, name, buy, sell
    name_map = {
        "Foreign_Investor":  "foreign",
        "Investment_Trust":  "trust",
        "Dealer_Self":       "dealer_self",
        "Dealer":            "dealer",
        "Dealer_Hedging":    "dealer_hedge",
    }
    raw["_key"] = raw["name"].map(name_map)
    raw = raw[raw["_key"].notna()].copy()
    raw["net"] = raw["buy"].astype(float) - raw["sell"].astype(float)

    records = {}
    for _, row in raw.iterrows():
        d = row["date"]
        if d not in records:
            records[d] = {
                "trade_date": d,
                "stock_id": stock_id,
                "market": "TPEX",
                "foreign_buy": 0, "foreign_sell": 0, "foreign_net": 0,
                "trust_buy": 0, "trust_sell": 0, "trust_net": 0,
                "dealer_self_net": 0, "dealer_hedge_net": 0,
                "dealer_net": 0, "total_net": 0,
            }
        k = row["_key"]
        buy = int(row["buy"])
        sell = int(row["sell"])
        net = int(row["net"])
        if k == "foreign":
            records[d]["foreign_buy"] = buy
            records[d]["foreign_sell"] = sell
            records[d]["foreign_net"] = net
        elif k == "trust":
            records[d]["trust_buy"] = buy
            records[d]["trust_sell"] = sell
            records[d]["trust_net"] = net
        elif k == "dealer_self":
            records[d]["dealer_self_net"] = net
        elif k == "dealer_hedge":
            records[d]["dealer_hedge_net"] = net
        elif k == "dealer":
            records[d]["dealer_net"] = net

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(list(records.values()))
    # total_net = 三大合計
    df["total_net"] = df["foreign_net"] + df["trust_net"] + df["dealer_net"]
    return df.sort_values("trade_date").reset_index(drop=True)


# ── 增量同步 ─────────────────────────────────────────────────────────────────

def sync_inst_flow(
    conn: sqlite3.Connection,
    backfill_days: int = 0,
    progress_cb: Optional[Callable] = None,
) -> None:
    """
    同步三大法人資料（缺漏回填語義）。
    掃描窗口 = [today - backfill_days, today]；backfill_days=0 時窗口從
    DB 最新 trade_date+1 起（日常增量）。窗口內每個交易日，TWSE / TPEX
    各自缺資料才抓 → 可回補歷史、可補單邊失敗、天然冪等。
    今天台北時間 < 18:00 不抓今天（盤後資料未發布）；
    過去日期確認無資料者記 sync_log status='empty'，之後不再重試。
    """
    from pytz import timezone

    tw_now = datetime.now(timezone("Asia/Taipei"))
    today = tw_now.date()

    if backfill_days > 0:
        window_start = today - timedelta(days=backfill_days)
    else:
        row = conn.execute("SELECT MAX(trade_date) FROM inst_flow").fetchone()
        window_start = (
            date.fromisoformat(row[0]) + timedelta(days=1) if row and row[0] else today
        )

    def _done_dates(market: str, source: str) -> set:
        done = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT trade_date FROM inst_flow "
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

    twse_done = _done_dates("TWSE", "T86")
    tpex_done = _done_dates("TPEX", "TPEx")

    def _log(source: str, d: date, status: str, n: int, msg: str = None):
        conn.execute(
            "INSERT OR REPLACE INTO sync_log VALUES (?,?,?,?,?)",
            (source, d.isoformat(), status, n, msg),
        )
        conn.commit()

    d = window_start
    while d <= today:
        # 今天且現在 < 18:00 → 盤後資料未發布
        if d == today and tw_now.hour < 18:
            print(f"[sync] {d} Taipei time {tw_now.strftime('%H:%M')} < 18:00, skip today")
            break

        if not is_trading_day(conn, d):
            d += timedelta(days=1)
            continue

        iso = d.isoformat()
        need_twse = iso not in twse_done
        need_tpex = iso not in tpex_done
        if not (need_twse or need_tpex):
            d += timedelta(days=1)  # 已有資料 → 不發請求、不 sleep
            continue

        total_rows = 0

        if need_twse:
            try:
                df_twse = fetch_t86(d)
                if df_twse is not None and not df_twse.empty:
                    total_rows += upsert(conn, "inst_flow", df_twse)
                    _log("T86", d, "ok", total_rows)
                elif d != today:
                    _log("T86", d, "empty", 0, "stat!=OK（臨時休市或無資料）")
            except Exception as e:
                _log("T86", d, "error", 0, str(e))

        if need_tpex:
            try:
                df_tpex = fetch_tpex_inst(d)
                if df_tpex is not None and not df_tpex.empty:
                    n = upsert(conn, "inst_flow", df_tpex)
                    total_rows += n
                    _log("TPEx", d, "ok", n)
                elif df_tpex is None:
                    _log("TPEx", d, "blocked", 0, "兩端點均 403/失敗")
                elif d != today:
                    _log("TPEx", d, "empty", 0, "回應無資料")
            except Exception as e:
                _log("TPEx", d, "error", 0, str(e))

        if progress_cb:
            progress_cb(d, total_rows)
        else:
            print(f"[sync] {d} rows={total_rows}")

        time.sleep(5 + random.random())
        d += timedelta(days=1)


# ── 查詢函式 ──────────────────────────────────────────────────────────────────

def get_inst_flow(
    conn: sqlite3.Connection, stock_id: str, days: int = 60
) -> pd.DataFrame:
    """回傳指定個股近 N 筆法人資料，按日期升冪。"""
    sql = """
        SELECT * FROM inst_flow
        WHERE stock_id = ?
        ORDER BY trade_date DESC
        LIMIT ?
    """
    df = pd.read_sql_query(sql, conn, params=(stock_id, days))
    return df.sort_values("trade_date").reset_index(drop=True)


def get_inst_summary(conn: sqlite3.Connection, days: int = 5) -> pd.DataFrame:
    """
    以表內最近 N 個 distinct trade_date 為窗，
    回傳每股 SUM(foreign_net)/SUM(trust_net)/SUM(total_net)。
    """
    # 先取最近 N 個交易日
    dates_sql = f"""
        SELECT DISTINCT trade_date FROM inst_flow
        ORDER BY trade_date DESC LIMIT {days}
    """
    dates_df = pd.read_sql_query(dates_sql, conn)
    if dates_df.empty:
        return pd.DataFrame(columns=["stock_id", "f_net", "t_net", "all_net"])

    placeholders = ",".join(["?"] * len(dates_df))
    sql = f"""
        SELECT stock_id,
               SUM(foreign_net) AS f_net,
               SUM(trust_net)   AS t_net,
               SUM(total_net)   AS all_net
        FROM inst_flow
        WHERE trade_date IN ({placeholders})
        GROUP BY stock_id
    """
    return pd.read_sql_query(
        sql, conn, params=tuple(dates_df["trade_date"].tolist())
    )


def get_trust_streak(conn: sqlite3.Connection, lookback: int = 15) -> pd.DataFrame:
    """
    投信連續買超天數（從最近交易日往回數，trust_net>0 連續天數）。
    回傳 DataFrame(stock_id, streak)。
    """
    sql = f"""
        SELECT stock_id, trade_date, trust_net FROM inst_flow
        WHERE trade_date IN (
            SELECT DISTINCT trade_date FROM inst_flow
            ORDER BY trade_date DESC LIMIT {lookback}
        )
        ORDER BY stock_id, trade_date DESC
    """
    df = pd.read_sql_query(sql, conn)
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "streak"])

    results = []
    for sid, grp in df.groupby("stock_id"):
        streak = 0
        for _, row in grp.iterrows():
            if row["trust_net"] > 0:
                streak += 1
            else:
                break
        results.append({"stock_id": sid, "streak": streak})
    return pd.DataFrame(results)


def get_data_status(conn: sqlite3.Connection) -> dict:
    """回傳資料涵蓋狀況摘要，供 UI 系統頁顯示。"""
    def scalar(sql, params=()):
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else None

    return {
        "inst_flow": {
            "latest_date": scalar("SELECT MAX(trade_date) FROM inst_flow"),
            "total_rows": scalar("SELECT COUNT(*) FROM inst_flow"),
            "stock_count": scalar("SELECT COUNT(DISTINCT stock_id) FROM inst_flow"),
        },
        "monthly_revenue": {
            "latest_month": scalar("SELECT MAX(year_month) FROM monthly_revenue"),
            "stock_count": scalar(
                "SELECT COUNT(DISTINCT stock_id) FROM monthly_revenue"
            ),
        },
    }

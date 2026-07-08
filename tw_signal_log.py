"""
tw_signal_log.py -- 複合訊號日誌（樣本外追蹤）＋盤後 Telegram 推播

原則：
- 只記「當日新觸發」（onset：今天成立、前一交易日不成立），與回測同語義。
- 只記最新交易日、不回填歷史——日誌的價值在「樣本外」，必須用當下可知資訊。
  （已知小缺陷：onset 判斷用今日營收表回看昨日，營收公告當天觸發的訊號
  可能被誤判為「昨日已成立」而漏記；影響極小、方向保守。）
- 報酬結算與 backtest_composite 同方法論：T+1 開盤進場、T+1+h 收盤出場、
  成本 0.585%、跨除權息以 1/factor 還原（含息報酬）。
- 參數固定為已驗證的 app 預設，快照存 json 供日後追溯。
"""

import json
import sqlite3
from typing import Optional

import pandas as pd

import tw_config
from tw_db import upsert

COST = tw_config.COST_ROUNDTRIP
STRATEGY = "composite_v1"
PARAMS = {**tw_config.COMPOSITE, "liq_min": tw_config.LIQ_MIN}


def _trade_dates(conn, upto: str = None, last_n: int = None) -> list:
    sql = "SELECT DISTINCT trade_date FROM daily_price"
    params = ()
    if upto:
        sql += " WHERE trade_date <= ?"
        params = (upto,)
    sql += " ORDER BY trade_date"
    rows = [r[0] for r in conn.execute(sql, params)]
    return rows[-last_n:] if last_n else rows


def compute_hits(conn: sqlite3.Connection, d: str) -> pd.DataFrame:
    """以日期 d 盤後可知資訊計算複合訊號 onset 名單。"""
    win = _trade_dates(conn, upto=d, last_n=18)
    if len(win) < PARAMS["inst_days"] + 2 or win[-1] != d:
        return pd.DataFrame()

    ph = ",".join("?" * len(win))
    inst = pd.read_sql_query(
        f"SELECT trade_date, stock_id, foreign_net, trust_net FROM inst_flow "
        f"WHERE trade_date IN ({ph})", conn, params=win)
    if inst.empty:
        return pd.DataFrame()
    f = inst.pivot(index="trade_date", columns="stock_id",
                   values="foreign_net").reindex(win).fillna(0)
    t = inst.pivot(index="trade_date", columns="stock_id",
                   values="trust_net").reindex(win).fillna(0)
    N = PARAMS["inst_days"]
    f_roll, t_roll = f.rolling(N).sum(), t.rolling(N).sum()
    b = t > 0
    cs = b.cumsum()
    streak = (cs - cs.where(~b).ffill().fillna(0)).astype(int)

    from tw_revenue import get_revenue_signals
    rev = get_revenue_signals(conn)
    yoy = rev.set_index("stock_id")["yoy_pct"] if not rev.empty else pd.Series(dtype=float)

    def cond(day):
        c = (t_roll.loc[day] > 0) & (streak.loc[day] >= PARAMS["min_streak"])
        if PARAMS["require_foreign"]:
            c &= (f_roll.loc[day] > 0)
        y = yoy.reindex(c.index)
        return c & (y > PARAMS["yoy_thr"]).fillna(False)

    onset = cond(d) & ~cond(win[-2])
    ids = list(onset[onset].index)
    if not ids:
        return pd.DataFrame()

    ph2 = ",".join("?" * len(ids))
    px = pd.read_sql_query(
        f"SELECT stock_id, close, turnover FROM daily_price "
        f"WHERE trade_date=? AND stock_id IN ({ph2})",
        conn, params=[d] + ids).set_index("stock_id")

    rows = []
    for s in ids:
        if s not in px.index:
            continue
        tov = px.at[s, "turnover"] or 0
        if tov < PARAMS["liq_min"]:
            continue
        rows.append({
            "signal_date": d, "stock_id": s, "strategy": STRATEGY,
            "params": json.dumps(PARAMS, ensure_ascii=False),
            "close_at_signal": px.at[s, "close"],
            "f_net": int(f_roll.at[d, s]), "t_net": int(t_roll.at[d, s]),
            "streak": int(streak.at[d, s]),
            "yoy_pct": float(yoy.get(s)) if s in yoy.index else None,
            "entry_open": None, "ret_5": None, "ret_20": None, "ret_60": None,
        })
    return pd.DataFrame(rows)


def get_cycle_phase():
    """大盤相位 1=牛/0=熊（^TWII 收盤>SMA60，與 app/walk-forward 同規則）。
    實證（walkforward_validation 2026-07-08）：牛相位訊號 h20 +7.3%/勝54.7%、
    熊相位 −1.6%/勝46.8% → 相位隨訊號入庫，熊相位推播帶警語。失敗回 None。"""
    try:
        import yfinance as yf
        tw = yf.Ticker("^TWII").history(period="1y", auto_adjust=True)
        if tw is None or len(tw) < 60:
            return None
        sma60 = tw["Close"].rolling(60).mean().iloc[-1]
        return int(float(tw["Close"].iloc[-1]) > float(sma60))
    except Exception:
        return None


def log_signals(conn: sqlite3.Connection, d: str = None) -> pd.DataFrame:
    """記錄最新交易日的新訊號（冪等：重跑同日結果覆蓋、已回填報酬不動——
    重跑只在當日資料重抓時發生，屆時 ret_* 尚未回填）。"""
    if d is None:
        row = conn.execute("SELECT MAX(trade_date) FROM daily_price").fetchone()
        d = row[0] if row else None
    if not d:
        return pd.DataFrame()
    hits = compute_hits(conn, d)
    if not hits.empty:
        hits["cycle_phase"] = get_cycle_phase()
        upsert(conn, "signal_log", hits)
    return hits


def update_signal_returns(conn: sqlite3.Connection) -> int:
    """回填已到期的 entry_open 與 ret_5/20/60（含成本、含息還原）。"""
    dates = _trade_dates(conn)
    gpos = {x: i for i, x in enumerate(dates)}
    pend = pd.read_sql_query(
        "SELECT * FROM signal_log WHERE ret_60 IS NULL", conn)
    if pend.empty:
        return 0
    div = pd.read_sql_query(
        "SELECT stock_id, ex_date, factor FROM dividend_events "
        "WHERE factor IS NOT NULL AND factor > 0", conn)
    divmap = {}
    for r in div.itertuples():
        divmap.setdefault(r.stock_id, []).append((r.ex_date, r.factor))

    def px_at(sid, dt, col):
        row = conn.execute(
            f"SELECT {col} FROM daily_price WHERE stock_id=? AND trade_date=?",
            (sid, dt)).fetchone()
        return row[0] if row and row[0] else None

    def _clean(v):
        """SQL NULL 經 pandas 讀出可能是 None 或 NaN（依整欄 dtype 而定）——
        一律正規化成 None，否則 NaN 會被 `or`/`is not None` 當成有效值。"""
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

    n_upd = 0
    for r in pend.itertuples():
        g = gpos.get(r.signal_date)
        if g is None or g + 1 >= len(dates):
            continue
        entry_date = dates[g + 1]
        entry_prev = _clean(r.entry_open)
        entry = entry_prev if entry_prev else px_at(r.stock_id, entry_date, "open")
        if not entry:
            continue
        vals = {"entry_open": entry,
                "ret_5": _clean(r.ret_5), "ret_20": _clean(r.ret_20),
                "ret_60": _clean(r.ret_60)}
        changed = entry_prev is None
        for h, col in [(5, "ret_5"), (20, "ret_20"), (60, "ret_60")]:
            if vals[col] is not None:
                continue
            xi = g + 1 + h
            if xi >= len(dates):
                continue
            exit_c = px_at(r.stock_id, dates[xi], "close")
            if not exit_c:
                continue
            inv = 1.0
            for exd, fac in divmap.get(r.stock_id, []):
                if entry_date < exd <= dates[xi]:
                    inv /= fac
            vals[col] = exit_c * inv / entry - 1 - COST
            changed = True
        if changed:
            conn.execute(
                "UPDATE signal_log SET entry_open=?, ret_5=?, ret_20=?, ret_60=? "
                "WHERE signal_date=? AND stock_id=? AND strategy=?",
                (vals["entry_open"], vals["ret_5"], vals["ret_20"], vals["ret_60"],
                 r.signal_date, r.stock_id, r.strategy))
            n_upd += 1
    conn.commit()
    return n_upd


def get_journal(conn: sqlite3.Connection):
    """回傳 (summary dict, 明細 DataFrame 由新到舊)。"""
    df = pd.read_sql_query(
        "SELECT l.signal_date, l.stock_id, n.name, l.close_at_signal, "
        "       l.streak, l.yoy_pct, l.cycle_phase, l.ret_5, l.ret_20, l.ret_60 "
        "FROM signal_log l LEFT JOIN stock_names n ON n.stock_id = l.stock_id "
        "WHERE l.strategy = ? "
        "ORDER BY l.signal_date DESC, l.stock_id", conn, params=(STRATEGY,))
    s20 = df["ret_20"].dropna()
    s60 = df["ret_60"].dropna()
    summary = {
        "n_total": len(df),
        "n20": len(s20),
        "avg20": s20.mean() * 100 if len(s20) else None,
        "win20": (s20 > 0).mean() * 100 if len(s20) else None,
        "avg60": s60.mean() * 100 if len(s60) else None,
    }
    return summary, df


def push_daily_signals(conn: sqlite3.Connection, hits: pd.DataFrame, d: str) -> bool:
    """盤後推播當日新訊號到 Telegram；無憑證時安靜跳過（回 False）。"""
    from tw_notifications import send_tg_message  # 延遲載入（含 streamlit）

    names = dict(conn.execute("SELECT stock_id, name FROM stock_names"))
    phase = (int(hits["cycle_phase"].iloc[0])
             if hits is not None and not hits.empty
             and pd.notna(hits["cycle_phase"].iloc[0]) else get_cycle_phase())
    ph_txt = {1: "🐂 牛相位", 0: "🐻 熊相位"}.get(phase, "—")
    lines = [
        "🎯 *籌碼×營收 複合訊號（樣本外日誌）*",
        f"📅 {d} 盤後 | 參數：10日雙買/投信連買≥5/YoY>20 | 大盤：{ph_txt}",
        "─────────────────────",
    ]
    if phase == 0:
        lines.append("⚠️ *熊相位警示*：實證熊相位訊號 20 日平均 −1.6%、勝率 47%"
                     "——本日訊號僅供觀察，不建議進場（等大盤收復季線）。")
    if hits is None or hits.empty:
        lines.append("本日無新觸發訊號。")
    else:
        for r in hits.itertuples():
            nm = names.get(r.stock_id, "")
            yoy = f"{r.yoy_pct:.0f}%" if r.yoy_pct is not None else "—"
            lines.append(
                f"• *{r.stock_id} {nm}*  收盤 {r.close_at_signal}  "
                f"投信連買{r.streak}日  YoY {yoy}")
        lines.append("")
        lines.append(f"共 {len(hits)} 檔（回測參考：持有 20–60 日、全訊號分散）")
    lines += ["─────────────────────", "⚠️ 僅供研究參考，不構成投資建議"]
    return send_tg_message("\n".join(lines))

"""
tw_portfolio_alerts.py -- 持倉例外推播（事件驅動，非每日轟炸）

設計原則（詳見 2026-07-05 持倉功能深度分析）：
- 只在「需要行動」時推：破停損／過熱減碼／進場條件瓦解／融資使用率突破。
  無事件的日子完全安靜——通知的價值與頻率成反比。
- 週五盤後附一則組合週結（回顧用，不催促行動）。
- 持倉在 Google Sheets：headless 執行需本機 ~/.streamlit/secrets.toml 含
  gcp_service_account（與 TG_TOKEN 同一個檔）。無憑證 → 安靜跳過並提示。
- 訊號計算與 app 完全同源：daily_price 建 OHLCV → calculate_indicators →
  get_strategy（market_info 用 DB 5 日序列，與線上因子一致）。
"""

import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

from tw_indicators import calculate_indicators
from tw_strategy import get_strategy


def _load_portfolio_headless() -> Optional[pd.DataFrame]:
    """讀 Google Sheets 持倉（st.secrets 於 bare mode 讀 ~/.streamlit/secrets.toml）。"""
    try:
        import gspread
        import streamlit as st
        sa = st.secrets.get("gcp_service_account")
        if not sa:
            return None
        gc = gspread.service_account_from_dict(dict(sa))
        df = pd.DataFrame(gc.open("Streamlit TW Stock").sheet1.get_all_records())
        if df.empty or "Symbol" not in df.columns:
            return None
        df["Symbol"] = df["Symbol"].astype(str).str.zfill(4)
        df["Shares"] = pd.to_numeric(df.get("Shares"), errors="coerce").fillna(0)
        df["Cost"] = pd.to_numeric(df.get("Cost"), errors="coerce").fillna(0)
        return df[df["Shares"] > 0]
    except Exception as e:
        print(f"[pf-alert] portfolio unavailable: {type(e).__name__}")
        return None


def _market_info_from_db(conn, sym: str) -> dict:
    """與 get_market_data 注入欄位同義的 DB 版 market_info（headless 不打 wespai）。"""
    mi = {}
    try:
        rows = conn.execute(
            "SELECT trade_date, foreign_net, trust_net FROM inst_flow "
            "WHERE stock_id=? ORDER BY trade_date DESC LIMIT 15", (sym,)).fetchall()
        if rows:
            f5 = sum(r[1] or 0 for r in rows[:5]) / 1000
            t5 = sum(r[2] or 0 for r in rows[:5]) / 1000
            streak = 0
            for r in rows:
                if (r[2] or 0) > 0:
                    streak += 1
                else:
                    break
            mi.update({"f_net_5d": f5, "t_net_5d": t5, "trust_streak": streak})
        u = conn.execute(
            "SELECT CAST(margin_balance AS REAL)/NULLIF(margin_quota,0)*100 "
            "FROM margin_trading WHERE stock_id=? "
            "ORDER BY trade_date DESC LIMIT 1", (sym,)).fetchone()
        if u and u[0] is not None:
            mi["margin_util"] = round(u[0], 1)
    except Exception:
        pass
    return mi


def _price_df(conn, sym: str, days: int = 300) -> Optional[pd.DataFrame]:
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, volume FROM daily_price "
        "WHERE stock_id=? ORDER BY trade_date DESC LIMIT ?",
        conn, params=(sym, days))
    if len(df) < 80:
        return None
    df = df.sort_values("trade_date").set_index("trade_date")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df.dropna(subset=["Close", "High", "Low"])


def check_portfolio_alerts(conn: sqlite3.Connection):
    """回傳 (alerts list[str], summary dict) 或 (None, None)=無持倉憑證。"""
    pf = _load_portfolio_headless()
    if pf is None:
        return None, None
    try:
        rev = pd.read_sql_query(
            "SELECT stock_id, year_month, yoy_pct FROM monthly_revenue "
            "WHERE yoy_pct IS NOT NULL", conn)
        yoy_map = dict(
            rev.sort_values("year_month").groupby("stock_id").tail(1)
            [["stock_id", "yoy_pct"]].itertuples(index=False, name=None))
    except Exception:
        yoy_map = {}
    names = dict(conn.execute("SELECT stock_id, name FROM stock_names"))

    alerts, rows = [], []
    for r in pf.itertuples():
        sym = r.Symbol
        nm = names.get(sym, "")
        dfp = _price_df(conn, sym)
        if dfp is None:
            continue
        ind = calculate_indicators(dfp)
        if ind is None or ind.empty:
            continue
        mi = _market_info_from_db(conn, sym)
        strat = get_strategy(ind, held_shares=float(r.Shares),
                             held_cost=float(r.Cost), market_info=mi)
        close = float(ind["Close"].iloc[-1])
        pl_pct = (close / r.Cost - 1) * 100 if r.Cost else 0
        rows.append({"sym": sym, "name": nm, "close": close, "pl_pct": pl_pct,
                     "score": strat["score"], "mv": close * r.Shares})

        if strat["action"] == "SELL_EXIT":
            alerts.append(f"🔴 *{sym} {nm}* 跌破停損防線（收盤 {close:.2f}，"
                          f"損益 {pl_pct:+.1f}%）——紀律出場檢查")
        elif strat["action"] == "SELL_PARTIAL":
            alerts.append(f"🟠 *{sym} {nm}* 高檔過熱（KD/RSI 極端，"
                          f"損益 {pl_pct:+.1f}%）——建議減碼鎖利")
        f5, t5 = mi.get("f_net_5d"), mi.get("t_net_5d")
        if f5 is not None and t5 is not None and f5 < 0 and t5 < 0:
            alerts.append(f"⚠️ *{sym} {nm}* 外資投信5日轉同賣"
                          f"（外資{f5:+,.0f}／投信{t5:+,.0f}張）——進場理由檢查")
        yoy = yoy_map.get(sym)
        if yoy is not None and yoy < 0:
            alerts.append(f"⚠️ *{sym} {nm}* 最新月營收 YoY 轉負（{yoy:+.1f}%）")
        util = mi.get("margin_util")
        if util is not None and util >= 15:
            alerts.append(f"⚠️ *{sym} {nm}* 融資使用率 {util:.1f}%（偏高）")

    summary = None
    if rows:
        tot_mv = sum(x["mv"] for x in rows)
        best = max(rows, key=lambda x: x["pl_pct"])
        worst = min(rows, key=lambda x: x["pl_pct"])
        summary = {
            "n": len(rows), "mv": tot_mv,
            "avg_score": sum(x["score"] for x in rows) / len(rows),
            "hot_n": sum(1 for x in rows if x["score"] >= 8),
            "best": best, "worst": worst,
        }
    return alerts, summary


def push_portfolio_alerts(conn: sqlite3.Connection, weekly: bool = False) -> bool:
    """例外才推；weekly=True（週五）另附組合週結。無 TG/持倉憑證安靜跳過。"""
    from tw_notifications import send_tg_message

    alerts, summary = check_portfolio_alerts(conn)
    if alerts is None:
        print("[pf-alert] skipped（本機無 gcp_service_account 憑證）")
        return False
    if not alerts and not weekly:
        print("[pf-alert] no exceptions today — 不推播")
        return False

    lines = ["💼 *持倉例外提醒*",
             f"📅 {datetime.now():%Y-%m-%d} 盤後", "─────────────────────"]
    if alerts:
        lines += alerts
    else:
        lines.append("本週無例外事件。")
    if weekly and summary:
        lines += ["", "📊 *組合週結*",
                  f"持股 {summary['n']} 檔・平均評分 {summary['avg_score']:.1f}／10"
                  f"・過熱 {summary['hot_n']} 檔",
                  f"最佳：{summary['best']['sym']} {summary['best']['name']} "
                  f"{summary['best']['pl_pct']:+.1f}%",
                  f"最弱：{summary['worst']['sym']} {summary['worst']['name']} "
                  f"{summary['worst']['pl_pct']:+.1f}%"]
    lines += ["─────────────────────", "⚠️ 僅供研究參考，不構成投資建議"]
    return send_tg_message("\n".join(lines))

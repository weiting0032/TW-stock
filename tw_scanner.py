"""掃描引擎、市場廣度、AutoScan 日誌"""
import concurrent.futures
from datetime import datetime

import pandas as pd
import pytz
import streamlit as st

from tw_data import get_gsheet_client, fetch_stock_history, PORTFOLIO_SHEET_TITLE
from tw_strategy import get_strategy

_TW = pytz.timezone("Asia/Taipei")

# ── 半導體關鍵字 ──────────────────────────────────────────────────────────────
SEMI_KEYWORDS = [
    "半導體", "電子零組件", "光電", "通信網路", "電腦及週邊", "其他電子",
    "積體電路", "IC設計", "ic設計", "晶圓", "晶圓代工",
    "封裝", "測試", "封裝測試", "OSAT",
    "記憶體", "DRAM", "Flash", "NAND", "NOR", "SRAM", "HBM",
    "功率元件", "電源管理", "類比", "PMIC", "MOSFET", "IGBT", "驅動IC", "混合訊號",
    "RF", "射頻", "藍牙", "WiFi", "無線", "5G晶片",
    "伺服器", "Server", "資料中心", "AI加速", "GPU", "NPU",
    "CoWoS", "ABF", "先進封裝",
    "基板", "載板", "導線框", "引線框", "印刷電路板", "電路板", "PCB",
    "散熱", "液冷", "均溫板", "熱管", "水冷",
    "化合物半導體", "碳化矽", "SiC", "氮化鎵", "GaN", "矽晶圓", "磊晶", "砷化鎵",
    "光罩", "光阻", "研磨", "CMP", "濺鍍", "蝕刻", "半導體設備", "晶圓設備",
    "被動元件", "電感", "MLCC", "電容", "電阻", "連接器",
    "矽智財", "IP矽", "EDA", "IP授權",
    "感測器", "CMOS感測", "影像感測", "ToF", "LiDAR",
]

SEMI_SCORE_MIN = 5.0
SEMI_SCAN_MAX  = 1500


def _sf(val, default=float("nan")) -> float:
    try:
        f = float(val)
        return f if not pd.isna(f) else default
    except Exception:
        return default


def is_semiconductor(industry: str) -> bool:
    ind = str(industry).strip()
    if not ind or ind in ("nan", "None", ""):
        return False
    il = ind.lower()
    return any(kw.lower() in il for kw in SEMI_KEYWORDS)


def is_tw_trading_day() -> bool:
    return datetime.now(_TW).weekday() < 5


def _build_scan_pool(market_map: dict, pe_lim: float, pb_lim: float,
                     max_price: float, sector_filter: list) -> dict:
    pool = {}
    for code, info in market_map.items():
        price = _sf(info.get("現價"))
        pe    = _sf(info.get("PE"))
        pb    = _sf(info.get("PB"))
        ind   = str(info.get("產業", ""))
        if pd.isna(price) or price <= 0 or price > max_price:  continue
        if pd.isna(pe)    or pe    <= 0 or pe    > pe_lim:     continue
        if pd.isna(pb)    or pb    <= 0 or pb    > pb_lim:     continue
        if sector_filter and ind not in sector_filter:          continue
        pool[code] = info
    return pool


def _make_candidate(code: str, info: dict, strat: dict, h_df) -> dict:
    return {
        "代碼":    code,
        "名稱":    info.get("名稱", ""),
        "產業":    info.get("產業", ""),
        "現價":    _sf(info.get("現價"), 0.0),
        "PE":      round(_sf(info.get("PE"), 0.0), 1),
        "PB":      round(_sf(info.get("PB"), 0.0), 2),
        "score":   strat["score"],
        "action":  strat["action"],
        "sl":      strat["sl"],
        "tp":      strat["tp"],
        "reasons": strat["reasons"],
        "strat":   strat,
        "df":      h_df,
    }


def run_scan(market_map: dict, scan_list: list, min_score: float,
             max_workers: int = 12, status_placeholder=None):
    """通用掃描核心，回傳 (candidates, total_scanned)"""
    total      = max(len(scan_list), 1)
    candidates = []
    done       = [0]

    def _scan_one(args):
        code, info = args
        try:
            h_df  = fetch_stock_history(code)
            if h_df is None:
                return None
            strat = get_strategy(h_df)
            return _make_candidate(code, info, strat, h_df) if strat["score"] >= min_score else None
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_scan_one, item): item[0] for item in scan_list}
        for fut in concurrent.futures.as_completed(futures):
            done[0] += 1
            if status_placeholder:
                pct = done[0] / total
                status_placeholder.progress(
                    pct, text=f"掃描 {done[0]}/{total} ({pct*100:.0f}%) — 發現 {len(candidates)} 個強勢標的",
                )
            r = fut.result()
            if r:
                candidates.append(r)

    candidates.sort(key=lambda x: -x["score"])
    return candidates, total


def run_semiconductor_scan(market_map: dict, status_placeholder=None):
    semi_pool = {
        code: info for code, info in market_map.items()
        if is_semiconductor(info.get("產業", "")) and _sf(info.get("現價"), 0) > 0
    }
    scan_list = list(semi_pool.items())[:SEMI_SCAN_MAX]
    return run_scan(market_map, scan_list, SEMI_SCORE_MIN,
                    max_workers=12, status_placeholder=status_placeholder)


def compute_market_breadth(portfolio_df: pd.DataFrame,
                           scan_results: list | None) -> dict | None:
    """
    從已快取的持倉 + 掃描結果計算市場廣度（不觸發新的 API 呼叫）。
    回傳 None 若可用資料不足。
    """
    records = []

    # From portfolio
    for _, r in portfolio_df.iterrows():
        h_df = fetch_stock_history(r["Symbol"])
        if h_df is not None and len(h_df) >= 2:
            records.append(h_df)

    # From scan results (already fetched)
    if scan_results:
        for c in scan_results:
            if c.get("df") is not None:
                records.append(c["df"])

    if len(records) < 3:
        return None

    advance = decline = flat = above_ma60 = new_high = new_low = 0
    for h_df in records:
        last  = h_df.iloc[-1]
        prev  = h_df.iloc[-2]
        close = float(last["Close"])
        prev_c = float(prev["Close"])
        ma60   = float(last.get("SMA60") or 0)
        _h52   = last.get("High52W")
        _l52   = last.get("Low52W")
        h52    = float(_h52) if _h52 and not pd.isna(_h52) else close
        l52    = float(_l52) if _l52 and not pd.isna(_l52) else close

        if close > prev_c:   advance += 1
        elif close < prev_c: decline += 1
        else:                flat    += 1

        if ma60 > 0 and close > ma60:
            above_ma60 += 1
        if close >= h52 * 0.99:
            new_high += 1
        if close <= l52 * 1.01:
            new_low  += 1

    total = max(advance + decline + flat, 1)
    return {
        "advance":        advance,
        "decline":        decline,
        "flat":           flat,
        "ad_ratio":       round(advance / max(decline, 1), 2),
        "above_ma60_pct": round(above_ma60 / total * 100, 1),
        "new_high":       new_high,
        "new_low":        new_low,
        "total":          total,
    }


# ── AutoScan 日誌 ─────────────────────────────────────────────────────────────

def _get_auto_scan_ws():
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        try:
            return sh.worksheet("AutoScan")
        except Exception:
            ws = sh.add_worksheet("AutoScan", rows=500, cols=6)
            ws.append_row(["Date", "Status", "ScanCount", "HitCount", "SentAt", "Note"])
            return ws
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_last_auto_scan_date() -> str:
    ws = _get_auto_scan_ws()
    if ws is None:
        return ""
    try:
        for row in reversed(ws.get_all_values()[1:]):
            if len(row) >= 2 and row[1] == "OK":
                return row[0]
    except Exception:
        pass
    return ""


def log_auto_scan_result(scan_count: int, hit_count: int, note: str = ""):
    ws = _get_auto_scan_ws()
    if ws is None:
        return
    today   = datetime.now(_TW).strftime("%Y-%m-%d")
    sent_at = datetime.now(_TW).strftime("%H:%M:%S")
    try:
        ws.append_row([today, "OK", scan_count, hit_count, sent_at, note])
        get_last_auto_scan_date.clear()
    except Exception:
        pass


def check_and_trigger_auto_scan(market_map: dict):
    """頁面載入時靜默檢查是否需執行當日自動掃描（18:00 後）"""
    import threading
    from tw_notifications import format_semi_tg_messages, send_tg_message

    tw_now = datetime.now(_TW)
    if not is_tw_trading_day() or tw_now.hour < 18:
        return

    today_str = tw_now.strftime("%Y-%m-%d")
    if get_last_auto_scan_date() == today_str:
        return
    if st.session_state.get("auto_scan_done_today") == today_str:
        return

    st.session_state.auto_scan_done_today = today_str

    def _bg():
        try:
            cands, n = run_semiconductor_scan(market_map)
            msgs = format_semi_tg_messages(cands, n, len(cands), today_str)
            ok   = all(send_tg_message(m) for m in msgs)
            log_auto_scan_result(n, len(cands), "auto" + ("_ok" if ok else "_tg_fail"))
        except Exception as e:
            log_auto_scan_result(0, 0, f"error:{e}")

    threading.Thread(target=_bg, daemon=True).start()

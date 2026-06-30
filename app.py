"""
台股戰情指揮中心 V16
Multi-factor · MTF · K線型態 · 回測 · 自選股 · 市場廣度 · 資產歷史
"""
import math
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st

from tw_data import (
    fetch_stock_history, fetch_weekly_history, get_market_data,
    fetch_taiex_cycle,
    load_portfolio, load_portfolio_history, load_watchlist,
    save_portfolio_snapshot, add_to_watchlist, remove_from_watchlist,
    get_gsheet_client, PORTFOLIO_SHEET_TITLE,
)
from tw_indicators import detect_candlestick_patterns
from tw_notifications import format_semi_tg_messages, send_tg_message
from tw_scanner import (
    _build_scan_pool, _sf, check_and_trigger_auto_scan,
    compute_market_breadth, get_last_auto_scan_date,
    is_semiconductor, is_tw_trading_day, log_auto_scan_result,
    run_scan, run_semiconductor_scan,
    SEMI_KEYWORDS, SEMI_SCORE_MIN, SEMI_SCAN_MAX,
)
from tw_strategy import _default_strat, get_strategy, get_strategy_mtf, run_backtest
from tw_ui import (
    accent_colour, get_tw_session, make_backtest_chart,
    make_breadth_gauge, make_portfolio_performance_chart,
    make_tw_chart, make_weekly_chart, pl_colour,
    score_bar_html, signal_badge_html,
)

_TW = pytz.timezone("Asia/Taipei")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="台股戰情中心",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Noto+Sans+TC:wght@300;400;500;700;900&display=swap');

:root {
  --bg:#0A0C12; --surface:#111318; --surface2:#181B23;
  --border:rgba(255,255,255,0.07); --border2:rgba(255,255,255,0.13);
  --text:#E6E8F0; --muted:#5A6072;
  --up:#E8192C; --down:#00B050; --gold:#F5A623; --blue:#3D8EFF;
  --purple:#9B6DFF; --cyan:#00D4FF;
  --mono:'JetBrains Mono',monospace; --sans:'Noto Sans TC','DM Sans',sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:var(--sans);}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebarNav"]{display:none;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px;}

[data-testid="stMetricValue"]{font-family:var(--mono)!important;font-size:1.3rem!important;font-weight:700!important;color:var(--text)!important;letter-spacing:-0.02em;}
[data-testid="stMetricLabel"]{font-size:0.68rem!important;color:var(--muted)!important;text-transform:uppercase;letter-spacing:0.06em;font-family:var(--sans)!important;}
[data-testid="stMetricDelta"]{font-size:0.78rem!important;font-family:var(--mono)!important;}
.stMetric{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:14px 16px!important;}

[data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:12px!important;padding:4px!important;gap:2px!important;border:1px solid var(--border)!important;flex-wrap:wrap!important;}
[data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:8px!important;font-size:0.75rem!important;font-weight:700!important;font-family:var(--sans)!important;padding:6px 10px!important;transition:all 0.2s;}
[aria-selected="true"][data-baseweb="tab"]{background:var(--up)!important;color:#fff!important;}

.stButton>button{background:var(--surface2)!important;border:1px solid var(--border2)!important;color:var(--text)!important;border-radius:10px!important;font-family:var(--sans)!important;font-weight:700!important;font-size:0.85rem!important;transition:all 0.2s;}
.stButton>button:hover{border-color:var(--up)!important;color:var(--up)!important;}

.stTextInput>div>div>input,.stNumberInput>div>div>input,.stSelectbox>div>div{background:var(--surface2)!important;border:1px solid var(--border2)!important;border-radius:10px!important;color:var(--text)!important;font-family:var(--mono)!important;font-size:0.9rem!important;}
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important;}

.tw-header{display:flex;align-items:center;justify-content:space-between;padding:12px 0 20px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.tw-logo{font-family:var(--mono);font-size:1.1rem;font-weight:700;color:var(--up);letter-spacing:-0.02em;}
.tw-logo span{color:var(--muted);font-weight:400;}

.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:999px;font-size:0.68rem;font-weight:700;font-family:var(--sans);letter-spacing:0.04em;text-transform:uppercase;}
.badge-up{background:rgba(232,25,44,0.15);color:var(--up);border:1px solid rgba(232,25,44,0.35);}
.badge-down{background:rgba(0,176,80,0.15);color:var(--down);border:1px solid rgba(0,176,80,0.35);}
.badge-flat{background:rgba(90,96,114,0.2);color:var(--muted);border:1px solid var(--border2);}
.badge-gold{background:rgba(245,166,35,0.12);color:var(--gold);border:1px solid rgba(245,166,35,0.3);}
.badge-blue{background:rgba(61,142,255,0.12);color:var(--blue);border:1px solid rgba(61,142,255,0.3);}

.sc{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px;margin-bottom:10px;position:relative;overflow:hidden;}
.sc-accent{position:absolute;left:0;top:0;bottom:0;width:3px;border-radius:16px 0 0 16px;}
.sc-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;}
.sc-name{font-size:1.0rem;font-weight:900;color:var(--text);font-family:var(--sans);}
.sc-code{font-family:var(--mono);font-size:0.72rem;color:var(--muted);font-weight:400;}
.sc-price{font-family:var(--mono);font-size:1.5rem;font-weight:700;}
.sc-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px 14px;margin-top:10px;}
.sc-kv-label{font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;font-family:var(--sans);}
.sc-kv-value{font-family:var(--mono);font-size:0.85rem;color:var(--text);font-weight:600;}
.sc-action{margin-top:10px;padding:9px 12px;background:var(--surface2);border-radius:9px;font-size:0.8rem;font-family:var(--sans);}
.sc-divider{border:none;border-top:1px solid var(--border);margin:10px 0;}

.wbar-bg{background:var(--surface2);border-radius:999px;height:3px;margin-top:8px;}
.wbar-fill{height:3px;border-radius:999px;}

.sig-buy{color:var(--up);font-weight:900;}
.sig-sell{color:var(--down);font-weight:900;}
.sig-hold{color:var(--gold);font-weight:900;}
.sig-watch{color:var(--muted);font-weight:900;}

.sk-card{background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:12px 15px;margin-bottom:7px;display:flex;align-items:center;gap:12px;}
.sk-rank{font-family:var(--mono);font-size:0.72rem;color:var(--muted);min-width:20px;}
.sk-ticker{font-family:var(--mono);font-size:0.95rem;font-weight:700;color:var(--text);}
.sk-name{font-size:0.75rem;color:var(--muted);margin-top:1px;font-family:var(--sans);}
.sk-score{font-family:var(--mono);font-size:0.9rem;font-weight:700;color:var(--up);}
.sk-reason{font-size:0.7rem;color:var(--muted);margin-top:2px;font-family:var(--sans);}
.sbar{background:var(--surface2);border-radius:999px;height:3px;flex:1;}
.sbar-fill{height:3px;border-radius:999px;background:linear-gradient(90deg,var(--up),var(--gold));}

.ps-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.ps{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px;}
.ps-label{font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;font-family:var(--sans);}
.ps-value{font-family:var(--mono);font-size:1.2rem;font-weight:700;margin-top:4px;}

.pattern-tag{display:inline-block;padding:2px 8px;border-radius:6px;font-size:0.7rem;font-family:var(--sans);font-weight:600;margin:2px;background:rgba(61,142,255,0.12);color:var(--blue);border:1px solid rgba(61,142,255,0.3);}

.qdiv{border:none;border-top:1px solid var(--border);margin:16px 0;}
.qsec{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);font-family:var(--sans);font-weight:700;margin:16px 0 10px;}

@media(max-width:600px){
  [data-testid="stMetricValue"]{font-size:1.05rem!important;}
  .sc-price{font-size:1.2rem;}
  .sc{padding:13px;}
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 快取資料（只在需要時載入）
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_stock_options(market_map_size: int) -> list:
    """market_map_size 作為 cache key 以便市場資料更新時重算"""
    return [f"{k} {v['名稱']} ({v['產業']})" for k, v in MARKET_MAP.items()]


MARKET_MAP    = get_market_data()
STOCK_OPTIONS = _cached_stock_options(len(MARKET_MAP))
TAIEX_CYCLE   = fetch_taiex_cycle()


# ─────────────────────────────────────────────────────────────────────────────
# Session state 初始化
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "df_portfolio":      None,
    "df_watchlist":      None,
    "scan_results":      None,
    "diag_plot":         None,
    "diag_weekly":       None,
    "diag_bt":           None,
    "tab1_chart_sym":    "",
    "tab1_chart_data":   None,
    "tab2_chart_sym":    "",
    "tab2_chart_data":   None,
    "wl_chart_sym":      "",
    "wl_chart_data":     None,
    "breadth":           None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.df_portfolio is None:
    st.session_state.df_portfolio = load_portfolio()
if st.session_state.df_watchlist is None:
    st.session_state.df_watchlist = load_watchlist()

# ─────────────────────────────────────────────────────────────────────────────
# Auto-scan trigger
# ─────────────────────────────────────────────────────────────────────────────
check_and_trigger_auto_scan(MARKET_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
session_now = get_tw_session()
_sc_cls = {"交易中": "badge-up", "盤前": "badge-gold", "盤後": "badge-blue", "休市": "badge-flat"}

# ── TAIEX 週期徽章 ─────────────────────────────────────────────────────────────
_cyc = TAIEX_CYCLE
if _cyc:
    _cyc_badge_cls = "badge-up" if _cyc["phase"] == 1 else "badge-down"
    _risk_cls = {"high": "badge-up" if _cyc["phase"] == 0 else "badge-down",
                 "medium": "badge-gold", "safe": "badge-blue"}[_cyc["flip_risk"]]
    _cyc_badges = (
        f'<span class="badge {_cyc_badge_cls}">{_cyc["phase_label"]}</span>'
        f'<span class="badge badge-gold" style="font-family:\'JetBrains Mono\'">第 {_cyc["days_in_cycle"]} 天</span>'
        f'<span class="badge {_risk_cls}" style="font-family:\'JetBrains Mono\'">'
        f'{"翻轉風險↑" if _cyc["flip_risk"]=="high" else "留意" if _cyc["flip_risk"]=="medium" else "穩定"}'
        f' {_cyc["dist_pct"]:+.1f}%</span>'
    )
else:
    _cyc_badges = ""

st.markdown(f"""
<div class="tw-header">
  <div class="tw-logo">台股<span>戰情中心</span> <span style="font-size:0.65rem;color:var(--muted)">V16</span></div>
  <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
    <span class="badge {_sc_cls.get(session_now,'badge-flat')}">{session_now}</span>
    <span class="badge badge-flat" style="font-family:'JetBrains Mono'">{datetime.now().strftime('%m/%d %H:%M')}</span>
    {_cyc_badges}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level metrics
# ─────────────────────────────────────────────────────────────────────────────
portfolio     = st.session_state.df_portfolio
total_mv = total_cost = 0.0
if not portfolio.empty:
    for _, r in portfolio.iterrows():
        m = MARKET_MAP.get(r["Symbol"])
        if m:
            total_mv   += m["現價"] * r["Shares"]
            total_cost += r["Cost"] * r["Shares"]

total_pl = total_mv - total_cost
pl_pct   = (total_pl / total_cost * 100) if total_cost > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("持倉市值",   f"${total_mv:,.0f}")
c2.metric("未實現損益", f"${total_pl:,.0f}", f"{pl_pct:+.2f}%")
c3.metric("投入成本",   f"${total_cost:,.0f}")
c4.metric("持倉檔數",   f"{len(portfolio)} 檔")

st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 庫存動態", "💰 潛力快篩", "🔍 個股診斷", "📝 庫存管理", "⚙️ 系統"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Portfolio Monitor
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if portfolio.empty:
        st.info("📭 庫存為空。請至「庫存管理」新增持股。")
    else:
        details = []
        for _, r in portfolio.iterrows():
            m = MARKET_MAP.get(r["Symbol"])
            if not m:
                continue
            cp    = m["現價"]
            h_df  = fetch_stock_history(r["Symbol"])
            strat = get_strategy(h_df, r["Shares"], r["Cost"], market_info=m) if h_df is not None else _default_strat("連線失敗", "#5A6072")
            details.append({"r": r, "m": m, "cp": cp,
                             "mv": cp * r["Shares"], "cv": r["Cost"] * r["Shares"],
                             "strat": strat, "df": h_df})

        details.sort(key=lambda x: (0 if "SELL" in x["strat"]["action"] else 1, -x["strat"]["score"]))

        # ── 資產歷史折線 ──────────────────────────────────────────────────────
        with st.expander("📈 資產歷史走勢", expanded=False):
            hist_df = load_portfolio_history()
            perf_fig = make_portfolio_performance_chart(hist_df)
            if perf_fig:
                st.plotly_chart(perf_fig, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"共 {len(hist_df)} 筆快照 | 最早：{hist_df['Date'].iloc[0].strftime('%Y-%m-%d') if not hist_df.empty else '—'}")
            else:
                st.caption("歷史資料不足，收盤後系統自動儲存每日快照。")

        # ── 存一筆今日快照（收盤後）──────────────────────────────────────────
        _tw_now = datetime.now(_TW)
        if is_tw_trading_day() and _tw_now.hour >= 14 and total_mv > 0:
            if st.session_state.get("snapshot_saved_today") != _tw_now.strftime("%Y-%m-%d"):
                save_portfolio_snapshot(total_mv, total_cost, total_pl)
                st.session_state.snapshot_saved_today = _tw_now.strftime("%Y-%m-%d")

        # ── 資產配置圓餅圖 ────────────────────────────────────────────────────
        with st.expander("資產配置圓餅圖", expanded=False):
            pie_labels  = [d["r"]["Name"] for d in details]
            pie_values  = [d["mv"] for d in details]
            pie_colours = ["#E8192C","#F5A623","#3D8EFF","#9B6DFF","#00B050",
                           "#FF8C42","#00D4FF","#FF6B9D"][:len(details)]
            pie_fig = go.Figure(go.Pie(
                labels=pie_labels, values=pie_values, hole=0.55,
                marker=dict(colors=pie_colours, line=dict(color="#0A0C12", width=2)),
                textfont=dict(family="Noto Sans TC", size=11),
            ))
            pie_fig.update_layout(template="plotly_dark", height=240,
                                  margin=dict(l=0,r=0,t=5,b=0),
                                  paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="qsec">持倉明細 (依訊號優先排序)</div>', unsafe_allow_html=True)

        for item in details:
            r, m, cp, strat = item["r"], item["m"], item["cp"], item["strat"]
            p_pct    = (cp - r["Cost"]) / r["Cost"] * 100 if r["Cost"] > 0 else 0
            diff_val = (cp - r["Cost"]) * r["Shares"]
            sl_str   = f"${strat['sl']:.2f}" if strat.get("sl") else "—"
            tp_str   = f"${strat['tp']:.2f}" if strat.get("tp") else "—"
            a        = strat["action"]
            vol_r    = float(item["df"].iloc[-1].get("VOL_Ratio", 1.0)) if item["df"] is not None else 1.0

            # K線型態
            patterns   = detect_candlestick_patterns(item["df"]) if item["df"] is not None else {}
            pattern_html = "".join(f'<span class="pattern-tag">{v}</span>' for v in list(patterns.values())[:2])

            if a in ("BUY", "BUY_WATCH"):
                action_line = (f"<span class='sig-buy'>🛒 建議買入 {strat['suggest_lots']} 張"
                               f"，停損 {sl_str}，目標 {tp_str}</span>")
            elif a == "SELL_EXIT":
                action_line = f"<span class='sig-sell'>⚠️ 跌破停損！建議出清 {strat['suggest_lots']} 張</span>"
            elif a == "SELL_PARTIAL":
                action_line = f"<span class='sig-hold'>💰 高檔過熱，建議減碼 {strat['suggest_lots']} 張獲利了結</span>"
            elif a == "HOLD":
                action_line = f"<span class='sig-hold'>🛡️ 持股續抱，跌破 {sl_str} 再出場</span>"
            else:
                action_line = "<span class='sig-watch'>☕ 觀望中，無明確動作</span>"

            # Pre-compute all conditional values to avoid blank-line HTML rendering issues
            _inst_net   = float(m.get("三大合計", 0) or 0) if m else 0.0
            _inst_col   = "var(--up)" if _inst_net > 0 else "var(--down)" if _inst_net < 0 else "var(--muted)"
            _inst_str   = f"+{int(_inst_net)}張" if _inst_net > 0 else (f"{int(_inst_net)}張" if _inst_net < 0 else "—")
            _day_chg    = float(m.get("漲跌幅", 0) or 0) if m else 0.0
            _chg_col    = "var(--up)" if _day_chg > 0 else "var(--down)" if _day_chg < 0 else "var(--muted)"
            _chg_str    = f"{_day_chg:+.2f}%" if _day_chg != 0 else "—"
            _vol_r_col  = "var(--up)" if vol_r >= 1.5 else "var(--muted)"
            _reasons_s  = "、".join(strat['reasons'][:2]) if strat['reasons'] else "—"
            _pct_dir    = "▲" if p_pct > 0 else "▼" if p_pct < 0 else "—"
            _pl_dir     = "▲" if diff_val >= 0 else "▼"
            _ptag_html  = f'<div style="margin-top:4px">{pattern_html}</div>' if pattern_html else ''
            st.markdown(f"""
<div class="sc">
  <div class="sc-accent" style="background:{accent_colour(strat)}"></div>
  <div class="sc-top">
    <div><div class="sc-name">{r['Name']} <span class="sc-code">{r['Symbol']}</span></div><div style="margin-top:3px;font-size:0.7rem;color:var(--muted)">{m['產業']} · PE {m['PE']} · PB {m['PB']}</div>{_ptag_html}</div>
    <div style="text-align:right">{signal_badge_html(strat)}<div style="margin-top:4px;font-family:var(--mono);font-size:0.7rem;color:var(--muted)">分數 {strat['score']:.1f}/10</div></div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;"><span class="sc-price" style="color:{pl_colour(diff_val)}">${cp:.2f}</span><span style="font-family:var(--mono);font-size:0.88rem;color:{pl_colour(p_pct)};font-weight:700">{_pct_dir}{abs(p_pct):.2f}%</span></div>
  <div style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);margin:2px 0 6px">損益 {_pl_dir}${abs(diff_val):,.0f} | 成本 ${r['Cost']:.2f} | {r['Shares']:,.0f}股</div>
  {score_bar_html(strat['score'])}
  <div class="sc-grid">
    <div><span class="sc-kv-label">停損</span><br><span class="sc-kv-value" style="color:var(--down)">{sl_str}</span></div>
    <div><span class="sc-kv-label">目標</span><br><span class="sc-kv-value" style="color:var(--up)">{tp_str}</span></div>
    <div><span class="sc-kv-label">量比</span><br><span class="sc-kv-value" style="color:{_vol_r_col}">{vol_r:.1f}x</span></div>
    <div><span class="sc-kv-label">法人流向</span><br><span class="sc-kv-value" style="color:{_inst_col}">{_inst_str}</span></div>
    <div><span class="sc-kv-label">今日漲跌</span><br><span class="sc-kv-value" style="color:{_chg_col}">{_chg_str}</span></div>
    <div><span class="sc-kv-label">分析依據</span><br><span class="sc-kv-value" style="font-size:0.68rem">{_reasons_s}</span></div>
  </div>
  <div class="sc-action">{action_line}</div>
</div>
""", unsafe_allow_html=True)

            if item["df"] is not None:
                if st.button(f"📊 技術圖表 {r['Symbol']}", key=f"chart_{r['Symbol']}", use_container_width=True):
                    cur = st.session_state.get("tab1_chart_sym", "")
                    if cur == r["Symbol"]:
                        st.session_state.tab1_chart_sym  = ""
                        st.session_state.tab1_chart_data = None
                    else:
                        st.session_state.tab1_chart_sym  = r["Symbol"]
                        st.session_state.tab1_chart_data = (item["df"], r["Name"], strat)

            if st.session_state.get("tab1_chart_sym") == r["Symbol"]:
                t1d = st.session_state.get("tab1_chart_data")
                if t1d:
                    t1_df, t1_name, t1_strat = t1d
                    t1_last = t1_df.iloc[-1]
                    ta, tb, tc, td = st.columns(4)
                    ta.metric("現價",  f"${t1_df['Close'].iloc[-1]:.2f}")
                    tb.metric("K / D", f"{float(t1_last['K']):.0f} / {float(t1_last['D']):.0f}")
                    tc.metric("RSI",   f"{float(t1_last['RSI']):.1f}")
                    td.metric("量比",  f"{float(t1_last.get('VOL_Ratio',1.0)):.1f}x")
                    st.markdown(
                        f'<div class="sc-action" style="border-left:3px solid {t1_strat["color"]};margin-bottom:10px">'
                        f'{t1_strat["html"]}</div>', unsafe_allow_html=True)
                    if t1_strat.get("warnings"):
                        st.warning("⚠️ " + "；".join(t1_strat["warnings"]))
                    st.plotly_chart(make_tw_chart(t1_df, t1_name, t1_strat),
                                    use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Smart Screener
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="qsec">多因子篩選條件</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    pe_lim    = f1.number_input("PE 上限", value=18.0, step=1.0)
    pb_lim    = f2.number_input("PB 上限", value=1.5,  step=0.1)
    min_score = f3.slider("最低技術分 (0-10)", 0.0, 10.0, 4.0, 0.5)

    f4, f5 = st.columns(2)
    sector_options = sorted({str(v["產業"]) for v in MARKET_MAP.values()
                              if v.get("產業") and str(v.get("產業")) not in ("nan", "", "None", "NaN")})
    sector_filter  = f4.multiselect("篩選產業 (空=全部)", options=sector_options)
    max_price      = f5.number_input("股價上限 (TWD)", value=500.0, min_value=1.0, step=10.0)

    _pool_preview = _build_scan_pool(MARKET_MAP, pe_lim, pb_lim, max_price, sector_filter)
    pool_n        = len(_pool_preview)
    # 修正後的時間估算（10 workers, 每股約 0.5–3s）
    est_sec_lo = max(5, pool_n * 5 // 10)
    est_sec_hi = max(10, pool_n * 30 // 10)
    est_str    = (f"{est_sec_lo}–{est_sec_hi} 秒" if est_sec_hi < 120
                  else f"{est_sec_lo//60}–{est_sec_hi//60} 分鐘")
    st.caption(
        f"📋 通過基本面條件：**{pool_n} 檔** 將接受技術分析，"
        f"預估掃描時間 **{est_str}**（平行 10 worker）。"
    )

    sa, sb = st.columns(2)
    scan_all     = sa.checkbox("✅ 全部掃描（不限上限）", value=False)
    scan_cap     = sb.number_input("或設定上限 (掃描前 N 檔)",
                                    min_value=50, max_value=2000, value=200, step=50, disabled=scan_all)
    max_workers_ui = st.slider("平行抓取 Worker 數", 3, 20, 10)

    if st.button("🔍 啟動大盤掃描", use_container_width=True):
        scan_list  = list(_pool_preview.items()) if scan_all else list(_pool_preview.items())[:int(scan_cap)]
        progress   = st.progress(0)
        candidates, _ = run_scan(MARKET_MAP, scan_list, min_score,
                                  max_workers=max_workers_ui, status_placeholder=progress)
        progress.empty()
        candidates.sort(key=lambda x: -x["score"])
        st.session_state.scan_results    = candidates
        st.session_state.tab2_chart_sym  = ""
        st.session_state.tab2_chart_data = None
        st.session_state.breadth         = None  # 清除廣度快取，下次重算

    res = st.session_state.scan_results
    if res is not None:
        buys  = [c for c in res if "BUY" in c["action"]]
        holds = [c for c in res if c["action"] == "HOLD"]

        if not buys and not holds:
            st.warning("無強勢買進或續抱標的。建議放寬技術分門檻或調整篩選條件。")

        if buys:
            st.markdown(f'<div class="qsec">🟥 買進機會 ({len(buys)} 檔)</div>', unsafe_allow_html=True)
            for i, c in enumerate(buys[:20], 1):
                reason_short = "、".join(c["reasons"][:3]) if c["reasons"] else "—"
                st.markdown(f"""
<div class="sk-card" style="border-color:rgba(232,25,44,0.3)">
  <div class="sk-rank">{i}</div>
  <div style="flex:1">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
      <div>
        <div class="sk-ticker">{c['代碼']} {c['名稱']} <span style="font-size:0.65rem;color:var(--muted)">{c['產業']}</span></div>
        <div class="sk-reason">{reason_short}</div>
      </div>
      <div style="text-align:right;min-width:80px">
        <div class="sk-score">{c['score']:.1f}</div>
        <div style="font-family:var(--mono);font-size:0.72rem;color:var(--muted)">PE {c['PE']:.1f} PB {c['PB']:.1f}</div>
      </div>
    </div>
    {score_bar_html(c['score'])}
  </div>
</div>""", unsafe_allow_html=True)
                if st.button(f"📊 診斷 {c['代碼']}", key=f"sc_diag_{c['代碼']}", use_container_width=True):
                    cur = st.session_state.get("tab2_chart_sym", "")
                    if cur == c["代碼"]:
                        st.session_state.tab2_chart_sym  = ""
                        st.session_state.tab2_chart_data = None
                    else:
                        st.session_state.tab2_chart_sym  = c["代碼"]
                        st.session_state.tab2_chart_data = (c["df"], c["名稱"], c["strat"])

                if st.session_state.get("tab2_chart_sym") == c["代碼"]:
                    t2d = st.session_state.get("tab2_chart_data")
                    if t2d:
                        t2_df, t2_name, t2_strat = t2d
                        t2_last = t2_df.iloc[-1]
                        ta, tb, tc, td = st.columns(4)
                        ta.metric("現價",  f"${t2_df['Close'].iloc[-1]:.2f}")
                        tb.metric("K / D", f"{float(t2_last['K']):.0f} / {float(t2_last['D']):.0f}")
                        tc.metric("RSI",   f"{float(t2_last['RSI']):.1f}")
                        td.metric("量比",  f"{float(t2_last.get('VOL_Ratio',1.0)):.1f}x")
                        st.markdown(
                            f'<div class="sc-action" style="border-left:3px solid {t2_strat["color"]};margin-bottom:10px">'
                            f'{t2_strat["html"]}</div>', unsafe_allow_html=True)
                        if t2_strat.get("warnings"):
                            st.warning("⚠️ " + "；".join(t2_strat["warnings"]))
                        st.plotly_chart(make_tw_chart(t2_df, t2_name, t2_strat),
                                        use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

        if holds:
            st.markdown(f'<div class="qsec">🟡 續抱觀察 ({len(holds)} 檔)</div>', unsafe_allow_html=True)
            for c in holds[:10]:
                st.markdown(f"""
<div class="sk-card">
  <div class="sk-rank">＝</div>
  <div style="flex:1">
    <div class="sk-ticker">{c['代碼']} {c['名稱']}</div>
    <div class="sk-reason">{'、'.join(c['reasons'][:2])}</div>
    {score_bar_html(c['score'])}
  </div>
  <div class="sk-score" style="color:var(--gold)">{c['score']:.1f}</div>
</div>""", unsafe_allow_html=True)

    elif res is None:
        st.info("設定篩選條件後點擊「啟動大盤掃描」。")
    elif len(res) == 0:
        st.warning("無符合條件的標的，請放寬篩選參數。")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Diagnosis (深度診斷 + MTF + K線型態 + 回測)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="qsec">個股深度診斷</div>', unsafe_allow_html=True)

    selection  = st.selectbox("搜尋台股標的", options=["請選擇..."] + STOCK_OPTIONS,
                               label_visibility="collapsed")
    use_mtf    = st.checkbox("啟用多時間框架（MTF）週線確認", value=True)

    if st.button("🚀 執行深度診斷", use_container_width=True) and selection != "請選擇...":
        parts = selection.split(" ")
        code, name = parts[0], parts[1] if len(parts) > 1 else parts[0]
        m_info = MARKET_MAP.get(code)
        with st.spinner(f"分析 {name}({code}) …"):
            df       = fetch_stock_history(code)
            weekly_df = fetch_weekly_history(code) if use_mtf else None
            if df is not None:
                strat = get_strategy_mtf(df, weekly_df, market_info=m_info) if use_mtf else get_strategy(df, market_info=m_info)
                bt    = run_backtest(df)
                st.session_state.diag_plot   = (df, name, strat)
                st.session_state.diag_weekly = weekly_df
                st.session_state.diag_bt     = bt
            else:
                st.error(f"❌ 無法取得 {name}({code}) 資料，請稍後再試。")

    plot_data = st.session_state.get("diag_plot")
    if plot_data:
        p_df, p_name, p_strat = plot_data
        weekly_df = st.session_state.get("diag_weekly")
        bt        = st.session_state.get("diag_bt")
        last_row  = p_df.iloc[-1]

        # ── 信號摘要 ──────────────────────────────────────────────────────────
        sc     = p_strat["score"]
        sc_col = "var(--up)" if sc >= 5 else "var(--gold)" if sc >= 3 else "var(--muted)"

        _mtf_note_html = f'<div style="margin-top:6px;font-size:0.75rem;color:var(--blue)">📡 {p_strat.get("mtf_note","")}</div>' if p_strat.get("mtf_note") else ''
        st.markdown(f"""
<div class="sc" style="margin-bottom:14px;">
  <div class="sc-top">
    <div><div class="sc-name">{p_name}</div><div style="margin-top:3px">{signal_badge_html(p_strat)}</div>{_mtf_note_html}</div>
    <div style="text-align:right"><div style="font-family:var(--mono);font-size:1.8rem;font-weight:700;color:{sc_col}">{sc:.1f}</div><div style="font-size:0.65rem;color:var(--muted)">/ 10 分</div></div>
  </div>{score_bar_html(sc)}
</div>
""", unsafe_allow_html=True)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("現價",     f"${p_df['Close'].iloc[-1]:.2f}")
        k2.metric("RSI(14)",  f"{float(last_row['RSI']):.1f}")
        k3.metric("K / D",    f"{float(last_row['K']):.0f} / {float(last_row['D']):.0f}")
        k4.metric("量比",      f"{float(last_row.get('VOL_Ratio',1.0)):.1f}x")
        k5.metric("停損防線", f"${p_strat['sl']:.2f}" if p_strat.get("sl") else "—")
        k6.metric("目標價",   f"${p_strat['tp']:.2f}" if p_strat.get("tp") else "—")

        st.markdown(f"""
<div class="sc-action" style="border-left:3px solid {p_strat['color']};margin-bottom:12px">
  {p_strat['html']}
</div>""", unsafe_allow_html=True)

        if p_strat.get("warnings"):
            st.warning("⚠️ 風險注意：" + "；".join(p_strat["warnings"]))

        # ── K線型態 ───────────────────────────────────────────────────────────
        patterns = detect_candlestick_patterns(p_df)
        if patterns:
            st.markdown('<div class="qsec">K線型態識別</div>', unsafe_allow_html=True)
            tags = "".join(f'<span class="pattern-tag">{v}</span>' for v in patterns.values())
            st.markdown(f'<div style="margin-bottom:12px">{tags}</div>', unsafe_allow_html=True)

        # ── 主圖 ──────────────────────────────────────────────────────────────
        st.plotly_chart(make_tw_chart(p_df, p_name, p_strat),
                        use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

        # ── MTF 雙框架圖 ───────────────────────────────────────────────────────
        if weekly_df is not None:
            with st.expander("📅 多時間框架對比（日線 vs 週線）", expanded=False):
                st.markdown(
                    f'<div style="font-size:0.78rem;color:var(--muted);margin-bottom:8px">'
                    f'週線訊號：<b style="color:var(--blue)">{p_strat.get("mtf_note","—")}</b>'
                    f'　週線分數：<b>{p_strat.get("weekly_score","—")}</b>/10</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(make_weekly_chart(p_df, weekly_df, p_name),
                                use_container_width=True, config={"displayModeBar": False})

        # ── 回測 ─────────────────────────────────────────────────────────────
        if bt:
            with st.expander("📊 策略回測（近2年）", expanded=False):
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("總報酬",     f"{bt['total_return']:+.1f}%")
                b2.metric("年化報酬",   f"{bt['ann_return']:+.1f}%")
                b3.metric("最大回撤",   f"-{bt['max_drawdown']:.1f}%")
                b4.metric("勝率",       f"{bt['win_rate']:.0f}% ({bt['trade_count']}筆)")
                st.plotly_chart(make_backtest_chart(bt),
                                use_container_width=True, config={"displayModeBar": False})
                st.caption("⚠️ 回測採 KD 黃金/死亡交叉策略，僅供參考，不代表未來績效。")

        if st.button("✖ 清除圖表", use_container_width=True):
            st.session_state.diag_plot   = None
            st.session_state.diag_weekly = None
            st.session_state.diag_bt     = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Portfolio Management + Watchlist
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    port_tab, wl_tab = st.tabs(["📋 庫存管理", "👀 自選股清單"])

    # ── 庫存管理 ─────────────────────────────────────────────────────────────
    with port_tab:
        st.markdown('<div class="qsec">新增持股</div>', unsafe_allow_html=True)
        with st.expander("➕ 新增持股", expanded=True):
            c1, c2, c3 = st.columns(3)
            new_sel    = c1.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS, key="mgmt_sel")
            new_cost   = c2.number_input("買入單價 (TWD)", min_value=0.01, step=0.1, key="mgmt_cost")
            new_shares = c3.number_input("買入股數 (1張=1000)", min_value=1, step=1000, value=1000, key="mgmt_shares")
            if st.button("＋ 暫存至庫存表", use_container_width=True):
                if new_sel != "請選擇...":
                    n_code = new_sel.split(" ")[0]
                    n_name = new_sel.split(" ")[1] if len(new_sel.split(" ")) > 1 else n_code
                    st.session_state.df_portfolio = pd.concat(
                        [st.session_state.df_portfolio,
                         pd.DataFrame([{"Symbol": n_code, "Name": n_name,
                                        "Cost": new_cost, "Shares": new_shares, "Note": ""}])],
                        ignore_index=True,
                    )
                    st.success(f"✅ 已暫存 {n_name}。請點擊「儲存至 Google Sheets」。")
                else:
                    st.warning("請先選擇標的。")

        st.markdown('<div class="qsec">庫存清單編輯器</div>', unsafe_allow_html=True)
        edited_df = st.data_editor(st.session_state.df_portfolio,
                                    hide_index=True, use_container_width=True, key="portfolio_editor")

        c_save, c_reload = st.columns(2)
        if c_save.button("💾 儲存至 Google Sheets", use_container_width=True, type="primary"):
            final_df = edited_df[edited_df["Shares"] > 0].copy()
            with st.spinner("同步至雲端 …"):
                try:
                    gc = get_gsheet_client()
                    ws = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                    ws.clear()
                    # 轉換 numpy 型別避免序列化錯誤
                    rows = [[str(v) if hasattr(v, 'item') else v for v in row]
                            for row in final_df.values.tolist()]
                    ws.update("A1", [final_df.columns.tolist()] + rows)
                    st.session_state.df_portfolio = final_df
                    load_portfolio.clear()  # 只清持倉快取，不清市場報價
                    st.success("🎉 同步成功！")
                    time.sleep(0.8)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 寫入失敗：{e}")

        if c_reload.button("🔄 重新載入庫存", use_container_width=True):
            load_portfolio.clear()
            st.session_state.df_portfolio = load_portfolio()
            st.rerun()

    # ── 自選股清單 ────────────────────────────────────────────────────────────
    with wl_tab:
        st.markdown('<div class="qsec">新增自選股</div>', unsafe_allow_html=True)
        wa, wb, wc = st.columns([2, 1, 1])
        wl_sel  = wa.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS, key="wl_add_sel")
        wl_note = wb.text_input("備註", key="wl_add_note")
        wb.write("")

        if wc.button("➕ 加入自選股", use_container_width=True):
            if wl_sel != "請選擇...":
                wl_code = wl_sel.split(" ")[0]
                wl_name = wl_sel.split(" ")[1] if len(wl_sel.split(" ")) > 1 else wl_code
                ok = add_to_watchlist(wl_code, wl_name, wl_note)
                if ok:
                    st.session_state.df_watchlist = load_watchlist()
                    st.success(f"✅ 已加入 {wl_name}")
                else:
                    st.warning("該標的已在自選股清單中。")
            else:
                st.warning("請先選擇標的。")

        wl_df = st.session_state.df_watchlist
        st.markdown('<div class="qsec">自選股清單</div>', unsafe_allow_html=True)

        if wl_df.empty:
            st.info("自選股清單為空，請新增標的。")
        else:
            for _, wr in wl_df.iterrows():
                m     = MARKET_MAP.get(wr["Symbol"])
                h_df  = fetch_stock_history(wr["Symbol"])
                strat = get_strategy(h_df, market_info=m) if h_df is not None else _default_strat("連線失敗", "#5A6072")
                cp    = m["現價"] if m else 0.0
                patterns = detect_candlestick_patterns(h_df) if h_df is not None else {}
                ptag = "".join(f'<span class="pattern-tag">{v}</span>' for v in list(patterns.values())[:1])

                _wl_ptag = f'<div style="margin-top:3px">{ptag}</div>' if ptag else ''
                st.markdown(f"""
<div class="sk-card">
  <div style="flex:1">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <div><div class="sk-ticker">{wr['Symbol']} {wr['Name']}</div><div style="font-size:0.68rem;color:var(--muted)">{wr.get('Note','')}</div>{_wl_ptag}</div>
      <div style="text-align:right"><div style="font-family:var(--mono);font-size:1.1rem;font-weight:700">${cp:.2f}</div>{signal_badge_html(strat)}<div style="font-size:0.68rem;color:var(--muted);margin-top:2px">分數 {strat['score']:.1f}/10</div></div>
    </div>{score_bar_html(strat['score'])}
  </div>
</div>""", unsafe_allow_html=True)

                col_chart, col_rm = st.columns(2)
                if col_chart.button(f"📊 圖表 {wr['Symbol']}", key=f"wl_chart_{wr['Symbol']}", use_container_width=True):
                    cur = st.session_state.get("wl_chart_sym", "")
                    if cur == wr["Symbol"]:
                        st.session_state.wl_chart_sym  = ""
                        st.session_state.wl_chart_data = None
                    else:
                        st.session_state.wl_chart_sym  = wr["Symbol"]
                        st.session_state.wl_chart_data = (h_df, wr["Name"], strat)

                if col_rm.button(f"🗑️ 移除", key=f"wl_rm_{wr['Symbol']}", use_container_width=True):
                    remove_from_watchlist(wr["Symbol"])
                    st.session_state.df_watchlist = load_watchlist()
                    st.rerun()

                if st.session_state.get("wl_chart_sym") == wr["Symbol"] and h_df is not None:
                    wl_d = st.session_state.get("wl_chart_data")
                    if wl_d:
                        wl_df2, wl_name, wl_strat = wl_d
                        st.plotly_chart(make_tw_chart(wl_df2, wl_name, wl_strat),
                                        use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — System & Settings
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    sys_tab, breadth_tab = st.tabs(["⚙️ 系統控制", "📡 市場廣度"])

    # ── 系統控制 ─────────────────────────────────────────────────────────────
    with sys_tab:
        st.markdown('<div class="qsec">系統控制</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        if col_a.button("🔄 強制刷新全部快取", use_container_width=True):
            st.cache_data.clear()
            for k in ["scan_results","diag_plot","diag_weekly","diag_bt",
                      "tab1_chart_sym","tab1_chart_data","breadth",
                      "df_portfolio","df_watchlist"]:
                st.session_state[k] = None if k not in ("tab1_chart_sym",) else ""
            st.rerun()
        if col_b.button("🗑️ 清除診斷圖表", use_container_width=True):
            st.session_state.diag_plot   = None
            st.session_state.diag_weekly = None
            st.session_state.diag_bt     = None
            st.rerun()

        st.markdown('<div class="qsec">策略說明</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="sc" style="font-size:0.82rem;line-height:1.8">
  <div style="font-weight:900;margin-bottom:8px;font-family:var(--sans)">多因子評分系統 (0–10 分)</div>
  <table width="100%" cellpadding="4">
    <tr><td style="color:var(--muted);width:40%">均線多頭 (MA5/20/60)</td><td>最高 +2.5</td></tr>
    <tr><td style="color:var(--muted)">年線 SMA240</td><td>±1.0</td></tr>
    <tr><td style="color:var(--muted)">KD 低檔黃金交叉</td><td>最高 +2.5</td></tr>
    <tr><td style="color:var(--muted)">RSI 超賣翻揚</td><td>最高 +1.5</td></tr>
    <tr><td style="color:var(--muted)">BB 壓縮放量突破</td><td>+2.5</td></tr>
    <tr><td style="color:var(--muted)">成交量比 ≥2x + OBV 承接</td><td>最高 +1.5</td></tr>
    <tr><td style="color:var(--muted)">MACD 零軸翻多</td><td>最高 +1.5</td></tr>
    <tr><td style="color:var(--muted)">10 日價格動能</td><td>±0.5</td></tr>
    <tr><td style="color:var(--muted)">三大法人買超</td><td>最高 +1.0</td></tr>
    <tr><td style="color:var(--muted)">接近年高/年低</td><td>+0.5</td></tr>
  </table>
  <div style="margin-top:12px;font-weight:700">MTF 週線濾網</div>
  <div style="color:var(--muted);font-size:0.75rem">
    週線分數 &lt; 3 → 日線 BUY 降級為 BUY_WATCH｜週線分數 ≥ 5 → 雙重確認升級
  </div>
  <div style="margin-top:12px;font-weight:700">回測規則</div>
  <div style="color:var(--muted);font-size:0.75rem">
    進場：KD黃金交叉 (K&lt;60) + 站上MA20｜出場：KD死亡交叉 (K&gt;50) 或跌破MA60×97%
  </div>
  <div style="margin-top:10px;color:var(--muted);font-size:0.75rem">
    ⚠️ 本系統僅供輔助參考，不構成投資建議。
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="qsec">資料來源狀態</div>', unsafe_allow_html=True)
        sa, sb = st.columns(2)
        sa.metric("市場報價標的數", len(MARKET_MAP))
        sb.metric("持倉檔數",       len(portfolio))

        # ── 半導體自動掃描 ────────────────────────────────────────────────────
        st.markdown('<hr class="qdiv">', unsafe_allow_html=True)
        st.markdown('<div class="qsec">🤖 半導體族群自動掃描推播</div>', unsafe_allow_html=True)

        _tw_now  = datetime.now(_TW)
        _today_s = _tw_now.strftime("%Y-%m-%d")
        _last_s  = get_last_auto_scan_date()
        _is_today = (_last_s == _today_s)
        _is_tday  = is_tw_trading_day()
        _after18  = _tw_now.hour >= 18

        _semi_kw_n    = sum(1 for v in MARKET_MAP.values() if is_semiconductor(v.get("產業","")))
        _semi_price_n = sum(1 for v in MARKET_MAP.values()
                            if is_semiconductor(v.get("產業","")) and _sf(v.get("現價"),0) > 0)

        st.markdown(f"""
<div class="sc" style="font-size:0.82rem;line-height:2.2;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 12px;margin-bottom:12px;">
    <div><span class="sc-kv-label">關鍵字命中</span><br>
         <span class="sc-kv-value" style="font-size:1.1rem">{_semi_kw_n} 檔</span></div>
    <div><span class="sc-kv-label">有即時報價</span><br>
         <span class="sc-kv-value" style="font-size:1.1rem;color:var(--gold)">{_semi_price_n} 檔</span></div>
    <div><span class="sc-kv-label">今日狀態</span><br>
         <span class="sc-kv-value" style="color:{'var(--down)' if _is_today else 'var(--muted)'}">
           {'✅ 已發送 ('+_last_s+')' if _is_today else '⏳ 尚未執行'}</span></div>
    <div><span class="sc-kv-label">排程時間</span><br>
         <span class="sc-kv-value">每交易日 18:00（台灣）</span></div>
    <div><span class="sc-kv-label">目前台灣時間</span><br>
         <span class="sc-kv-value">{_tw_now.strftime('%H:%M')} {'(交易日)' if _is_tday else '(非交易日)'}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="qsec">手動立即掃描</div>', unsafe_allow_html=True)
        _m1, _m2 = st.columns(2)
        _force   = _m1.checkbox("強制重新掃描（忽略今日已發送記錄）", value=False)
        _dry_run = _m2.checkbox("Dry Run（掃描但不發 TG）",           value=False)

        if st.button("🚀 立即執行半導體掃描推播", use_container_width=True):
            if _is_today and not _force:
                st.warning("今日已發送過推播。勾選「強制重新掃描」可重新執行。")
            else:
                _prog = st.progress(0, text="準備掃描 …")
                _cands, _scan_n = run_semiconductor_scan(MARKET_MAP, _prog)
                _prog.empty()
                _hit_n = len(_cands)
                st.success(f"掃描完成！共 {_scan_n} 檔，{_hit_n} 檔達到技術門檻。")
                _msgs = format_semi_tg_messages(_cands, _scan_n, _hit_n, _today_s)
                with st.expander(f"📨 Telegram 預覽（共 {len(_msgs)} 則）", expanded=True):
                    for _mi, _m in enumerate(_msgs, 1):
                        st.caption(f"第 {_mi} 則（{len(_m)} 字元）")
                        st.code(_m, language=None)
                if not _dry_run:
                    _ok = all(send_tg_message(_m) for _m in _msgs)
                    if _ok:
                        log_auto_scan_result(_scan_n, _hit_n, "manual_ok")
                        st.success("✅ Telegram 推播成功！")
                    else:
                        st.error("❌ 推播失敗，請確認 TG_TOKEN / TG_CHAT_ID。")
                else:
                    st.info("Dry Run 模式：未實際發送。")
                if _cands:
                    st.markdown('<div class="qsec">掃描結果（依分數排序）</div>', unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame([{
                        "代碼": c["代碼"], "名稱": c["名稱"], "產業": c["產業"],
                        "現價": c["現價"], "PE": c["PE"], "PB": c["PB"],
                        "技術分": c["score"], "訊號": c["action"],
                        "停損": round(c["sl"], 2) if c.get("sl") else None,
                        "目標": round(c["tp"], 2) if c.get("tp") else None,
                    } for c in _cands]), use_container_width=True, hide_index=True)

    # ── 市場廣度 ──────────────────────────────────────────────────────────────
    with breadth_tab:
        # ── 大盤週期面板 ──────────────────────────────────────────────────────
        st.markdown('<div class="qsec">📈 加權指數牛熊週期</div>', unsafe_allow_html=True)
        _cyc2 = TAIEX_CYCLE
        if _cyc2:
            _over_avg = _cyc2["days_in_cycle"] > _cyc2["avg_same_days"]
            _pct_of_avg = _cyc2["days_in_cycle"] / max(_cyc2["avg_same_days"], 1) * 100

            _phase_col  = "var(--up)"   if _cyc2["phase"] == 1 else "var(--down)"
            _risk_colour = {"high": "var(--up)" if _cyc2["phase"]==0 else "var(--down)",
                            "medium": "var(--gold)", "safe": "var(--blue)"}[_cyc2["flip_risk"]]
            _sma240_row = (
                f'<div><span class="sc-kv-label">年線SMA240</span><br>'
                f'<span class="sc-kv-value">{_cyc2["sma240"]:,.0f}</span></div>'
            ) if _cyc2.get("sma240") else ""

            _bar_w = min(int(_pct_of_avg), 100)
            _bar_col = "var(--down)" if _over_avg else "var(--up)"

            st.markdown(f"""
<div class="sc" style="margin-bottom:14px;">
  <div class="sc-top">
    <div>
      <div style="font-size:1.3rem;font-weight:900;font-family:var(--sans);color:{_phase_col}">
        {"🔺" if _cyc2["phase"]==1 else "🔻"} {_cyc2["phase_label"]}
      </div>
      <div style="font-size:0.72rem;color:var(--muted);margin-top:3px">
        起始日：{_cyc2["cycle_start"]}　|　加權指數 {_cyc2["close"]:,.0f}
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-family:var(--mono);font-size:2rem;font-weight:700;color:{_phase_col}">{_cyc2["days_in_cycle"]}</div>
      <div style="font-size:0.65rem;color:var(--muted)">天</div>
    </div>
  </div>

  <div style="margin:10px 0 4px;font-size:0.68rem;color:var(--muted)">
    目前進度 vs 歷史同向均值 ({_cyc2["avg_same_days"]} 天)
    — {"⚠️ 已超過歷史均值，週期延長中" if _over_avg else f"歷史均值剩餘估計 {_cyc2['avg_same_days'] - _cyc2['days_in_cycle']} 天"}
  </div>
  <div class="wbar-bg">
    <div class="wbar-fill" style="width:{_bar_w}%;background:{_bar_col}"></div>
  </div>

  <div class="sc-grid" style="margin-top:12px;">
    <div><span class="sc-kv-label">SMA60（翻轉線）</span><br><span class="sc-kv-value">{_cyc2["sma60"]:,.0f}</span></div>
    <div><span class="sc-kv-label">距翻轉線</span><br>
         <span class="sc-kv-value" style="color:{_risk_colour}">{_cyc2["dist_pct"]:+.1f}%</span></div>
    {_sma240_row}
    <div><span class="sc-kv-label">MACD Histogram</span><br>
         <span class="sc-kv-value" style="color:{'var(--up)' if _cyc2['hist']>0 else 'var(--down)'}">
           {_cyc2["hist"]:+.0f}
         </span></div>
    <div><span class="sc-kv-label">52W 高點</span><br><span class="sc-kv-value">{_cyc2["high52w"]:,.0f}</span></div>
    <div><span class="sc-kv-label">52W 低點</span><br><span class="sc-kv-value">{_cyc2["low52w"]:,.0f}</span></div>
  </div>

  <div class="sc-action" style="margin-top:12px;border-left:3px solid {_risk_colour}">
    {_cyc2["flip_msg"]}
  </div>
</div>
""", unsafe_allow_html=True)

            # 翻轉條件說明
            if _cyc2["phase"] == 1:
                st.info("**翻轉為下跌週期條件**：加權指數日收盤跌破 SMA60（"
                        f"{_cyc2['sma60']:,.0f}）並維持 2 日以上，MACD Histogram 同步轉負視為確認。")
            else:
                st.success("**翻轉為上漲週期條件**：加權指數日收盤站回 SMA60（"
                           f"{_cyc2['sma60']:,.0f}）並維持 2 日以上，MACD Histogram 同步轉正視為確認。")
        else:
            st.warning("無法取得加權指數資料（^TWII），請稍後重整頁面。")

        st.markdown('<hr class="qdiv">', unsafe_allow_html=True)
        st.markdown('<div class="qsec">市場廣度（持倉 + 掃描結果）</div>', unsafe_allow_html=True)
        st.caption("廣度計算來源：已快取的持倉標的 + 最近一次掃描結果，不額外觸發 API。")

        if st.button("🔄 計算市場廣度", use_container_width=True):
            with st.spinner("計算中 …"):
                st.session_state.breadth = compute_market_breadth(
                    portfolio, st.session_state.scan_results
                )

        bd = st.session_state.get("breadth")
        if bd:
            ba, bb, bc = st.columns(3)
            ba.metric("漲家數",        f"{bd['advance']} 家")
            bb.metric("跌家數",        f"{bd['decline']} 家")
            bc.metric("漲跌比 A/D",    f"{bd['ad_ratio']:.2f}")

            bd2, be2, bf2 = st.columns(3)
            bd2.metric("站上 MA60 佔比", f"{bd['above_ma60_pct']}%")
            be2.metric("逼近52週高點",   f"{bd['new_high']} 家")
            bf2.metric("逼近52週低點",   f"{bd['new_low']} 家")

            st.plotly_chart(make_breadth_gauge(bd["ad_ratio"]),
                            use_container_width=True, config={"displayModeBar": False})

            ad = bd["ad_ratio"]
            if ad >= 1.5:
                st.success("🟥 市場偏多，強勢格局，可積極佈局。")
            elif ad >= 0.8:
                st.info("🟡 漲跌平衡，選股為主，跟隨強勢族群。")
            else:
                st.warning("🟢 市場偏弱，保守觀望，控制倉位。")
            st.caption(f"統計範圍：{bd['total']} 檔（含持倉 {len(portfolio)} 檔）")
        else:
            st.info("點擊「計算市場廣度」以分析目前持倉與掃描結果的市場強弱。")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 6px;font-size:0.62rem;color:var(--muted);font-family:'JetBrains Mono'">
  台股戰情中心 V16 · {datetime.now().strftime('%Y/%m/%d %H:%M')} · 紅漲綠跌 · 僅供參考勿作投資依據
</div>
""", unsafe_allow_html=True)

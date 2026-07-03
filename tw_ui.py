"""UI 輔助函數與圖表生成（純 Plotly/HTML，無 Streamlit 依賴）"""
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── 文字 / 顏色工具 ──────────────────────────────────────────────────────────

def pl_colour(val: float) -> str:
    return "var(--up)" if val > 0 else "var(--down)" if val < 0 else "var(--muted)"


def signal_badge_html(strat: dict) -> str:
    a = strat["action"]
    if a in ("BUY", "BUY_WATCH"):           return f'<span class="badge badge-up">▲ {strat["name"]}</span>'
    if a in ("SELL_EXIT", "SELL_PARTIAL"):  return f'<span class="badge badge-down">▼ {strat["name"]}</span>'
    if a == "HOLD":                         return f'<span class="badge badge-gold">＝ {strat["name"]}</span>'
    return f'<span class="badge badge-flat">— {strat["name"]}</span>'


def score_bar_html(score: float) -> str:
    pct = min(100, score / 10 * 100)
    col = "var(--up)" if score >= 5 else "var(--gold)" if score >= 3 else "var(--muted)"
    return f'<div class="sbar"><div class="sbar-fill" style="width:{pct:.0f}%;background:{col}"></div></div>'


def accent_colour(strat: dict) -> str:
    a = strat["action"]
    if a in ("BUY", "BUY_WATCH"):            return "var(--up)"
    if a in ("SELL_EXIT", "SELL_PARTIAL"):   return "var(--down)"
    if a == "HOLD":                          return "var(--gold)"
    return "var(--muted)"


def get_tw_session() -> str:
    import pytz
    tw  = pytz.timezone("Asia/Taipei")
    now = datetime.now(tw)
    if now.weekday() >= 5:
        return "休市"
    t = now.time()
    if datetime.strptime("08:30", "%H:%M").time() <= t < datetime.strptime("09:00", "%H:%M").time():
        return "盤前"
    if datetime.strptime("09:00", "%H:%M").time() <= t < datetime.strptime("13:30", "%H:%M").time():
        return "交易中"
    if datetime.strptime("13:30", "%H:%M").time() <= t < datetime.strptime("14:00", "%H:%M").time():
        return "盤後"
    return "休市"


# ── 主圖表：K線 + 量 + KD + MACD ────────────────────────────────────────────

def make_tw_chart(df: pd.DataFrame, name: str, strat: dict) -> go.Figure:
    p   = df.tail(120)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.15, 0.20, 0.20],
        vertical_spacing=0.025,
        subplot_titles=(f"{name} — K線 + BB", "成交量", "KD 隨機指標", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=p.index, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"],
        name="K", increasing_fillcolor="#E8192C", increasing_line_color="#E8192C",
        decreasing_fillcolor="#00B050", decreasing_line_color="#00B050",
    ), row=1, col=1)

    for col_name, colour, dash in [
        ("SMA5", "#F5A623", "solid"), ("SMA20", "#3D8EFF", "dot"), ("SMA60", "#9B6DFF", "dot")
    ]:
        fig.add_trace(go.Scatter(x=p.index, y=p[col_name],
                                 line=dict(color=colour, width=1.2, dash=dash), name=col_name), row=1, col=1)

    fig.add_trace(go.Scatter(x=p.index, y=p["BB_Upper"],
                             line=dict(color="rgba(255,255,255,0.12)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["BB_Lower"], fill="tonexty",
                             fillcolor="rgba(61,142,255,0.04)",
                             line=dict(color="rgba(255,255,255,0.12)", width=1), showlegend=False), row=1, col=1)

    if strat.get("sl"):
        fig.add_hline(y=strat["sl"], line_dash="dot", line_color="#00B050", row=1, col=1,
                      annotation_text=f"SL {strat['sl']:.2f}", annotation_font_color="#00B050")
    if strat.get("tp"):
        fig.add_hline(y=strat["tp"], line_dash="dot", line_color="#E8192C", row=1, col=1,
                      annotation_text=f"TP {strat['tp']:.2f}", annotation_font_color="#E8192C")

    vol_colours = ["#E8192C" if p["Close"].iloc[i] >= p["Open"].iloc[i] else "#00B050" for i in range(len(p))]
    fig.add_trace(go.Bar(x=p.index, y=p["Volume"], marker_color=vol_colours, name="量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["VOL_SMA20"],
                             line=dict(color="#F5A623", width=1.2), name="均量"), row=2, col=1)

    fig.add_trace(go.Scatter(x=p.index, y=p["K"], line=dict(color="#3D8EFF", width=1.5), name="K"), row=3, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["D"], line=dict(color="#F5A623", width=1.5), name="D"), row=3, col=1)
    for lvl, c in [(80, "rgba(232,25,44,0.25)"), (20, "rgba(0,176,80,0.25)")]:
        fig.add_hline(y=lvl, line_color=c, line_dash="dot", row=3, col=1)

    hist_cols = ["#E8192C" if v >= 0 else "#00B050" for v in p["Hist"]]
    fig.add_trace(go.Bar(x=p.index, y=p["Hist"], marker_color=hist_cols, name="OSC"), row=4, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["MACD"],   line=dict(color="#3D8EFF", width=1.2), name="DIF"), row=4, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["Signal"], line=dict(color="#F5A623", width=1.2), name="MACD"), row=4, col=1)

    _apply_dark_layout(fig, height=560)
    return fig


# ── 週線圖（雙時間框架對比）────────────────────────────────────────────────

def make_weekly_chart(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, name: str) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2, shared_xaxes=False,
        subplot_titles=(f"{name} 日線 (近90日)", f"{name} 週線 (近52週)"),
    )
    for col_idx, (df, bars) in enumerate([(daily_df.tail(90), 90), (weekly_df.tail(52), 52)], 1):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_fillcolor="#E8192C", increasing_line_color="#E8192C",
            decreasing_fillcolor="#00B050", decreasing_line_color="#00B050",
            showlegend=False, name=("日線" if col_idx == 1 else "週線"),
        ), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"],
                                 line=dict(color="#3D8EFF", width=1.2), showlegend=False), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA60"],
                                 line=dict(color="#9B6DFF", width=1.2), showlegend=False), row=1, col=col_idx)

    _apply_dark_layout(fig, height=360)
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    return fig


# ── 績效歷史折線圖 ──────────────────────────────────────────────────────────

def make_portfolio_performance_chart(hist_df: pd.DataFrame):
    if hist_df is None or hist_df.empty or len(hist_df) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["Date"], y=hist_df["MarketValue"],
        name="市值", line=dict(color="#E8192C", width=2),
        fill="tozeroy", fillcolor="rgba(232,25,44,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=hist_df["Date"], y=hist_df["Cost"],
        name="成本", line=dict(color="#5A6072", width=1.5, dash="dot"),
    ))
    _apply_dark_layout(fig, height=260)
    fig.update_layout(legend=dict(orientation="h", y=1.1))
    return fig


# ── 回測圖 ──────────────────────────────────────────────────────────────────

def make_backtest_chart(bt: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04,
                        subplot_titles=("資產曲線", "每筆出場損益"))

    fig.add_trace(go.Scatter(
        x=bt["dates"], y=bt["equity"],
        line=dict(color="#E8192C", width=2),
        fill="tozeroy", fillcolor="rgba(232,25,44,0.06)", name="資產",
    ), row=1, col=1)
    fig.add_hline(y=bt["initial"], line_dash="dot", line_color="#5A6072", row=1, col=1,
                  annotation_text="初始資金", annotation_font_color="#5A6072")

    sell_trades = [t for t in bt["trades"] if t["type"] == "SELL"]
    if sell_trades:
        pnl_vals = [t.get("pnl", 0) for t in sell_trades]
        colours  = ["#E8192C" if p >= 0 else "#00B050" for p in pnl_vals]
        fig.add_trace(go.Bar(
            x=[t["date"] for t in sell_trades], y=pnl_vals,
            marker_color=colours, name="損益",
        ), row=2, col=1)

    _apply_dark_layout(fig, height=400)
    return fig


# ── 市場廣度 Gauge ───────────────────────────────────────────────────────────

def make_breadth_gauge(ad_ratio: float) -> go.Figure:
    colour = "#E8192C" if ad_ratio >= 1.5 else "#F5A623" if ad_ratio >= 0.8 else "#00B050"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ad_ratio,
        title={"text": "漲跌比 (A/D Ratio)", "font": {"size": 13, "color": "#5A6072"}},
        gauge={
            "axis": {"range": [0, 3], "tickwidth": 1, "tickcolor": "#5A6072"},
            "bar":  {"color": colour},
            "steps": [
                {"range": [0, 0.8],  "color": "rgba(0,176,80,0.15)"},
                {"range": [0.8, 1.5],"color": "rgba(245,166,35,0.15)"},
                {"range": [1.5, 3],  "color": "rgba(232,25,44,0.15)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "value": 1.0},
        },
        number={"font": {"color": colour, "size": 32}},
    ))
    _apply_dark_layout(fig, height=220)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ── 法人籌碼圖：外資/投信 grouped bar + 三大合計累積折線 ─────────────────────

def make_inst_flow_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    輸入 get_inst_flow() 的結果 DataFrame。
    外資/投信 grouped bar（單位：張＝股/1000）+ 三大合計累積折線（secondary_y）。
    """
    if df is None or df.empty:
        return None

    dates = df["trade_date"].astype(str)
    f_net = (df["foreign_net"] / 1000).round(0)
    t_net = (df["trust_net"]  / 1000).round(0)
    total = (df["total_net"]  / 1000).round(0)
    cum_total = total.cumsum()

    def _bar_colors(series):
        return ["#E8192C" if v >= 0 else "#00B050" for v in series]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=dates, y=f_net, name="外資(張)",
        marker_color=_bar_colors(f_net),
        marker_opacity=0.85, offsetgroup=0,
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=dates, y=t_net, name="投信(張)",
        marker_color=_bar_colors(t_net),
        marker_opacity=0.65, offsetgroup=1,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=dates, y=cum_total, name="三大累積(張)",
        line=dict(color="#F5A623", width=1.8),
        mode="lines",
    ), secondary_y=True)

    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9)),
        font=dict(family="JetBrains Mono", size=9),
    )
    fig.update_yaxes(title_text="買賣超(張)", secondary_y=False,
                     gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="累積(張)", secondary_y=True,
                     gridcolor="rgba(255,255,255,0.02)")
    return fig


# ── 月營收圖：月營收 bar + YoY% 折線 ─────────────────────────────────────────

def make_revenue_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    輸入 get_revenue_history() 的結果 DataFrame。
    月營收 bar（億元）+ YoY% 折線（secondary_y）+ y=0 虛線。
    """
    if df is None or df.empty:
        return None

    x    = df["year_month"].astype(str)
    rev  = df["revenue"] / 1e5  # 千元 → 億元
    yoy  = df["yoy_pct"]

    rev_colors = ["#E8192C" if (yoy.iloc[i] or 0) >= 0 else "#5A6072"
                  for i in range(len(rev))]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=x, y=rev, name="月營收(億)",
        marker_color=rev_colors, marker_opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x, y=yoy, name="YoY%",
        line=dict(color="#3D8EFF", width=2),
        mode="lines+markers",
        marker=dict(size=4),
    ), secondary_y=True)

    # y=0 虛線
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)",
                  secondary_y=True)

    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9)),
        font=dict(family="JetBrains Mono", size=9),
    )
    fig.update_yaxes(title_text="月營收(億元)", secondary_y=False,
                     gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="YoY%", secondary_y=True,
                     gridcolor="rgba(255,255,255,0.02)")
    return fig


# ── 共用 Layout 設定 ─────────────────────────────────────────────────────────

def _apply_dark_layout(fig: go.Figure, height: int = 400):
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, showlegend=False,
        font=dict(family="JetBrains Mono", size=9),
    )
    for attr in ["xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3", "xaxis4", "yaxis4"]:
        fig.update_layout(**{attr: dict(gridcolor="rgba(255,255,255,0.04)")})

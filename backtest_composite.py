"""
backtest_composite.py -- 複合選股（籌碼×營收）參數網格回測

方法論（防前視偏誤的關鍵假設全列在此）：
- 訊號日 T 的可用資訊：法人買賣超到 T 當日（盤後 ~18:00 公布）、
  月營收取「T 時點已公告」的最新月——以法定申報期限保守近似：
  M 月營收自 (M+1) 月 11 日起才視為可用（早公告的公司會被低估時效，可接受）。
- 進場：T+1 交易日「開盤價」（T 的訊號盤後才產生，隔日開盤是最早可執行點）。
- 出場：進場後第 h 個交易日收盤（h = 5/10/20/60），不設停損——目的是評估
  「因子」本身，不是交易系統。
- 觸發去重（onset）：條件在 T 成立且 T-1 不成立才記一次訊號，
  避免連續成立的日子重複計數膨脹樣本。
- 成本：往返 0.585%（買賣手續費各 0.1425% + 證交稅 0.3%，未打折，保守）。
- 基準：0050 同窗口報酬（可投資基準），excess = 個股報酬 - 0050 報酬。
- 流動性防呆：訊號日成交金額 >= 2,000 萬元。

用法：
    python -X utf8 backtest_composite.py                # 全網格
    python -X utf8 backtest_composite.py --min-n 50     # 過濾樣本數
輸出：data/backtest_results.csv + 終端摘要表
"""

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd

from tw_db import get_conn

COST = 0.00585          # 往返成本
LIQ_MIN = 20_000_000    # 訊號日最低成交金額（元）
HORIZONS = [5, 10, 20, 60]
BENCH_ID = "0050"

GRID_N = [3, 5, 10]             # 法人視窗（交易日）
GRID_STREAK = [2, 3, 5]         # 投信連買門檻
GRID_FOREIGN = [True, False]    # 是否要求外資同買
GRID_REV = [                    # (模式, YoY 門檻)
    ("either", 20.0),
    ("yoy", 20.0),
    ("yoy", 30.0),
    ("turnaround", None),
]


def _consecutive_true(b: pd.DataFrame) -> pd.DataFrame:
    """逐欄計算「連續 True 天數」（截至每列當日）。"""
    cs = b.cumsum()
    reset = cs.where(~b).ffill().fillna(0)
    return (cs - reset).astype(int)


def load_data(conn):
    print("loading panels from DB ...")
    px = pd.read_sql_query(
        "SELECT trade_date, stock_id, open, close, turnover FROM daily_price", conn
    )
    inst = pd.read_sql_query(
        "SELECT trade_date, stock_id, foreign_net, trust_net FROM inst_flow", conn
    )
    rev = pd.read_sql_query(
        "SELECT stock_id, year_month, yoy_pct FROM monthly_revenue "
        "WHERE yoy_pct IS NOT NULL", conn
    )
    return px, inst, rev


def build_panels(px, inst, rev):
    dates = sorted(px["trade_date"].unique())
    didx = pd.Index(dates, name="trade_date")

    open_p = px.pivot(index="trade_date", columns="stock_id", values="open").reindex(didx)
    close_p = px.pivot(index="trade_date", columns="stock_id", values="close").reindex(didx)
    tov_p = px.pivot(index="trade_date", columns="stock_id", values="turnover").reindex(didx)

    f_p = inst.pivot(index="trade_date", columns="stock_id", values="foreign_net") \
              .reindex(didx).fillna(0)
    t_p = inst.pivot(index="trade_date", columns="stock_id", values="trust_net") \
              .reindex(didx).fillna(0)

    # 對齊欄位（只回測「價格與法人都有」的股票）
    cols = open_p.columns.intersection(f_p.columns)
    open_p, close_p, tov_p = open_p[cols], close_p[cols], tov_p[cols]
    f_p, t_p = f_p[cols], t_p[cols]

    f_roll = {n: f_p.rolling(n, min_periods=n).sum() for n in GRID_N}
    t_roll = {n: t_p.rolling(n, min_periods=n).sum() for n in GRID_N}
    streak_p = _consecutive_true(t_p > 0)

    # 營收可用時點面板：M 月營收自 (M+1) 月 11 日起可用
    rev = rev.copy()
    ym = pd.to_datetime(rev["year_month"], format="%Y-%m")
    rev["avail"] = (ym + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)).dt.strftime("%Y-%m-%d")
    rev = rev.sort_values(["stock_id", "year_month"])
    rev["prev_yoy"] = rev.groupby("stock_id")["yoy_pct"].shift(1)
    rev["turn"] = ((rev["yoy_pct"] > 0) & (rev["prev_yoy"].fillna(0) <= 0)).astype(float)

    def _step_panel(value_col):
        p = rev.pivot_table(index="avail", columns="stock_id", values=value_col,
                            aggfunc="last")
        # 以「可用日 ∪ 交易日」為索引 ffill，再取交易日切片
        full = p.reindex(p.index.union(didx)).ffill()
        return full.reindex(didx)[cols.intersection(p.columns)] \
                   .reindex(columns=cols)

    yoy_panel = _step_panel("yoy_pct")
    turn_panel = _step_panel("turn")

    return dict(
        dates=didx, cols=cols,
        open=open_p, close=close_p, tov=tov_p,
        f_roll=f_roll, t_roll=t_roll, streak=streak_p,
        yoy=yoy_panel, turn=turn_panel,
    )


def horizon_returns(panels):
    """預先算好各 horizon 的（entry=T+1 開盤 → T+1+h 收盤）報酬與基準報酬。"""
    open_p, close_p = panels["open"], panels["close"]
    entry = open_p.shift(-1)
    rets, bench = {}, {}
    b_open = open_p[BENCH_ID] if BENCH_ID in open_p.columns else None
    for h in HORIZONS:
        exit_c = close_p.shift(-(1 + h))
        rets[h] = exit_c / entry - 1 - COST
        if b_open is not None:
            b_exit = close_p[BENCH_ID].shift(-(1 + h))
            bench[h] = b_exit / b_open.shift(-1) - 1
        else:
            bench[h] = pd.Series(np.nan, index=open_p.index)
    return rets, bench


def run_grid(panels, rets, bench, min_n):
    liq_ok = panels["tov"] >= LIQ_MIN
    rows = []

    def _eval(cond, label_dict):
        onset = cond & ~cond.shift(1, fill_value=False)
        n_onset = int(onset.values.sum())
        for h in HORIZONS:
            r = rets[h].where(onset)
            flat = r.stack().dropna()
            if len(flat) == 0:
                continue
            ex = (rets[h].sub(bench[h], axis=0)).where(onset).stack().dropna()
            rows.append({
                **label_dict, "h": h,
                "n": len(flat),
                "win%": round((flat > 0).mean() * 100, 1),
                "avg%": round(flat.mean() * 100, 2),
                "med%": round(flat.median() * 100, 2),
                "excess%": round(ex.mean() * 100, 2),
                "n_unmeasured": n_onset - len(flat),
            })

    # 基準列：僅流動性條件（市場全體，供對照）
    _eval(liq_ok, {"N": "-", "streak": "-", "foreign": "-", "rev": "baseline_all"})

    for n in GRID_N:
        f_ok = panels["f_roll"][n] > 0
        t_ok = panels["t_roll"][n] > 0
        for smin in GRID_STREAK:
            s_ok = panels["streak"] >= smin
            for freq in GRID_FOREIGN:
                chip = (t_ok & s_ok & f_ok) if freq else (t_ok & s_ok)
                # 籌碼-only 對照
                _eval(chip & liq_ok,
                      {"N": n, "streak": smin, "foreign": freq, "rev": "chip_only"})
                for mode, thr in GRID_REV:
                    if mode == "yoy":
                        r_ok = panels["yoy"] > thr
                        tag = f"yoy>{thr:.0f}"
                    elif mode == "turnaround":
                        r_ok = panels["turn"] > 0
                        tag = "turnaround"
                    else:
                        r_ok = (panels["yoy"] > thr) | (panels["turn"] > 0)
                        tag = f"either>{thr:.0f}"
                    _eval(chip & r_ok & liq_ok,
                          {"N": n, "streak": smin, "foreign": freq, "rev": tag})

    df = pd.DataFrame(rows)
    if min_n:
        df = df[(df["n"] >= min_n) | (df["rev"] == "baseline_all")]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=30,
                    help="摘要表最低樣本數（CSV 仍輸出全部）")
    args = ap.parse_args()

    conn = get_conn()
    px, inst, rev = load_data(conn)
    n_days = px["trade_date"].nunique()
    print(f"price days={n_days}, inst rows={len(inst)}, rev rows={len(rev)}")
    if n_days < 80:
        print("價格資料不足（<80 交易日），請先跑 backfill.py --price-days 730")
        sys.exit(1)

    panels = build_panels(px, inst, rev)
    rets, bench = horizon_returns(panels)
    result = run_grid(panels, rets, bench, min_n=0)

    out = "data/backtest_results.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nfull results -> {out}  ({len(result)} rows)")

    # 終端摘要：20 日 excess 最佳前 15（n >= min-n）
    view = result[(result["h"] == 20) & (result["n"] >= args.min_n)]
    view = view.sort_values("excess%", ascending=False)
    print(f"\n=== 20 日持有、樣本數>={args.min_n}、依超額報酬排序（前 15）===")
    print(view.head(15).to_string(index=False))
    base = result[(result["rev"] == "baseline_all")]
    print("\n=== 市場基準（僅流動性條件）===")
    print(base.to_string(index=False))


if __name__ == "__main__":
    main()

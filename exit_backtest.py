"""
exit_backtest.py -- 出場規則歷史模擬：現行 vs 三層 vs 固定60日

進場樣本：複合訊號 onset（視窗10/連買5/外資開/YoY>20，還原價，流動性>=2000萬），
T+1 開盤進場，成本 0.585% 往返。出場皆「觸發日收盤確認 → 次日開盤成交」。

A 現行邏輯   ：收盤 < SMA60×0.98（get_strategy SELL_EXIT 的實際語義）
B 三層出場   ：任一觸發即出——
               1) 硬停損 = max(進場價×0.92, 進場日收盤−2×ATR14)（固定不動）
               2) 獲利保護 = 獲利曾達 +15% 後啟動：收盤 < 波段峰值收盤×0.90
               3) 趨勢出場 = 收盤 < SMA60×0.98
C 固定 60 日 ：基準對照
A/B 皆設 120 交易日強制出場上限。
"""

import numpy as np
import pandas as pd

from backtest_composite import LIQ_MIN, load_data
from tw_db import get_conn
from tw_dividends import apply_adjustment, get_adjustment_factors

COST = 0.00585
HARD_PCT = 0.92          # 成本 −8%
TRAIL_TRIG = 0.15        # 獲利 +15% 啟動保護
TRAIL_PCT = 0.90         # 峰值回吐 10%
TREND_PCT = 0.98         # SMA60 × 0.98
MAX_HOLD = 120


def _consecutive_true(b):
    cs = b.cumsum()
    return (cs - cs.where(~b).ffill().fillna(0)).astype(int)


def main():
    conn = get_conn()
    px, inst, rev = load_data(conn)
    hl = pd.read_sql_query(
        "SELECT trade_date, stock_id, high, low FROM daily_price", conn)
    factors = get_adjustment_factors(conn)

    dates = sorted(px["trade_date"].unique())
    didx = pd.Index(dates, name="trade_date")

    P = {}
    for col, src in [("open", px), ("close", px), ("turnover", px),
                     ("high", hl), ("low", hl)]:
        P[col] = src.pivot(index="trade_date", columns="stock_id",
                           values=col).reindex(didx)
    for col in ["open", "close", "high", "low"]:
        P[col] = apply_adjustment(P[col], factors)

    cols = P["close"].columns
    close_v, open_v = P["close"].values, P["open"].values
    sma60 = P["close"].rolling(60).mean().values
    prev_c = P["close"].shift(1)
    tr = pd.concat([(P["high"] - P["low"]).stack(),
                    (P["high"] - prev_c).abs().stack(),
                    (P["low"] - prev_c).abs().stack()], axis=1).max(axis=1)
    atr = tr.unstack().reindex(didx)[cols].rolling(14).mean().values

    # 複合 onset（最佳組）
    f = inst.pivot(index="trade_date", columns="stock_id",
                   values="foreign_net").reindex(didx).fillna(0)
    t = inst.pivot(index="trade_date", columns="stock_id",
                   values="trust_net").reindex(didx).fillna(0)
    common = cols.intersection(f.columns)
    f, t = f[common].reindex(columns=cols), t[common].reindex(columns=cols)
    f10, t10 = f.rolling(10).sum(), t.rolling(10).sum()
    streak = _consecutive_true(t > 0)
    rev = rev.copy()
    ym = pd.to_datetime(rev["year_month"], format="%Y-%m")
    rev["avail"] = (ym + pd.offsets.MonthBegin(1)
                    + pd.Timedelta(days=10)).dt.strftime("%Y-%m-%d")
    rev = rev.sort_values(["stock_id", "year_month"])
    yp = rev.pivot_table(index="avail", columns="stock_id",
                         values="yoy_pct", aggfunc="last")
    yoy = yp.reindex(yp.index.union(didx)).ffill().reindex(didx) \
            .reindex(columns=cols)
    cond = ((f10 > 0) & (t10 > 0) & (streak >= 5) & (yoy > 20)
            & (P["turnover"] >= LIQ_MIN))
    onset = (cond & ~cond.shift(1, fill_value=False)).values

    col_ix = {s: i for i, s in enumerate(cols)}
    n_days = len(dates)
    sigs = [(g, ci) for g, ci in zip(*np.nonzero(onset)) if g + 2 < n_days]
    print(f"onset signals: {len(sigs)}")

    def _exit_at(d, ci):
        """觸發日 d 收盤確認 → 次日開盤成交（缺開盤往後找，最多3天）"""
        for k in range(1, 4):
            if d + k < n_days and not np.isnan(open_v[d + k, ci]):
                return open_v[d + k, ci], d + k
        return close_v[d, ci], d

    def simulate(use_hard=False, hard_pct=0.92, use_trail=False,
                 trail_trig=0.15, trail_pct=0.90, use_trend=True):
        out = []
        for g, ci in sigs:
            e = open_v[g + 1, ci]
            ec = close_v[g + 1, ci]
            if np.isnan(e) or e <= 0 or np.isnan(ec):
                continue
            a0 = atr[g + 1, ci]
            hard = (max(e * hard_pct, ec - 2 * a0)
                    if not np.isnan(a0) else e * hard_pct)
            peak = ec
            exit_px, hold = None, None
            last = min(g + 1 + MAX_HOLD, n_days - 1)
            for d in range(g + 2, last + 1):
                c = close_v[d, ci]
                if np.isnan(c):
                    continue
                eff = -np.inf
                if use_trend:
                    s60 = sma60[d, ci]
                    if not np.isnan(s60):
                        eff = max(eff, s60 * TREND_PCT)
                if use_hard:
                    eff = max(eff, hard)
                if use_trail and peak / e - 1 >= trail_trig:
                    eff = max(eff, peak * trail_pct)
                if c < eff:
                    exit_px, xd = _exit_at(d, ci)
                    hold = xd - (g + 1)
                    break
                peak = max(peak, c)
            if exit_px is None:
                exit_px, hold = close_v[last, ci], last - (g + 1)
                if np.isnan(exit_px):
                    continue
            out.append((exit_px / e - 1 - COST, hold))
        return pd.DataFrame(out, columns=["ret", "hold"])

    def fixed60():
        out = []
        for g, ci in sigs:
            e = open_v[g + 1, ci]
            x = g + 61
            if np.isnan(e) or e <= 0 or x >= n_days:
                continue
            c = close_v[x, ci]
            if np.isnan(c):
                continue
            out.append((c / e - 1 - COST, 60))
        return pd.DataFrame(out, columns=["ret", "hold"])

    res = {
        "A 現行(僅季線)":  simulate(),
        "B 原提案三層":    simulate(use_hard=True, use_trail=True),
        "B2 季線+硬停損":  simulate(use_hard=True),
        "B3 季線+寬保護":  simulate(use_trail=True, trail_trig=0.30, trail_pct=0.85),
        "B4 硬10%+寬保護": simulate(use_hard=True, hard_pct=0.90,
                                    use_trail=True, trail_trig=0.30, trail_pct=0.85),
        "C 固定60日":      fixed60(),
    }
    print(f"\n{'方案':<12}{'n':>5}{'平均%':>8}{'中位%':>8}{'勝率%':>7}"
          f"{'P10%':>8}{'最差%':>8}{'均持有日':>9}")
    for k, df in res.items():
        r = df["ret"]
        print(f"{k:<12}{len(df):>5}{r.mean()*100:>8.2f}{r.median()*100:>8.2f}"
              f"{(r>0).mean()*100:>7.1f}{r.quantile(0.1)*100:>8.2f}"
              f"{r.min()*100:>8.2f}{df['hold'].mean():>9.1f}")
    pd.concat({k: v for k, v in res.items()}).to_csv(
        "data/exit_backtest.csv", encoding="utf-8-sig")
    print("\n-> data/exit_backtest.csv")


if __name__ == "__main__":
    main()

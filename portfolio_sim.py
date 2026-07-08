"""
portfolio_sim.py -- 組合層模擬（審計 S4）：複合訊號當一個帳戶跑兩年長什麼樣

規則（貼近散戶實務）：
- 初始資金 100 萬、最多同時 5 檔、每檔目標配置=權益/5、整張成交
- 進場：訊號日收盤確認 → 次日開盤買（槽位有空、現金足 1 張才進；
  多訊號依投信10日買超額排序取前者）
- 出場：收盤破季線−2%（exit_backtest 實測最佳）或持有滿 120 日 → 次日開盤賣
- 成本：買 0.1425%、賣 0.4425%；還原價（含息）
- 變體：A 全訊號照單全收 vs B 牛相位閘門（訊號日大盤收盤>季線才進）
- 基準：0050 買進持有（還原）
"""

import numpy as np
import pandas as pd

import tw_config
from backtest_composite import build_panels, horizon_returns, load_data
from tw_db import get_conn
from tw_dividends import get_adjustment_factors

INIT = 1_000_000
MAX_POS = 5
BUY_C, SELL_C = 0.001425, 0.004425
MAX_HOLD = 120
LIVE = tw_config.COMPOSITE


def main():
    conn = get_conn()
    px, inst, rev = load_data(conn)
    factors = get_adjustment_factors(conn)
    panels = build_panels(px, inst, rev, factors)
    dates = list(panels["open"].index)
    cols = list(panels["close"].columns)
    open_v = panels["open"].values
    close_v = panels["close"].values
    close_ff = panels["close"].ffill().values
    sma60_v = panels["close"].rolling(60).mean().values
    n_days = len(dates)

    N, smin, thr = LIVE["inst_days"], LIVE["min_streak"], LIVE["yoy_thr"]
    cond = ((panels["t_roll"][N] > 0) & (panels["f_roll"][N] > 0)
            & (panels["streak"] >= smin) & (panels["yoy"] > thr)
            & (panels["tov"] >= tw_config.LIQ_MIN))
    onset = (cond & ~cond.shift(1, fill_value=False)).values
    t_rank = panels["t_roll"][N].values

    import yfinance as yf
    tw = yf.Ticker("^TWII").history(period="3y", auto_adjust=True)
    tw.index = tw.index.tz_localize(None).strftime("%Y-%m-%d")
    bull_map = dict((tw["Close"] > tw["Close"].rolling(60).mean()).items())

    def run(gated: bool):
        cash, pos = float(INIT), {}          # sid -> [shares, entry_i, entry_px]
        pend_buy, pend_sell = [], []
        equity, trades = [], []
        for i in range(n_days):
            # 1) 執行昨日排單（今日開盤價）
            for ci in list(pend_sell):
                sh, ei, epx = pos.pop(cols[ci])
                pxo = open_v[i, ci]
                px_exec = pxo if not np.isnan(pxo) else close_ff[i, ci]
                cash += sh * px_exec * (1 - SELL_C)
                trades.append((px_exec / epx - 1 - BUY_C - SELL_C, i - ei))
            pend_sell.clear()
            for ci in pend_buy:
                sid = cols[ci]
                if sid in pos or len(pos) >= MAX_POS:
                    continue
                pxo = open_v[i, ci]
                if np.isnan(pxo) or pxo <= 0:
                    continue
                eq_now = cash + sum(p[0] * close_ff[i, c2]
                                    for sid2, p in pos.items()
                                    for c2 in [cols.index(sid2)])
                lots = int(min(cash, eq_now / MAX_POS) // (pxo * 1000))
                if lots < 1:
                    continue
                sh = lots * 1000
                cost = sh * pxo * (1 + BUY_C)
                if cost > cash:
                    continue
                cash -= cost
                pos[sid] = [sh, i, pxo]
            pend_buy.clear()

            # 2) 今日收盤：出場檢查
            for sid, (sh, ei, epx) in list(pos.items()):
                ci = cols.index(sid)
                c = close_v[i, ci]
                s60 = sma60_v[i, ci]
                if (not np.isnan(c) and not np.isnan(s60)
                        and c < s60 * tw_config.TREND_EXIT_PCT) or i - ei >= MAX_HOLD:
                    pend_sell.append(ci)

            # 3) 今日新訊號 → 排明日買單
            if not (gated and not bull_map.get(dates[i], False)):
                sigs = np.nonzero(onset[i])[0]
                if len(sigs):
                    sigs = sorted(sigs, key=lambda c: -t_rank[i, c])
                    pend_buy.extend(sigs[:MAX_POS])

            equity.append(cash + sum(p[0] * close_ff[i, cols.index(s)]
                                     for s, p in pos.items()))

        eq = pd.Series(equity, index=dates)
        ret = eq.iloc[-1] / INIT - 1
        yrs = len(eq) / 244
        dd = (eq / eq.cummax() - 1).min()
        dr = eq.pct_change().dropna()
        sharpe = dr.mean() / dr.std() * np.sqrt(244) if dr.std() > 0 else 0
        tr = pd.DataFrame(trades, columns=["ret", "hold"])
        return dict(eq=eq, total=ret * 100, cagr=((1 + ret) ** (1 / yrs) - 1) * 100,
                    mdd=dd * 100, sharpe=sharpe, n=len(tr),
                    win=(tr["ret"] > 0).mean() * 100 if len(tr) else 0,
                    hold=tr["hold"].mean() if len(tr) else 0)

    res = {"A 全訊號": run(False), "B 牛相位閘門": run(True)}

    # 0050 買進持有
    b = panels["close"]["0050"].dropna()
    b_ret = b.iloc[-1] / b.iloc[0] - 1
    b_dd = (b / b.cummax() - 1).min() * 100
    b_dr = b.pct_change().dropna()
    b_sh = b_dr.mean() / b_dr.std() * np.sqrt(244)

    print(f"{'方案':<10}{'總報酬%':>9}{'CAGR%':>8}{'MDD%':>8}{'Sharpe':>8}"
          f"{'交易數':>7}{'勝率%':>7}{'均持有':>7}")
    for k, r in res.items():
        print(f"{k:<10}{r['total']:>9.1f}{r['cagr']:>8.1f}{r['mdd']:>8.1f}"
              f"{r['sharpe']:>8.2f}{r['n']:>7}{r['win']:>7.1f}{r['hold']:>7.1f}")
    print(f"{'0050買持':<10}{b_ret*100:>9.1f}{((1+b_ret)**(1/(len(b)/244))-1)*100:>8.1f}"
          f"{b_dd:>8.1f}{b_sh:>8.2f}")

    pd.DataFrame({k: r["eq"] for k, r in res.items()}).to_csv(
        "data/portfolio_sim.csv", encoding="utf-8-sig")
    print("\n-> data/portfolio_sim.csv（權益曲線）")


if __name__ == "__main__":
    main()

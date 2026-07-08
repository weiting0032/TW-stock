"""
walkforward_validation.py -- 複合參數的樣本外嚴謹性檢驗（審計 S1/S2/S3）

S1 選擇偏誤：參數 (10/5/外資/yoy20) 是全樣本網格第一名——本腳本改用
   walk-forward：只用前一年（train）選參數，在後一年（test）驗證；
   並掃描 live 參數的鄰域敏感度（穩健因子應對參數不敏感）。
S2 週期閘門：以 ^TWII 收盤>SMA60 重建歷史牛熊相位（與 app 同規則），
   將 live 參數的訊號按「訊號日相位」分桶，驗證「牛市才執行」紀律。
S3 漲停偏誤：T+1 開盤漲幅 >9.3% 的訊號實務上可能買不到，量化其
   佔比與報酬貢獻（高估程度的上界）。
"""

import numpy as np
import pandas as pd

import tw_config
from backtest_composite import (GRID_FOREIGN, GRID_N, GRID_REV, GRID_STREAK,
                                build_panels, horizon_returns, load_data)
from tw_db import get_conn
from tw_dividends import get_adjustment_factors

SPLIT = "2025-07-01"
LIQ = tw_config.LIQ_MIN
LIVE = (10, 5, True, "yoy", 20.0)


def main():
    conn = get_conn()
    px, inst, rev = load_data(conn)
    factors = get_adjustment_factors(conn)
    panels = build_panels(px, inst, rev, factors)
    rets, bench = horizon_returns(panels)
    liq_ok = panels["tov"] >= LIQ

    def onset_of(n, smin, foreign, mode, thr):
        c = (panels["t_roll"][n] > 0) & (panels["streak"] >= smin)
        if foreign:
            c &= panels["f_roll"][n] > 0
        if mode == "yoy":
            c &= panels["yoy"] > thr
        elif mode == "turnaround":
            c &= panels["turn"] > 0
        else:
            c &= (panels["yoy"] > thr) | (panels["turn"] > 0)
        c &= liq_ok
        return c & ~c.shift(1, fill_value=False)

    def perf(onset, h, lo=None, hi=None):
        o = onset
        if lo:
            o = o.loc[o.index >= lo]
        if hi:
            o = o.loc[o.index < hi]
        r = rets[h].where(o).stack().dropna()
        ex = (rets[h].sub(bench[h], axis=0)).where(o).stack().dropna()
        if len(r) == 0:
            return dict(n=0)
        return dict(n=len(r), win=round((r > 0).mean() * 100, 1),
                    avg=round(r.mean() * 100, 2), ex=round(ex.mean() * 100, 2))

    # ── S1a. walk-forward：train 期選參數 → test 期驗證 ─────────────────────
    combos = []
    for n in GRID_N:
        for smin in GRID_STREAK:
            for fr in GRID_FOREIGN:
                for mode, thr in GRID_REV:
                    combos.append((n, smin, fr, mode, thr or 0))
    rows = []
    for cb in combos:
        o = onset_of(*cb[:3], cb[3], cb[4])
        tr = perf(o, 20, hi=SPLIT)
        te = perf(o, 20, lo=SPLIT)
        if tr["n"] >= 80:
            rows.append({"combo": cb, "tr_n": tr["n"], "tr_ex": tr["ex"],
                         "te_n": te.get("n", 0), "te_ex": te.get("ex")})
    wf = pd.DataFrame(rows).sort_values("tr_ex", ascending=False)
    best = wf.iloc[0]
    live_row = wf[wf["combo"].apply(lambda c: c == LIVE)]
    print("=== S1a Walk-forward（h=20，excess vs 0050）===")
    print(f"train期第一名: {best['combo']}  train_ex={best['tr_ex']}% "
          f"→ test_ex={best['te_ex']}% (n={best['te_n']})")
    if not live_row.empty:
        lr = live_row.iloc[0]
        rank = int((wf["tr_ex"] > lr["tr_ex"]).sum()) + 1
        print(f"live參數 {LIVE}: train_ex={lr['tr_ex']}%（train期第{rank}名/共{len(wf)}）"
              f" → test_ex={lr['te_ex']}% (n={lr['te_n']})")
    top5 = wf.head(5)
    print(f"train前5名的test期平均: {top5['te_ex'].mean():.2f}%"
          f"（穩健性：前段班整體樣本外仍{'為正' if top5['te_ex'].mean() > 0 else '轉負'}）")

    # ── S1b. live 參數鄰域敏感度（全樣本 h=20/60）────────────────────────────
    print("\n=== S1b 鄰域敏感度（live 參數逐項擾動，全期間）===")
    neigh = {
        "live(10/5/外資/yoy20)": LIVE,
        "視窗 5": (5, 5, True, "yoy", 20.0),
        "視窗 3": (3, 5, True, "yoy", 20.0),
        "連買 3": (10, 3, True, "yoy", 20.0),
        "外資關": (10, 5, False, "yoy", 20.0),
        "yoy30": (10, 5, True, "yoy", 30.0),
    }
    for name, cb in neigh.items():
        o = onset_of(*cb[:3], cb[3], cb[4])
        p20, p60 = perf(o, 20), perf(o, 60)
        print(f"{name:<18} h20: n={p20['n']:>4} ex={p20['ex']:>6}%   "
              f"h60: ex={p60['ex']:>6}%")

    # ── S2. 週期閘門條件化（^TWII 收盤>SMA60 = 牛相位，與 app 同規則）────────
    print("\n=== S2 週期閘門驗證（live 參數，依訊號日大盤相位分桶）===")
    try:
        import yfinance as yf
        tw = yf.Ticker("^TWII").history(period="3y", auto_adjust=True)
        tw.index = tw.index.tz_localize(None)
        bull = (tw["Close"] > tw["Close"].rolling(60).mean())
        bull.index = bull.index.strftime("%Y-%m-%d")
        o_live = onset_of(*LIVE[:3], LIVE[3], LIVE[4])
        phase = pd.Series(o_live.index.map(lambda d: bull.get(d)),
                          index=o_live.index)
        for tag, msk in [("牛相位(收>SMA60)", phase == True),
                         ("熊相位(收<SMA60)", phase == False)]:
            o_p = o_live.loc[msk[msk].index] if msk.any() else o_live.iloc[0:0]
            for h in (20, 60):
                p = perf(o_p, h)
                if p["n"]:
                    print(f"{tag} h{h}: n={p['n']:>4} win={p['win']}% "
                          f"avg={p['avg']:>7}% ex={p['ex']:>6}%")
                else:
                    print(f"{tag} h{h}: 無樣本")
    except Exception as e:
        print(f"^TWII 取得失敗（{type(e).__name__}）——S2 略過")

    # ── S3. 漲停不可成交偏誤（live 參數）────────────────────────────────────
    print("\n=== S3 進場日跳空／漲停偏誤（live 參數）===")
    o_live = onset_of(*LIVE[:3], LIVE[3], LIVE[4])
    open_p, close_p = panels["open"], panels["close"]
    gap = open_p.shift(-1) / close_p - 1          # T+1 開盤 vs T 收盤
    r20 = rets[20]
    st_all = r20.where(o_live).stack().dropna()
    lockup = (gap >= 0.093)
    st_lock = r20.where(o_live & lockup).stack().dropna()
    if len(st_all):
        share = len(st_lock) / len(st_all) * 100
        print(f"進場日開盤即漲停(>9.3%)訊號: {len(st_lock)}/{len(st_all)}"
              f"（{share:.1f}%）")
        if len(st_lock):
            print(f"  這些訊號的 20 日均報酬 {st_lock.mean()*100:+.2f}% "
                  f"vs 其餘 {st_all.drop(st_lock.index).mean()*100:+.2f}%")
            adj_avg = st_all.drop(st_lock.index).mean() * 100
            print(f"  若全數買不到 → 修正後全樣本均報酬 {adj_avg:+.2f}%"
                  f"（原 {st_all.mean()*100:+.2f}%）")


if __name__ == "__main__":
    main()

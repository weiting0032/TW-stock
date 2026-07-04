"""
compare_inst_factor.py -- 法人因子新舊版 A/B 歷史回放

同一份樣本（同日同股同指標）各算兩版分數：
- old：market_info={"三大合計": 當日張數}（生產環境原行為，wespai 快照等價）
- new：market_info={"f_net_5d","t_net_5d","trust_streak"}（DB 5 日序列版）
融資警示兩版皆關閉（資料回補中，保持 A/B 乾淨——只比法人邏輯差異）。
評估：20 日 IC、BUY 閘門桶（score>=5 且站上 SMA20）alpha、7 分箱 alpha。
"""

import numpy as np
import pandas as pd

from tw_db import get_conn
from tw_dividends import apply_adjustment, get_adjustment_factors
from tw_indicators import calculate_indicators
from tw_strategy import get_strategy

COST = 0.00585
LIQ_MIN = 20_000_000
STRIDE = 5
WARMUP = 65


def _consecutive_true(b):
    cs = b.cumsum()
    return (cs - cs.where(~b).ffill().fillna(0)).astype(int)


def main():
    conn = get_conn()
    print("loading ...")
    px = pd.read_sql_query(
        "SELECT trade_date, stock_id, open, high, low, close, volume, turnover "
        "FROM daily_price", conn)
    inst = pd.read_sql_query(
        "SELECT trade_date, stock_id, foreign_net, trust_net, total_net "
        "FROM inst_flow", conn)
    factors = get_adjustment_factors(conn)

    dates = sorted(px["trade_date"].unique())
    didx = pd.Index(dates, name="trade_date")
    gpos = {d: i for i, d in enumerate(dates)}

    panels = {}
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        panels[col] = px.pivot(index="trade_date", columns="stock_id",
                               values=col).reindex(didx)
    for col in ["open", "high", "low", "close"]:
        panels[col] = apply_adjustment(panels[col], factors)

    tot_lots = inst.pivot(index="trade_date", columns="stock_id",
                          values="total_net").reindex(didx) / 1000.0
    f_lots = inst.pivot(index="trade_date", columns="stock_id",
                        values="foreign_net").reindex(didx).fillna(0) / 1000.0
    t_lots = inst.pivot(index="trade_date", columns="stock_id",
                        values="trust_net").reindex(didx).fillna(0) / 1000.0
    f5_p, t5_p = f_lots.rolling(5).sum(), t_lots.rolling(5).sum()
    streak_p = _consecutive_true(t_lots > 0)

    sample_set = set(dates[WARMUP::STRIDE])
    stocks = [s for s in panels["close"].columns
              if panels["close"][s].notna().sum() >= 130]
    print(f"days={len(dates)}, stocks={len(stocks)}")

    open_v, close_v = panels["open"].values, panels["close"].values
    col_ix = {s: i for i, s in enumerate(panels["close"].columns)}

    recs = []
    for k, sid in enumerate(stocks):
        if k % 400 == 0:
            print(f"  {k}/{len(stocks)}")
        raw = pd.DataFrame({
            "Open": panels["open"][sid], "High": panels["high"][sid],
            "Low": panels["low"][sid], "Close": panels["close"][sid],
            "Volume": panels["volume"][sid],
        }).dropna(subset=["Close", "High", "Low"])
        if len(raw) < 80:
            continue
        ind = calculate_indicators(raw)
        if ind is None or ind.empty:
            continue
        ci = col_ix[sid]
        in_inst = sid in tot_lots.columns
        for i, d in enumerate(ind.index):
            if d not in sample_set:
                continue
            g = gpos[d]
            tov = panels["turnover"].iat[g, ci]
            if pd.isna(tov) or tov < LIQ_MIN:
                continue
            lots = float(tot_lots[sid].iat[g]) if in_inst and pd.notna(tot_lots[sid].iat[g]) else 0.0
            mi_old = {"三大合計": lots}
            sub = ind.iloc[: i + 1]
            s_old = get_strategy(sub, market_info=mi_old)
            if in_inst and pd.notna(f5_p[sid].iat[g]):
                mi_new = {"f_net_5d": float(f5_p[sid].iat[g]),
                          "t_net_5d": float(t5_p[sid].iat[g]),
                          "trust_streak": int(streak_p[sid].iat[g])}
            else:
                mi_new = mi_old
            s_new = get_strategy(sub, market_info=mi_new)
            row = {"date": d, "stock_id": sid,
                   "score_old": s_old["score"], "score_new": s_new["score"],
                   "above": float(sub["Close"].iat[-1]) >= float(sub["SMA20"].iat[-1])}
            eg = g + 1
            entry = open_v[eg, ci] if eg < len(dates) else np.nan
            for h in (20, 60):
                xg = eg + h
                row[f"fwd{h}"] = (close_v[xg, ci] / entry - 1 - COST
                                  if xg < len(dates) and pd.notna(entry) and entry > 0
                                  and pd.notna(close_v[xg, ci]) else np.nan)
            recs.append(row)

    df = pd.DataFrame(recs)
    print("samples:", len(df))
    for h in (20, 60):
        df[f"alpha{h}"] = df[f"fwd{h}"] - df.groupby("date")[f"fwd{h}"].transform("mean")

    def report(sub, tag):
        print(f"\n===== {tag}（n={len(sub)}）=====")
        for ver in ("old", "new"):
            sc = sub[f"score_{ver}"]
            ics = []
            for d, grp in sub.groupby("date"):
                if grp["fwd20"].notna().sum() >= 100 and grp[f"score_{ver}"].nunique() > 1:
                    ic = grp[f"score_{ver}"].corr(grp["fwd20"], method="spearman")
                    if pd.notna(ic):
                        ics.append(ic)
            ics = pd.Series(ics)
            t = ics.mean() / (ics.std() / np.sqrt(len(ics))) if len(ics) > 5 else float("nan")
            buy = sub[(sc >= 5.0) & sub["above"]]
            top = sub[sc.clip(0, 9.99).astype(int) == 7]
            print(f"[{ver}] IC20={ics.mean():+.4f} t={t:+.2f} 正比率={(ics>0).mean()*100:.0f}% | "
                  f"BUY桶 n={len(buy)} alpha20={buy['alpha20'].mean()*100:+.2f}% "
                  f"alpha60={buy['alpha60'].mean()*100:+.2f}% | "
                  f"7分箱 n={len(top)} alpha20={top['alpha20'].mean()*100:+.2f}%")

    report(df, "全期間")
    report(df[df["date"] >= "2025-07-01"], "2025-07 起")
    df.to_csv("data/inst_factor_ab.csv", index=False, encoding="utf-8-sig")
    print("\nfull -> data/inst_factor_ab.csv")


if __name__ == "__main__":
    main()

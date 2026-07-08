"""
factor_attribution.py -- 10 因子歸因（審計 S4-b）

方法：歷史回放 get_strategy（還原價、DB 法人序列、每 5 日取樣、流動性>=2000萬），
記錄每個樣本觸發的 reasons/warnings 標籤與其 20 日「當日等權 alpha」。
每個標籤的貢獻 = 含該標籤樣本的平均 alpha − 不含的平均 alpha（uplift）。
正向因子 uplift 應為正、警示因子應為負；接近零者為雜訊候選。
"""

from datetime import datetime

import numpy as np
import pandas as pd

from tw_db import get_conn
from tw_dividends import apply_adjustment, get_adjustment_factors
from tw_indicators import calculate_indicators
from tw_strategy import get_strategy

COST = 0.00585
LIQ = 20_000_000
STRIDE, WARMUP, H = 5, 65, 20

TAGS = [  # (前綴, 類型)
    ("均線多頭", "R"), ("5日線向上", "R"), ("站上年線", "R"),
    ("KD低檔黃金交叉", "R"), ("KD低檔翻揚", "R"),
    ("RSI超賣", "R"), ("RSI健康", "R"),
    ("BB壓縮放量突破", "R"), ("大量上漲", "R"), ("OBV籌碼承接", "R"),
    ("MACD零軸翻多", "R"), ("MACD動能增強", "R"),
    ("接近年高", "R"), ("接近年低支撐", "R"), ("10日動能", "R"),
    ("外資投信5日同買", "R"), ("法人5日淨買超", "R"), ("投信連買", "R"),
    ("均線空頭", "W"), ("年線下方", "W"), ("KD高檔超買", "W"),
    ("KD超買+J值過熱", "W"), ("RSI超買", "W"), ("成交量萎縮", "W"),
    ("OBV走弱", "W"), ("MACD動能轉弱", "W"), ("10日走弱", "W"),
    ("外資投信5日同賣", "W"), ("法人5日淨賣超", "W"),
]


def main():
    conn = get_conn()
    px = pd.read_sql_query(
        "SELECT trade_date, stock_id, open, high, low, close, volume, turnover "
        "FROM daily_price", conn)
    inst = pd.read_sql_query(
        "SELECT trade_date, stock_id, foreign_net, trust_net FROM inst_flow", conn)
    factors = get_adjustment_factors(conn)

    dates = sorted(px["trade_date"].unique())
    didx = pd.Index(dates, name="trade_date")
    gpos = {d: i for i, d in enumerate(dates)}
    P = {c: px.pivot(index="trade_date", columns="stock_id", values=c)
            .reindex(didx) for c in ["open", "high", "low", "close",
                                     "volume", "turnover"]}
    for c in ["open", "high", "low", "close"]:
        P[c] = apply_adjustment(P[c], factors)

    f = inst.pivot(index="trade_date", columns="stock_id",
                   values="foreign_net").reindex(didx).fillna(0) / 1000
    t = inst.pivot(index="trade_date", columns="stock_id",
                   values="trust_net").reindex(didx).fillna(0) / 1000
    f5, t5 = f.rolling(5).sum(), t.rolling(5).sum()
    b = t > 0
    cs = b.cumsum()
    stk = (cs - cs.where(~b).ffill().fillna(0)).astype(int)

    sample_set = set(dates[WARMUP::STRIDE])
    stocks = [s for s in P["close"].columns if P["close"][s].notna().sum() >= 130]
    open_v, close_v = P["open"].values, P["close"].values
    cix = {s: i for i, s in enumerate(P["close"].columns)}

    recs = []
    t0 = datetime.now()
    for k, sid in enumerate(stocks):
        if k % 400 == 0:
            print(f"  {k}/{len(stocks)} ({(datetime.now()-t0).seconds}s)")
        raw = pd.DataFrame({"Open": P["open"][sid], "High": P["high"][sid],
                            "Low": P["low"][sid], "Close": P["close"][sid],
                            "Volume": P["volume"][sid]}
                           ).dropna(subset=["Close", "High", "Low"])
        if len(raw) < 80:
            continue
        ind = calculate_indicators(raw)
        if ind is None or ind.empty:
            continue
        ci = cix[sid]
        has_inst = sid in f5.columns
        for i, d in enumerate(ind.index):
            if d not in sample_set:
                continue
            g = gpos[d]
            tov = P["turnover"].iat[g, ci]
            if pd.isna(tov) or tov < LIQ:
                continue
            mi = ({"f_net_5d": float(f5[sid].iat[g]),
                   "t_net_5d": float(t5[sid].iat[g]),
                   "trust_streak": int(stk[sid].iat[g])}
                  if has_inst and pd.notna(f5[sid].iat[g]) else None)
            s = get_strategy(ind.iloc[:i + 1], market_info=mi)
            eg = g + 1
            xg = eg + H
            if xg >= len(dates):
                continue
            e = open_v[eg, ci]
            x = close_v[xg, ci]
            if pd.isna(e) or e <= 0 or pd.isna(x):
                continue
            recs.append({"date": d,
                         "tags": tuple(s["reasons"]) + tuple(
                             "W:" + w for w in s["warnings"]),
                         "fwd": x / e - 1 - COST})
    df = pd.DataFrame(recs)
    print("samples:", len(df))
    df["alpha"] = df["fwd"] - df.groupby("date")["fwd"].transform("mean")

    rows = []
    for prefix, kind in TAGS:
        pat = ("W:" + prefix) if kind == "W" else prefix
        m = df["tags"].apply(lambda ts: any(x.startswith(pat) for x in ts))
        n_w = int(m.sum())
        if n_w < 200:
            continue
        up = (df.loc[m, "alpha"].mean() - df.loc[~m, "alpha"].mean()) * 100
        rows.append({"因子": prefix, "型": "警示" if kind == "W" else "加分",
                     "n": n_w, "佔比%": round(n_w / len(df) * 100, 1),
                     "uplift(20日alpha,pp)": round(up, 2)})
    out = pd.DataFrame(rows).sort_values("uplift(20日alpha,pp)", ascending=False)
    print(out.to_string(index=False))
    out.to_csv("data/factor_attribution.csv", index=False, encoding="utf-8-sig")
    print("\n-> data/factor_attribution.csv")


if __name__ == "__main__":
    main()

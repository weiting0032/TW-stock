"""
score_v2_validation.py -- Score v2 重定權：train 期定權、test 期對決（預註冊規則）

規則（跑之前已釘死，見 2026-07-10 對話）：
1. 只用 train（<2025-07-01）的因子 uplift 修權重：
   「uplift 符號與權重符號矛盾、|uplift|>=0.5pp、n>=500」→ 權重歸零（不翻號）；
   加強僅允許 hi52（train uplift>=1.5pp → 0.5→1.0）。
2. 採用判準（test 期）：v2 的 IC20 與 BUY桶 alpha20 皆 >= v1，且分箱單調性不劣化。
3. 法人/融資因子（已獨立 A/B 驗證）不在本次範圍。

v2 分數以「觸發分支 × 新權重」算術重建——分支選擇只依條件不依權重，
與真跑 get_strategy(weights=V2) 完全等價（並抽樣驗證）。
"""

from datetime import datetime

import numpy as np
import pandas as pd

import tw_config
from tw_db import get_conn
from tw_dividends import apply_adjustment, get_adjustment_factors
from tw_indicators import calculate_indicators
from tw_strategy import get_strategy

COST, LIQ, STRIDE, WARMUP = 0.00585, 20_000_000, 5, 65
SPLIT = "2025-07-01"
V1 = tw_config.SCORE_WEIGHTS_V1

KEY_TAG = [  # (權重鍵, 標籤前綴, 是否警示)
    ("trend", "均線多頭", 0), ("trend_5d", "5日線向上", 0),
    ("trend_pen", "均線空頭", 1),
    ("annual", "站上年線", 0), ("annual_pen", "年線下方", 1),
    ("kd_golden", "KD低檔黃金交叉", 0), ("kd_turn", "KD低檔翻揚", 0),
    ("kd_ob", "KD高檔超買", 1), ("kdj_hot", "KD超買+J值過熱", 1),
    ("rsi_os", "RSI超賣", 0), ("rsi_mid", "RSI健康", 0), ("rsi_ob", "RSI超買", 1),
    ("bb_break", "BB壓縮放量突破", 0), ("vol_surge", "大量上漲", 0),
    ("obv", "OBV籌碼承接", 0),
    ("macd_zero", "MACD零軸翻多", 0), ("macd_up", "MACD動能增強", 0),
    ("macd_down", "MACD動能轉弱", 1),
    ("hi52", "接近年高", 0), ("lo52", "接近年低支撐", 0),
    ("mom10", "10日動能", 0), ("mom10_pen", "10日走弱", 1),
]
INST_TAGS = ["外資投信5日同買", "法人5日淨買超", "投信連買",
             "外資投信5日同賣", "法人5日淨賣超"]


def inst_score(f5, t5, streak):
    s = 0.0
    if f5 > 0 and t5 > 0:
        s += 1.0
    elif f5 + t5 > 500:
        s += 0.5
    elif f5 < 0 and t5 < 0:
        s -= 1.0
    elif f5 + t5 < -500:
        s -= 0.5
    if streak >= 3:
        s += 0.5
    return s


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
    f5p, t5p = f.rolling(5).sum(), t.rolling(5).sum()
    b = t > 0
    cs = b.cumsum()
    stkp = (cs - cs.where(~b).ffill().fillna(0)).astype(int)

    sample_set = set(dates[WARMUP::STRIDE])
    stocks = [s for s in P["close"].columns if P["close"][s].notna().sum() >= 130]
    open_v, close_v = P["open"].values, P["close"].values
    cix = {s: i for i, s in enumerate(P["close"].columns)}

    recs, t0 = [], datetime.now()
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
        has_i = sid in f5p.columns
        for i, d in enumerate(ind.index):
            if d not in sample_set or i + 1 < 20:   # <20 根 get_strategy 回「資料不足」
                continue
            g = gpos[d]
            tov = P["turnover"].iat[g, ci]
            if pd.isna(tov) or tov < LIQ:
                continue
            if has_i and pd.notna(f5p[sid].iat[g]):
                fv, tv, sv = (float(f5p[sid].iat[g]), float(t5p[sid].iat[g]),
                              int(stkp[sid].iat[g]))
                mi = {"f_net_5d": fv, "t_net_5d": tv, "trust_streak": sv}
            else:
                fv = tv = sv = 0.0
                mi = None
            sub = ind.iloc[:i + 1]
            s1 = get_strategy(sub, market_info=mi, weights=V1)
            tags = set(s1["reasons"]) | {"W:" + w for w in s1["warnings"]}
            fired = [key for key, pre, isw in KEY_TAG
                     if any(x.startswith(("W:" + pre) if isw else pre)
                            for x in tags)]
            eg, x20, x60 = g + 1, g + 21, g + 61
            e = open_v[eg, ci] if eg < len(dates) else np.nan
            recs.append({
                "date": d, "fired": tuple(fired),
                "inst": inst_score(fv, tv, sv) if mi else 0.0,
                "score1": s1["score"],
                "above": float(sub["Close"].iat[-1]) >= float(sub["SMA20"].iat[-1]),
                "fwd20": (close_v[x20, ci] / e - 1 - COST
                          if x20 < len(dates) and pd.notna(e) and e > 0
                          and pd.notna(close_v[x20, ci]) else np.nan),
                "fwd60": (close_v[x60, ci] / e - 1 - COST
                          if x60 < len(dates) and pd.notna(e) and e > 0
                          and pd.notna(close_v[x60, ci]) else np.nan),
            })
    df = pd.DataFrame(recs)
    print("samples:", len(df))
    for h in (20, 60):
        df[f"a{h}"] = df[f"fwd{h}"] - df.groupby("date")[f"fwd{h}"].transform("mean")

    # 重建 sanity：v1 權重算術分數 == get_strategy 分數
    df["score1_re"] = df.apply(
        lambda r: max(0, min(10, sum(V1[k] for k in r["fired"]) + r["inst"])), axis=1)
    bad = (df["score1_re"] - df["score1"]).abs().max()
    print(f"重建 sanity: max|diff|={bad:.6f}（應為 0）")
    assert bad < 1e-9

    # ── train 期定權（預註冊規則）────────────────────────────────────────────
    tr = df[df["date"] < SPLIT]
    V2 = dict(V1)
    changes = []
    for key, pre, isw in KEY_TAG:
        m = tr["fired"].apply(lambda fs: key in fs)
        n = int(m.sum())
        if n < 500:
            continue
        up = (tr.loc[m, "a20"].mean() - tr.loc[~m, "a20"].mean()) * 100
        w = V1[key]
        if w > 0 and up <= -0.5:
            V2[key] = 0.0
            changes.append(f"{key}({pre}): +{w} → 0  [train uplift {up:+.2f}pp, n={n}]")
        elif w < 0 and up >= 0.5:
            V2[key] = 0.0
            changes.append(f"{key}({pre}): {w} → 0  [train uplift {up:+.2f}pp, n={n}]")
        elif key == "hi52" and up >= 1.5:
            V2[key] = 1.0
            changes.append(f"hi52(接近年高): +0.5 → +1.0  [train uplift {up:+.2f}pp, n={n}]")
    print("\n=== 預註冊規則產生的 v2 修訂 ===")
    for c in changes:
        print(" ", c)

    df["score2"] = df.apply(
        lambda r: max(0, min(10, sum(V2[k] for k in r["fired"]) + r["inst"])), axis=1)

    # ── 指標對決 ────────────────────────────────────────────────────────────
    def metrics(sub, col):
        ics = []
        for d, g in sub.groupby("date"):
            if g["fwd20"].notna().sum() >= 100 and g[col].nunique() > 1:
                ic = g[col].corr(g["fwd20"], method="spearman")
                if pd.notna(ic):
                    ics.append(ic)
        ics = pd.Series(ics)
        t = ics.mean() / (ics.std() / np.sqrt(len(ics))) if len(ics) > 5 else np.nan
        buy = sub[(sub[col] >= 5.0) & sub["above"]]
        return (f"IC={ics.mean():+.4f}(t={t:+.2f}) | BUY桶 n={len(buy)} "
                f"a20={buy['a20'].mean()*100:+.2f}% a60={buy['a60'].mean()*100:+.2f}%")

    for tag, sub in [("TRAIN", df[df["date"] < SPLIT]),
                     ("TEST ", df[df["date"] >= SPLIT])]:
        print(f"\n=== {tag} ===")
        print("v1:", metrics(sub, "score1"))
        print("v2:", metrics(sub, "score2"))

    te = df[df["date"] >= SPLIT]
    print("\n=== TEST 期分箱 alpha20%（單調性檢查）===")
    for col in ("score1", "score2"):
        g = te.groupby(te[col].clip(0, 9.99).astype(int))["a20"].agg(["mean", "size"])
        print(col, {i: f"{v*100:+.2f}({n})" for i, (v, n) in
                    g[["mean", "size"]].iterrows()})

    print("\nV2 =", {k: v for k, v in V2.items() if v != V1[k]})
    df.to_csv("data/score_v2_validation.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

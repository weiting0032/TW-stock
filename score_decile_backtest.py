"""
score_decile_backtest.py -- 10 因子評分（get_strategy）歷史預測力驗證

方法：
- 還原價（除權息+拆分）重建每檔 OHLCV → calculate_indicators →
  每 5 個交易日取樣，對每檔呼叫「生產環境同一份」get_strategy 取得分數。
- 忠實度：因子 1–9 完整重現；因子 10 法人買超用 inst_flow 真實當日張數
  （比生產的 wespai 快照更準）；融資率歷史無資料 → 以 0 中性化（少了融資
  警示扣分，對高融資股評分略偏高，已揭露）。SMA240 在 2025-06 前為部分均值
  （min_periods=60，與資料起點有關）→ 另列 2025-07 起子期間驗證。
- 前視防護：分數用 T 收盤後資訊 → T+1 開盤進場 → T+1+h 收盤出場，
  成本 0.585%，流動性 門檻 2,000 萬。
- 評估：分數整數分箱（0..9）之遠期報酬單調性；BUY 規則桶（score>=5 且
  收盤>=SMA20，即 app 的進場閘門）vs 其他；每日橫斷面 Spearman IC 與 t 值；
  「vs 當日等權」= 減去當日全體受評股平均（純選股 alpha，不受大盤方向影響）。

輸出：data/score_decile_results.csv + 終端摘要。
"""

import sys
from datetime import datetime

import numpy as np
import pandas as pd

from tw_db import get_conn
from tw_dividends import apply_adjustment, get_adjustment_factors
from tw_indicators import calculate_indicators
from tw_strategy import get_strategy

COST = 0.00585
LIQ_MIN = 20_000_000
HORIZONS = [5, 20, 60]
STRIDE = 5          # 每 5 個交易日取樣一次
WARMUP = 65         # 全域日index 起算前保留（指標暖機）


def main():
    conn = get_conn()
    print("loading ...")
    px = pd.read_sql_query(
        "SELECT trade_date, stock_id, open, high, low, close, volume, turnover "
        "FROM daily_price", conn)
    inst = pd.read_sql_query(
        "SELECT trade_date, stock_id, total_net FROM inst_flow", conn)
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

    net_lots = inst.pivot(index="trade_date", columns="stock_id",
                          values="total_net").reindex(didx) / 1000.0

    sample_dates = dates[WARMUP::STRIDE]
    sample_set = set(sample_dates)
    stocks = [s for s in panels["close"].columns
              if panels["close"][s].notna().sum() >= 130]
    print(f"days={len(dates)}, stocks={len(stocks)}, "
          f"sample_dates={len(sample_dates)}, adj_events={len(factors)}")

    open_v, close_v = panels["open"].values, panels["close"].values
    col_ix = {s: i for i, s in enumerate(panels["close"].columns)}

    records = []
    t0 = datetime.now()
    for k, sid in enumerate(stocks):
        if k % 300 == 0:
            print(f"  scoring {k}/{len(stocks)} ({(datetime.now()-t0).seconds}s)")
        raw = pd.DataFrame({
            "Open":   panels["open"][sid],  "High": panels["high"][sid],
            "Low":    panels["low"][sid],   "Close": panels["close"][sid],
            "Volume": panels["volume"][sid],
        }).dropna(subset=["Close", "High", "Low"])
        if len(raw) < 80:
            continue
        ind = calculate_indicators(raw)
        if ind is None or ind.empty:
            continue
        ci = col_ix[sid]
        lots_s = net_lots[sid] if sid in net_lots.columns else None
        idx_list = list(ind.index)
        for i, d in enumerate(idx_list):
            if d not in sample_set:
                continue
            g = gpos[d]
            tov = panels["turnover"].iat[g, ci]
            if pd.isna(tov) or tov < LIQ_MIN:
                continue
            lots = 0.0
            if lots_s is not None:
                v = lots_s.iat[g]
                lots = 0.0 if pd.isna(v) else float(v)
            strat = get_strategy(ind.iloc[: i + 1],
                                 market_info={"三大合計": lots, "融資率": 0.0})
            row = {"date": d, "stock_id": sid, "score": strat["score"],
                   "action": strat["action"],
                   "above_ma20": float(ind["Close"].iat[i]) >= float(ind["SMA20"].iat[i])}
            entry_g = g + 1
            if entry_g < len(dates):
                entry = open_v[entry_g, ci]
                for h in HORIZONS:
                    xg = entry_g + h
                    if xg < len(dates) and not pd.isna(entry) and entry > 0:
                        ex = close_v[xg, ci]
                        row[f"fwd{h}"] = (ex / entry - 1 - COST) if not pd.isna(ex) else np.nan
                    else:
                        row[f"fwd{h}"] = np.nan
            records.append(row)

    df = pd.DataFrame(records)
    print(f"scored samples: {len(df)}  ({(datetime.now()-t0).seconds}s)")
    if df.empty:
        sys.exit(1)

    # 當日等權 alpha
    for h in HORIZONS:
        df[f"alpha{h}"] = df[f"fwd{h}"] - df.groupby("date")[f"fwd{h}"].transform("mean")

    df["bin"] = df["score"].clip(0, 9.99).astype(int)
    df["is_buy"] = (df["score"] >= 5.0) & df["above_ma20"]
    df.to_csv("data/score_decile_results.csv", index=False, encoding="utf-8-sig")

    def summarize(sub, label):
        print(f"\n=== {label}（樣本 {len(sub)}）===")
        g = sub.groupby("bin").agg(
            n=("score", "size"),
            win20=("fwd20", lambda x: (x > 0).mean() * 100),
            fwd20=("fwd20", "mean"), alpha20=("alpha20", "mean"),
            fwd60=("fwd60", "mean"), alpha60=("alpha60", "mean"),
        )
        for c in ["win20"]:
            g[c] = g[c].round(1)
        for c in ["fwd20", "alpha20", "fwd60", "alpha60"]:
            g[c] = (g[c] * 100).round(2)
        print(g.to_string())
        buy = sub[sub["is_buy"]]
        rest = sub[~sub["is_buy"]]
        for h in [20, 60]:
            b, r = buy[f"fwd{h}"].mean() * 100, rest[f"fwd{h}"].mean() * 100
            ba = buy[f"alpha{h}"].mean() * 100
            print(f"BUY規則桶 h={h}: n={buy[f'fwd{h}'].notna().sum()} "
                  f"avg={b:+.2f}% vs其他={b - r:+.2f}% 當日等權alpha={ba:+.2f}%")
        # IC
        ics = []
        for d, grp in sub.groupby("date"):
            if grp["fwd20"].notna().sum() >= 100:
                ic = grp["score"].corr(grp["fwd20"], method="spearman")
                if not pd.isna(ic):
                    ics.append(ic)
        ics = pd.Series(ics)
        if len(ics) > 5:
            t = ics.mean() / (ics.std() / np.sqrt(len(ics)))
            print(f"IC(20日): mean={ics.mean():+.4f}  正IC比率={(ics > 0).mean() * 100:.0f}%  "
                  f"t={t:+.2f}  (n={len(ics)} 期)")

    summarize(df, "全期間 2024-10 起")
    summarize(df[df["date"] >= "2025-07-01"], "子期間 2025-07 起（SMA240 完整）")


if __name__ == "__main__":
    main()

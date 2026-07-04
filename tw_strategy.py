"""策略引擎、多時間框架確認（MTF）、回測（純計算，無 Streamlit 依賴）"""
import math
import pandas as pd


def _default_strat(name: str, color: str) -> dict:
    return {
        "action": "WATCH", "name": name, "color": color, "score": 0,
        "tp": None, "sl": None, "suggest_lots": 0,
        "reasons": [], "warnings": [],
        "html": f"<span style='color:{color}'>{name}</span>",
    }


def get_strategy(df: pd.DataFrame, held_shares: float = 0, held_cost: float = 0,
                 market_info: dict = None) -> dict:
    """多因子評分策略 (0–10分)"""
    if df is None or df.empty or len(df) < 20:
        return _default_strat("資料不足", "#5A6072")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    close  = float(last["Close"])
    sma5   = float(last["SMA5"])
    sma20  = float(last["SMA20"])
    sma60  = float(last["SMA60"])
    _r240  = last.get("SMA240")
    sma240 = 0.0 if _r240 is None or pd.isna(_r240) else float(_r240)
    atr    = float(last.get("ATR", 0) or 0)
    rsi    = float(last["RSI"])
    k, d   = float(last["K"]), float(last["D"])
    j      = float(last["J"])
    macd_h = float(last["Hist"])
    bb_w   = float(last["BB_Width"])
    vol_r  = float(last.get("VOL_Ratio", 1.0))
    _h52   = last.get("High52W")
    _l52   = last.get("Low52W")
    high52 = close if _h52 is None or pd.isna(_h52) else float(_h52)
    low52  = close if _l52 is None or pd.isna(_l52) else float(_l52)

    # Market info（三大法人、融資）
    inst_net    = float(market_info.get("三大合計", 0) or 0) if market_info else 0.0
    margin_rate = float(market_info.get("融資率",  0) or 0) if market_info else 0.0

    stop_loss   = max(close - 2.0 * atr, sma60 * 0.98, close * 0.88) if atr else close * 0.88
    take_profit = min(high52, close + 3.0 * atr) if atr and high52 > close else (close + 3.0 * atr if atr else close * 1.12)
    # ATR-based position sizing（每筆風險上限 30,000 TWD）
    _sl_dist  = max(close - stop_loss, close * 0.05)
    base_lots = max(1, min(5, int(30_000 / (_sl_dist * 1000))))

    score, reasons, warnings = 0.0, [], []

    # 1. Trend
    if close > sma20 > sma60:
        score += 2.0; reasons.append("均線多頭")
        if sma5 > sma20:
            score += 0.5; reasons.append("5日線向上")
    elif close < sma20 < sma60:
        score -= 1.0; warnings.append("均線空頭")

    # 2. Annual line
    if sma240 > 0:
        if close > sma240:
            score += 1.0; reasons.append("站上年線")
        else:
            score -= 0.5; warnings.append("年線下方")

    # 3. KD
    k_prev = float(prev["K"]); d_prev = float(prev["D"])
    if k < 20 and d < 20 and k > d and k_prev < d_prev:
        score += 2.5; reasons.append("KD低檔黃金交叉")
    elif k < 30 and k > d:
        score += 1.5; reasons.append("KD低檔翻揚")
    elif k > 80 and d > 80:
        score -= 1.0; warnings.append("KD高檔超買")
    if k > 85 and j > 90:
        score -= 1.5; warnings.append("KD超買+J值過熱")

    # 4. RSI
    if rsi < 30:
        score += 1.5; reasons.append(f"RSI超賣({rsi:.0f})")
    elif 40 <= rsi <= 65:
        score += 0.5; reasons.append("RSI健康")
    elif rsi > 80:
        score -= 1.0; warnings.append(f"RSI超買({rsi:.0f})")

    # 5. BB breakout
    prev_bb_w = float(prev.get("BB_Width", 1.0))
    if close > float(last["BB_Upper"]) and vol_r > 1.5 and prev_bb_w < 0.08:
        score += 2.5; reasons.append("BB壓縮放量突破")
    elif bb_w < 0.08:
        reasons.append("BB醞釀蓄力")

    # 6. Volume
    if vol_r >= 2.0 and close >= sma20:
        score += 1.0; reasons.append(f"大量上漲({vol_r:.1f}倍)")
    elif vol_r < 0.5:
        warnings.append("成交量萎縮")

    # 6b. OBV trend（籌碼承接確認）
    if len(df) >= 6:
        obv_5d  = float(df["OBV"].iloc[-6])
        obv_now = float(last["OBV"])
        if obv_now > obv_5d and close > sma20:
            score += 0.5; reasons.append("OBV籌碼承接")
        elif obv_now < obv_5d and close > sma20:
            warnings.append("OBV走弱，留意出貨")

    # 7. MACD
    prev_hist = float(prev["Hist"])
    if prev_hist < 0 <= macd_h:              # 零軸翻多（最強訊號）
        score += 1.5; reasons.append("MACD零軸翻多")
    elif macd_h > 0 and macd_h > prev_hist:  # 正區間動能增強
        score += 0.5; reasons.append("MACD動能增強")
    elif macd_h < 0 and macd_h < prev_hist:  # 負區間繼續惡化
        score -= 0.5; warnings.append("MACD動能轉弱")

    # 8. 52W position
    if close >= high52 * 0.95:
        score += 0.5; reasons.append("接近年高")
    if (close - low52) / max(high52 - low52, 1e-9) < 0.15:
        score += 0.5; reasons.append("接近年低支撐")

    # 9. 10-day price momentum
    if len(df) >= 12:
        mom10 = (close / float(df["Close"].iloc[-11]) - 1) * 100
        if mom10 > 5.0:
            score += 0.5; reasons.append(f"10日動能({mom10:.1f}%)")
        elif mom10 < -8.0:
            score -= 0.5; warnings.append(f"10日走弱({mom10:.1f}%)")

    # 10. Institutional flow（三大法人 + 融資警示）
    if market_info:
        f5 = market_info.get("f_net_5d")   # 外資近5日淨買超（張，來自本地 DB）
        t5 = market_info.get("t_net_5d")   # 投信近5日淨買超（張）
        if f5 is not None and t5 is not None:
            # 新版（2026-07）：DB 真實 5 日序列取代 wespai 當日快照，
            # 邏輯與複合選股同源（外資投信同買 + 投信連買，均經回測驗證）
            f5, t5 = float(f5), float(t5)
            streak5 = int(market_info.get("trust_streak", 0) or 0)
            if f5 > 0 and t5 > 0:
                score += 1.0; reasons.append("外資投信5日同買")
            elif f5 + t5 > 500:
                score += 0.5; reasons.append(f"法人5日淨買超{int(f5 + t5)}張")
            elif f5 < 0 and t5 < 0:
                score -= 1.0; warnings.append("外資投信5日同賣")
            elif f5 + t5 < -500:
                score -= 0.5; warnings.append(f"法人5日淨賣超{int(abs(f5 + t5))}張")
            if streak5 >= 3:
                score += 0.5; reasons.append(f"投信連買{streak5}日")
        else:
            # 舊版 fallback：無 DB 序列時維持原行為（wespai 當日快照）
            if inst_net > 500:
                score += 1.0; reasons.append(f"法人大幅買超{int(inst_net)}張")
            elif inst_net > 100:
                score += 0.5; reasons.append(f"法人買超{int(inst_net)}張")
            elif inst_net < -500:
                score -= 1.0; warnings.append(f"法人大幅賣超{int(abs(inst_net))}張")
            elif inst_net < -100:
                score -= 0.5; warnings.append(f"法人賣超{int(abs(inst_net))}張")
        # 融資警示：優先用 DB 資使用率（margin_trading），否則 wespai 融資率
        _mrate = market_info.get("margin_util")
        _mrate = margin_rate if _mrate is None else float(_mrate)
        if _mrate > 20.0:
            score -= 1.0; warnings.append(f"融資使用率過高({_mrate:.1f}%)")
        elif _mrate > 15.0:
            score -= 0.5; warnings.append(f"融資使用率偏高({_mrate:.1f}%)")

    score       = max(0.0, min(10.0, score))
    reason_str  = "、".join(reasons[:4]) if reasons else "無明顯因子"
    warning_str = "；".join(warnings)    if warnings else ""

    if held_shares > 0 and close < stop_loss:
        return {
            "action": "SELL_EXIT", "name": "停損出場", "color": "#00B050", "score": score,
            "tp": take_profit, "sl": stop_loss, "suggest_lots": math.ceil(held_shares / 1000),
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-sell'>⚠️ 跌破 ATR 停損防線 ${stop_loss:.2f}</span><br>"
                     f"紀律執行出場，<b>{math.ceil(held_shares/1000)} 張</b>全部出清。{warning_str}"),
        }

    if held_shares > 0 and k > 85 and j > 90 and rsi > 78:
        sell_lots = max(1, math.floor(held_shares / 1000 / 2))
        return {
            "action": "SELL_PARTIAL", "name": "高檔減碼", "color": "#F5A623", "score": score,
            "tp": take_profit, "sl": stop_loss, "suggest_lots": sell_lots,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-hold'>💰 技術面高檔過熱 (KD {k:.0f}/{d:.0f}，RSI {rsi:.0f})</span><br>"
                     f"建議先減碼 <b>{sell_lots} 張</b>鎖利，保留剩餘部位。"),
        }

    if score >= 5.0 and close >= sma20:
        return {
            "action": "BUY", "name": "強勢進場", "color": "#E8192C", "score": score,
            "tp": take_profit, "sl": stop_loss, "suggest_lots": base_lots,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-buy'>🚀 多因子共振，進場訊號！(分數 {score:.1f}/10)</span><br>"
                     f"<b>理由</b>：{reason_str}<br>"
                     f"建議買入 <b>{base_lots} 張</b>，停損 ${stop_loss:.2f}，目標 ${take_profit:.2f}"),
        }

    if 3.0 <= score < 5.0 and close >= sma20:
        return {
            "action": "BUY_WATCH", "name": "留意機會", "color": "#F5A623", "score": score,
            "tp": take_profit, "sl": stop_loss, "suggest_lots": 1,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-hold'>📊 有一定支撐 (分數 {score:.1f}/10)</span><br>"
                     f"<b>理由</b>：{reason_str}<br>可小量試單，等待突破確認。"),
        }

    if held_shares > 0 and close > sma20:
        return {
            "action": "HOLD", "name": "多頭續抱", "color": "#F5A623", "score": score,
            "tp": take_profit, "sl": stop_loss, "suggest_lots": 0,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-hold'>🛡️ 趨勢向上，持股不動</span><br>"
                     f"跌破季線 ${sma60:.2f} 或停損 ${stop_loss:.2f} 再出場。"),
        }

    return {
        "action": "WATCH", "name": "觀望整理", "color": "#5A6072", "score": score,
        "tp": take_profit, "sl": stop_loss, "suggest_lots": 0,
        "reasons": reasons, "warnings": warnings,
        "html": (f"<span class='sig-watch'>☕ 訊號不明確 (分數 {score:.1f}/10)</span><br>"
                 f"等待均線、KD、量能三者共振後再進場。"),
    }


def get_strategy_mtf(daily_df: pd.DataFrame, weekly_df,
                     held_shares: float = 0, held_cost: float = 0,
                     market_info: dict = None) -> dict:
    """日線策略 + 週線濾網：雙重確認才升級訊號"""
    strat = get_strategy(daily_df, held_shares, held_cost, market_info=market_info)

    if weekly_df is None or (hasattr(weekly_df, "empty") and weekly_df.empty):
        strat["mtf_note"] = "週線資料不足，僅採日線判斷"
        return strat

    w = get_strategy(weekly_df)
    strat["weekly_score"]  = w["score"]
    strat["weekly_action"] = w["action"]

    # 週線弱 → 降級 BUY → BUY_WATCH
    if w["score"] < 3.0 and strat["action"] == "BUY":
        strat["action"] = "BUY_WATCH"
        strat["name"]   = "留意機會"
        strat["color"]  = "#F5A623"
        strat["html"]   = (
            f"<span class='sig-hold'>📊 日線共振，但週線較弱（週線 {w['score']:.1f}/10）</span><br>"
            f"建議等週線轉強後再進場，可小量試單。"
        )
        strat["mtf_note"] = f"⚠️ 週線訊號偏弱（{w['name']}），日線訊號降級"
    # 週線強 → 升級確認
    elif w["score"] >= 5.0 and strat["action"] == "BUY":
        prefix = (f"<span class='sig-buy'>🚀 日線 + 週線雙重共振！"
                  f"（日 {strat['score']:.1f} / 週 {w['score']:.1f}）</span><br>")
        detail = strat["html"].split("<br>", 1)[-1] if "<br>" in strat["html"] else strat["html"]
        strat["html"]   = prefix + detail
        strat["mtf_note"] = f"✅ 週線強勢確認（{w['name']}，{w['score']:.1f}/10）"
    else:
        strat["mtf_note"] = f"週線：{w['name']}（{w['score']:.1f}/10）"

    return strat


def run_backtest(df: pd.DataFrame, initial_capital: float = 500_000,
                 commission: float = 0.001425) -> dict | None:
    """
    基於預計算指標的向量化回測。
    進場：KD 黃金交叉（K<60）+ 收盤站上 MA20
    出場：KD 死亡交叉（K>50）或收盤跌破 MA60
    """
    if df is None or len(df) < 60:
        return None

    capital, position, entry_price = initial_capital, 0, 0.0
    trades, equity = [], []

    closes = df["Close"].values
    ks     = df["K"].values
    ds     = df["D"].values
    ma20s  = df["SMA20"].values
    ma60s  = df["SMA60"].values

    for i in range(1, len(df)):
        price = float(closes[i])
        k, d  = float(ks[i]), float(ds[i])
        kp    = float(ks[i - 1])
        dp    = float(ds[i - 1])
        ma20  = float(ma20s[i])
        ma60  = float(ma60s[i])

        buy_signal  = kp < dp and k > d and k < 60 and price > ma20
        sell_signal = (kp > dp and k < d and k > 50) or price < ma60 * 0.97

        if buy_signal and position == 0:
            lots   = max(1, int(capital / (price * 1000)))
            shares = lots * 1000
            cost   = shares * price * (1 + commission)
            if cost <= capital:
                capital    -= cost
                position    = shares
                entry_price = price
                trades.append({"type": "BUY", "date": df.index[i], "price": price, "shares": shares})

        elif sell_signal and position > 0:
            proceeds = position * price * (1 - commission)
            pnl      = (price - entry_price) * position
            capital  += proceeds
            trades.append({"type": "SELL", "date": df.index[i], "price": price,
                           "shares": position, "pnl": pnl})
            position = 0

        equity.append(capital + position * price)

    if not trades:
        return None

    sell_trades = [t for t in trades if t["type"] == "SELL"]
    wins        = [t for t in sell_trades if t.get("pnl", 0) > 0]

    peak, max_dd = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / max(peak, 1e-9) * 100)

    days      = (df.index[-1] - df.index[0]).days
    total_ret = (equity[-1] - initial_capital) / initial_capital * 100
    ann_ret   = ((1 + total_ret / 100) ** (365 / max(days, 1)) - 1) * 100

    return {
        "trades":       trades,
        "equity":       equity,
        "dates":        df.index[1:],
        "initial":      initial_capital,
        "final":        equity[-1],
        "total_return": total_ret,
        "ann_return":   ann_ret,
        "max_drawdown": max_dd,
        "win_rate":     len(wins) / max(len(sell_trades), 1) * 100,
        "trade_count":  len(sell_trades),
    }

"""
tw_summary.py -- 個股綜合體檢（規則彙總引擎）

設計原則：
- 每個燈號可追溯：規則只引用「已驗證」的結論（10因子評分十分位回測、
  複合選股 2 年回測、法人因子 A/B），未驗證訊號（毛利率轉折）明確標注。
- 結論是「分級 + 依據」，不是買賣指令。
- 任一資料層缺失 → 該維度亮 ⚪ 並照常輸出，絕不拋例外。

燈號：g=🟢 有利 / y=🟡 中性或需留意 / r=🔴 不利 / na=⚪ 無資料
"""

from typing import Optional

import pandas as pd

LIGHT = {"g": "🟢", "y": "🟡", "r": "🔴", "na": "⚪"}


def _safe_row(df: pd.DataFrame, code: str) -> Optional[pd.Series]:
    if df is None or df.empty or "stock_id" not in df.columns:
        return None
    m = df[df["stock_id"] == code]
    return m.iloc[0] if not m.empty else None


def build_stock_summary(code: str, strat: dict, close: float, sma20: float,
                        market_info: dict, cycle: dict) -> dict:
    """回傳 {'dims': [...], 'gate': {...}, 'verdict': {...}, 'journal': {...}}"""
    from tw_data import (cached_get_fin_signals, cached_get_margin_history,
                         cached_get_revenue_signals, cached_get_signal_journal)

    mi = market_info or {}
    score = float(strat.get("score", 0) or 0)
    above_ma = bool(close and sma20 and close >= sma20)
    dims = []

    # ── 1. 技術面（依十分位回測：0-7 單調、8+ 過熱）─────────────────────────
    if score >= 8:
        dims.append(("技術面", "y", f"⚠️ 過熱區 {score:.1f}/10",
                     "回測：8分以上 20 日期望轉負——訊號全亮常是買在人聲鼎沸處，等回檔"))
    elif 5 <= score < 8 and above_ma:
        dims.append(("技術面", "g", f"甜蜜區 {score:.1f}/10・站上月線",
                     "回測：6–7 分為預測力最高區（20日等權alpha約+2~3%）"))
    elif 3 <= score < 5:
        dims.append(("技術面", "y", f"中性 {score:.1f}/10",
                     "訊號未共振，等均線/KD/量能同步"))
    else:
        dims.append(("技術面", "r", f"偏弱 {score:.1f}/10",
                     "低分組歷史報酬落後市場均值"))

    # ── 2. 法人籌碼（A/B 驗證的新版因子輸入）────────────────────────────────
    f5, t5 = mi.get("f_net_5d"), mi.get("t_net_5d")
    streak = int(mi.get("trust_streak", 0) or 0)
    if f5 is None or t5 is None:
        dims.append(("法人籌碼", "na", "無 DB 序列", "非法人統計範圍或資料未同步"))
        chip_ok = False
    else:
        f5, t5 = float(f5), float(t5)
        stk_txt = f"・投信連買{streak}日" if streak >= 3 else ""
        detail = f"外資5日 {f5:+,.0f} 張／投信5日 {t5:+,.0f} 張"
        if f5 > 0 and t5 > 0:
            dims.append(("法人籌碼", "g", f"外資投信同買{stk_txt}", detail))
        elif f5 > 0 or t5 > 0:
            dims.append(("法人籌碼", "y", f"單邊買超{stk_txt}", detail))
        else:
            dims.append(("法人籌碼", "r", "外資投信同賣", detail))
        chip_ok = f5 > 0 and t5 > 0 and streak >= 5   # 複合選股已驗證組合

    # ── 3. 散戶籌碼（融資券）────────────────────────────────────────────────
    try:
        mg = cached_get_margin_history(code, days=10)
    except Exception:
        mg = None
    if mg is None or mg.empty:
        dims.append(("散戶籌碼", "na", "無融資券資料", "—"))
        margin_risk = False
    else:
        last = mg.iloc[-1]
        util = last.get("margin_util")
        smr = last.get("short_margin_ratio")
        chg5 = (int(last["margin_balance"] - mg.iloc[0]["margin_balance"])
                if len(mg) > 1 else 0)
        d = f"融資餘額 {int(last['margin_balance']):,} 張（5日 {chg5:+,}）"
        if pd.notna(smr):
            d += f"・券資比 {smr:.1f}%"
            if smr > 30:
                d += "（偏高，具軋空彈性）"
        margin_risk = pd.notna(util) and util >= 15
        if margin_risk:
            dims.append(("散戶籌碼", "r", f"融資使用率 {util:.1f}%（過高）",
                         d + "——籌碼浮動，回檔易多殺多"))
        elif pd.notna(util) and util >= 10:
            dims.append(("散戶籌碼", "y", f"融資使用率 {util:.1f}%", d))
        else:
            dims.append(("散戶籌碼", "g",
                         f"融資水位低（{util:.1f}%）" if pd.notna(util) else "融資水位低", d))

    # ── 4. 營收動能（月營收）────────────────────────────────────────────────
    try:
        rev = _safe_row(cached_get_revenue_signals(), code)
    except Exception:
        rev = None
    if rev is None or pd.isna(rev.get("yoy_pct")):
        dims.append(("營收動能", "na", "無月營收資料", "—"))
        rev_ok = False
    else:
        yoy, mom = float(rev["yoy_pct"]), rev.get("mom_pct")
        ym = rev.get("year_month", "")
        tag = "・轉機(由負轉正)" if rev.get("turnaround") else (
              "・連續加速" if rev.get("accelerating") else "")
        d = f"{ym}：YoY {yoy:+.1f}%" + (f"／MoM {mom:+.1f}%" if pd.notna(mom) else "")
        rev_ok = yoy > 20
        if rev_ok:
            dims.append(("營收動能", "g", f"YoY {yoy:+.0f}%{tag}", d))
        elif yoy > 0:
            dims.append(("營收動能", "y", f"YoY {yoy:+.0f}%{tag}", d))
        else:
            dims.append(("營收動能", "r", f"YoY {yoy:+.0f}%{tag}",
                         d + "——回測：轉機模式未獲支持，衰退中勿只靠籌碼"))

    # ── 5. 獲利品質（季報 EPS/毛利率）───────────────────────────────────────
    try:
        fin = _safe_row(cached_get_fin_signals(), code)
    except Exception:
        fin = None
    if fin is None:
        dims.append(("獲利品質", "na", "無季報資料", "金融業毛利率欄不適用"))
    else:
        ttm = fin.get("ttm_eps")
        gmd = fin.get("gm_yoy_delta")
        pe = (close / ttm) if ttm and pd.notna(ttm) and ttm > 0 and close else None
        d = f"{fin.get('quarter','')}：TTM EPS {ttm:.2f}" if pd.notna(ttm) else "TTM EPS —"
        if pe:
            d += f"・PE(TTM) {pe:.1f}x"
        if pd.notna(gmd):
            d += f"・毛利率同季YoY {gmd:+.1f}pp"
        if fin.get("margin_up") and pd.notna(ttm) and ttm > 0:
            dims.append(("獲利品質", "g", "毛利率轉折向上（未回測訊號）", d))
        elif pd.notna(ttm) and ttm > 0:
            dims.append(("獲利品質", "y", "獲利穩定", d))
        else:
            dims.append(("獲利品質", "r", "獲利疲弱（TTM EPS ≤ 0 或缺）", d))

    # ── 大盤週期閘門（回測：因子選股不擇時，弱市絕對報酬可能為負）────────────
    gate = {"light": "na", "title": "大盤週期資料缺", "detail": "—", "bull": True}
    if cycle:
        bull = cycle.get("phase") == 1
        days = cycle.get("days_in_cycle", 0)
        avg = cycle.get("avg_same_days", 0)
        over = days - avg if avg else 0
        if bull and over <= 0:
            gate = {"light": "g", "title": f"上漲週期第 {days} 天",
                    "detail": f"均值 {avg} 天——環境有利", "bull": True}
        elif bull:
            gate = {"light": "y", "title": f"上漲週期第 {days} 天（超均值 +{over}）",
                    "detail": "週期已過歷史均壽，新倉紀律從嚴、既有部位顧好停損", "bull": True}
        else:
            gate = {"light": "r", "title": "下跌週期",
                    "detail": "回測：弱市中選股因子絕對報酬可能為負——現金也是部位",
                    "bull": False}

    # ── 綜合結論（規則彙總，可追溯）─────────────────────────────────────────
    sweet = 5 <= score < 8 and above_ma
    hot = score >= 8
    if chip_ok and rev_ok and sweet and gate["bull"]:
        verdict = {"level": "A", "color": "#E8192C", "title": "🎯 符合已驗證複合條件",
                   "text": ("籌碼（外資投信同買＋投信連買≥5）×營收（YoY>20%）×技術甜蜜區"
                            "×牛市環境全數成立——與 2 年回測最佳組同構"
                            "（歷史 20 日勝率 52%／平均 +4.9%，60 日勝率 59%／+17.5%，"
                            "含成本）。紀律：持有 20–60 日、同類訊號分散接、破停損就走。")}
    elif chip_ok and rev_ok and hot:
        verdict = {"level": "B", "color": "#F5A623", "title": "✅ 條件成立・技術過熱",
                   "text": "籌碼與營收條件皆符，但評分落在 8 分以上過熱區"
                           "（歷史 20 日期望轉負）——列入追蹤，等拉回月線附近再評估。"}
    elif chip_ok and rev_ok and not gate["bull"]:
        verdict = {"level": "B", "color": "#F5A623", "title": "✅ 條件成立・大盤逆風",
                   "text": "個股條件符合複合訊號，但大盤處下跌週期——回測顯示弱市中"
                           "絕對報酬可能為負，倉位減半或等週期翻多。"}
    elif chip_ok and rev_ok:
        verdict = {"level": "B", "color": "#F5A623", "title": "✅ 條件成立・技術未確認",
                   "text": "籌碼與營收到位、技術面尚未進入甜蜜區——放進觀察清單，"
                           "等站穩月線或評分升上 5 分。"}
    elif (chip_ok or rev_ok) and score >= 3:
        _done, _todo = [], []
        (_done if chip_ok else _todo).append("籌碼（外資投信同買＋投信連買≥5）")
        (_done if rev_ok else _todo).append("營收（YoY>20%）")
        (_done if sweet else _todo).append("技術甜蜜區（5–8分且站上月線）")
        verdict = {"level": "C", "color": "#4A7BA6", "title": "👀 觀察（部分條件成立）",
                   "text": (f"已成立：{'、'.join(_done)}。待補：{'、'.join(_todo)}。"
                            "複合訊號的價值在多面向同時確認，等到位再行動。")}
    elif score < 3 and dims[1][1] == "r":
        verdict = {"level": "D", "color": "#00B050", "title": "🚫 多面向偏弱・迴避",
                   "text": "技術低分＋法人同賣——歷史上此組合顯著落後市場，"
                           "沒有持股就不需要動作，有持股檢視停損紀律。"}
    else:
        verdict = {"level": "C", "color": "#5A6072", "title": "☕ 訊號不足・中性",
                   "text": "各面向未形成一致方向，維持觀望。"}
    if margin_risk and verdict["level"] in ("A", "B"):
        verdict["text"] += "（⚠️ 融資使用率偏高，追價風險放大）"

    # ── 此標的的訊號日誌戰績 ────────────────────────────────────────────────
    journal = {"n": 0, "last": None}
    try:
        _, jrows = cached_get_signal_journal()
        if jrows is not None and not jrows.empty:
            mine = jrows[jrows["stock_id"] == code]
            if not mine.empty:
                lastr = mine.iloc[0]
                journal = {"n": len(mine),
                           "last": {"date": lastr["signal_date"],
                                    "ret_20": lastr.get("ret_20")}}
    except Exception:
        pass

    return {"dims": dims, "gate": gate, "verdict": verdict, "journal": journal}

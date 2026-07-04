"""Telegram 推播（依賴 st.secrets）"""
import requests
import streamlit as st

SEMI_SCORE_MIN = 5.0


def _get_tg_creds():
    try:
        tok = str(st.secrets.get("TG_TOKEN", "")).strip()
        cid = str(st.secrets.get("TG_CHAT_ID", "")).strip()
        if tok and cid:
            return tok, cid
    except Exception:
        pass
    # headless 排程備援：環境變數或本機 ~/.streamlit/secrets.toml（st.secrets 亦讀得到）
    import os
    return os.environ.get("TG_TOKEN", "").strip(), os.environ.get("TG_CHAT_ID", "").strip()


def send_tg_message(text: str) -> bool:
    token, chat_id = _get_tg_creds()
    if not token or not chat_id:
        return False
    url    = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
    ok     = True
    for chunk in chunks:
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": chunk,
                                          "parse_mode": "Markdown"}, timeout=15)
            if not r.json().get("ok"):
                ok = False
        except Exception:
            ok = False
    return ok


def _build_candidate_block(i: int, c: dict, sig_emoji: dict) -> list:
    RANKS = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    rank  = RANKS[i] if i < len(RANKS) else f"{i+1}."
    sig   = sig_emoji.get(c["action"], "⚪ 觀望")
    stars = "⭐" * max(1, round(c["score"] / 2))
    return [
        f"{rank} *{c['代碼']} {c['名稱']}*  {stars}",
        f"   {sig}  |  分數 *{c['score']:.1f}*/10",
        f"   現價 ${c['現價']:.2f}  |  {c['產業']}",
        f"   📈 {'、'.join(c['reasons'][:3]) if c['reasons'] else '—'}",
        f"   🛑 停損 {'$'+str(round(c['sl'],1)) if c.get('sl') else '—'}  |  "
        f"🎯 目標 {'$'+str(round(c['tp'],1)) if c.get('tp') else '—'}",
        f"   PE {c.get('PE','—')}  |  PB {c.get('PB','—')}",
        "",
    ]


def format_semi_tg_messages(candidates: list, scan_count: int,
                              hit_count: int, scan_date: str) -> list:
    SIG_EMOJI = {
        "BUY": "🔴 強勢進場", "BUY_WATCH": "🟡 留意機會",
        "HOLD": "🟠 多頭續抱", "SELL_PARTIAL": "🟢 高檔減碼",
        "SELL_EXIT": "🟢 停損出場", "WATCH": "⚪ 觀望",
    }
    MAX = 4000
    sep = "\n"

    header = [
        "📊 *台股半導體族群 · 日收盤自動掃描*",
        f"📅 {scan_date} 18:00 | 市場已收盤",
        "─────────────────────",
        f"掃描：{scan_count} 檔　│　入選：{hit_count} 檔　│　門檻：技術分 ≥ {SEMI_SCORE_MIN}",
        "─────────────────────", "",
    ]
    footer = ["─────────────────────", "⚠️ 本訊息僅供參考，不構成投資建議"]

    if not candidates:
        return [sep.join(header + ["⚠️ 本日無符合條件的強勢標的，建議觀望。"] + footer)]

    messages, cur_lines, is_first = [], list(header), True
    for i, c in enumerate(candidates):
        block    = _build_candidate_block(i, c, SIG_EMOJI)
        cur_text = sep.join(cur_lines)
        if len(cur_text) + len(sep.join(block)) + 1 > MAX and not is_first:
            messages.append(cur_text)
            cur_lines = [f"📊 *台股半導體族群 · 續篇 ({len(messages)+1})*", ""]
        cur_lines.extend(block)
        is_first = False

    cur_lines.extend(footer)
    messages.append(sep.join(cur_lines))
    return messages

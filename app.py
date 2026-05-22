"""
台股戰情指揮中心 V15 Pro
Multi-factor strategy · KD · 年線 · 量比 · 台股紅漲綠跌配色
Dark precision UI · Mobile-first · 5-tab layout
"""
import math
import random
import time
from datetime import datetime, timedelta

import gspread
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PORTFOLIO_SHEET_TITLE = "Streamlit TW Stock"

# ── 半導體族群關鍵字（對應 Wespai 產業欄位模糊比對）─────────────────────────
# ── 半導體族群關鍵字 ────────────────────────────────────────────────────────
#
# 台灣半導體生態系上市+上櫃約 450–600 支，分屬多個 Wespai 官方產業分類。
# 必須同時匹配「Wespai 官方產業名稱」與「公司業務描述詞」才能完整覆蓋。
#
# 【官方產業分類 (Wespai 產業欄直接對應)】
#   半導體業、電子零組件業、光電業、電腦及週邊設備業、通信網路業 …
# 【業務描述詞 (公司簡介/產品描述內含)】
#   IC設計、晶圓代工、封裝測試、記憶體、功率元件 …
# ─────────────────────────────────────────────────────────────────────────────
SEMI_KEYWORDS = [
    # ── Wespai 官方產業分類名稱（最關鍵！直接對到分類欄位）─────────────────
    "半導體",           # 半導體業（TSMC、聯電、世界先進等）
    "電子零組件",       # 電子零組件業（大量IC、被動、連接器公司）
    "光電",             # 光電業（感測器、LED、雷射、CMOS）
    "通信網路",         # 通信網路業（WiFi/BT/5G晶片、網通設備）
    "電腦及週邊",       # 電腦及週邊設備業（伺服器、AI伺服器、儲存）
    "其他電子",         # 其他電子業（catch-all 大量電子公司）

    # ── 核心半導體製程 ──────────────────────────────────────────────────────
    "積體電路", "IC設計", "ic設計", "晶圓", "晶圓代工",
    "封裝", "測試", "封裝測試", "OSAT",

    # ── 記憶體 ──────────────────────────────────────────────────────────────
    "記憶體", "DRAM", "Flash", "NAND", "NOR", "SRAM", "HBM",

    # ── 功率/電源/類比 ───────────────────────────────────────────────────────
    "功率元件", "電源管理", "類比", "PMIC", "MOSFET", "IGBT",
    "驅動IC", "混合訊號",

    # ── 無線/射頻 ────────────────────────────────────────────────────────────
    "RF", "射頻", "藍牙", "WiFi", "無線", "5G晶片",

    # ── AI / 伺服器基礎設施 ──────────────────────────────────────────────────
    "伺服器", "Server", "資料中心", "AI加速", "GPU", "NPU",
    "CoWoS", "ABF", "HBM", "先進封裝",

    # ── 基板 / 載板 / PCB ────────────────────────────────────────────────────
    "基板", "載板", "導線框", "引線框", "印刷電路板", "電路板", "PCB",

    # ── 散熱 / 機構 ──────────────────────────────────────────────────────────
    "散熱", "液冷", "均溫板", "熱管", "水冷",

    # ── 化合物半導體 / 新材料 ────────────────────────────────────────────────
    "化合物半導體", "碳化矽", "SiC", "氮化鎵", "GaN",
    "矽晶圓", "磊晶", "砷化鎵",

    # ── 半導體材料 / 設備 ────────────────────────────────────────────────────
    "光罩", "光阻", "研磨", "CMP", "濺鍍", "蝕刻",
    "半導體設備", "晶圓設備",

    # ── 被動元件 / 連接器（電子零組件業子類） ───────────────────────────────
    "被動元件", "電感", "MLCC", "電容", "電阻", "連接器",

    # ── IP / 設計工具 ────────────────────────────────────────────────────────
    "矽智財", "IP矽", "EDA", "IP授權",

    # ── 感測器 / 影像 ────────────────────────────────────────────────────────
    "感測器", "CMOS感測", "影像感測", "ToF", "LiDAR",
]

# ── 估值門檻 ──────────────────────────────────────────────────────────────────
# 台灣半導體各子族群估值參考（2024–2025 市場均值）：
#   晶圓代工   PE 12–25  PB 2–5
#   IC設計     PE 15–40  PB 2–8   ← 成長股 PE 可達 40+
#   封裝測試   PE 10–20  PB 1–3
#   伺服器/AI  PE 20–60  PB 2–6   ← AI 題材本益比高
#   被動元件   PE 15–30  PB 1–4
# ─────────────────────────────────────────────────────────────────────────────
SEMI_PE_MAX    = 60.0    # 上調至 60，涵蓋 AI/伺服器成長股（原 45 太嚴）
SEMI_PB_MAX    = 10.0    # 上調至 10，涵蓋高品質 IC 設計領頭羊（原 8）
SEMI_PE_MIN    = -999.0  # 允許虧損股進入族群（PE<0 也掃，但技術分會過濾）
SEMI_SCORE_MIN = 5.0     # 強勢股技術分門檻（≥5/10 需多因子共振）
SEMI_SCAN_MAX  = 1500    # 大幅提高上限，確保全掃（台灣半導體生態系約 500–600 支）

# ── Telegram（從 Streamlit Secrets 讀取）────────────────────────────────────
def _get_tg_creds():
    try:
        token   = st.secrets.get("TG_TOKEN", "")
        chat_id = st.secrets.get("TG_CHAT_ID", "")
        return str(token).strip(), str(chat_id).strip()
    except Exception:
        return "", ""


st.set_page_config(
    page_title="台股戰情中心",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — Taiwan financial terminal dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Noto+Sans+TC:wght@300;400;500;700;900&display=swap');

:root {
  --bg:       #0A0C12;
  --surface:  #111318;
  --surface2: #181B23;
  --border:   rgba(255,255,255,0.07);
  --border2:  rgba(255,255,255,0.13);
  --text:     #E6E8F0;
  --muted:    #5A6072;
  --up:       #E8192C;   /* 台股：紅漲 */
  --down:     #00B050;   /* 台股：綠跌 */
  --gold:     #F5A623;
  --blue:     #3D8EFF;
  --purple:   #9B6DFF;
  --cyan:     #00D4FF;
  --mono:     'JetBrains Mono', monospace;
  --sans:     'Noto Sans TC', 'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important; color: var(--text); font-family: var(--sans);
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important; font-size: 1.3rem !important;
  font-weight: 700 !important; color: var(--text) !important; letter-spacing: -0.02em;
}
[data-testid="stMetricLabel"] {
  font-size: 0.68rem !important; color: var(--muted) !important;
  text-transform: uppercase; letter-spacing: 0.06em; font-family: var(--sans) !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; font-family: var(--mono) !important; }
.stMetric {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; padding: 14px 16px !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
  background: var(--surface) !important; border-radius: 12px !important;
  padding: 4px !important; gap: 2px !important; border: 1px solid var(--border) !important;
  flex-wrap: wrap !important;
}
[data-baseweb="tab"] {
  background: transparent !important; color: var(--muted) !important;
  border-radius: 8px !important; font-size: 0.75rem !important; font-weight: 700 !important;
  font-family: var(--sans) !important; padding: 6px 10px !important; transition: all 0.2s;
}
[aria-selected="true"][data-baseweb="tab"] {
  background: var(--up) !important; color: #fff !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  color: var(--text) !important; border-radius: 10px !important;
  font-family: var(--sans) !important; font-weight: 700 !important;
  font-size: 0.85rem !important; transition: all 0.2s;
}
.stButton > button:hover { border-color: var(--up) !important; color: var(--up) !important; }

/* ── Inputs ───────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
  background: var(--surface2) !important; border: 1px solid var(--border2) !important;
  border-radius: 10px !important; color: var(--text) !important;
  font-family: var(--mono) !important; font-size: 0.9rem !important;
}

/* ── Expander ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* ── Custom: Header ───────────────────────────────────────────────────────── */
.tw-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 0 20px; border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.tw-logo {
  font-family: var(--mono); font-size: 1.1rem; font-weight: 700;
  color: var(--up); letter-spacing: -0.02em;
}
.tw-logo span { color: var(--muted); font-weight: 400; }

/* ── Badge ────────────────────────────────────────────────────────────────── */
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 9px; border-radius: 999px;
  font-size: 0.68rem; font-weight: 700; font-family: var(--sans);
  letter-spacing: 0.04em; text-transform: uppercase;
}
.badge-up   { background: rgba(232,25,44,0.15); color: var(--up);   border: 1px solid rgba(232,25,44,0.35); }
.badge-down { background: rgba(0,176,80,0.15);  color: var(--down); border: 1px solid rgba(0,176,80,0.35); }
.badge-flat { background: rgba(90,96,114,0.2);  color: var(--muted); border: 1px solid var(--border2); }
.badge-gold { background: rgba(245,166,35,0.12); color: var(--gold); border: 1px solid rgba(245,166,35,0.3); }
.badge-blue { background: rgba(61,142,255,0.12); color: var(--blue); border: 1px solid rgba(61,142,255,0.3); }

/* ── Stock card ───────────────────────────────────────────────────────────── */
.sc {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; padding: 16px; margin-bottom: 10px;
  position: relative; overflow: hidden;
}
.sc-accent { position: absolute; left: 0; top: 0; bottom: 0; width: 3px; border-radius: 16px 0 0 16px; }
.sc-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; }
.sc-name { font-size: 1.0rem; font-weight: 900; color: var(--text); font-family: var(--sans); }
.sc-code { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); font-weight: 400; }
.sc-price { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; }
.sc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px 14px; margin-top: 10px; }
.sc-kv-label { font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-family: var(--sans); }
.sc-kv-value { font-family: var(--mono); font-size: 0.85rem; color: var(--text); font-weight: 600; }
.sc-action {
  margin-top: 10px; padding: 9px 12px; background: var(--surface2);
  border-radius: 9px; font-size: 0.8rem; font-family: var(--sans);
}
.sc-divider { border: none; border-top: 1px solid var(--border); margin: 10px 0; }

/* weight bar */
.wbar-bg { background: var(--surface2); border-radius: 999px; height: 3px; margin-top: 8px; }
.wbar-fill { height: 3px; border-radius: 999px; }

/* signal text colors */
.sig-buy  { color: var(--up); font-weight: 900; }
.sig-sell { color: var(--down); font-weight: 900; }
.sig-hold { color: var(--gold); font-weight: 900; }
.sig-watch{ color: var(--muted); font-weight: 900; }

/* ── Scanner row ──────────────────────────────────────────────────────────── */
.sk-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 13px; padding: 12px 15px; margin-bottom: 7px;
  display: flex; align-items: center; gap: 12px;
}
.sk-rank { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); min-width: 20px; }
.sk-ticker { font-family: var(--mono); font-size: 0.95rem; font-weight: 700; color: var(--text); }
.sk-name { font-size: 0.75rem; color: var(--muted); margin-top: 1px; font-family: var(--sans); }
.sk-score { font-family: var(--mono); font-size: 0.9rem; font-weight: 700; color: var(--up); }
.sk-reason { font-size: 0.7rem; color: var(--muted); margin-top: 2px; font-family: var(--sans); }
.sbar { background: var(--surface2); border-radius: 999px; height: 3px; flex: 1; }
.sbar-fill { height: 3px; border-radius: 999px; background: linear-gradient(90deg, var(--up), var(--gold)); }

/* ── Perf stats ───────────────────────────────────────────────────────────── */
.ps-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.ps { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.ps-label { font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-family: var(--sans); }
.ps-value { font-family: var(--mono); font-size: 1.2rem; font-weight: 700; margin-top: 4px; }

.qdiv { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
.qsec { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-family: var(--sans); font-weight: 700; margin: 16px 0 10px; }

@media (max-width: 600px) {
  [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
  .sc-price { font-size: 1.2rem; }
  .sc { padding: 13px; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────────────────────
def get_gsheet_client():
    return gspread.service_account_from_dict(st.secrets["gcp_service_account"])


@st.cache_data(ttl=300, show_spinner=False)
def load_portfolio() -> pd.DataFrame:
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        df = pd.DataFrame(sh.sheet1.get_all_records())
        df["Symbol"] = df["Symbol"].astype(str).str.zfill(4)
        df["Cost"]   = pd.to_numeric(df["Cost"],   errors="coerce").fillna(0.0)
        df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce").fillna(0)
        return df
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Name", "Cost", "Shares", "Note"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data() -> dict:
    """Wespai 市場報價 + PE/PB"""
    url = "https://stock.wespai.com/lists"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        df = pd.read_html(res.text)[0]
        data = df.iloc[:, [0, 1, 2, 3, 14, 15]].copy()
        data.columns = ["代碼", "名稱", "產業", "現價", "PE", "PB"]
        data["代碼"] = data["代碼"].astype(str).str.zfill(4)
        data["現價"] = pd.to_numeric(data["現價"], errors="coerce")
        data["PE"]   = pd.to_numeric(data["PE"],   errors="coerce").fillna(999.0)
        data["PB"]   = pd.to_numeric(data["PB"],   errors="coerce").fillna(999.0)
        return data.set_index("代碼").to_dict("index")
    except Exception as e:
        st.error(f"市場報價抓取失敗: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# ★ Enhanced indicator engine ★
# ─────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    """計算全套技術指標：MA5/20/60/240、BB、ATR、KD(9)、RSI、MACD、OBV、量比"""
    if df is None or len(df) < 60:
        return None
    df = df.copy()

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["SMA5"]  = df["Close"].rolling(5).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()
    df["SMA240"] = df["Close"].rolling(min(240, len(df))).mean()  # 年線

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["SMA20"] + 1e-9)

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ── KD Stochastic (9,3,3) ────────────────────────────────────────────────
    n = 9
    low_n  = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    df["RSV"] = (df["Close"] - low_n) / (high_n - low_n + 1e-9) * 100
    df["K"] = df["RSV"].ewm(com=2, adjust=False).mean()
    df["D"] = df["K"].ewm(com=2, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ── MACD (12,26,9) ────────────────────────────────────────────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]

    # ── Volume indicators ─────────────────────────────────────────────────────
    df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
    df["VOL_Ratio"] = df["Volume"] / (df["VOL_SMA20"] + 1e-9)  # 量比

    # ── OBV ───────────────────────────────────────────────────────────────────
    obv = [0]
    cls, vols = df["Close"].tolist(), df["Volume"].tolist()
    for i in range(1, len(df)):
        if cls[i] > cls[i - 1]:   obv.append(obv[-1] + vols[i])
        elif cls[i] < cls[i - 1]: obv.append(obv[-1] - vols[i])
        else:                      obv.append(obv[-1])
    df["OBV"] = obv

    # ── 52W High/Low ─────────────────────────────────────────────────────────
    df["High52W"] = df["High"].rolling(min(252, len(df))).max()
    df["Low52W"]  = df["Low"].rolling(min(252, len(df))).min()

    return df.dropna(subset=["SMA20", "SMA60", "K", "D", "RSI"]).copy()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_history(symbol: str) -> pd.DataFrame | None:
    """資料抓取引擎 (FinMind 優先 → YFinance 上市/上櫃備援)"""
    # 1. FinMind
    try:
        time.sleep(random.uniform(0.05, 0.2))
        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=760)).strftime("%Y-%m-%d")
        res = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockPrice", "data_id": symbol,
                    "start_date": start_date, "end_date": end_date},
            timeout=5,
        )
        if res.status_code == 200:
            data = res.json()
            if data.get("msg") == "success" and data.get("data"):
                df = pd.DataFrame(data["data"])
                df = df.rename(columns={
                    "date": "Date", "open": "Open", "max": "High",
                    "min": "Low", "close": "Close", "trading_volume": "Volume"
                })
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                result = calculate_indicators(df)
                if result is not None:
                    return result
    except Exception:
        pass

    # 2. YFinance (上市 .TW / 上櫃 .TWO)
    for suffix in [".TW", ".TWO"]:
        try:
            tk = yf.Ticker(f"{symbol}{suffix}")
            df = tk.history(period="3y", auto_adjust=True)
            if not df.empty:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                result = calculate_indicators(df)
                if result is not None:
                    return result
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ★ Multi-factor strategy engine ★
# ─────────────────────────────────────────────────────────────────────────────
def get_strategy(df: pd.DataFrame, held_shares: float = 0, held_cost: float = 0) -> dict:
    """
    多因子評分策略 (0–10分)
    因子：趨勢、KD、RSI、BB突破、量比、OBV、年線位置
    訊號：BUY / HOLD / SELL_PARTIAL / SELL_EXIT / WATCH
    """
    if df is None or df.empty or len(df) < 20:
        return _default_strat("資料不足", "#5A6072")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    close    = float(last["Close"])
    sma5     = float(last["SMA5"])
    sma20    = float(last["SMA20"])
    sma60    = float(last["SMA60"])
    sma240   = float(last.get("SMA240", 0) or 0)
    atr      = float(last.get("ATR", 0) or 0)
    rsi      = float(last["RSI"])
    k, d     = float(last["K"]), float(last["D"])
    j        = float(last["J"])
    macd_h   = float(last["Hist"])
    bb_w     = float(last["BB_Width"])
    vol_r    = float(last.get("VOL_Ratio", 1.0))
    high52   = float(last.get("High52W", close))
    low52    = float(last.get("Low52W",  close))

    # ── Stop / Take Profit ────────────────────────────────────────────────────
    stop_loss   = max(close - 2.0 * atr, sma60 * 0.98, close * 0.88) if atr else close * 0.88
    take_profit = close + 3.0 * atr if atr else close * 1.12
    # Dynamic TP based on 52W high
    if high52 > close:
        take_profit = min(high52, take_profit)

    # ── Lot sizing (台股 1張=1000股) ─────────────────────────────────────────
    base_lots = 1 if close > 150 else 2 if close > 60 else 3

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR SCORING (滿分 10)
    # ═══════════════════════════════════════════════════════════════════════
    score    = 0.0
    reasons  = []
    warnings = []

    # 1. Trend alignment (均線多頭排列)
    if close > sma20 > sma60:
        score += 2.0; reasons.append("均線多頭")
        if sma5 > sma20:
            score += 0.5; reasons.append("5日線向上")
    elif close < sma20 < sma60:
        score -= 1.0; warnings.append("均線空頭")

    # 2. Annual line position (年線 — 長線趨勢濾網)
    if sma240 > 0:
        if close > sma240:
            score += 1.0; reasons.append("站上年線")
        else:
            score -= 0.5; warnings.append("年線下方")

    # 3. KD indicator
    k_gc_prev = float(prev["K"]) < float(prev["D"])  # previous K below D
    k_gc_now  = k > d                                  # current K above D = 黃金交叉
    if k < 20 and d < 20 and k_gc_now and k_gc_prev:
        score += 2.5; reasons.append("KD低檔黃金交叉")
    elif k < 30 and k > d:
        score += 1.5; reasons.append("KD低檔翻揚")
    elif k > 80 and d > 80:
        score -= 1.0; warnings.append("KD高檔超買")
    elif k > 85 and j > 90:
        score -= 1.5; warnings.append("KD超買+J值過熱")

    # 4. RSI zone
    if rsi < 30:
        score += 1.5; reasons.append(f"RSI超賣({rsi:.0f})")
    elif 40 <= rsi <= 65:
        score += 0.5; reasons.append("RSI健康")
    elif rsi > 80:
        score -= 1.0; warnings.append(f"RSI超買({rsi:.0f})")

    # 5. BB squeeze breakout
    prev_bb_w = float(prev.get("BB_Width", 1.0))
    is_squeeze = bb_w < 0.08
    is_breakout = (close > float(last["BB_Upper"])) and (vol_r > 1.5) and (prev_bb_w < 0.10)
    if is_breakout:
        score += 2.5; reasons.append("BB壓縮放量突破")
    elif is_squeeze:
        reasons.append("BB醞釀蓄力")

    # 6. Volume ratio (量比)
    if vol_r >= 2.0 and close >= sma20:
        score += 1.0; reasons.append(f"大量上漲({vol_r:.1f}倍)")
    elif vol_r < 0.5:
        warnings.append("成交量萎縮")

    # 7. MACD histogram direction
    if macd_h > 0 and float(prev["Hist"]) < macd_h:
        score += 0.5; reasons.append("MACD翻多")
    elif macd_h < 0 and float(prev["Hist"]) > macd_h:
        score -= 0.5; warnings.append("MACD轉弱")

    # 8. Near 52W high (市場強勢股過濾)
    if close >= high52 * 0.95:
        score += 0.5; reasons.append("接近年高")
    near_bottom = (close - low52) / max(high52 - low52, 1e-9)
    if near_bottom < 0.15:
        score += 0.5; reasons.append("接近年低支撐")

    # Clamp
    score = max(0.0, min(10.0, score))

    # ═══════════════════════════════════════════════════════════════════════
    # SIGNAL DECISION TREE
    # ═══════════════════════════════════════════════════════════════════════
    reason_str   = "、".join(reasons[:4]) if reasons else "無明顯因子"
    warning_str  = "；".join(warnings)    if warnings else ""

    # ── 優先：出場保護 ────────────────────────────────────────────────────
    if held_shares > 0 and close < stop_loss:
        return {
            "action": "SELL_EXIT", "name": "停損出場",
            "color": "#00B050", "score": score,
            "tp": take_profit, "sl": stop_loss,
            "suggest_lots": math.ceil(held_shares / 1000),
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-sell'>⚠️ 跌破 ATR 停損防線 ${stop_loss:.2f}</span><br>"
                     f"紀律執行出場，<b>{math.ceil(held_shares/1000)} 張</b>全部出清。{warning_str}"),
        }

    # ── 高檔獲利了結 ──────────────────────────────────────────────────────
    if held_shares > 0 and (k > 85 and j > 90 and rsi > 78):
        sell_lots = max(1, math.floor(held_shares / 1000 / 2))
        return {
            "action": "SELL_PARTIAL", "name": "高檔減碼",
            "color": "#F5A623", "score": score,
            "tp": take_profit, "sl": stop_loss,
            "suggest_lots": sell_lots,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-hold'>💰 技術面高檔過熱 (KD {k:.0f}/{d:.0f}，RSI {rsi:.0f})</span><br>"
                     f"建議先減碼 <b>{sell_lots} 張</b>鎖利，保留剩餘部位。"),
        }

    # ── 強力進場 ─────────────────────────────────────────────────────────
    if score >= 5.0 and close >= sma20:
        return {
            "action": "BUY", "name": "強勢進場",
            "color": "#E8192C", "score": score,
            "tp": take_profit, "sl": stop_loss,
            "suggest_lots": base_lots,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-buy'>🚀 多因子共振，進場訊號！(分數 {score:.1f}/10)</span><br>"
                     f"<b>理由</b>：{reason_str}<br>"
                     f"建議買入 <b>{base_lots} 張</b>，停損 ${stop_loss:.2f}，目標 ${take_profit:.2f}"),
        }

    # ── 中度進場 ─────────────────────────────────────────────────────────
    if 3.0 <= score < 5.0 and close >= sma20:
        return {
            "action": "BUY_WATCH", "name": "留意機會",
            "color": "#F5A623", "score": score,
            "tp": take_profit, "sl": stop_loss,
            "suggest_lots": 1,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-hold'>📊 有一定支撐 (分數 {score:.1f}/10)</span><br>"
                     f"<b>理由</b>：{reason_str}<br>可小量試單，等待突破確認。"),
        }

    # ── 續抱 ─────────────────────────────────────────────────────────────
    if held_shares > 0 and close > sma20:
        return {
            "action": "HOLD", "name": "多頭續抱",
            "color": "#F5A623", "score": score,
            "tp": take_profit, "sl": stop_loss,
            "suggest_lots": 0,
            "reasons": reasons, "warnings": warnings,
            "html": (f"<span class='sig-hold'>🛡️ 趨勢向上，持股不動</span><br>"
                     f"跌破季線 ${sma60:.2f} 或停損 ${stop_loss:.2f} 再出場。"),
        }

    # ── 觀望 ─────────────────────────────────────────────────────────────
    return {
        "action": "WATCH", "name": "觀望整理",
        "color": "#5A6072", "score": score,
        "tp": take_profit, "sl": stop_loss,
        "suggest_lots": 0,
        "reasons": reasons, "warnings": warnings,
        "html": (f"<span class='sig-watch'>☕ 訊號不明確 (分數 {score:.1f}/10)</span><br>"
                 f"等待均線、KD、量能三者共振後再進場。"),
    }


def _default_strat(name, color):
    return {
        "action": "WATCH", "name": name, "color": color, "score": 0,
        "tp": None, "sl": None, "suggest_lots": 0,
        "reasons": [], "warnings": [],
        "html": f"<span style='color:{color}'>{name}</span>",
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
MARKET_MAP = get_market_data()
STOCK_OPTIONS = [f"{k} {v['名稱']} ({v['產業']})" for k, v in MARKET_MAP.items()]


def fmt_tw_pl(val: float) -> str:
    if val > 0: return f'<span style="color:var(--up);font-weight:900">▲{val:,.2f}</span>'
    if val < 0: return f'<span style="color:var(--down);font-weight:900">▼{abs(val):,.2f}</span>'
    return f'<span style="color:var(--muted)">—{val:,.2f}</span>'


def pl_colour(val: float) -> str:
    return "var(--up)" if val > 0 else "var(--down)" if val < 0 else "var(--muted)"


def signal_badge_html(strat: dict) -> str:
    a = strat["action"]
    if a in ("BUY", "BUY_WATCH"):   return f'<span class="badge badge-up">▲ {strat["name"]}</span>'
    if a in ("SELL_EXIT", "SELL_PARTIAL"): return f'<span class="badge badge-down">▼ {strat["name"]}</span>'
    if a == "HOLD":  return f'<span class="badge badge-gold">＝ {strat["name"]}</span>'
    return f'<span class="badge badge-flat">— {strat["name"]}</span>'


def score_bar_html(score: float) -> str:
    pct = min(100, score / 10 * 100)
    col = "var(--up)" if score >= 5 else "var(--gold)" if score >= 3 else "var(--muted)"
    return f'<div class="sbar"><div class="sbar-fill" style="width:{pct:.0f}%;background:{col}"></div></div>'


def accent_colour(strat: dict) -> str:
    a = strat["action"]
    if a in ("BUY", "BUY_WATCH"):        return "var(--up)"
    if a in ("SELL_EXIT", "SELL_PARTIAL"): return "var(--down)"
    if a == "HOLD":                       return "var(--gold)"
    return "var(--muted)"


def get_tw_session() -> str:
    import pytz
    tw = pytz.timezone("Asia/Taipei")
    now = datetime.now(tw)
    if now.weekday() >= 5:
        return "休市"
    t = now.time()
    if datetime.strptime("08:30", "%H:%M").time() <= t < datetime.strptime("09:00", "%H:%M").time():
        return "盤前"
    if datetime.strptime("09:00", "%H:%M").time() <= t < datetime.strptime("13:30", "%H:%M").time():
        return "交易中"
    if datetime.strptime("13:30", "%H:%M").time() <= t < datetime.strptime("14:00", "%H:%M").time():
        return "盤後"
    return "休市"


def make_tw_chart(df: pd.DataFrame, name: str, strat: dict) -> go.Figure:
    """4-panel chart: K線BB / 量 / KD / MACD"""
    p = df.tail(120)
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.15, 0.20, 0.20],
        vertical_spacing=0.025,
        subplot_titles=(f"{name} — K線 + BB", "成交量", "KD 隨機指標", "MACD"),
    )

    # ── K線 ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=p.index, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"],
        name="K", increasing_fillcolor="#E8192C", increasing_line_color="#E8192C",
        decreasing_fillcolor="#00B050", decreasing_line_color="#00B050",
    ), row=1, col=1)

    for col_name, colour, dash in [
        ("SMA5", "#F5A623", "solid"), ("SMA20", "#3D8EFF", "dot"), ("SMA60", "#9B6DFF", "dot")
    ]:
        fig.add_trace(go.Scatter(x=p.index, y=p[col_name], line=dict(color=colour, width=1.2, dash=dash), name=col_name), row=1, col=1)

    fig.add_trace(go.Scatter(x=p.index, y=p["BB_Upper"], line=dict(color="rgba(255,255,255,0.12)", width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["BB_Lower"], fill="tonexty", fillcolor="rgba(61,142,255,0.04)",
                             line=dict(color="rgba(255,255,255,0.12)", width=1), showlegend=False), row=1, col=1)
    # SL / TP lines
    if strat.get("sl"):
        fig.add_hline(y=strat["sl"], line_dash="dot", line_color="#00B050", row=1, col=1,
                      annotation_text=f"SL {strat['sl']:.2f}", annotation_font_color="#00B050")
    if strat.get("tp"):
        fig.add_hline(y=strat["tp"], line_dash="dot", line_color="#E8192C", row=1, col=1,
                      annotation_text=f"TP {strat['tp']:.2f}", annotation_font_color="#E8192C")

    # ── Volume ────────────────────────────────────────────────────────────
    vol_colours = ["#E8192C" if p["Close"].iloc[i] >= p["Open"].iloc[i] else "#00B050" for i in range(len(p))]
    fig.add_trace(go.Bar(x=p.index, y=p["Volume"], marker_color=vol_colours, name="量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["VOL_SMA20"], line=dict(color="#F5A623", width=1.2), name="均量"), row=2, col=1)

    # ── KD ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=p.index, y=p["K"], line=dict(color="#3D8EFF", width=1.5), name="K"), row=3, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["D"], line=dict(color="#F5A623", width=1.5), name="D"), row=3, col=1)
    for lvl, c in [(80, "rgba(232,25,44,0.25)"), (20, "rgba(0,176,80,0.25)")]:
        fig.add_hline(y=lvl, line_color=c, line_dash="dot", row=3, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────
    hist_cols = ["#E8192C" if v >= 0 else "#00B050" for v in p["Hist"]]
    fig.add_trace(go.Bar(x=p.index, y=p["Hist"], marker_color=hist_cols, name="OSC"), row=4, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["MACD"],   line=dict(color="#3D8EFF", width=1.2), name="DIF"), row=4, col=1)
    fig.add_trace(go.Scatter(x=p.index, y=p["Signal"], line=dict(color="#F5A623", width=1.2), name="MACD"), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", height=560,
        margin=dict(l=0, r=0, t=28, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False, showlegend=False,
        font=dict(family="JetBrains Mono", size=9),
    )
    for i in range(1, 5):
        fig.update_layout(**{f"xaxis{i if i > 1 else ''}": dict(gridcolor="rgba(255,255,255,0.04)"),
                              f"yaxis{i if i > 1 else ''}": dict(gridcolor="rgba(255,255,255,0.04)")})
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# Screening helpers (module-level so they work outside button callbacks)
# ─────────────────────────────────────────────────────────────────────────────
def _sf(val, default=float("nan")) -> float:
    """Safely cast any value to float; returns default on failure or NaN."""
    try:
        f = float(val)
        return f if not pd.isna(f) else default
    except Exception:
        return default

def _build_scan_pool(market_map: dict, pe_lim: float, pb_lim: float,
                     max_price: float, sector_filter: list) -> dict:
    """
    Apply fundamental filters and return candidates as {code: info}.

    Conditions (all must pass — NaN / zero / placeholder values are excluded):
      1. 股價  > 0  AND  <= max_price
      2. PE    > 0  AND  <= pe_lim      (虧損股 PE<0 排除；無效 999 排除)
      3. PB    > 0  AND  <= pb_lim
      4. 產業  in sector_filter          (sector_filter 為空 = 全通過)
    """
    pool = {}
    for code, info in market_map.items():
        price = _sf(info.get("現價"))
        pe    = _sf(info.get("PE"))
        pb    = _sf(info.get("PB"))
        ind   = str(info.get("產業", ""))

        if pd.isna(price) or price <= 0 or price > max_price:   continue
        if pd.isna(pe)    or pe    <= 0 or pe    > pe_lim:      continue
        if pd.isna(pb)    or pb    <= 0 or pb    > pb_lim:      continue
        if sector_filter and ind not in sector_filter:           continue

        pool[code] = info
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# 半導體族群自動掃描 & Telegram 推播
# ─────────────────────────────────────────────────────────────────────────────
def is_semiconductor(industry: str) -> bool:
    """
    判斷 Wespai 產業欄位是否屬於半導體生態系族群。
    SEMI_KEYWORDS 已包含官方產業分類名稱（如「電子零組件」「光電」）
    與業務描述詞（如「IC設計」「晶圓」），一個函數統一比對。
    """
    ind = str(industry).strip()
    if not ind or ind in ("nan", "None", ""):
        return False
    ind_lower = ind.lower()
    return any(kw.lower() in ind_lower for kw in SEMI_KEYWORDS)


def is_tw_trading_day() -> bool:
    """簡易判斷：今日是否為台股交易日（週一至週五；未納入國定假日）"""
    import pytz
    tw_now = datetime.now(pytz.timezone("Asia/Taipei"))
    return tw_now.weekday() < 5      # 0=Mon … 4=Fri


def get_auto_scan_worksheet():
    try:
        gc = get_gsheet_client()
        sh = gc.open(PORTFOLIO_SHEET_TITLE)
        try:
            return sh.worksheet("AutoScan")
        except Exception:
            ws = sh.add_worksheet("AutoScan", rows=500, cols=6)
            ws.append_row(["Date", "Status", "ScanCount", "HitCount", "SentAt", "Note"])
            return ws
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_last_auto_scan_date() -> str:
    """從 AutoScan 工作表讀取最後一次成功掃描的日期 (YYYY-MM-DD)"""
    ws = get_auto_scan_worksheet()
    if ws is None:
        return ""
    try:
        rows = ws.get_all_values()
        for row in reversed(rows[1:]):     # 跳過 header，從最新往回找
            if len(row) >= 2 and row[1] == "OK":
                return row[0]
    except Exception:
        pass
    return ""


def log_auto_scan_result(scan_count: int, hit_count: int, note: str = ""):
    ws = get_auto_scan_worksheet()
    if ws is None:
        return
    import pytz
    today = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d")
    sent_at = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%H:%M:%S")
    try:
        ws.append_row([today, "OK", scan_count, hit_count, sent_at, note])
        get_last_auto_scan_date.clear()
    except Exception:
        pass


def send_tg_message(text: str) -> bool:
    """發送 Telegram 訊息，超過 4096 字自動分段"""
    token, chat_id = _get_tg_creds()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    MAX = 4096
    chunks = [text[i:i+MAX] for i in range(0, len(text), MAX)]
    ok = True
    for chunk in chunks:
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": chunk,
                                          "parse_mode": "Markdown"}, timeout=15)
            if not r.json().get("ok"):
                ok = False
        except Exception:
            ok = False
    return ok


def format_semi_tg_message(candidates: list, scan_count: int, hit_count: int,
                             scan_date: str) -> str:
    """將掃描結果格式化為 Telegram Markdown 訊息"""
    lines = [
        f"📊 *台股半導體族群 · 日收盤自動掃描*",
        f"📅 {scan_date} 18:00 | 市場已收盤",
        f"─────────────────────",
        f"掃描：{scan_count} 檔　│　入選：{hit_count} 檔　│　門檻：技術分 ≥ {SEMI_SCORE_MIN}",
        f"基本面：PE ≤ {SEMI_PE_MAX}、PB ≤ {SEMI_PB_MAX}",
        f"─────────────────────",
        "",
    ]

    if not candidates:
        lines.append("⚠️ 本日無符合條件的強勢標的，建議觀望。")
        sep = "\n"
        return sep.join(lines)

    # 訊號 emoji 對應
    SIG_EMOJI = {
        "BUY":          "🔴 強勢進場",
        "BUY_WATCH":    "🟡 留意機會",
        "HOLD":         "🟠 多頭續抱",
        "SELL_PARTIAL": "🟢 高檔減碼",
        "SELL_EXIT":    "🟢 停損出場",
        "WATCH":        "⚪ 觀望",
    }
    RANK_EMOJI = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]

    for i, c in enumerate(candidates[:15]):          # 最多推播 15 檔
        rank  = RANK_EMOJI[i] if i < len(RANK_EMOJI) else f"{i+1}."
        strat = c["strat"]
        sig   = SIG_EMOJI.get(c["action"], "⚪ 觀望")
        stars = "⭐" * max(1, round(c["score"] / 2))
        reasons_str = "、".join(c["reasons"][:3]) if c["reasons"] else "—"
        sl_str = f"${c['sl']:.1f}" if c.get("sl") else "—"
        tp_str = f"${c['tp']:.1f}" if c.get("tp") else "—"

        block = [
            f"{rank} *{c['代碼']} {c['名稱']}*  {stars}",
            f"   {sig}  |  分數 *{c['score']:.1f}*/10",
            f"   現價 ${c['現價']:.2f}  |  {c['產業']}",
            f"   📈 {reasons_str}",
            f"   🛑 停損 {sl_str}  |  🎯 目標 {tp_str}",
            f"   PE {c['PE']}  |  PB {c['PB']}",
            "",
        ]
        lines.extend(block)

    if len(candidates) > 15:
        lines.append(f"_… 另有 {len(candidates)-15} 檔符合條件，請至 App 查看完整清單_")

    lines += [
        "─────────────────────",
        "⚠️ 本訊息僅供參考，不構成投資建議",
    ]
    sep = "\n"
    return sep.join(lines)


def run_semiconductor_scan(market_map: dict, status_placeholder=None) -> list:
    """
    掃描所有半導體族群標的，回傳依分數排序的候選清單。
    使用 ThreadPoolExecutor 平行抓取歷史資料。
    """
    import concurrent.futures

    # 1. 產業關鍵字過濾（瞬間完成，不再用 PE/PB 濾掉入池標的）
    #    PE/PB 只做資訊顯示，入池只要：是半導體族群 + 股價 > 0
    #    技術分 (SEMI_SCORE_MIN) 才是真正的強勢過濾器
    semi_pool = {
        code: info for code, info in market_map.items()
        if is_semiconductor(info.get("產業", ""))
        and _sf(info.get("現價"), 0) > 0
    }
    scan_list  = list(semi_pool.items())[:SEMI_SCAN_MAX]
    total_scan = max(len(scan_list), 1)
    candidates = []
    done_count = [0]

    def _scan_one(args):
        code, info = args
        try:
            h_df = fetch_stock_history(code)
            if h_df is None:
                return None
            strat = get_strategy(h_df)
            if strat["score"] < SEMI_SCORE_MIN:
                return None
            return {
                "代碼":    code,
                "名稱":    info.get("名稱", ""),
                "產業":    info.get("產業", ""),
                "現價":    _sf(info.get("現價"), 0.0),
                "PE":      round(_sf(info.get("PE"), 0.0), 1),
                "PB":      round(_sf(info.get("PB"), 0.0), 2),
                "score":   strat["score"],
                "action":  strat["action"],
                "sl":      strat["sl"],
                "tp":      strat["tp"],
                "reasons": strat["reasons"],
                "strat":   strat,
                "df":      h_df,
            }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_scan_one, item): item[0] for item in scan_list}
        for fut in concurrent.futures.as_completed(futures):
            done_count[0] += 1
            if status_placeholder:
                pct = done_count[0] / total_scan
                status_placeholder.progress(
                    pct,
                    text=f"掃描 {done_count[0]}/{total_scan} ({pct*100:.0f}%) — "
                         f"已發現 {len(candidates)} 個強勢標的",
                )
            result = fut.result()
            if result:
                candidates.append(result)

    candidates.sort(key=lambda x: -x["score"])
    return candidates, total_scan


def check_and_trigger_auto_scan():
    """
    在頁面載入時呼叫：檢查是否該執行當日自動掃描。
    條件：台股交易日 + 台灣時間 18:00–23:59 + 今日尚未掃描
    """
    import pytz
    tw_now = datetime.now(pytz.timezone("Asia/Taipei"))
    if not is_tw_trading_day():
        return
    if tw_now.hour < 18:
        return

    today_str = tw_now.strftime("%Y-%m-%d")
    last_date  = get_last_auto_scan_date()
    if last_date == today_str:
        return                  # 今日已掃描，跳過

    # 避免同一個 session 重複執行
    if st.session_state.get("auto_scan_done_today") == today_str:
        return

    st.session_state.auto_scan_done_today = today_str

    # ── 後台執行（避免阻塞 UI）─────────────────────────────────────────────
    import threading
    def _bg():
        try:
            candidates, scan_n = run_semiconductor_scan(MARKET_MAP)
            hit_n    = len(candidates)
            msg      = format_semi_tg_message(candidates, scan_n, hit_n, today_str)
            ok       = send_tg_message(msg)
            log_auto_scan_result(scan_n, hit_n, "auto" + ("_ok" if ok else "_tg_fail"))
        except Exception as e:
            log_auto_scan_result(0, 0, f"error:{e}")

    threading.Thread(target=_bg, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "df_portfolio" not in st.session_state:
    st.session_state.df_portfolio = load_portfolio()
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "diag_plot" not in st.session_state:
    st.session_state.diag_plot = None
if "tab1_chart_sym" not in st.session_state:
    st.session_state.tab1_chart_sym = ""
if "tab1_chart_data" not in st.session_state:
    st.session_state.tab1_chart_data = None
if "tab2_chart_sym" not in st.session_state:
    st.session_state.tab2_chart_sym = ""
if "tab2_chart_data" not in st.session_state:
    st.session_state.tab2_chart_data = None


# ─────────────────────────────────────────────────────────────────────────────
# Auto-scan trigger (每次頁面載入時靜默檢查是否需要執行當日掃描)
# ─────────────────────────────────────────────────────────────────────────────
check_and_trigger_auto_scan()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
session_now = get_tw_session()
session_colours = {
    "交易中": "badge-up", "盤前": "badge-gold", "盤後": "badge-blue", "休市": "badge-flat"
}

st.markdown(f"""
<div class="tw-header">
  <div class="tw-logo">台股<span>戰情中心</span></div>
  <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
    <span class="badge {session_colours.get(session_now,'badge-flat')}">{session_now}</span>
    <span class="badge badge-flat" style="font-family:'JetBrains Mono'">{datetime.now().strftime('%m/%d %H:%M')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level metrics
# ─────────────────────────────────────────────────────────────────────────────
portfolio = st.session_state.df_portfolio
total_mv = total_cost = 0.0
if not portfolio.empty:
    for _, r in portfolio.iterrows():
        m = MARKET_MAP.get(r["Symbol"])
        if m:
            total_mv   += m["現價"] * r["Shares"]
            total_cost += r["Cost"] * r["Shares"]

total_pl     = total_mv - total_cost
pl_pct       = (total_pl / total_cost * 100) if total_cost > 0 else 0
holding_cnt  = len(portfolio)

c1, c2, c3, c4 = st.columns(4)
c1.metric("持倉市值", f"${total_mv:,.0f}")
c2.metric("未實現損益", f"${total_pl:,.0f}", f"{pl_pct:+.2f}%")
c3.metric("投入成本", f"${total_cost:,.0f}")
c4.metric("持倉檔數", f"{holding_cnt} 檔")

st.markdown("<hr class='qdiv'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 庫存動態", "💰 潛力快篩", "🔍 個股診斷", "📝 庫存管理", "⚙️ 系統"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Portfolio Monitor
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if portfolio.empty:
        st.info("📭 庫存為空。請至「庫存管理」新增持股。")
    else:
        details = []
        for _, r in portfolio.iterrows():
            m = MARKET_MAP.get(r["Symbol"])
            if not m:
                continue
            cp = m["現價"]
            mv = cp * r["Shares"]
            cv = r["Cost"] * r["Shares"]
            h_df = fetch_stock_history(r["Symbol"])
            strat = get_strategy(h_df, r["Shares"], r["Cost"]) if h_df is not None else _default_strat("連線失敗", "#5A6072")
            details.append({"r": r, "m": m, "cp": cp, "mv": mv, "cv": cv,
                             "strat": strat, "df": h_df})

        # Sort: exits first, then by score desc
        details.sort(key=lambda x: (0 if "SELL" in x["strat"]["action"] else 1, -x["strat"]["score"]))

        # Pie chart
        with st.expander("資產配置圓餅圖", expanded=False):
            pie_labels  = [d["r"]["Name"] for d in details]
            pie_values  = [d["mv"] for d in details]
            pie_colours = ["#E8192C", "#F5A623", "#3D8EFF", "#9B6DFF", "#00B050",
                           "#FF8C42", "#00D4FF", "#FF6B9D"][:len(details)]
            pie_fig = go.Figure(go.Pie(
                labels=pie_labels, values=pie_values, hole=0.55,
                marker=dict(colors=pie_colours, line=dict(color="#0A0C12", width=2)),
                textfont=dict(family="Noto Sans TC", size=11),
            ))
            pie_fig.update_layout(
                template="plotly_dark", height=240, margin=dict(l=0, r=0, t=5, b=0),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            )
            st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="qsec">持倉明細 (依訊號優先排序)</div>', unsafe_allow_html=True)

        for item in details:
            r, m, cp, strat = item["r"], item["m"], item["cp"], item["strat"]
            p_pct    = (cp - r["Cost"]) / r["Cost"] * 100 if r["Cost"] > 0 else 0
            diff_val = (cp - r["Cost"]) * r["Shares"]
            sl_str   = f"${strat['sl']:.2f}" if strat.get("sl") else "—"
            tp_str   = f"${strat['tp']:.2f}" if strat.get("tp") else "—"
            a = strat["action"]

            # Action strip
            if a in ("BUY", "BUY_WATCH"):
                action_line = (f"<span class='sig-buy'>🛒 建議買入 {strat['suggest_lots']} 張"
                               f"，停損 {sl_str}，目標 {tp_str}</span>")
            elif a == "SELL_EXIT":
                action_line = f"<span class='sig-sell'>⚠️ 跌破停損！建議出清 {strat['suggest_lots']} 張</span>"
            elif a == "SELL_PARTIAL":
                action_line = f"<span class='sig-hold'>💰 高檔過熱，建議減碼 {strat['suggest_lots']} 張獲利了結</span>"
            elif a == "HOLD":
                action_line = f"<span class='sig-hold'>🛡️ 持股續抱，跌破 {sl_str} 再出場</span>"
            else:
                action_line = "<span class='sig-watch'>☕ 觀望中，無明確動作</span>"

            kd_str = f"K{strat.get('score', 0):.0f}" if strat["action"] != "WATCH" else ""
            vol_r = float(item["df"].iloc[-1].get("VOL_Ratio", 1.0)) if item["df"] is not None else 1.0

            st.markdown(f"""
<div class="sc">
  <div class="sc-accent" style="background:{accent_colour(strat)}"></div>
  <div class="sc-top">
    <div>
      <div class="sc-name">{r['Name']} <span class="sc-code">{r['Symbol']}</span></div>
      <div style="margin-top:3px;font-size:0.7rem;color:var(--muted)">{m['產業']} · PE {m['PE']} · PB {m['PB']}</div>
    </div>
    <div style="text-align:right">
      {signal_badge_html(strat)}
      <div style="margin-top:4px;font-family:var(--mono);font-size:0.7rem;color:var(--muted)">分數 {strat['score']:.1f}/10</div>
    </div>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <span class="sc-price" style="color:{pl_colour(diff_val)}">${cp:.2f}</span>
    <span style="font-family:var(--mono);font-size:0.88rem;color:{pl_colour(p_pct)};font-weight:700">
      {'▲' if p_pct > 0 else '▼' if p_pct < 0 else '—'}{abs(p_pct):.2f}%
    </span>
  </div>
  <div style="font-family:var(--mono);font-size:0.75rem;color:var(--muted);margin:2px 0 6px">
    損益 {'▲' if diff_val >= 0 else '▼'}${abs(diff_val):,.0f} | 成本 ${r['Cost']:.2f} | {r['Shares']:,.0f}股
  </div>
  {score_bar_html(strat['score'])}
  <div class="sc-grid">
    <div><span class="sc-kv-label">停損</span><br><span class="sc-kv-value" style="color:var(--down)">{sl_str}</span></div>
    <div><span class="sc-kv-label">目標</span><br><span class="sc-kv-value" style="color:var(--up)">{tp_str}</span></div>
    <div><span class="sc-kv-label">量比</span><br><span class="sc-kv-value" style="color:{'var(--up)' if vol_r>=1.5 else 'var(--muted)'}">{vol_r:.1f}x</span></div>
    <div><span class="sc-kv-label">分析依據</span><br><span class="sc-kv-value" style="font-size:0.68rem">{'、'.join(strat['reasons'][:2]) if strat['reasons'] else '—'}</span></div>
  </div>
  <div class="sc-action">{action_line}</div>
</div>
""", unsafe_allow_html=True)

            # Chart toggle — render inline in Tab1 (no rerun needed)
            if item["df"] is not None:
                if st.button(f"📊 技術圖表 {r['Symbol']}", key=f"chart_{r['Symbol']}", use_container_width=True):
                    # Toggle: click same ticker again to close
                    current = st.session_state.get("tab1_chart_sym", "")
                    if current == r["Symbol"]:
                        st.session_state.tab1_chart_sym  = ""
                        st.session_state.tab1_chart_data = None
                    else:
                        st.session_state.tab1_chart_sym  = r["Symbol"]
                        st.session_state.tab1_chart_data = (item["df"], r["Name"], strat)

            # ── Inline chart (renders immediately, same tab, no rerun) ──────
            if st.session_state.get("tab1_chart_sym") == r["Symbol"]:
                t1_data = st.session_state.get("tab1_chart_data")
                if t1_data:
                    t1_df, t1_name, t1_strat = t1_data
                    t1_last = t1_df.iloc[-1]
                    t1_k  = float(t1_last["K"])
                    t1_d  = float(t1_last["D"])
                    t1_rsi = float(t1_last["RSI"])
                    t1_vol = float(t1_last.get("VOL_Ratio", 1.0))

                    ta, tb, tc, td = st.columns(4)
                    ta.metric("現價",    f"${t1_df['Close'].iloc[-1]:.2f}")
                    tb.metric("K / D",   f"{t1_k:.0f} / {t1_d:.0f}")
                    tc.metric("RSI",     f"{t1_rsi:.1f}")
                    td.metric("量比",    f"{t1_vol:.1f}x")

                    st.markdown(
                        f'<div class="sc-action" style="border-left:3px solid {t1_strat["color"]};margin-bottom:10px">'
                        f'{t1_strat["html"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if t1_strat.get("warnings"):
                        st.warning("⚠️ " + "；".join(t1_strat["warnings"]))

                    st.plotly_chart(
                        make_tw_chart(t1_df, t1_name, t1_strat),
                        use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": False},
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Smart Screener
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="qsec">多因子篩選條件</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    pe_lim  = f1.number_input("PE 上限", value=18.0, step=1.0)
    pb_lim  = f2.number_input("PB 上限", value=1.5, step=0.1)
    min_score = f3.slider("最低技術分 (0-10)", 0.0, 10.0, 4.0, 0.5)

    f4, f5 = st.columns(2)
    sector_options = sorted(list(set(
        str(v["產業"]) for v in MARKET_MAP.values()
        if v.get("產業") and str(v.get("產業")) not in ("nan", "", "None", "NaN")
    )))
    sector_filter  = f4.multiselect("篩選產業 (空=全部)", options=sector_options)
    max_price      = f5.number_input("股價上限 (TWD)", value=500.0, min_value=1.0, step=10.0)

    # ── 即時顯示通過基本面篩選的數量 ─────────────────────────────────────
    _pool_preview = _build_scan_pool(MARKET_MAP, pe_lim, pb_lim, max_price, sector_filter)
    pool_n = len(_pool_preview)
    est_min_lo = max(1, pool_n // 10 // 60)       # 10 workers, ~0.6s/stock
    est_min_hi = max(1, pool_n * 6 // 10 // 60)   # pessimistic 6s/stock
    st.caption(
        f"📋 通過基本面條件（PE≤{pe_lim}、PB≤{pb_lim}、股價≤{max_price}、"
        f"{'產業：' + '、'.join(sector_filter) if sector_filter else '全產業'}）："
        f" **{pool_n} 檔** 將接受技術分析，預估掃描時間 "
        f"**{est_min_lo}–{est_min_hi} 分鐘**（平行 10 worker）。"
    )

    sa, sb = st.columns(2)
    scan_all   = sa.checkbox("✅ 全部掃描（不限上限）", value=False)
    scan_cap   = sb.number_input(
        "或設定上限 (掃描前 N 檔)",
        min_value=50, max_value=2000, value=200, step=50,
        disabled=scan_all,
    )
    max_workers_ui = st.slider("平行抓取 Worker 數（越多越快，但較易觸發 API 限流）", 3, 20, 10)

    if st.button("🔍 啟動大盤掃描", use_container_width=True):
        import concurrent.futures

        scan_list  = list(_pool_preview.items()) if scan_all else list(_pool_preview.items())[:int(scan_cap)]
        total_scan = len(scan_list)
        candidates = []

        progress  = st.progress(0)
        prog_text = st.empty()
        done_count = [0]          # mutable counter for thread-safe increment

        def _scan_one(args):
            code, info = args
            h_df = fetch_stock_history(code)
            if h_df is None:
                return None
            strat = get_strategy(h_df)
            if strat["score"] < min_score:
                return None
            return {
                "代碼":    code,
                "名稱":    info.get("名稱", ""),
                "產業":    info.get("產業", ""),
                "現價":    _sf(info.get("現價"), 0.0),
                "PE":      round(_sf(info.get("PE"), 0.0), 1),
                "PB":      round(_sf(info.get("PB"), 0.0), 2),
                "score":   strat["score"],
                "action":  strat["action"],
                "sl":      strat["sl"],
                "tp":      strat["tp"],
                "reasons": strat["reasons"],
                "strat":   strat,
                "df":      h_df,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_ui) as executor:
            futures = {executor.submit(_scan_one, item): item[0] for item in scan_list}
            for fut in concurrent.futures.as_completed(futures):
                done_count[0] += 1
                pct = done_count[0] / max(total_scan, 1)
                progress.progress(pct)
                prog_text.caption(
                    f"掃描進度 {done_count[0]}/{total_scan} "
                    f"({pct*100:.0f}%) — 已發現 {len(candidates)} 個符合標的"
                )
                result = fut.result()
                if result:
                    candidates.append(result)

        progress.empty()
        prog_text.empty()

        candidates.sort(key=lambda x: -x["score"])
        st.session_state.scan_results    = candidates
        st.session_state.tab2_chart_sym  = ""
        st.session_state.tab2_chart_data = None

    res = st.session_state.scan_results
    if res is not None:
        buys  = [c for c in res if "BUY" in c["action"]]
        holds = [c for c in res if c["action"] == "HOLD"]

        st.markdown(f'<div class="qsec">🟥 買進機會 ({len(buys)} 檔)</div>', unsafe_allow_html=True)
        for i, c in enumerate(buys[:20], 1):
            reason_short = "、".join(c["reasons"][:3]) if c["reasons"] else "—"
            st.markdown(f"""
<div class="sk-card" style="border-color:rgba(232,25,44,0.3)">
  <div class="sk-rank">{i}</div>
  <div style="flex:1">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
      <div>
        <div class="sk-ticker">{c['代碼']} {c['名稱']} <span style="font-size:0.65rem;color:var(--muted)">{c['產業']}</span></div>
        <div class="sk-reason">{reason_short}</div>
      </div>
      <div style="text-align:right;min-width:80px">
        <div class="sk-score">{c['score']:.1f}</div>
        <div style="font-family:var(--mono);font-size:0.72rem;color:var(--muted)">PE {c['PE']:.1f} PB {c['PB']:.1f}</div>
      </div>
    </div>
    {score_bar_html(c['score'])}
  </div>
</div>""", unsafe_allow_html=True)
            if st.button(f"📊 診斷 {c['代碼']}", key=f"sc_diag_{c['代碼']}", use_container_width=True):
                cur = st.session_state.get("tab2_chart_sym", "")
                if cur == c["代碼"]:
                    st.session_state.tab2_chart_sym  = ""
                    st.session_state.tab2_chart_data = None
                else:
                    st.session_state.tab2_chart_sym  = c["代碼"]
                    st.session_state.tab2_chart_data = (c["df"], c["名稱"], c["strat"])

            # ── Inline chart for this candidate ────────────────────────────
            if st.session_state.get("tab2_chart_sym") == c["代碼"]:
                t2d = st.session_state.get("tab2_chart_data")
                if t2d:
                    t2_df, t2_name, t2_strat = t2d
                    t2_last = t2_df.iloc[-1]
                    ta, tb, tc, td = st.columns(4)
                    ta.metric("現價",   f"${t2_df['Close'].iloc[-1]:.2f}")
                    tb.metric("K / D",  f"{float(t2_last['K']):.0f} / {float(t2_last['D']):.0f}")
                    tc.metric("RSI",    f"{float(t2_last['RSI']):.1f}")
                    td.metric("量比",   f"{float(t2_last.get('VOL_Ratio', 1.0)):.1f}x")
                    st.markdown(
                        f'<div class="sc-action" style="border-left:3px solid {t2_strat["color"]};margin-bottom:10px">'
                        f'{t2_strat["html"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if t2_strat.get("warnings"):
                        st.warning("⚠️ " + "；".join(t2_strat["warnings"]))
                    st.plotly_chart(
                        make_tw_chart(t2_df, t2_name, t2_strat),
                        use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": False},
                    )

        if holds:
            st.markdown(f'<div class="qsec">🟡 續抱觀察 ({len(holds)} 檔)</div>', unsafe_allow_html=True)
            for c in holds[:10]:
                st.markdown(f"""
<div class="sk-card">
  <div class="sk-rank">＝</div>
  <div style="flex:1">
    <div class="sk-ticker">{c['代碼']} {c['名稱']}</div>
    <div class="sk-reason">{'、'.join(c['reasons'][:2])}</div>
    {score_bar_html(c['score'])}
  </div>
  <div class="sk-score" style="color:var(--gold)">{c['score']:.1f}</div>
</div>""", unsafe_allow_html=True)

    elif res is None:
        st.info("設定篩選條件後點擊「啟動大盤掃描」。")
    else:
        st.warning("無符合條件的標的，請放寬篩選參數。")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Diagnosis (single stock deep-dive)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="qsec">個股深度診斷</div>', unsafe_allow_html=True)

    selection = st.selectbox("搜尋台股標的", options=["請選擇..."] + STOCK_OPTIONS, label_visibility="collapsed")

    if st.button("🚀 執行深度診斷", use_container_width=True) and selection != "請選擇...":
        parts = selection.split(" ")
        code, name = parts[0], parts[1] if len(parts) > 1 else parts[0]
        with st.spinner(f"分析 {name}({code}) … FinMind → YFinance"):
            df = fetch_stock_history(code)
            if df is not None:
                strat = get_strategy(df)
                st.session_state.diag_plot = (df, name, strat)
            else:
                st.error(f"❌ 無法取得 {name}({code}) 資料，請稍後再試。")

# Global chart render lives inside the tab3 block above.
with tab3:
    plot_data = st.session_state.get("diag_plot")
    if plot_data:
        p_df, p_name, p_strat = plot_data

        # Summary strip
        last_row = p_df.iloc[-1]
        k_val  = float(last_row["K"])
        d_val  = float(last_row["D"])
        rsi_v  = float(last_row["RSI"])
        vol_r  = float(last_row.get("VOL_Ratio", 1.0))
        sc     = p_strat["score"]
        sc_col = "var(--up)" if sc >= 5 else "var(--gold)" if sc >= 3 else "var(--muted)"

        st.markdown(f"""
<div class="sc" style="margin-bottom:14px;">
  <div class="sc-top">
    <div>
      <div class="sc-name">{p_name}</div>
      <div style="margin-top:3px">{signal_badge_html(p_strat)}</div>
    </div>
    <div style="text-align:right">
      <div style="font-family:var(--mono);font-size:1.8rem;font-weight:700;color:{sc_col}">{sc:.1f}</div>
      <div style="font-size:0.65rem;color:var(--muted)">/ 10 分</div>
    </div>
  </div>
  {score_bar_html(sc)}
</div>
""", unsafe_allow_html=True)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("現價",     f"${p_df['Close'].iloc[-1]:.2f}")
        k2.metric("RSI(14)",  f"{rsi_v:.1f}")
        k3.metric("K / D",    f"{k_val:.0f} / {d_val:.0f}")
        k4.metric("量比",      f"{vol_r:.1f}x")
        k5.metric("停損防線", f"${p_strat['sl']:.2f}" if p_strat.get("sl") else "—")
        k6.metric("目標價",   f"${p_strat['tp']:.2f}" if p_strat.get("tp") else "—")

        st.markdown(f"""
<div class="sc-action" style="border-left:3px solid {p_strat['color']};margin-bottom:12px">
  {p_strat['html']}
</div>
""", unsafe_allow_html=True)

        if p_strat.get("warnings"):
            st.warning("⚠️ 風險注意：" + "；".join(p_strat["warnings"]))

        fig = make_tw_chart(p_df, p_name, p_strat)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

        if st.button("✖ 清除圖表", use_container_width=True):
            st.session_state.diag_plot = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Portfolio Management
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="qsec">新增交易紀錄</div>', unsafe_allow_html=True)

    with st.expander("➕ 新增持股", expanded=True):
        c1, c2, c3 = st.columns(3)
        new_sel    = c1.selectbox("搜尋標的", options=["請選擇..."] + STOCK_OPTIONS, key="mgmt_sel")
        new_cost   = c2.number_input("買入單價 (TWD)", min_value=0.01, step=0.1, key="mgmt_cost")
        new_shares = c3.number_input("買入股數 (1張=1000)", min_value=1, step=1000, value=1000, key="mgmt_shares")

        if st.button("＋ 暫存至庫存表", use_container_width=True):
            if new_sel != "請選擇...":
                n_code = new_sel.split(" ")[0]
                n_name = new_sel.split(" ")[1] if len(new_sel.split(" ")) > 1 else n_code
                new_row = {"Symbol": n_code, "Name": n_name, "Cost": new_cost, "Shares": new_shares, "Note": ""}
                st.session_state.df_portfolio = pd.concat(
                    [st.session_state.df_portfolio, pd.DataFrame([new_row])], ignore_index=True
                )
                st.success(f"✅ 已暫存 {n_name}。請記得點擊「儲存至 Google Sheets」。")
            else:
                st.warning("請先選擇標的。")

    st.markdown('<div class="qsec">庫存清單編輯器</div>', unsafe_allow_html=True)
    edited_df = st.data_editor(
        st.session_state.df_portfolio, hide_index=True, use_container_width=True, key="portfolio_editor"
    )

    c_save, c_reload = st.columns(2)
    if c_save.button("💾 儲存至 Google Sheets", use_container_width=True, type="primary"):
        final_df = edited_df[edited_df["Shares"] > 0].copy()
        with st.spinner("同步至雲端 …"):
            try:
                gc = get_gsheet_client()
                ws = gc.open(PORTFOLIO_SHEET_TITLE).sheet1
                ws.clear()
                ws.update("A1", [final_df.columns.tolist()] + final_df.values.tolist())
                st.session_state.df_portfolio = final_df
                st.cache_data.clear()
                st.success("🎉 同步成功！")
                time.sleep(0.8)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 寫入失敗：{e}")

    if c_reload.button("🔄 重新載入庫存", use_container_width=True):
        st.cache_data.clear()
        st.session_state.df_portfolio = load_portfolio()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — System & Settings
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="qsec">系統控制</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    if col_a.button("🔄 強制刷新全部快取", use_container_width=True):
        st.cache_data.clear()
        st.session_state.scan_results    = None
        st.session_state.diag_plot       = None
        st.session_state.tab1_chart_sym  = ""
        st.session_state.tab1_chart_data = None
        st.rerun()

    if col_b.button("🗑️ 清除診斷圖表", use_container_width=True):
        st.session_state.diag_plot = None
        st.rerun()

    st.markdown('<div class="qsec">策略說明</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sc" style="font-size:0.82rem;line-height:1.8">
  <div style="font-weight:900;margin-bottom:8px;font-family:var(--sans)">多因子評分系統 (0–10 分)</div>
  <table width="100%" cellpadding="4">
    <tr><td style="color:var(--muted);width:40%">均線多頭 (MA5/20/60)</td><td>最高 +2.5</td></tr>
    <tr><td style="color:var(--muted)">年線 SMA240</td><td>±1.0</td></tr>
    <tr><td style="color:var(--muted)">KD 低檔黃金交叉</td><td>最高 +2.5</td></tr>
    <tr><td style="color:var(--muted)">RSI 超賣翻揚</td><td>最高 +1.5</td></tr>
    <tr><td style="color:var(--muted)">BB 壓縮放量突破</td><td>+2.5</td></tr>
    <tr><td style="color:var(--muted)">成交量比 ≥2x</td><td>+1.0</td></tr>
    <tr><td style="color:var(--muted)">MACD 翻多</td><td>+0.5</td></tr>
    <tr><td style="color:var(--muted)">接近年高</td><td>+0.5</td></tr>
  </table>
  <div style="margin-top:10px;color:var(--muted);font-size:0.75rem">
    ⚠️ 本系統僅供輔助參考，不構成投資建議。台股停損以 2×ATR 或季線-2% 為基準。
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="qsec">資料來源狀態</div>', unsafe_allow_html=True)
    sa, sb = st.columns(2)
    sa.metric("市場報價標的數", len(MARKET_MAP))
    sb.metric("持倉檔數", len(portfolio))

    # ── 半導體自動掃描控制台 ────────────────────────────────────────────────
    st.markdown('<hr class="qdiv">', unsafe_allow_html=True)
    st.markdown('<div class="qsec">🤖 半導體族群自動掃描推播</div>', unsafe_allow_html=True)

    import pytz as _pytz
    _tw_now   = datetime.now(_pytz.timezone("Asia/Taipei"))
    _today_s  = _tw_now.strftime("%Y-%m-%d")
    _last_s   = get_last_auto_scan_date()
    _is_today = (_last_s == _today_s)
    _is_tday  = is_tw_trading_day()
    _after18  = _tw_now.hour >= 18

    # Status cards
    # ── 三層漏斗計數（關鍵字→有報價→估值合理） ──────────────────────────
    _semi_kw_n   = sum(1 for v in MARKET_MAP.values()
                       if is_semiconductor(v.get("產業", "")))
    _semi_price_n = sum(1 for v in MARKET_MAP.values()
                        if is_semiconductor(v.get("產業", ""))
                        and _sf(v.get("現價"), 0) > 0)
    _semi_val_n  = sum(1 for v in MARKET_MAP.values()
                       if is_semiconductor(v.get("產業", ""))
                       and _sf(v.get("現價"), 0) > 0
                       and 0 < _sf(v.get("PE"), 0) <= SEMI_PE_MAX
                       and 0 < _sf(v.get("PB"), 0) <= SEMI_PB_MAX)

    # ── 產業分布（Top 10）──────────────────────────────────────────────────
    from collections import Counter as _Counter
    _ind_counter = _Counter(
        str(v.get("產業", "未分類"))
        for v in MARKET_MAP.values()
        if is_semiconductor(v.get("產業", ""))
    )

    st.markdown(f"""
<div class="sc" style="font-size:0.82rem;line-height:2.2;">
  <div style="font-weight:900;margin-bottom:10px;font-family:var(--sans)">
    族群篩選漏斗
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 12px;margin-bottom:12px;">
    <div>
      <span class="sc-kv-label">① 關鍵字命中</span><br>
      <span class="sc-kv-value" style="font-size:1.1rem;color:var(--text)">{_semi_kw_n} 檔</span><br>
      <span style="font-size:0.65rem;color:var(--muted)">半導體生態系族群</span>
    </div>
    <div>
      <span class="sc-kv-label">② 有即時報價</span><br>
      <span class="sc-kv-value" style="font-size:1.1rem;color:var(--gold)">{_semi_price_n} 檔</span><br>
      <span style="font-size:0.65rem;color:var(--muted)">Wespai 有收盤價</span>
    </div>
    <div>
      <span class="sc-kv-label">③ 估值合理</span><br>
      <span class="sc-kv-value" style="font-size:1.1rem;color:var(--down)">{_semi_val_n} 檔</span><br>
      <span style="font-size:0.65rem;color:var(--muted)">PE≤{SEMI_PE_MAX} PB≤{SEMI_PB_MAX}</span>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 16px;">
    <div><span class="sc-kv-label">實際掃描範圍</span><br>
         <span class="sc-kv-value">全部 {_semi_price_n} 檔（不限 PE/PB）</span></div>
    <div><span class="sc-kv-label">技術分門檻</span><br>
         <span class="sc-kv-value">≥ {SEMI_SCORE_MIN}/10（強勢股）</span></div>
    <div><span class="sc-kv-label">排程時間</span><br>
         <span class="sc-kv-value">每交易日 18:00（台灣）</span></div>
    <div><span class="sc-kv-label">今日狀態</span><br>
         <span class="sc-kv-value" style="color:{'var(--down)' if _is_today else 'var(--muted)'}">
           {'✅ 已發送 (' + _last_s + ')' if _is_today else '⏳ 尚未執行'}</span></div>
    <div><span class="sc-kv-label">目前台灣時間</span><br>
         <span class="sc-kv-value">{_tw_now.strftime('%H:%M')}
           {'（交易日）' if _is_tday else '（非交易日）'}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 產業分布明細（可展開） ──────────────────────────────────────────────
    with st.expander(f"🔬 產業分布明細（共 {len(_ind_counter)} 種分類）", expanded=False):
        _ind_df = pd.DataFrame(
            [{"產業分類": k, "標的數": v} for k, v in _ind_counter.most_common()],
        )
        st.dataframe(_ind_df, use_container_width=True, hide_index=True)
        st.caption(
            "若某產業分類標的數過少（0–5 檔），代表 Wespai 使用了不同分類名稱，"
            "可截圖回報以便加入新關鍵字。"
        )

    # 自動觸發說明
    if _is_tday and _after18 and not _is_today:
        st.info("⏱️ 自動掃描將在背景執行（頁面載入時已觸發），完成後 Telegram 將收到推播。")
    elif not _is_tday:
        st.caption("今日為非交易日，自動掃描不會觸發。")
    elif not _after18:
        st.caption(f"尚未到 18:00（目前 {_tw_now.strftime('%H:%M')}），收盤後自動掃描將在頁面載入時觸發。")

    # 手動立即掃描
    st.markdown('<div class="qsec">手動立即掃描</div>', unsafe_allow_html=True)
    _m1, _m2 = st.columns(2)
    _force    = _m1.checkbox("強制重新掃描（忽略今日已發送記錄）", value=False)
    _dry_run  = _m2.checkbox("Dry Run（掃描但不發 TG）", value=False)

    if st.button("🚀 立即執行半導體掃描推播", use_container_width=True):
        if _is_today and not _force:
            st.warning("今日已發送過推播。勾選「強制重新掃描」可重新執行。")
        else:
            _prog = st.progress(0, text="準備掃描半導體族群 …")
            with st.spinner("平行掃描中（12 workers）…"):
                _cands, _scan_n = run_semiconductor_scan(MARKET_MAP, _prog)
            _prog.empty()
            _hit_n = len(_cands)
            st.success(f"掃描完成！共 {_scan_n} 檔半導體標的，{_hit_n} 檔達到技術門檻。")

            _msg = format_semi_tg_message(_cands, _scan_n, _hit_n, _today_s)

            # Preview
            with st.expander("📨 Telegram 訊息預覽", expanded=True):
                st.code(_msg, language=None)

            if not _dry_run:
                _ok = send_tg_message(_msg)
                if _ok:
                    log_auto_scan_result(_scan_n, _hit_n, "manual_ok")
                    st.success("✅ Telegram 推播成功！")
                else:
                    st.error("❌ Telegram 推播失敗，請確認 TG_TOKEN / TG_CHAT_ID 是否正確設定於 Secrets。")
            else:
                st.info("Dry Run 模式：未實際發送 Telegram。")

            # Show table
            if _cands:
                st.markdown('<div class="qsec">掃描結果（依分數排序）</div>', unsafe_allow_html=True)
                _df_show = pd.DataFrame([{
                    "代碼": c["代碼"], "名稱": c["名稱"], "產業": c["產業"],
                    "現價": c["現價"], "PE": c["PE"], "PB": c["PB"],
                    "技術分": c["score"], "訊號": c["action"],
                    "停損": round(c["sl"], 2) if c.get("sl") else None,
                    "目標": round(c["tp"], 2) if c.get("tp") else None,
                } for c in _cands])
                st.dataframe(_df_show, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 6px;font-size:0.62rem;color:var(--muted);font-family:'JetBrains Mono'">
  台股戰情中心 V15 Pro · {datetime.now().strftime('%Y/%m/%d %H:%M')} · 紅漲綠跌 · 僅供參考勿作投資依據
</div>
""", unsafe_allow_html=True)

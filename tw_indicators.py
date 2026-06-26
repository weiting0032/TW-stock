"""技術指標計算 + K線型態識別（純計算，無 Streamlit 依賴）"""
import pandas as pd


def calculate_indicators(df: pd.DataFrame):
    if df is None or len(df) < 60:
        return None
    df = df.copy()

    df["SMA5"]   = df["Close"].rolling(5).mean()
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA60"]  = df["Close"].rolling(60).mean()
    df["SMA240"] = df["Close"].rolling(240, min_periods=60).mean()

    std20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["SMA20"] + 1e-9)

    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"]  - df["Close"].shift(1)).abs()
    df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    low_n  = df["Low"].rolling(9).min()
    high_n = df["High"].rolling(9).max()
    df["RSV"] = (df["Close"] - low_n) / (high_n - low_n + 1e-9) * 100
    df["K"]   = df["RSV"].ewm(com=2, adjust=False).mean()
    df["D"]   = df["K"].ewm(com=2, adjust=False).mean()
    df["J"]   = 3 * df["K"] - 2 * df["D"]

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Hist"]   = df["MACD"] - df["Signal"]

    df["VOL_SMA20"] = df["Volume"].rolling(20).mean()
    df["VOL_Ratio"] = df["Volume"] / (df["VOL_SMA20"] + 1e-9)

    direction = df["Close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["OBV"] = (direction * df["Volume"]).cumsum()

    df["High52W"] = df["High"].rolling(252, min_periods=60).max()
    df["Low52W"]  = df["Low"].rolling(252, min_periods=60).min()

    return df.dropna(subset=["SMA20", "SMA60", "K", "D", "RSI"]).copy()


def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """識別常見 K 線型態，回傳 {代碼: 中文說明} 字典"""
    if df is None or len(df) < 3:
        return {}

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]

    o,  h,  l,  c  = float(last["Open"]),  float(last["High"]),  float(last["Low"]),  float(last["Close"])
    o1, h1, l1, c1 = float(prev["Open"]),  float(prev["High"]),  float(prev["Low"]),  float(prev["Close"])
    o2, h2, l2, c2 = float(prev2["Open"]), float(prev2["High"]), float(prev2["Low"]), float(prev2["Close"])

    body       = abs(c - o)
    body_prev  = abs(c1 - o1)
    body_prev2 = abs(c2 - o2)
    rng        = h - l + 1e-9

    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)

    patterns = {}

    if body <= 0.1 * rng:
        patterns["Doji"] = "⬜ 十字星（趨勢猶豫，留意轉折）"

    if lower_shadow >= 2 * body and upper_shadow <= 0.15 * rng and c > l + rng * 0.4:
        patterns["Hammer"] = "🔨 錘子線（底部反轉訊號）"

    if upper_shadow >= 2 * body and lower_shadow <= 0.15 * rng and c < h - rng * 0.4:
        patterns["ShootingStar"] = "💫 流星線（高檔反轉訊號）"

    if c1 < o1 and c > o and c >= o1 and o <= c1:
        patterns["BullEngulfing"] = "🟩 多頭吞噬（強力底部反轉）"

    if c1 > o1 and c < o and c <= o1 and o >= c1:
        patterns["BearEngulfing"] = "🟥 空頭吞噬（高檔反轉向下）"

    if (c2 < o2 and body_prev <= 0.35 * body_prev2 and c > o and c > (o2 + c2) / 2):
        patterns["MorningStar"] = "🌅 晨星型態（三日底部反轉）"

    if (c2 > o2 and body_prev <= 0.35 * body_prev2 and c < o and c < (o2 + c2) / 2):
        patterns["EveningStar"] = "🌆 黃昏之星（三日頂部反轉）"

    if upper_shadow <= 0.02 * rng and lower_shadow <= 0.02 * rng:
        if c > o:
            patterns["BullMarubozu"] = "📊 紅色光頭光腳（強勢多頭）"
        else:
            patterns["BearMarubozu"] = "📊 綠色光頭光腳（強勢空頭）"

    return patterns

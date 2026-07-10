"""
tw_config.py -- 已驗證策略參數的單一事實源

每個常數都有回測出處；改動任何值屬「影響交易決策」紅線，需重跑對應
驗證（引擎備註於旁）並經使用者核准。消費端：tw_signal_log、tw_summary、
app.py（UI 預設與過熱標記）、tw_strategy（出場線）。
"""

# 交易成本（買賣手續費各 0.1425% + 證交稅 0.3%，未打折）
COST_ROUNDTRIP = 0.00585

# 流動性下限（訊號日成交金額，元）
LIQ_MIN = 20_000_000

# 複合選股最佳參數（backtest_composite.py，2026-07-04 還原價回測；
# walk-forward 穩健性見 walkforward_validation.py）
COMPOSITE = {
    "inst_days": 10,        # 法人視窗（交易日）
    "min_streak": 5,        # 投信連買門檻
    "require_foreign": True,
    "yoy_thr": 20.0,        # 月營收 YoY 門檻（%）
}

# 評分區間語義（score_v2_validation.py TEST 期分箱，2026-07-10）：
# v2 下 7 分箱 alpha 最高(+4.1%)、8 分次之(+3.4%)、9 分邊際趨零(+0.7%)
# → 過熱門檻由 v1 的 8 上移至 9
SWEET_MIN = 5.0
OVERHEAT_SCORE = 9.0

# 出場線（exit_backtest.py，2026-07-05）：趨勢單層出場為停損型規則最佳；
# 更緊的停損以平均報酬換尾部風險，已否決為預設
TREND_EXIT_PCT = 0.98       # SMA60 × 0.98

# 回測建議持有區間（交易日）
HOLD_SWEET_RANGE = (20, 60)

# ── 10 因子評分權重 ───────────────────────────────────────────────────────────
# v1：V16 原始手調權重（score_decile_backtest 驗證 0-7 單調有效）
SCORE_WEIGHTS_V1 = {
    "trend": 2.0, "trend_5d": 0.5, "trend_pen": -1.0,
    "annual": 1.0, "annual_pen": -0.5,
    "kd_golden": 2.5, "kd_turn": 1.5, "kd_ob": -1.0, "kdj_hot": -1.5,
    "rsi_os": 1.5, "rsi_mid": 0.5, "rsi_ob": -1.0,
    "bb_break": 2.5, "vol_surge": 1.0, "obv": 0.5,
    "macd_zero": 1.5, "macd_up": 0.5, "macd_down": -0.5,
    "hi52": 0.5, "lo52": 0.5, "mom10": 0.5, "mom10_pen": -0.5,
}

# v2（2026-07-10 定案）：train 期（2024-07~2025-06）預註冊規則修訂，
# test 期（2025-07 起）判準通過（IC +0.039→+0.044、BUY桶 a20 +2.35→+2.42%
# 且訊號數 +30%、9分箱 −0.20→+0.73%）。詳 score_v2_validation.py。
SCORE_WEIGHTS_V2 = {**SCORE_WEIGHTS_V1,
    "kd_turn": 0.0,    # train uplift −0.53pp（低檔翻揚實為弱勢特徵）
    "kd_ob": 0.0,      # train +1.92pp——超買懲罰在懲罰強勢股
    "kdj_hot": 0.0,    # train +1.88pp
    "rsi_ob": 0.0,     # train +1.25pp
    "hi52": 1.0,       # train +1.56pp（接近年高為最強動能特徵）
}

ACTIVE_SCORE_WEIGHTS = SCORE_WEIGHTS_V2     # 切回 v1：改指向 SCORE_WEIGHTS_V1

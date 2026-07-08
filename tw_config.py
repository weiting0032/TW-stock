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

# 評分區間語義（score_decile_backtest.py，2026-07-04）：
# 0–7 分單調遞增；8 分以上過熱（20 日期望轉負）
SWEET_MIN = 5.0
OVERHEAT_SCORE = 8.0

# 出場線（exit_backtest.py，2026-07-05）：趨勢單層出場為停損型規則最佳；
# 更緊的停損以平均報酬換尾部風險，已否決為預設
TREND_EXIT_PCT = 0.98       # SMA60 × 0.98

# 回測建議持有區間（交易日）
HOLD_SWEET_RANGE = (20, 60)

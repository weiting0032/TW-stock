"""
核心數學與語義的回歸測試（stdlib unittest，零新依賴）。

目的：所有「已驗證的回測結論」都建立在這些函式的正確性上——
還原乘數、EPS 累計制換算、連買/onset 語義、出場線等價性、
NaN 正規化（曾出過真 bug）。改壞任何一個都會無聲污染結論。

執行：python -m unittest discover tests -v
"""

import sqlite3
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tw_config
import tw_db
from backtest_composite import _consecutive_true
from tw_dividends import _roc_to_iso, apply_adjustment
from tw_fundamentals import expected_quarter, get_fin_history, get_fin_signals
from tw_signal_log import PARAMS, update_signal_returns
from tw_strategy import get_strategy


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(tw_db._DDL)
    return conn


def _synthetic_ind(close=100.0, n=80, sma60_level=None):
    """造一份 calculate_indicators 輸出等價的最小欄位 DataFrame。"""
    idx = [f"2026-01-{i:02d}" for i in range(1, 32)] + \
          [f"2026-02-{i:02d}" for i in range(1, 29)] + \
          [f"2026-03-{i:02d}" for i in range(1, 22)]
    idx = idx[:n]
    df = pd.DataFrame(index=idx)
    df["Close"] = close
    df["SMA5"] = close
    df["SMA20"] = close * 0.99
    df["SMA60"] = sma60_level if sma60_level is not None else close * 0.95
    df["SMA240"] = close * 0.9
    df["ATR"] = close * 0.02
    df["RSI"] = 55.0
    df["K"], df["D"], df["J"] = 50.0, 48.0, 54.0
    df["Hist"] = 0.1
    df["BB_Width"] = 0.1
    df["BB_Upper"] = close * 1.05
    df["VOL_Ratio"] = 1.0
    df["OBV"] = np.arange(n, dtype=float)
    df["High52W"], df["Low52W"] = close * 1.2, close * 0.7
    return df


class TestAdjustment(unittest.TestCase):
    def test_dividend_back_adjust(self):
        panel = pd.DataFrame({"2330": [100.0, 100.0, 90.0, 90.0]},
                             index=["d1", "d2", "d3", "d4"])
        fac = pd.DataFrame({"ex_date": ["d3"], "stock_id": ["2330"],
                            "factor": [0.9]})
        adj = apply_adjustment(panel, fac)
        # ex_date 之前 ×0.9，當日起不變
        self.assertAlmostEqual(adj.loc["d1", "2330"], 90.0)
        self.assertAlmostEqual(adj.loc["d2", "2330"], 90.0)
        self.assertAlmostEqual(adj.loc["d3", "2330"], 90.0)
        self.assertAlmostEqual(adj.loc["d4", "2330"], 90.0)

    def test_split_and_reverse_split(self):
        panel = pd.DataFrame({"0050": [200.0, 50.0], "2380": [10.0, 36.0]},
                             index=["d1", "d2"])
        fac = pd.DataFrame({
            "ex_date": ["d2", "d2"], "stock_id": ["0050", "2380"],
            "factor": [0.25, 3.6],   # 一拆四 / 減資併縮
        })
        adj = apply_adjustment(panel, fac)
        self.assertAlmostEqual(adj.loc["d1", "0050"], 50.0)
        self.assertAlmostEqual(adj.loc["d1", "2380"], 36.0)

    def test_roc_to_iso(self):
        self.assertEqual(_roc_to_iso("115年06月01日"), "2026-06-01")
        self.assertEqual(_roc_to_iso("115/07/06"), "2026-07-06")
        self.assertEqual(_roc_to_iso("1150706"), "2026-07-06")
        self.assertIsNone(_roc_to_iso("garbage"))


class TestFundamentals(unittest.TestCase):
    def _seed(self, conn):
        rows = [  # (quarter, eps_cum, gross_margin)
            ("2025Q1", 5.0, 50.0), ("2025Q2", 11.0, 51.0),
            ("2025Q3", 18.0, 52.0), ("2025Q4", 26.0, 53.0),
            ("2026Q1", 7.0, 56.0),
        ]
        for q, e, g in rows:
            conn.execute(
                "INSERT INTO quarterly_fin (stock_id, quarter, eps_cum, "
                "gross_margin) VALUES ('9999', ?, ?, ?)", (q, e, g))
        conn.commit()

    def test_eps_q_ttm_gm_delta(self):
        conn = _mem_conn()
        self._seed(conn)
        h = get_fin_history(conn, "9999", quarters=8)
        eq = dict(zip(h["quarter"], h["eps_q"]))
        self.assertAlmostEqual(eq["2025Q2"], 6.0)    # 11-5
        self.assertAlmostEqual(eq["2025Q4"], 8.0)    # 26-18
        self.assertAlmostEqual(eq["2026Q1"], 7.0)    # Q1=累計
        s = get_fin_signals(conn)
        r = s[s.stock_id == "9999"].iloc[0]
        # TTM = 2026Q1累計 + 2025Q4累計 − 2025Q1累計 = 7+26−5 = 28
        self.assertAlmostEqual(r["ttm_eps"], 28.0)
        self.assertAlmostEqual(r["gm_yoy_delta"], 6.0)   # 56 − 50
        self.assertTrue(bool(r["margin_up"]))

    def test_expected_quarter_boundaries(self):
        cases = [
            (date(2026, 5, 15), (2025, 4)), (date(2026, 5, 16), (2026, 1)),
            (date(2026, 8, 14), (2026, 1)), (date(2026, 8, 15), (2026, 2)),
            (date(2026, 11, 14), (2026, 2)), (date(2026, 11, 15), (2026, 3)),
            (date(2026, 3, 31), (2025, 3)), (date(2026, 4, 1), (2025, 4)),
        ]
        for d, exp in cases:
            self.assertEqual(expected_quarter(d), exp, msg=str(d))


class TestSignalSemantics(unittest.TestCase):
    def test_consecutive_true(self):
        b = pd.DataFrame({"a": [True, True, False, True],
                          "b": [False, True, True, True]})
        r = _consecutive_true(b)
        self.assertEqual(r["a"].tolist(), [1, 2, 0, 1])
        self.assertEqual(r["b"].tolist(), [0, 1, 2, 3])

    def test_onset_dedup(self):
        cond = pd.Series([False, True, True, False, True])
        onset = cond & ~cond.shift(1, fill_value=False)
        self.assertEqual(onset.tolist(), [False, True, False, False, True])

    def test_update_signal_returns_nan_regression(self):
        """混合 dtype（部分已回填→NULL 變 NaN）不得卡死後續回填。曾為真 bug。"""
        conn = _mem_conn()
        dates = [f"2026-06-{i:02d}" for i in range(1, 13)]
        for i, d in enumerate(dates):
            conn.execute(
                "INSERT INTO daily_price (trade_date, stock_id, market, open, "
                "close) VALUES (?, '9999', 'TWSE', ?, ?)",
                (d, 100.0 + i, 101.0 + i))
        for sig in ("2026-06-01", "2026-06-03"):
            conn.execute(
                "INSERT INTO signal_log (signal_date, stock_id, strategy, "
                "params) VALUES (?, '9999', 't', '{}')", (sig,))
        # 先讓第一列部分回填 → 產生混合 dtype 情境
        conn.execute("UPDATE signal_log SET entry_open=101.0, ret_5=0.05 "
                     "WHERE signal_date='2026-06-01'")
        conn.commit()
        update_signal_returns(conn)
        rows = conn.execute(
            "SELECT signal_date, entry_open, ret_5 FROM signal_log "
            "ORDER BY signal_date").fetchall()
        self.assertIsNotNone(rows[1][1], "第二列 entry_open 未回填（NaN bug 回歸）")
        self.assertIsNotNone(rows[1][2], "第二列 ret_5 未回填（NaN bug 回歸）")
        # 手動預填值不得被覆蓋
        self.assertAlmostEqual(rows[0][2], 0.05)

    def test_upsert_idempotent(self):
        conn = _mem_conn()
        df = pd.DataFrame({"stock_id": ["1101"], "name": ["台泥"]})
        tw_db.upsert(conn, "stock_names", df)
        tw_db.upsert(conn, "stock_names", df)
        n = conn.execute("SELECT COUNT(*) FROM stock_names").fetchone()[0]
        self.assertEqual(n, 1)


class TestStrategy(unittest.TestCase):
    def test_exit_line_is_trend_line(self):
        ind = _synthetic_ind(close=100.0, sma60_level=95.0)
        s = get_strategy(ind, held_shares=1000, held_cost=80.0)
        self.assertAlmostEqual(s["sl"], 95.0 * tw_config.TREND_EXIT_PCT, places=6)
        self.assertNotEqual(s["action"], "SELL_EXIT")

    def test_sell_exit_triggers_below_trend_line(self):
        ind = _synthetic_ind(close=100.0, sma60_level=105.0)  # 線在 102.9
        s = get_strategy(ind, held_shares=1000, held_cost=80.0)
        self.assertEqual(s["action"], "SELL_EXIT")

    def test_factor10_new_path_and_fallback(self):
        ind = _synthetic_ind()
        s_new = get_strategy(ind, market_info={
            "f_net_5d": 500, "t_net_5d": 300, "trust_streak": 5})
        self.assertTrue(any("同買" in r for r in s_new["reasons"]))
        self.assertTrue(any("連買5" in r for r in s_new["reasons"]))
        s_old = get_strategy(ind, market_info={"三大合計": 600})
        self.assertTrue(any("法人大幅買超" in r for r in s_old["reasons"]))

    def test_config_single_source(self):
        self.assertEqual(PARAMS["inst_days"], tw_config.COMPOSITE["inst_days"])
        self.assertEqual(PARAMS["min_streak"], tw_config.COMPOSITE["min_streak"])
        self.assertEqual(PARAMS["yoy_thr"], tw_config.COMPOSITE["yoy_thr"])
        self.assertEqual(PARAMS["liq_min"], tw_config.LIQ_MIN)


if __name__ == "__main__":
    unittest.main()

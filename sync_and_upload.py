"""
sync_and_upload.py -- 盤後同步 + 快照上傳（供 Windows 工作排程器每交易日 18:30 執行）

用法：
    python -X utf8 sync_and_upload.py              # 同步法人+營收，然後推送快照
    python -X utf8 sync_and_upload.py --no-push    # 只同步不推送
    python -X utf8 sync_and_upload.py --push-only  # 只推送現有 DB 快照
"""

import sys
from datetime import datetime

from tw_db import get_conn
from tw_dividends import sync_dividends
from tw_institutional import get_data_status, sync_inst_flow
from tw_prices import sync_daily_prices
from tw_revenue import sync_monthly_revenue
from tw_snapshot import make_and_push_snapshot


def main():
    no_push = "--no-push" in sys.argv
    push_only = "--push-only" in sys.argv
    print(f"=== sync_and_upload {datetime.now():%Y-%m-%d %H:%M} ===")

    if not push_only:
        conn = get_conn()
        sync_inst_flow(conn)
        sync_daily_prices(conn)
        sync_dividends(conn)
        from tw_margin import sync_margin
        from tw_tdcc import sync_tdcc
        from tw_fundamentals import sync_quarterly_fin
        sync_margin(conn)
        n_tdcc = sync_tdcc(conn)
        print(f"[tdcc] rows={n_tdcc}")
        sync_quarterly_fin(conn)
        sync_monthly_revenue(conn, months_back=2)

        # 訊號日誌：記當日新觸發 → 回填到期報酬 → TG 推播（無憑證自動跳過）
        from tw_signal_log import log_signals, push_daily_signals, update_signal_returns
        latest = conn.execute("SELECT MAX(trade_date) FROM daily_price").fetchone()[0]
        hits = log_signals(conn, latest)
        n_upd = update_signal_returns(conn)
        print(f"[signal] {latest} new={len(hits)} returns_updated={n_upd}")
        try:
            sent = push_daily_signals(conn, hits, latest)
            print(f"[signal] telegram {'sent' if sent else 'skipped (無憑證或發送失敗)'}")
        except Exception as e:
            print(f"[signal] telegram error: {e}")

        # 持倉例外推播（事件驅動；週五附組合週結）
        try:
            from tw_portfolio_alerts import push_portfolio_alerts
            is_fri = datetime.now().weekday() == 4
            sent = push_portfolio_alerts(conn, weekly=is_fri)
            print(f"[pf-alert] {'sent' if sent else 'no-op'}")
        except Exception as e:
            print(f"[pf-alert] error: {e}")

        status = get_data_status(conn)
        conn.close()
        inst = status.get("inst_flow", {})
        print(f"synced: inst latest={inst.get('latest_date')} rows={inst.get('total_rows')}")

    if no_push:
        print("skip push (--no-push)")
        return
    print(make_and_push_snapshot())


if __name__ == "__main__":
    main()

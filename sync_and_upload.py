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
from tw_institutional import get_data_status, sync_inst_flow
from tw_revenue import sync_monthly_revenue
from tw_snapshot import make_and_push_snapshot


def main():
    no_push = "--no-push" in sys.argv
    push_only = "--push-only" in sys.argv
    print(f"=== sync_and_upload {datetime.now():%Y-%m-%d %H:%M} ===")

    if not push_only:
        conn = get_conn()
        sync_inst_flow(conn)
        sync_monthly_revenue(conn, months_back=2)
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

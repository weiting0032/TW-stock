"""
backfill.py -- CLI data backfill tool for tw_market.db
Usage:
    python backfill.py --inst-days 500 --revenue-months 24
    python backfill.py --inst-days 30       # institutional only
    python backfill.py --revenue-months 24  # revenue only
    python backfill.py                      # status only
"""

import argparse
import sys

from tw_db import get_conn
from tw_institutional import get_data_status, sync_inst_flow
from tw_prices import sync_daily_prices
from tw_revenue import backfill_revenue


def progress(d, rows):
    print(f"  {d}  rows={rows}")


def main():
    parser = argparse.ArgumentParser(description="TW stock local DB backfill")
    parser.add_argument(
        "--inst-days",
        type=int,
        default=0,
        metavar="N",
        help="Backfill institutional flow N calendar days (default 0=skip)",
    )
    parser.add_argument(
        "--revenue-months",
        type=int,
        default=0,
        metavar="M",
        help="Backfill monthly revenue M months (default 0=skip)",
    )
    parser.add_argument(
        "--price-days",
        type=int,
        default=0,
        metavar="N",
        help="Backfill daily prices N calendar days (default 0=skip)",
    )
    parser.add_argument(
        "--margin-days",
        type=int,
        default=0,
        metavar="N",
        help="Backfill margin trading N calendar days (default 0=skip)",
    )
    args = parser.parse_args()

    conn = get_conn()
    print("DB initialized.")

    # -- Institutional backfill ------------------------------------------------
    if args.inst_days > 0:
        print(f"\nBackfilling institutional flow ({args.inst_days} calendar days)...")
        sync_inst_flow(conn, backfill_days=args.inst_days, progress_cb=progress)
        print("Institutional backfill done.")
    else:
        print("\n--inst-days=0, skipping institutional backfill.")

    # -- Price backfill --------------------------------------------------------
    if args.price_days > 0:
        print(f"\nBackfilling daily prices ({args.price_days} calendar days)...")
        sync_daily_prices(conn, backfill_days=args.price_days, progress_cb=progress)
        print("Price backfill done.")
    else:
        print("\n--price-days=0, skipping price backfill.")

    # -- Margin backfill -------------------------------------------------------
    if args.margin_days > 0:
        from tw_margin import sync_margin
        print(f"\nBackfilling margin trading ({args.margin_days} calendar days)...")
        sync_margin(conn, backfill_days=args.margin_days, progress_cb=progress)
        print("Margin backfill done.")

    # -- Revenue backfill ------------------------------------------------------
    if args.revenue_months > 0:
        print(f"\nBackfilling monthly revenue ({args.revenue_months} months)...")
        backfill_revenue(conn, months=args.revenue_months)
        print("Revenue backfill done.")
    else:
        print("\n--revenue-months=0, skipping revenue backfill.")

    # -- Final status ----------------------------------------------------------
    print("\n=== DB Status ===")
    status = get_data_status(conn)
    inst = status["inst_flow"]
    rev = status["monthly_revenue"]
    print(f"inst_flow   : latest={inst['latest_date']}  "
          f"rows={inst['total_rows']}  stocks={inst['stock_count']}")
    print(f"monthly_rev : latest={rev['latest_month']}  "
          f"stocks={rev['stock_count']}")

    conn.close()


if __name__ == "__main__":
    main()

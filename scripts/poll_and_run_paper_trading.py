"""Polls for NSE's end-of-day data to become available, then runs the daily
paper-trading step as soon as it does -- instead of waiting out a fixed
6:30 PM trigger regardless of whether NSE has actually published yet.

Meant to be invoked repeatedly (every 10-15 minutes) by Windows Task
Scheduler across a window starting shortly after NSE's 3:30 PM close.
Each invocation is a cheap no-op once today's step has already run, or if
today's data isn't out yet -- the check is a single-symbol, uncached fetch,
not the full 50-symbol universe every account otherwise pulls. Past
FALLBACK_HOUR_IST the step runs regardless of the check, so a canary-check
false negative can't cause the whole day to be silently skipped -- that
matches the old fixed-time trigger's behavior as a last resort.

    python scripts/poll_and_run_paper_trading.py
"""
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loaders import get_raw_ohlcv  # noqa: E402
from execution.accounts import ACCOUNTS, load_account_state  # noqa: E402
from scripts.run_daily_paper_trading import main as run_daily_paper_trading  # noqa: E402

CANARY_SYMBOL = "RELIANCE"
# The scheduled polling window is 4:00-6:45 PM IST (every 15 min) -- this
# must be <= the hour of that window's last poll, or the fallback would
# never actually get an invocation to fire from.
FALLBACK_HOUR_IST = 18


def _already_ran_today(today_str: str) -> bool:
    return all(load_account_state(meta)["last_run_date"] == today_str for meta in ACCOUNTS)


def _eod_data_is_out(today: date) -> bool:
    try:
        df = get_raw_ohlcv(CANARY_SYMBOL, today, today, use_cache=False)
    except Exception:
        return False
    return not df.empty and today in df.index.date


def main(today: date | None = None, now: datetime | None = None) -> int:
    today = today or date.today()
    now = now or datetime.now()
    today_str = today.isoformat()

    if _already_ran_today(today_str):
        print(f"Already ran for {today_str}, nothing to do.")
        return 0

    if not _eod_data_is_out(today):
        if now.hour < FALLBACK_HOUR_IST:
            print(f"NSE EOD data for {today_str} isn't out yet, will check again next poll.")
            return 0
        print(f"NSE EOD data for {today_str} still isn't out at the {FALLBACK_HOUR_IST}:00 fallback -- running anyway.")

    print(f"Running the daily paper-trading step for {today_str}.")
    return run_daily_paper_trading([])


if __name__ == "__main__":
    sys.exit(main())

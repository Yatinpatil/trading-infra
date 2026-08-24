"""One-off/periodic bulk backfill: pull historical OHLCV + corporate actions
for every symbol in a given index into the project's SQLite store
(data/trading.db), so backtests and paper trading never have to hit NSE
symbol-by-symbol on a cold cache.

Deliberately sequential with a delay between symbols — NSE's unofficial API
(via jugaad-data) is already documented elsewhere in this repo as flaky, and
hammering it concurrently risks a temporary IP block. A single run is
resumable: get_ohlcv's cache only fetches the date ranges it doesn't already
have, so re-running after a partial failure just tops up what's missing.

    python scripts/backfill_universe.py --index "NIFTY 50" --start 2021-01-01
"""
import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders import get_ohlcv
from universe.constituents import get_current_constituents


def backfill(index_name: str, start, end, delay_seconds: float = 1.5) -> dict:
    symbols = get_current_constituents(index_name)
    print(f"{index_name}: {len(symbols)} constituents")

    succeeded, failed = [], []
    for i, symbol in enumerate(symbols, 1):
        try:
            df = get_ohlcv(symbol, start, end, adjust=True)
            print(f"[{i}/{len(symbols)}] {symbol}: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})" if not df.empty else f"[{i}/{len(symbols)}] {symbol}: no data returned")
            succeeded.append(symbol)
        except Exception as exc:
            print(f"[{i}/{len(symbols)}] {symbol}: FAILED - {type(exc).__name__}: {exc}")
            failed.append(symbol)
        if i < len(symbols):
            time.sleep(delay_seconds)

    print(f"\nDone: {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed symbols (re-run this script to retry, cache resumes from where it left off): {failed}")

    return {"succeeded": succeeded, "failed": failed}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV + corporate actions for an index's constituents.")
    parser.add_argument("--index", default="NIFTY 50", help="NSE index name, e.g. 'NIFTY 50' or 'NIFTY 500'")
    parser.add_argument("--start", default="2021-01-01", help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD (default: today)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds to sleep between symbols")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    end = args.end or date.today().isoformat()
    return backfill(args.index, args.start, end, delay_seconds=args.delay)


if __name__ == "__main__":
    main()

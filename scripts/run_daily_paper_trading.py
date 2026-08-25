"""Daily orchestrator: runs one paper-trading step for every configured
strategy account, then regenerates the local dashboard. Each account runs
in its own subprocess so one crashing or hanging account can't block the
rest -- this is the one script Windows Task Scheduler should point at.

    python scripts/run_daily_paper_trading.py
    python scripts/run_daily_paper_trading.py --as-of 2026-08-25
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

ACCOUNTS = [
    "mean_reversion_nifty50",
    "momentum_nifty50",
    "breakout_nifty50",
    "buy_and_hold_nifty50",
    "ml_strategy_nifty50",
    "rsi_mean_reversion_nifty50",
    "bollinger_breakout_nifty50",
    "adx_trend_nifty50",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every paper-trading account's daily step, then rebuild the dashboard.")
    parser.add_argument("--as-of", help="Date to run the step for, YYYY-MM-DD (default: today)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    python = str(PYTHON) if PYTHON.exists() else sys.executable

    results = []
    for config in ACCOUNTS:
        cmd = [python, str(ROOT / "run_paper_trading.py"), "--config", config]
        if args.as_of:
            cmd += ["--as-of", args.as_of]
        print(f"--- {config} ---")
        proc = subprocess.run(cmd, cwd=ROOT)
        results.append((config, proc.returncode))
        print()

    failed = [name for name, code in results if code != 0]
    if failed:
        print(f"FAILED accounts: {failed}")
    else:
        print("All accounts stepped successfully.")

    print("\n--- regenerating dashboard ---")
    dash_cmd = [python, str(ROOT / "scripts" / "generate_dashboard.py")]
    dash_proc = subprocess.run(dash_cmd, cwd=ROOT)
    if dash_proc.returncode != 0:
        print("Dashboard generation FAILED.")

    return 1 if (failed or dash_proc.returncode != 0) else 0


if __name__ == "__main__":
    sys.exit(main())

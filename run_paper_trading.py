"""Daily paper-trading CLI — Phase 7's operational entry point. Invoke once
per trading day (e.g. from cron / Windows Task Scheduler) after NSE market
close; PaperTradingEngine (execution/paper_trading.py) picks up exactly
where the previous invocation left off via a PaperBroker's persisted state
under execution/state/.

    python run_paper_trading.py --config mean_reversion --symbol RELIANCE
    python run_paper_trading.py --config mean_reversion              # portfolio mode, universe from config
    python run_paper_trading.py --config mean_reversion --report     # tear sheet from the ledger so far

Every run is logged to logs/paper_trading.log (in addition to stdout) since
a scheduled job's console output is easy to lose, and a per-account lock
file guards against two overlapping invocations (e.g. a hung previous run
still going when cron fires the next one) corrupting the same JSON state.
"""
import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from analytics.report import generate_report
from configs import load_config
from data.loaders import get_ohlcv
from execution.broker import PaperBroker
from execution.paper_trading import PaperTradingEngine
from main import build_strategy, resolve_portfolio_symbols
from strategies.ml_strategy import MLStrategy

STATE_DIR = Path(__file__).parent / "execution" / "state"
LOG_DIR = Path(__file__).parent / "logs"
STALE_LOCK_SECONDS = 2 * 60 * 60  # a lock older than this is assumed to be from a crashed run

logger = logging.getLogger("run_paper_trading")


class AlreadyRunningError(Exception):
    pass


def _ensure_file_logging() -> None:
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / "paper_trading.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _acquire_lock(lock_path: Path) -> None:
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age < STALE_LOCK_SECONDS:
            raise AlreadyRunningError(
                f"{lock_path} exists and is only {age:.0f}s old -- another run may still be in "
                f"progress for this account. Remove it manually if you're sure it isn't."
            )
        logger.warning("Stale lock %s (%.0fs old) -- assuming a previous run crashed; proceeding.", lock_path, age)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()))


def _release_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


class LedgerResult:
    """Adapts a PaperBroker's persisted ledgers to the `.equity_curve`/
    `.trades` duck type analytics.report.generate_report expects."""

    def __init__(self, broker: PaperBroker):
        self.equity_curve = broker.equity_curve()
        self.trades = broker.trades()
        self.final_equity = self.equity_curve.iloc[-1] if len(self.equity_curve) else None


def account_name(config_name: str, symbol: str | None) -> str:
    return f"{config_name}_{symbol}" if symbol else f"{config_name}_portfolio"


def resolve_strategy(
    config: dict,
    symbols: list[str],
    as_of,
    model_path: Path,
    train_days: int,
    refit_days: int,
):
    """`build_strategy(config)` for every ordinary strategy. For `ml_strategy`
    specifically: load the persisted, fitted model if it's still within
    `refit_days` of when it was last fit, otherwise fit a fresh one on the
    trailing `train_days` of history (the last `horizon` rows embargoed, so
    the fit never uses a label that depends on data at or after `as_of`) and
    persist it — mirroring the walk-forward evaluation's refit cadence
    (scripts/evaluate_ml_strategy.py) so a live account doesn't quietly keep
    running on a model that's gone stale.
    """
    if config.get("strategy") != "ml_strategy":
        return build_strategy(config)

    as_of_date = pd.Timestamp(as_of or date.today()).date()

    if model_path.exists():
        strategy = MLStrategy.load(model_path)
        if strategy.fitted_at is not None:
            fitted_date = datetime.strptime(strategy.fitted_at, "%Y-%m-%d").date()
            age_days = (as_of_date - fitted_date).days
            if age_days < refit_days:
                logger.info("Using ml_strategy model fit on %s (%d days old)", strategy.fitted_at, age_days)
                return strategy
            logger.info(
                "ml_strategy model fit on %s is %d days old (>= refit_days=%d) -- refitting",
                strategy.fitted_at, age_days, refit_days,
            )

    logger.info("Fitting a new ml_strategy model on the trailing %d days for %d symbols", train_days, len(symbols))
    train_start = as_of_date - timedelta(days=int(train_days * 1.6))  # calendar-day buffer for weekends/holidays
    strategy = MLStrategy(config.get("params"))
    horizon = strategy.params["horizon"]

    ohlcv_by_symbol = {}
    for symbol in symbols:
        df = get_ohlcv(symbol, train_start, as_of_date)
        if len(df) > horizon:
            ohlcv_by_symbol[symbol] = df.tail(train_days)
    embargoed = {s: df.iloc[:-horizon] for s, df in ohlcv_by_symbol.items()}

    strategy.fit(embargoed)
    strategy.fitted_at = as_of_date.isoformat()
    strategy.save(model_path)
    logger.info("Fitted and saved ml_strategy model to %s", model_path)
    return strategy


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one daily paper-trading step, or report on an existing account.")
    parser.add_argument("--config", default="mean_reversion", help="Config name under configs/ (without .yaml)")
    parser.add_argument("--symbol", help="Paper-trade a single symbol instead of the configured universe")
    parser.add_argument("--as-of", help="Date to run the step for, YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Starting capital for a brand-new account")
    parser.add_argument("--report", action="store_true", help="Write a tear sheet from the ledger so far instead of running a step")
    parser.add_argument("--output-dir", default="reports", help="Directory to write the tear sheet into")
    parser.add_argument("--train-days", type=int, default=750, help="ml_strategy only: trading days of history to fit on")
    parser.add_argument("--refit-days", type=int, default=126, help="ml_strategy only: refit once the model is this many days old")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _ensure_file_logging()
    config = load_config(args.config)
    name = account_name(args.config, args.symbol)
    broker = PaperBroker(STATE_DIR / f"{name}.json", initial_capital=args.capital)

    if args.report:
        report_path = generate_report(name, LedgerResult(broker), args.output_dir)
        logger.info("Report written to %s", report_path)
        print(f"Report written to {report_path}")
        return broker

    lock_path = STATE_DIR / f"{name}.lock"
    try:
        _acquire_lock(lock_path)
    except AlreadyRunningError as exc:
        logger.error(str(exc))
        print(f"Error: {exc}")
        return {"skipped": True, "reason": str(exc), "as_of": args.as_of}

    try:
        if args.symbol:
            symbols = [args.symbol]
        else:
            symbols = resolve_portfolio_symbols(config, args.as_of or date.today())

        model_path = STATE_DIR / f"{name}_model.joblib"
        strategy = resolve_strategy(config, symbols, args.as_of, model_path, args.train_days, args.refit_days)
        engine = PaperTradingEngine(strategy, config, broker, symbols)
        try:
            summary = engine.run_daily_step(args.as_of)
        except Exception:
            logger.exception("Daily step failed for account %s", name)
            raise
    finally:
        _release_lock(lock_path)

    if summary["skipped"]:
        logger.info("Skipped: %s", summary["reason"])
        print(f"Skipped: {summary['reason']}")
    else:
        logger.info(
            "[%s] equity=%.2f cash=%.2f open=%s trades=%d",
            summary["as_of"], summary["equity"], summary["cash"],
            summary["open_positions"], len(summary["trades_today"]),
        )
        print(f"[{summary['as_of']}] equity={summary['equity']:,.2f} cash={summary['cash']:,.2f}")
        print(f"Open positions: {summary['open_positions']}")
        print(f"Trades filled today: {len(summary['trades_today'])}")
        print(f"Queued for tomorrow's open -- entries: {summary['pending_entries_tomorrow']}, exits: {summary['pending_exits_tomorrow']}")

    return summary


if __name__ == "__main__":
    main()

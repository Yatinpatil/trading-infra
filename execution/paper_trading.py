"""Paper-trading engine — Phase 7 of the project plan, the "live/paper
execution bridge" explicitly deferred out of Phases 0-6's scope.

There is no live intraday feed here (jugaad-data only gives end-of-day NSE
bars) and no real broker connection. `run_daily_step()` is meant to be
invoked once per trading day, after market close, once today's OHLCV bar is
final — which is exactly the information engine/portfolio.py's next-bar-open
convention needs: a signal decided on day t's close fills at day t+1's open.
Because this only runs once day t's data is final, "tomorrow's open" becomes
"today's open" by the time the *next* invocation runs, so the timing
convention is identical to the backtest engines, just spread across process
restarts instead of one in-memory loop. Point a scheduled job (cron / Windows
Task Scheduler) at run_paper_trading.py once a day and PaperBroker's
persisted state picks up exactly where the previous run left off.

Money math (fills, stop checks, mark-to-market, position sizing) always uses
*raw* (unadjusted) OHLCV, never the split/bonus-adjusted series: adjustment
is a backward rescale computed fresh on every call relative to "today", so
the adjusted value for a fixed historical date silently changes the day a
new corporate action is registered — fine for a one-shot backtest (adjusted
once, consistently, at the end of a fixed window) but wrong for a position
whose entry_price is captured once and held fixed across many days of
re-fetching. Signals are still generated from the adjusted series, since
indicators need a continuous history free of fake ex-date jumps. A real
corporate action during an open paper position still isn't auto-reconciled
(no share-count/cost-basis adjustment) — see the warning logged below.

Swapping in a real broker later means replacing PaperBroker with a class
that talks to a broker API — this engine's step logic doesn't change.
"""
import logging
from datetime import date, datetime

import pandas as pd

from data.corporate_actions import get_corporate_actions
from data.loaders import get_ohlcv
from engine.costs import compute_transaction_cost
from execution.broker import PaperBroker
from risk.limits import has_room_for_position, within_correlation_limit, within_sector_limit
from risk.position_sizing import fixed_fractional_size

logger = logging.getLogger(__name__)

# Yahoo's fallback (data/yahoo_fallback.py) is only safe to use for `as_of`
# = today once NSE's trading day has genuinely concluded -- "NSE hasn't
# returned it yet" is trivially true at ANY time before NSE actually
# publishes, including minutes after market open, and Yahoo's same-day
# price before then is a live intraday snapshot, not a real close. This
# gate lives here rather than only in the poller's own canary check
# (scripts/poll_and_run_paper_trading.py) because every caller of
# run_daily_step -- the poller's forced fallback, but also a direct CLI
# run or the web UI's "Run Now"/"Run All" -- goes through this same path,
# and a canary check that only some callers use isn't a real guarantee.
# A strictly past `as_of` has no such ambiguity and is always safe.
YAHOO_FALLBACK_EARLIEST_HOUR = 18


def _yahoo_fallback_is_safe(as_of: pd.Timestamp, now: datetime) -> bool:
    return as_of.date() < now.date() or now.hour >= YAHOO_FALLBACK_EARLIEST_HOUR


class PaperTradingEngine:
    def __init__(
        self,
        strategy,
        config: dict,
        broker: PaperBroker,
        symbols: list[str],
        sector_map: dict[str, str] | None = None,
        history_days: int = 400,
    ):
        self.strategy = strategy
        self.config = config
        self.broker = broker
        self.symbols = symbols
        self.sector_map = sector_map or {}
        self.history_days = history_days

    def run_daily_step(self, as_of=None, now: datetime | None = None) -> dict:
        as_of = pd.Timestamp(as_of or date.today()).normalize()
        as_of_str = as_of.date().isoformat()
        now = now or datetime.now()
        broker = self.broker

        if broker.last_run_date == as_of_str:
            return {"skipped": True, "reason": f"already ran for {as_of_str}", "as_of": as_of_str}

        start = (as_of - pd.Timedelta(days=self.history_days)).date()
        allow_yahoo_fallback = _yahoo_fallback_is_safe(as_of, now)
        raw_by_symbol, adjusted_by_symbol = self._fetch_data(start, as_of, allow_yahoo_fallback)

        if not raw_by_symbol:
            return {
                "skipped": True,
                "reason": f"no trading data for any tracked symbol on {as_of_str} (holiday?)",
                "as_of": as_of_str,
            }

        self._warn_on_unreconciled_corporate_actions(as_of)

        costs_cfg = self.config.get("costs", {})
        risk_cfg = self.config.get("risk", {})
        stop_loss_pct = self.config.get("params", {}).get("stop_loss_pct")
        position_size_pct = risk_cfg.get("position_size_pct", 0.1)
        max_concurrent_positions = risk_cfg.get("max_concurrent_positions")
        max_exposure_per_sector_pct = risk_cfg.get("max_exposure_per_sector_pct")
        max_correlation = risk_cfg.get("max_correlation")

        trades_today = []

        # 1. Resolve pending exits queued from the previous step, at today's open
        for symbol in sorted(broker.pending_exits):
            if symbol not in broker.positions or symbol not in raw_by_symbol:
                continue
            exit_price = float(raw_by_symbol[symbol].loc[as_of, "OPEN"])
            exit_cost = compute_transaction_cost(broker.positions[symbol].quantity * exit_price, costs_cfg)
            trades_today.append(broker.close_position(symbol, as_of, exit_price, exit_cost, "signal"))
        broker.pending_exits = set()

        # 2. Resolve pending entries queued from the previous step, at today's open,
        #    subject to the same risk limits engine/portfolio.py applies
        for symbol in sorted(broker.pending_entries):
            if symbol in broker.positions or symbol not in raw_by_symbol:
                continue
            if not has_room_for_position(len(broker.positions), max_concurrent_positions):
                continue

            entry_price = float(raw_by_symbol[symbol].loc[as_of, "OPEN"])
            equity_now = self._mark_to_market(raw_by_symbol, as_of)
            quantity = fixed_fractional_size(equity_now, entry_price, position_size_pct)
            if quantity <= 0:
                continue

            sector = self.sector_map.get(symbol)
            if sector is not None and not within_sector_limit(
                self._sector_value(sector, raw_by_symbol, as_of),
                quantity * entry_price,
                equity_now,
                max_exposure_per_sector_pct,
            ):
                continue

            held_returns = {
                sym: adjusted_by_symbol[sym]["CLOSE"].pct_change()
                for sym in broker.positions if sym in adjusted_by_symbol
            }
            candidate_returns = adjusted_by_symbol[symbol]["CLOSE"].pct_change()
            if not within_correlation_limit(candidate_returns, held_returns, max_correlation):
                continue

            entry_cost = compute_transaction_cost(quantity * entry_price, costs_cfg)
            stop_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct else None
            broker.open_position(symbol, as_of, entry_price, quantity, entry_cost, stop_price)
        broker.pending_entries = set()

        # 3. Intrabar stop-loss check for everything still open
        for symbol in list(broker.positions.keys()):
            if symbol not in raw_by_symbol:
                continue
            position = broker.positions[symbol]
            if position.stop_price is not None and float(raw_by_symbol[symbol].loc[as_of, "LOW"]) <= position.stop_price:
                exit_cost = compute_transaction_cost(position.quantity * position.stop_price, costs_cfg)
                trades_today.append(
                    broker.close_position(symbol, as_of, position.stop_price, exit_cost, "stop_loss")
                )

        # 4. Evaluate today's close for signals (adjusted series), queue for tomorrow's open
        for symbol, df in adjusted_by_symbol.items():
            signals = self.strategy.generate_signals(df)
            if as_of not in signals.index:
                continue
            if symbol not in broker.positions:
                if bool(signals.loc[as_of, "entry_long"]):
                    broker.pending_entries.add(symbol)
            else:
                if bool(signals.loc[as_of, "exit_long"]):
                    broker.pending_exits.add(symbol)

        equity = self._mark_to_market(raw_by_symbol, as_of)
        broker.record_equity(as_of, equity)
        broker.last_run_date = as_of_str
        broker.save()

        return {
            "skipped": False,
            "as_of": as_of_str,
            "equity": equity,
            "cash": broker.cash,
            "open_positions": sorted(broker.positions),
            "trades_today": trades_today,
            "pending_entries_tomorrow": sorted(broker.pending_entries),
            "pending_exits_tomorrow": sorted(broker.pending_exits),
        }

    def _fetch_data(self, start, as_of, allow_yahoo_fallback: bool) -> tuple[dict, dict]:
        """Raw (money math) and adjusted (signals) OHLCV per symbol, through
        `as_of`. A symbol whose fetch fails (NSE hiccup, delisting, bad
        network day) is logged and skipped for this step rather than
        aborting the whole run — one bad symbol shouldn't block the rest of
        the universe.
        """
        raw_by_symbol, adjusted_by_symbol = {}, {}
        for symbol in self.symbols:
            try:
                raw = get_ohlcv(symbol, start, as_of.date(), adjust=False, allow_yahoo_fallback=allow_yahoo_fallback)
                adjusted = get_ohlcv(symbol, start, as_of.date(), adjust=True, allow_yahoo_fallback=allow_yahoo_fallback)
            except Exception:
                logger.warning("Skipping %s for %s: data fetch failed", symbol, as_of.date(), exc_info=True)
                continue
            if not raw.empty and as_of in raw.index:
                raw_by_symbol[symbol] = raw
                adjusted_by_symbol[symbol] = adjusted
        return raw_by_symbol, adjusted_by_symbol

    def _warn_on_unreconciled_corporate_actions(self, as_of) -> None:
        for symbol, position in self.broker.positions.items():
            try:
                actions = get_corporate_actions(symbol)
            except Exception:
                continue
            if actions.empty:
                continue
            entry_date = pd.Timestamp(position.entry_date)
            pending = actions[(actions["ex_date"] > entry_date) & (actions["ex_date"] <= as_of)]
            for _, row in pending.iterrows():
                logger.warning(
                    "%s had a %s (ratio %.3f) on %s while a paper position opened %s is still held -- "
                    "quantity/cost-basis are NOT auto-adjusted; true up execution/state manually if needed.",
                    symbol, row["action_type"], row["ratio"], row["ex_date"].date(), position.entry_date,
                )

    def _mark_to_market(self, raw_by_symbol: dict, as_of) -> float:
        equity = self.broker.cash
        for symbol, position in self.broker.positions.items():
            df = raw_by_symbol.get(symbol)
            price = float(df.loc[as_of, "CLOSE"]) if df is not None and as_of in df.index else position.entry_price
            equity += position.quantity * price
        return equity

    def _sector_value(self, sector: str, raw_by_symbol: dict, as_of) -> float:
        total = 0.0
        for symbol, position in self.broker.positions.items():
            if self.sector_map.get(symbol) != sector:
                continue
            df = raw_by_symbol.get(symbol)
            price = float(df.loc[as_of, "CLOSE"]) if df is not None and as_of in df.index else position.entry_price
            total += position.quantity * price
        return total

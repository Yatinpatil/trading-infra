"""Generic single-stock backtest engine — takes any Strategy and turns its
signals into simulated fills, applying costs and an optional stop-loss.

Timing convention (this is what keeps the engine lookahead-free):
  - A strategy's signal at row t is decided using only data through t.
  - Entries and signal-based exits fill at the *next* bar's OPEN — you can't
    trade on a close using information only available at that close.
  - A stop-loss, once a position is open, is checked against the *current*
    bar's LOW/HIGH and can fill same-day — that's a resting order, not a
    forecast, so it's not lookahead.
"""
from dataclasses import dataclass, field

import pandas as pd

from engine.costs import compute_transaction_cost


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    quantity: int
    entry_cost: float
    exit_cost: float
    exit_reason: str

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl(self) -> float:
        return self.gross_pnl - self.entry_cost - self.exit_cost

    @property
    def pnl_pct(self) -> float:
        basis = self.entry_price * self.quantity
        return self.pnl / basis if basis else 0.0

    @property
    def holding_days(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    final_equity: float
    signals: pd.DataFrame = field(repr=False, default=None)


def _trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_date", "entry_price", "exit_date", "exit_price", "quantity",
                "entry_cost", "exit_cost", "exit_reason", "pnl", "pnl_pct", "holding_days",
            ]
        )
    rows = [
        {
            "entry_date": t.entry_date,
            "entry_price": t.entry_price,
            "exit_date": t.exit_date,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "entry_cost": t.entry_cost,
            "exit_cost": t.exit_cost,
            "exit_reason": t.exit_reason,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "holding_days": t.holding_days,
        }
        for t in trades
    ]
    return pd.DataFrame(rows)


def run_single_backtest(strategy, df: pd.DataFrame, config: dict, initial_capital: float = 1_000_000.0) -> BacktestResult:
    signals = strategy.generate_signals(df)
    costs_cfg = config.get("costs", {})
    position_size_pct = config.get("risk", {}).get("position_size_pct", 1.0)
    stop_loss_pct = config.get("params", {}).get("stop_loss_pct")

    dates = df.index
    cash = initial_capital
    position = None  # dict: entry_date, entry_price, quantity, stop_price, entry_cost
    pending_entry = False
    pending_exit = False
    trades: list[Trade] = []
    equity_points = []

    for i, date in enumerate(dates):
        row = df.iloc[i]

        if pending_entry and position is None:
            entry_price = float(row["OPEN"])
            budget = cash * position_size_pct
            quantity = int(budget // entry_price)
            if quantity > 0:
                trade_value = quantity * entry_price
                entry_cost = compute_transaction_cost(trade_value, costs_cfg)
                cash -= trade_value + entry_cost
                stop_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct else None
                position = {
                    "entry_date": date,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "stop_price": stop_price,
                    "entry_cost": entry_cost,
                }
        pending_entry = False

        if pending_exit and position is not None:
            exit_price = float(row["OPEN"])
            cash = _close_position(position, date, exit_price, "signal", costs_cfg, cash, trades)
            position = None
        pending_exit = False

        if position is not None and position["stop_price"] is not None and row["LOW"] <= position["stop_price"]:
            exit_price = position["stop_price"]
            cash = _close_position(position, date, exit_price, "stop_loss", costs_cfg, cash, trades)
            position = None

        if position is None and not pending_entry:
            if bool(signals["entry_long"].iloc[i]):
                pending_entry = True
        elif position is not None:
            if bool(signals["exit_long"].iloc[i]):
                pending_exit = True

        mark_price = float(row["CLOSE"])
        equity = cash + (position["quantity"] * mark_price if position else 0.0)
        equity_points.append((date, equity))

    if position is not None:
        last_date = dates[-1]
        last_close = float(df.iloc[-1]["CLOSE"])
        cash = _close_position(position, last_date, last_close, "end_of_data", costs_cfg, cash, trades)
        if equity_points:
            equity_points[-1] = (equity_points[-1][0], cash)

    equity_curve = pd.Series(
        [v for _, v in equity_points], index=[d for d, _ in equity_points], name="equity"
    )
    return BacktestResult(
        equity_curve=equity_curve,
        trades=_trades_to_df(trades),
        final_equity=cash,
        signals=signals,
    )


def _close_position(position: dict, exit_date: pd.Timestamp, exit_price: float, reason: str, costs_cfg: dict, cash: float, trades: list[Trade]) -> float:
    trade_value = position["quantity"] * exit_price
    exit_cost = compute_transaction_cost(trade_value, costs_cfg)
    cash += trade_value - exit_cost
    trades.append(
        Trade(
            entry_date=position["entry_date"],
            entry_price=position["entry_price"],
            exit_date=exit_date,
            exit_price=exit_price,
            quantity=position["quantity"],
            entry_cost=position["entry_cost"],
            exit_cost=exit_cost,
            exit_reason=reason,
        )
    )
    return cash

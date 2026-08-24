"""Paper-trading account: cash, open positions, and orders queued for the
next daily step's open, persisted to disk so a scheduled job can run once a
day and pick up exactly where the previous run left off.

State (cash, positions, pending orders) lives as JSON at `state_path`. The
trade and equity ledgers are separate append-only CSVs next to it, so the
account has a permanent audit trail independent of the point-in-time JSON
snapshot — and `equity_curve()`/`trades()` hand that ledger back in the same
shape engine/single_stock.py and engine/portfolio.py produce, so
analytics/report.py's tear sheets work unchanged on a paper account.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def _date_str(d) -> str:
    if isinstance(d, (datetime, pd.Timestamp)):
        d = d.date()
    return d.isoformat() if isinstance(d, date) else str(d)


@dataclass
class PaperPosition:
    quantity: int
    entry_price: float
    entry_date: str
    entry_cost: float
    stop_price: float | None = None


TRADE_COLUMNS = [
    "symbol", "entry_date", "entry_price", "exit_date", "exit_price", "quantity",
    "entry_cost", "exit_cost", "exit_reason", "pnl", "pnl_pct",
]


class PaperBroker:
    def __init__(self, state_path, initial_capital: float = 1_000_000.0):
        self.state_path = Path(state_path)
        self.trades_path = self.state_path.with_name(self.state_path.stem + "_trades.csv")
        self.equity_path = self.state_path.with_name(self.state_path.stem + "_equity.csv")

        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            self.cash: float = state["cash"]
            self.positions: dict[str, PaperPosition] = {
                sym: PaperPosition(**pos) for sym, pos in state["positions"].items()
            }
            self.pending_entries: set[str] = set(state["pending_entries"])
            self.pending_exits: set[str] = set(state["pending_exits"])
            self.last_run_date: str | None = state["last_run_date"]
        else:
            self.cash = initial_capital
            self.positions = {}
            self.pending_entries = set()
            self.pending_exits = set()
            self.last_run_date = None

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cash": self.cash,
            "positions": {sym: asdict(pos) for sym, pos in self.positions.items()},
            "pending_entries": sorted(self.pending_entries),
            "pending_exits": sorted(self.pending_exits),
            "last_run_date": self.last_run_date,
        }
        self.state_path.write_text(json.dumps(state, indent=2))

    def open_position(
        self, symbol: str, entry_date, price: float, quantity: int, entry_cost: float, stop_price: float | None
    ) -> None:
        self.cash -= quantity * price + entry_cost
        self.positions[symbol] = PaperPosition(
            quantity=quantity, entry_price=price, entry_date=_date_str(entry_date),
            entry_cost=entry_cost, stop_price=stop_price,
        )

    def close_position(self, symbol: str, exit_date, price: float, exit_cost: float, reason: str) -> dict:
        position = self.positions.pop(symbol)
        trade_value = position.quantity * price
        self.cash += trade_value - exit_cost
        gross_pnl = (price - position.entry_price) * position.quantity
        pnl = gross_pnl - position.entry_cost - exit_cost
        basis = position.entry_price * position.quantity
        row = {
            "symbol": symbol,
            "entry_date": position.entry_date,
            "entry_price": position.entry_price,
            "exit_date": _date_str(exit_date),
            "exit_price": price,
            "quantity": position.quantity,
            "entry_cost": position.entry_cost,
            "exit_cost": exit_cost,
            "exit_reason": reason,
            "pnl": pnl,
            "pnl_pct": pnl / basis if basis else 0.0,
        }
        self._append_csv(self.trades_path, row)
        return row

    def record_equity(self, as_of, equity: float) -> None:
        self._append_csv(self.equity_path, {"date": _date_str(as_of), "equity": equity})

    def equity_curve(self) -> pd.Series:
        if not self.equity_path.exists():
            return pd.Series(dtype="float64", name="equity")
        df = pd.read_csv(self.equity_path, parse_dates=["date"]).drop_duplicates("date", keep="last")
        return df.set_index("date")["equity"].sort_index().rename("equity")

    def trades(self) -> pd.DataFrame:
        if not self.trades_path.exists():
            return pd.DataFrame(columns=TRADE_COLUMNS)
        return pd.read_csv(self.trades_path, parse_dates=["entry_date", "exit_date"])

    @staticmethod
    def _append_csv(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

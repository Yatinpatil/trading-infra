"""Paper-trading account: cash, open positions, and orders queued for the
next daily step's open, persisted in the project's SQLite store (db/) so a
scheduled job can run once a day and pick up exactly where the previous run
left off.

Cash/positions/pending-orders only persist when `save()` is called
explicitly. Closed trades and daily equity marks are written immediately by
`close_position()`/`record_equity()` regardless of `save()`, so the account
has a permanent audit trail independent of the point-in-time snapshot --
and `equity_curve()`/`trades()` hand that ledger back in the same shape
engine/single_stock.py and engine/portfolio.py produce, so
analytics/report.py's tear sheets work unchanged on a paper account.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from db.connection import connect


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
    def __init__(self, account: str, initial_capital: float = 1_000_000.0):
        self.account = account

        with connect() as conn:
            row = conn.execute(
                "SELECT cash, last_run_date FROM accounts WHERE name = ?", (account,)
            ).fetchone()

            if row is not None:
                self.cash: float = row["cash"]
                self.last_run_date: str | None = row["last_run_date"]

                self.positions: dict[str, PaperPosition] = {
                    r["symbol"]: PaperPosition(
                        quantity=r["quantity"], entry_price=r["entry_price"], entry_date=r["entry_date"],
                        entry_cost=r["entry_cost"], stop_price=r["stop_price"],
                    )
                    for r in conn.execute(
                        "SELECT symbol, quantity, entry_price, entry_date, entry_cost, stop_price "
                        "FROM positions WHERE account = ?",
                        (account,),
                    )
                }
                self.pending_entries: set[str] = {
                    r["symbol"]
                    for r in conn.execute(
                        "SELECT symbol FROM pending_orders WHERE account = ? AND side = 'entry'", (account,)
                    )
                }
                self.pending_exits: set[str] = {
                    r["symbol"]
                    for r in conn.execute(
                        "SELECT symbol FROM pending_orders WHERE account = ? AND side = 'exit'", (account,)
                    )
                }
            else:
                self.cash = initial_capital
                self.last_run_date = None
                self.positions = {}
                self.pending_entries = set()
                self.pending_exits = set()

    def save(self) -> None:
        with connect() as conn:
            conn.execute(
                "INSERT INTO accounts (name, cash, last_run_date) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET cash = excluded.cash, last_run_date = excluded.last_run_date",
                (self.account, self.cash, self.last_run_date),
            )

            conn.execute("DELETE FROM positions WHERE account = ?", (self.account,))
            if self.positions:
                conn.executemany(
                    "INSERT INTO positions (account, symbol, quantity, entry_price, entry_date, entry_cost, stop_price) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [
                        (self.account, sym, p.quantity, p.entry_price, p.entry_date, p.entry_cost, p.stop_price)
                        for sym, p in self.positions.items()
                    ],
                )

            conn.execute("DELETE FROM pending_orders WHERE account = ?", (self.account,))
            pending_rows = [(self.account, sym, "entry") for sym in self.pending_entries]
            pending_rows += [(self.account, sym, "exit") for sym in self.pending_exits]
            if pending_rows:
                conn.executemany(
                    "INSERT INTO pending_orders (account, symbol, side) VALUES (?, ?, ?)", pending_rows
                )

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
        with connect() as conn:
            conn.execute(
                "INSERT INTO trades (account, symbol, entry_date, entry_price, exit_date, exit_price, "
                "quantity, entry_cost, exit_cost, exit_reason, pnl, pnl_pct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    self.account, row["symbol"], row["entry_date"], row["entry_price"], row["exit_date"],
                    row["exit_price"], row["quantity"], row["entry_cost"], row["exit_cost"], row["exit_reason"],
                    row["pnl"], row["pnl_pct"],
                ),
            )
        return row

    def record_equity(self, as_of, equity: float) -> None:
        with connect() as conn:
            conn.execute(
                "INSERT INTO equity_history (account, date, equity) VALUES (?, ?, ?) "
                "ON CONFLICT(account, date) DO UPDATE SET equity = excluded.equity",
                (self.account, _date_str(as_of), equity),
            )

    def equity_curve(self) -> pd.Series:
        with connect() as conn:
            rows = conn.execute(
                "SELECT date, equity FROM equity_history WHERE account = ? ORDER BY date", (self.account,)
            ).fetchall()
        if not rows:
            return pd.Series(dtype="float64", name="equity")
        index = pd.to_datetime([r["date"] for r in rows])
        return pd.Series([r["equity"] for r in rows], index=index, name="equity")

    def trades(self) -> pd.DataFrame:
        with connect() as conn:
            rows = conn.execute(
                "SELECT symbol, entry_date, entry_price, exit_date, exit_price, quantity, "
                "entry_cost, exit_cost, exit_reason, pnl, pnl_pct FROM trades WHERE account = ? ORDER BY id",
                (self.account,),
            ).fetchall()
        df = pd.DataFrame([dict(r) for r in rows], columns=TRADE_COLUMNS)
        if not df.empty:
            df["entry_date"] = pd.to_datetime(df["entry_date"])
            df["exit_date"] = pd.to_datetime(df["exit_date"])
        return df

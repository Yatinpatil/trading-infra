"""Portfolio backtest engine: runs one Strategy across many symbols at once,
aggregating signals into a single combined equity curve subject to capital
allocation and risk limits (Portfolio/Risk Layer + Execution Simulator,
sharing the same timing convention as engine/single_stock.py).

Universe selection (which symbols are even eligible) is the caller's job —
see universe/constituents.py and universe/liquidity_filter.py — this engine
just simulates trading across whatever `ohlcv_by_symbol` it's given.
"""
from dataclasses import dataclass, field

import pandas as pd

from engine.costs import compute_transaction_cost
from risk.limits import has_room_for_position, within_correlation_limit, within_sector_limit
from risk.position_sizing import fixed_fractional_size


@dataclass
class PortfolioTrade:
    symbol: str
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
class PortfolioBacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    final_equity: float
    signals_by_symbol: dict = field(repr=False, default=None)


def _trades_to_df(trades: list[PortfolioTrade]) -> pd.DataFrame:
    columns = [
        "symbol", "entry_date", "entry_price", "exit_date", "exit_price", "quantity",
        "entry_cost", "exit_cost", "exit_reason", "pnl", "pnl_pct", "holding_days",
    ]
    if not trades:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        [
            {
                "symbol": t.symbol, "entry_date": t.entry_date, "entry_price": t.entry_price,
                "exit_date": t.exit_date, "exit_price": t.exit_price, "quantity": t.quantity,
                "entry_cost": t.entry_cost, "exit_cost": t.exit_cost, "exit_reason": t.exit_reason,
                "pnl": t.pnl, "pnl_pct": t.pnl_pct, "holding_days": t.holding_days,
            }
            for t in trades
        ],
        columns=columns,
    )


def run_portfolio_backtest(
    strategy,
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    config: dict,
    initial_capital: float = 1_000_000.0,
    sector_map: dict[str, str] | None = None,
) -> PortfolioBacktestResult:
    sector_map = sector_map or {}
    costs_cfg = config.get("costs", {})
    risk_cfg = config.get("risk", {})
    stop_loss_pct = config.get("params", {}).get("stop_loss_pct")
    position_size_pct = risk_cfg.get("position_size_pct", 0.1)
    max_concurrent_positions = risk_cfg.get("max_concurrent_positions")
    max_exposure_per_sector_pct = risk_cfg.get("max_exposure_per_sector_pct")
    max_correlation = risk_cfg.get("max_correlation")

    signals_by_symbol = {sym: strategy.generate_signals(df) for sym, df in ohlcv_by_symbol.items()}
    returns_by_symbol = {sym: df["CLOSE"].pct_change() for sym, df in ohlcv_by_symbol.items()}

    calendar = sorted(set().union(*(df.index for df in ohlcv_by_symbol.values())))

    cash = initial_capital
    positions: dict[str, dict] = {}
    last_price: dict[str, float] = {}
    pending_entries: set[str] = set()
    pending_exits: set[str] = set()
    trades: list[PortfolioTrade] = []
    equity_points = []

    def sector_value(sector: str) -> float:
        return sum(
            p["quantity"] * last_price.get(sym, p["entry_price"])
            for sym, p in positions.items()
            if sector_map.get(sym) == sector
        )

    def current_equity() -> float:
        return cash + sum(p["quantity"] * last_price.get(sym, p["entry_price"]) for sym, p in positions.items())

    for date in calendar:
        # 1. Fill pending exits at today's open (frees slots/exposure before new entries)
        for symbol in sorted(pending_exits):
            df = ohlcv_by_symbol[symbol]
            if symbol not in positions or date not in df.index:
                continue
            exit_price = float(df.loc[date, "OPEN"])
            cash = _close(positions.pop(symbol), symbol, date, exit_price, "signal", costs_cfg, cash, trades)
        pending_exits = set()

        # 2. Fill pending entries at today's open, subject to risk limits
        for symbol in sorted(pending_entries):
            df = ohlcv_by_symbol[symbol]
            if symbol in positions or date not in df.index:
                continue
            if not has_room_for_position(len(positions), max_concurrent_positions):
                continue

            entry_price = float(df.loc[date, "OPEN"])
            equity_now = current_equity()
            quantity = fixed_fractional_size(equity_now, entry_price, position_size_pct)
            if quantity <= 0:
                continue
            new_value = quantity * entry_price

            sector = sector_map.get(symbol)
            if sector is not None and not within_sector_limit(
                sector_value(sector), new_value, equity_now, max_exposure_per_sector_pct
            ):
                continue

            held_returns = {sym: returns_by_symbol[sym].loc[:date] for sym in positions}
            candidate_returns = returns_by_symbol[symbol].loc[:date]
            if not within_correlation_limit(candidate_returns, held_returns, max_correlation):
                continue

            trade_value = quantity * entry_price
            entry_cost = compute_transaction_cost(trade_value, costs_cfg)
            cash -= trade_value + entry_cost
            stop_price = entry_price * (1 - stop_loss_pct) if stop_loss_pct else None
            positions[symbol] = {
                "entry_date": date, "entry_price": entry_price, "quantity": quantity,
                "stop_price": stop_price, "entry_cost": entry_cost,
            }
        pending_entries = set()

        # 3. Stop-loss check, intrabar, for all open positions
        for symbol in list(positions.keys()):
            df = ohlcv_by_symbol[symbol]
            if date not in df.index:
                continue
            position = positions[symbol]
            if position["stop_price"] is not None and float(df.loc[date, "LOW"]) <= position["stop_price"]:
                cash = _close(positions.pop(symbol), symbol, date, position["stop_price"], "stop_loss", costs_cfg, cash, trades)

        # 4. Update last known prices and decide tomorrow's pending actions from today's signals
        for symbol, df in ohlcv_by_symbol.items():
            if date not in df.index:
                continue
            last_price[symbol] = float(df.loc[date, "CLOSE"])

            symbol_signals = signals_by_symbol[symbol]
            if date not in symbol_signals.index:
                continue
            if symbol not in positions:
                if bool(symbol_signals.loc[date, "entry_long"]):
                    pending_entries.add(symbol)
            else:
                if bool(symbol_signals.loc[date, "exit_long"]):
                    pending_exits.add(symbol)

        equity_points.append((date, current_equity()))

    # Close out anything still open at the end of the backtest
    for symbol in list(positions.keys()):
        exit_price = last_price.get(symbol, positions[symbol]["entry_price"])
        cash = _close(positions.pop(symbol), symbol, calendar[-1], exit_price, "end_of_data", costs_cfg, cash, trades)
    if equity_points:
        equity_points[-1] = (equity_points[-1][0], cash)

    equity_curve = pd.Series(
        [v for _, v in equity_points], index=[d for d, _ in equity_points], name="equity"
    )
    return PortfolioBacktestResult(
        equity_curve=equity_curve,
        trades=_trades_to_df(trades),
        final_equity=cash,
        signals_by_symbol=signals_by_symbol,
    )


def _close(position: dict, symbol: str, exit_date, exit_price: float, reason: str, costs_cfg: dict, cash: float, trades: list[PortfolioTrade]) -> float:
    trade_value = position["quantity"] * exit_price
    exit_cost = compute_transaction_cost(trade_value, costs_cfg)
    cash += trade_value - exit_cost
    trades.append(
        PortfolioTrade(
            symbol=symbol,
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

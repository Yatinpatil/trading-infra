"""Canonical list of paper-trading accounts (display label, DB account
name, the config that runs it, chart colors) plus a shared loader that
assembles one account's full state -- equity curve, positions, trades,
computed metrics -- as a JSON-serializable dict.

Both the static dashboard (scripts/generate_dashboard.py) and the web API
(api/) import this, so they never drift out of sync on which accounts
exist or what "stale" means.
"""
from datetime import date, datetime

from analytics.metrics import compute_metrics
from db.connection import connect
from execution.broker import PaperBroker

ACCOUNTS = [
    {
        "key": "mean_reversion", "label": "Mean Reversion", "config": "mean_reversion_nifty50",
        "account": "mean_reversion_nifty50_portfolio", "color_light": "#1F9C8C", "color_dark": "#3BB5A6",
    },
    {
        "key": "momentum", "label": "Momentum", "config": "momentum_nifty50",
        "account": "momentum_nifty50_portfolio", "color_light": "#BD3F5A", "color_dark": "#D8637E",
    },
    {
        "key": "breakout", "label": "Breakout", "config": "breakout_nifty50",
        "account": "breakout_nifty50_portfolio", "color_light": "#9B6FC4", "color_dark": "#AE8BD6",
    },
    {
        "key": "buy_and_hold", "label": "Buy & Hold", "config": "buy_and_hold_nifty50",
        "account": "buy_and_hold_nifty50_portfolio", "color_light": "#9C7D22", "color_dark": "#C7A43A",
    },
    {
        "key": "ml_strategy", "label": "ML Strategy", "config": "ml_strategy_nifty50",
        "account": "ml_strategy_nifty50_portfolio", "color_light": "#3B7DBF", "color_dark": "#5B9BDB",
    },
]

STALE_AFTER_DAYS = 3  # a run older than this (accounting for weekends) is flagged, not just "not today"


def account_by_key(key: str) -> dict:
    for meta in ACCOUNTS:
        if meta["key"] == key:
            return meta
    raise ValueError(f"Unknown account key '{key}'. Available: {[m['key'] for m in ACCOUNTS]}")


def load_account_state(meta: dict, trade_limit: int | None = 10) -> dict:
    """`trade_limit=None` returns every trade (an account detail view);
    the static dashboard passes a small limit to keep the page short."""
    with connect() as conn:
        started = conn.execute("SELECT 1 FROM accounts WHERE name = ?", (meta["account"],)).fetchone() is not None
    broker = PaperBroker(meta["account"])

    equity = broker.equity_curve()
    trades = broker.trades()
    metrics = compute_metrics(equity, trades) if len(equity) > 1 else None

    status = "not_started"
    age_days = None
    if started and broker.last_run_date:
        age_days = (date.today() - datetime.strptime(broker.last_run_date, "%Y-%m-%d").date()).days
        status = "stale" if age_days > STALE_AFTER_DAYS else "current"

    today_change_pct = None
    if len(equity) >= 2:
        today_change_pct = float(equity.iloc[-1] / equity.iloc[-2] - 1.0)

    positions = [
        {
            "symbol": sym,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "entry_date": pos.entry_date,
            "stop_price": pos.stop_price,
        }
        for sym, pos in sorted(broker.positions.items())
    ]

    trade_records = []
    if not trades.empty:
        ordered = trades.iloc[::-1]
        if trade_limit is not None:
            ordered = ordered.head(trade_limit)
        for _, row in ordered.iterrows():
            trade_records.append(
                {
                    "symbol": row["symbol"],
                    "entry_date": str(row["entry_date"].date()) if hasattr(row["entry_date"], "date") else str(row["entry_date"]),
                    "exit_date": str(row["exit_date"].date()) if hasattr(row["exit_date"], "date") else str(row["exit_date"]),
                    "quantity": int(row["quantity"]),
                    "entry_price": float(row["entry_price"]),
                    "exit_price": float(row["exit_price"]),
                    "pnl": float(row["pnl"]),
                    "pnl_pct": float(row["pnl_pct"]),
                    "exit_reason": row["exit_reason"],
                }
            )

    equity_weekly = equity.resample("W").last().dropna() if len(equity) > 1 else equity
    dates = [d.strftime("%Y-%m-%d") for d in equity_weekly.index]
    values = [round(float(v), 2) for v in equity_weekly.values]

    return {
        **meta,
        "started": started,
        "status": status,
        "age_days": age_days,
        "last_run_date": broker.last_run_date,
        "cash": broker.cash,
        "equity": float(equity.iloc[-1]) if len(equity) else None,
        "today_change_pct": today_change_pct,
        "num_open_positions": len(positions),
        "positions": positions,
        "num_trades": len(trades),
        "trades": trade_records,
        "metrics": metrics,
        "dates": dates,
        "values": values,
        "pending_entries": sorted(broker.pending_entries),
        "pending_exits": sorted(broker.pending_exits),
    }

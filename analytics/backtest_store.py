"""Persists the latest full-period portfolio backtest per strategy, so the
dashboard and web UI can show "how this strategy performed over its whole
backtest window" next to the live paper-trading numbers -- two genuinely
different things (a multi-year simulation vs. a days-old live ledger) that
must stay visibly separate, not blended into one number.
"""
from db.connection import connect


def save_backtest_result(strategy: str, start_date: str, end_date: str, computed_at: str, metrics: dict, final_equity: float) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO backtest_results
                (strategy, start_date, end_date, computed_at, final_equity, cagr, sharpe, sortino,
                 max_drawdown, win_rate, profit_factor, num_trades, total_return)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(strategy) DO UPDATE SET
                start_date=excluded.start_date, end_date=excluded.end_date, computed_at=excluded.computed_at,
                final_equity=excluded.final_equity, cagr=excluded.cagr, sharpe=excluded.sharpe,
                sortino=excluded.sortino, max_drawdown=excluded.max_drawdown, win_rate=excluded.win_rate,
                profit_factor=excluded.profit_factor, num_trades=excluded.num_trades, total_return=excluded.total_return
            """,
            (
                strategy, start_date, end_date, computed_at, final_equity,
                metrics["cagr"], metrics["sharpe"], metrics["sortino"], metrics["max_drawdown"],
                metrics["win_rate"], metrics["profit_factor"], metrics["num_trades"], metrics["total_return"],
            ),
        )


def load_backtest_results() -> dict[str, dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM backtest_results").fetchall()
    return {row["strategy"]: dict(row) for row in rows}

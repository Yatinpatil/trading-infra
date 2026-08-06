"""Benchmark comparison: does the strategy actually beat simple buy-and-hold
on the same instrument, after the same costs? A strategy that can't clear
this bar isn't worth the added complexity and trading risk, however good its
standalone metrics look.
"""
import pandas as pd

from analytics.metrics import compute_metrics
from engine.single_stock import run_single_backtest
from strategies.buy_and_hold import BuyAndHoldStrategy


def compare_to_buy_and_hold(strategy, df: pd.DataFrame, config: dict, initial_capital: float = 1_000_000.0) -> dict:
    """Runs both `strategy` and buy-and-hold over the same data/costs; returns
    each one's metrics plus the strategy's edge (its metric minus
    buy-and-hold's) for cagr/sharpe/max_drawdown.
    """
    strategy_result = run_single_backtest(strategy, df, config, initial_capital)
    strategy_metrics = compute_metrics(strategy_result.equity_curve, strategy_result.trades)

    benchmark_result = run_single_backtest(BuyAndHoldStrategy(), df, config, initial_capital)
    benchmark_metrics = compute_metrics(benchmark_result.equity_curve, benchmark_result.trades)

    edge = {
        f"edge_{key}": strategy_metrics[key] - benchmark_metrics[key]
        for key in ("cagr", "sharpe", "max_drawdown")
    }

    return {
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        **edge,
        "beats_benchmark_cagr": strategy_metrics["cagr"] > benchmark_metrics["cagr"],
    }

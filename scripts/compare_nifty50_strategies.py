"""Run every implemented strategy as a portfolio backtest over the same
NIFTY 50 universe/costs/risk settings and the freshly backfilled 2021-2026
data, then build a tear sheet per strategy plus a side-by-side comparison --
the same workflow Phase 6 (analytics/report.py) was built for.

    python scripts/compare_nifty50_strategies.py --start 2021-01-01 --end 2026-08-05
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.metrics import compute_metrics
from analytics.report import compare_strategies, generate_report
from configs import load_config
from main import build_strategy, resolve_portfolio_symbols
from data.loaders import get_ohlcv
from engine.portfolio import run_portfolio_backtest

STRATEGIES = {
    "mean_reversion": {"lookback": 20, "entry_zscore": -2.0, "exit_zscore": 0.0, "stop_loss_pct": 0.05},
    "momentum": {"lookback": 60, "entry_threshold": 0.10, "exit_threshold": 0.0, "stop_loss_pct": 0.05},
    "breakout": {"entry_lookback": 20, "exit_lookback": 10, "stop_loss_pct": 0.05},
    "buy_and_hold": {"stop_loss_pct": 0.05},
    "rsi_mean_reversion": {"lookback": 14, "entry_rsi": 30.0, "exit_rsi": 50.0, "stop_loss_pct": 0.05},
    "bollinger_breakout": {"lookback": 20, "num_std": 2.0, "stop_loss_pct": 0.05},
    "adx_trend": {"lookback": 14, "entry_adx": 25.0, "stop_loss_pct": 0.05},
}


def run_all(start: str, end: str, output_dir: str, initial_capital: float = 1_000_000.0) -> dict:
    base_config = load_config("mean_reversion_nifty50")

    symbols = resolve_portfolio_symbols(base_config, start)
    print(f"Universe: {len(symbols)} symbols eligible as of {start}: {symbols}\n")

    ohlcv_by_symbol = {}
    for symbol in symbols:
        df = get_ohlcv(symbol, start, end)
        if not df.empty:
            ohlcv_by_symbol[symbol] = df

    metrics_by_strategy = {}
    for name, params in STRATEGIES.items():
        config = {**base_config, "strategy": name, "params": params}
        strategy = build_strategy(config)
        result = run_portfolio_backtest(strategy, ohlcv_by_symbol, config, initial_capital)
        metrics = compute_metrics(result.equity_curve, result.trades)
        metrics_by_strategy[name] = metrics

        report_path = generate_report(f"{name}_nifty50", result, output_dir)
        print(
            f"{name:15s}  final_equity={result.final_equity:>14,.2f}  trades={len(result.trades):>4d}  "
            f"cagr={metrics['cagr']:.2%}  sharpe={metrics['sharpe']:.2f}  "
            f"max_dd={metrics['max_drawdown']:.2%}  -> {report_path}"
        )

    comparison = compare_strategies(metrics_by_strategy)
    comparison_path = Path(output_dir) / "nifty50_strategy_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path)
    print(f"\nComparison table written to {comparison_path}")
    print(comparison)

    return {"metrics": metrics_by_strategy, "comparison_path": comparison_path}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare all strategies over the NIFTY 50 universe.")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-08-05")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_all(args.start, args.end, args.output_dir, args.capital)


if __name__ == "__main__":
    main()

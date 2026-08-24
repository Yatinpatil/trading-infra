"""Walk-forward, out-of-sample evaluation of MLStrategy against the three
rule-based strategies and buy-and-hold, over the same NIFTY 50 universe used
by scripts/compare_nifty50_strategies.py.

MLStrategy is refit from scratch on each rolling training window (with the
last `horizon` rows of that window embargoed, per ml/features.py) and every
strategy is then scored only on the following, disjoint test window -- so
the headline numbers are what an investor would have actually realized
rebuilding the model every ~6 months, never an in-sample fit. Per-window
train-vs-test CAGR for MLStrategy is also reported, since a wide gap there
is the overfitting signal validation/walk_forward.py's own docstring warns
to watch for.

    python scripts/evaluate_ml_strategy.py
    python scripts/evaluate_ml_strategy.py --label regularized --max-depth 2 --min-samples-leaf 200 --l2 2.0 --learning-rate 0.03 --max-iter 100
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from analytics.metrics import compute_metrics
from analytics.report import generate_report
from configs import load_config
from data.loaders import get_ohlcv
from engine.portfolio import run_portfolio_backtest
from main import build_strategy, resolve_portfolio_symbols
from strategies.ml_strategy import MLStrategy

RULE_STRATEGIES = {
    "mean_reversion": {"lookback": 20, "entry_zscore": -2.0, "exit_zscore": 0.0, "stop_loss_pct": 0.05},
    "momentum": {"lookback": 60, "entry_threshold": 0.10, "exit_threshold": 0.0, "stop_loss_pct": 0.05},
    "breakout": {"entry_lookback": 20, "exit_lookback": 10, "stop_loss_pct": 0.05},
    "buy_and_hold": {"stop_loss_pct": 0.05},
}

TRAIN_SIZE = 750  # ~3 trading years
TEST_SIZE = 125  # ~6 months


class ChainedResult:
    """Adapts a chained equity curve + concatenated trades to the
    `.equity_curve`/`.trades` duck type analytics.report.generate_report
    expects."""

    def __init__(self, equity_curve: pd.Series, trades: pd.DataFrame):
        self.equity_curve = equity_curve
        self.trades = trades
        self.final_equity = equity_curve.iloc[-1] if len(equity_curve) else None


def walk_forward_windows_by_date(dates, train_size, test_size, step_size=None):
    """Same idea as validation.walk_forward.walk_forward_windows, but
    operating on one shared trading calendar rather than a single symbol's
    row count -- every symbol's DataFrame must be sliced to the SAME
    train/test date range in a multi-symbol portfolio evaluation.
    """
    step_size = step_size or test_size
    windows = []
    start = 0
    n = len(dates)
    while start + train_size + test_size <= n:
        windows.append((dates[start : start + train_size], dates[start + train_size : start + train_size + test_size]))
        start += step_size
    return windows


def chain_equity_curves(curves: list[pd.Series]) -> pd.Series:
    """Concatenate independently-run window equity curves into one
    continuously-compounding curve, rebasing each window to continue from
    the previous window's ending equity instead of resetting to the
    original initial_capital."""
    chained = []
    running_capital = None
    for curve in curves:
        if curve.empty:
            continue
        scaled = curve if running_capital is None else curve / curve.iloc[0] * running_capital
        chained.append(scaled)
        running_capital = scaled.iloc[-1]
    return pd.concat(chained) if chained else pd.Series(dtype="float64")


def parse_args(argv=None) -> argparse.Namespace:
    # defaults mirror MLStrategy.default_params -- the regularized config
    # that closed most of the walk-forward train/test gap. Pass
    # --max-depth 4 --min-samples-leaf 40 --l2 0.1 --learning-rate 0.05
    # --max-iter 200 to reproduce the original, overfit baseline.
    parser = argparse.ArgumentParser(description="Walk-forward evaluate MLStrategy against the rule-based strategies.")
    parser.add_argument("--label", default="", help="Suffix for report filenames, e.g. 'baseline'")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--entry-threshold", type=float, default=0.6)
    parser.add_argument("--exit-threshold", type=float, default=0.45)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-samples-leaf", type=int, default=200)
    parser.add_argument("--l2", type=float, default=2.0, help="l2_regularization")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ml_params = {
        "horizon": args.horizon,
        "entry_threshold": args.entry_threshold,
        "exit_threshold": args.exit_threshold,
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_samples_leaf": args.min_samples_leaf,
        "l2_regularization": args.l2,
    }
    suffix = f"_{args.label}" if args.label else ""
    print(f"ML params: {ml_params}\n")

    base_config = load_config("mean_reversion_nifty50")
    start, end = "2021-01-01", "2026-08-05"
    symbols = resolve_portfolio_symbols(base_config, start)

    ohlcv_by_symbol = {}
    for symbol in symbols:
        df = get_ohlcv(symbol, start, end)
        if not df.empty:
            ohlcv_by_symbol[symbol] = df

    calendar = sorted(set().union(*(df.index for df in ohlcv_by_symbol.values())))
    windows = walk_forward_windows_by_date(calendar, TRAIN_SIZE, TEST_SIZE)
    print(f"{len(windows)} walk-forward windows across {len(calendar)} trading days, {len(ohlcv_by_symbol)} symbols\n")

    strategy_names = list(RULE_STRATEGIES) + ["ml_strategy"]
    window_equity = {name: [] for name in strategy_names}
    window_trades = {name: [] for name in strategy_names}
    horizon = ml_params["horizon"]

    for w, (train_dates, test_dates) in enumerate(windows):
        train_start, train_end = train_dates[0], train_dates[-1]
        test_start, test_end = test_dates[0], test_dates[-1]

        train_by_symbol = {s: df.loc[train_start:train_end] for s, df in ohlcv_by_symbol.items()}
        train_by_symbol = {s: df for s, df in train_by_symbol.items() if len(df) > horizon + 30}
        test_by_symbol = {
            s: df.loc[test_start:test_end] for s, df in ohlcv_by_symbol.items() if s in train_by_symbol
        }
        test_by_symbol = {s: df for s, df in test_by_symbol.items() if not df.empty}

        print(f"window {w}: train {train_start.date()}..{train_end.date()}  test {test_start.date()}..{test_end.date()}")

        for name, params in RULE_STRATEGIES.items():
            config = {**base_config, "strategy": name, "params": params}
            strategy = build_strategy(config)
            result = run_portfolio_backtest(strategy, test_by_symbol, config, 1_000_000.0)
            window_equity[name].append(result.equity_curve)
            window_trades[name].append(result.trades)

        embargoed = {s: df.iloc[:-horizon] for s, df in train_by_symbol.items()}
        ml_strategy = MLStrategy(ml_params)
        ml_strategy.fit(embargoed)

        ml_config = {**base_config, "strategy": "ml_strategy", "params": ml_params}
        train_result = run_portfolio_backtest(ml_strategy, embargoed, ml_config, 1_000_000.0)
        test_result = run_portfolio_backtest(ml_strategy, test_by_symbol, ml_config, 1_000_000.0)
        train_cagr = compute_metrics(train_result.equity_curve, train_result.trades)["cagr"]
        test_cagr = compute_metrics(test_result.equity_curve, test_result.trades)["cagr"]
        print(f"    ml_strategy: train_cagr={train_cagr:+.2%}  test_cagr={test_cagr:+.2%}  (gap={train_cagr - test_cagr:+.2%})")

        window_equity["ml_strategy"].append(test_result.equity_curve)
        window_trades["ml_strategy"].append(test_result.trades)

    print("\n--- out-of-sample results (chained across all windows) ---\n")
    metrics_by_strategy = {}
    for name in strategy_names:
        equity = chain_equity_curves(window_equity[name])
        trades = pd.concat(window_trades[name], ignore_index=True) if window_trades[name] else pd.DataFrame()
        metrics = compute_metrics(equity, trades)
        metrics_by_strategy[name] = metrics
        report_name = f"{name}_oos{suffix}" if name == "ml_strategy" else f"{name}_oos"
        report_path = generate_report(report_name, ChainedResult(equity, trades), "reports")
        print(
            f"{name:15s}  cagr={metrics['cagr']:+.2%}  sharpe={metrics['sharpe']:.2f}  "
            f"max_dd={metrics['max_drawdown']:.2%}  trades={metrics['num_trades']:.0f}  -> {report_path}"
        )

    comparison = pd.DataFrame(metrics_by_strategy).T
    comparison.index.name = "strategy"
    comparison_path = Path("reports") / f"nifty50_oos_comparison{suffix}.csv"
    comparison.to_csv(comparison_path)
    print(f"\nComparison table written to {comparison_path}")

    return metrics_by_strategy


if __name__ == "__main__":
    main()

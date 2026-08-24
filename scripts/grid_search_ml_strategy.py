"""Small grid search over MLStrategy's regularization hyperparameters, using
the same walk-forward OOS methodology as evaluate_ml_strategy.py, to check
whether the regularized default found there (max_depth=2, min_samples_leaf=200,
l2_regularization=2.0) is a robust region or a single lucky pick. The
rule-based strategies are skipped here -- their OOS numbers don't depend on
these hyperparameters; see evaluate_ml_strategy.py for those.

    python scripts/grid_search_ml_strategy.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from analytics.metrics import compute_metrics
from configs import load_config
from data.loaders import get_ohlcv
from engine.portfolio import run_portfolio_backtest
from evaluate_ml_strategy import TEST_SIZE, TRAIN_SIZE, chain_equity_curves, walk_forward_windows_by_date
from main import resolve_portfolio_symbols
from strategies.ml_strategy import MLStrategy

GRID = {
    "max_depth": [1, 2, 4],
    "min_samples_leaf": [100, 200],
    "l2_regularization": [0.5, 2.0],
}
FIXED_PARAMS = {"horizon": 5, "entry_threshold": 0.6, "exit_threshold": 0.45, "max_iter": 100, "learning_rate": 0.03}


def main():
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
    horizon = FIXED_PARAMS["horizon"]

    # slice train/test data per window once, reused across every grid point
    sliced_windows = []
    for train_dates, test_dates in windows:
        train_start, train_end = train_dates[0], train_dates[-1]
        test_start, test_end = test_dates[0], test_dates[-1]
        train_by_symbol = {s: df.loc[train_start:train_end] for s, df in ohlcv_by_symbol.items()}
        train_by_symbol = {s: df for s, df in train_by_symbol.items() if len(df) > horizon + 30}
        test_by_symbol = {s: df.loc[test_start:test_end] for s, df in ohlcv_by_symbol.items() if s in train_by_symbol}
        test_by_symbol = {s: df for s, df in test_by_symbol.items() if not df.empty}
        embargoed = {s: df.iloc[:-horizon] for s, df in train_by_symbol.items()}
        sliced_windows.append((embargoed, test_by_symbol))

    keys = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"{len(combos)} combinations x {len(sliced_windows)} windows\n")

    rows = []
    for combo_values in combos:
        combo = dict(zip(keys, combo_values))
        params = {**FIXED_PARAMS, **combo}

        window_equity, window_trades = [], []
        for embargoed, test_by_symbol in sliced_windows:
            strategy = MLStrategy(params)
            strategy.fit(embargoed)
            config = {**base_config, "strategy": "ml_strategy", "params": params}
            result = run_portfolio_backtest(strategy, test_by_symbol, config, 1_000_000.0)
            window_equity.append(result.equity_curve)
            window_trades.append(result.trades)

        equity = chain_equity_curves(window_equity)
        trades = pd.concat(window_trades, ignore_index=True) if window_trades else pd.DataFrame()
        metrics = compute_metrics(equity, trades)
        rows.append({**combo, **metrics})
        print(
            f"{combo}  ->  cagr={metrics['cagr']:+.2%}  sharpe={metrics['sharpe']:.2f}  "
            f"max_dd={metrics['max_drawdown']:.2%}  trades={metrics['num_trades']:.0f}"
        )

    results = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    out_path = Path("reports") / "ml_strategy_grid_search.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nFull grid written to {out_path}")
    print(results[["max_depth", "min_samples_leaf", "l2_regularization", "cagr", "sharpe", "max_drawdown", "num_trades"]].round(4).to_string())


if __name__ == "__main__":
    main()

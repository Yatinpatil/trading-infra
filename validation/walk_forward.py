"""Train/test splitting, walk-forward validation, and parameter sensitivity
sweeps — the tools that separate a real edge from a curve-fit one.

Every function here is chronological only: splits never shuffle dates, and
parameters are always chosen on a train window before being scored on a
disjoint, later test window. That ordering is what makes the "out-of-sample"
numbers this module produces actually mean something.
"""
import itertools

import pandas as pd

from analytics.metrics import compute_metrics
from engine.single_stock import run_single_backtest


def train_test_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def walk_forward_windows(
    df: pd.DataFrame, train_size: int, test_size: int, step_size: int | None = None
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling (train_size, test_size) windows, advancing by `step_size` bars
    (defaults to test_size, i.e. non-overlapping test windows).
    """
    step_size = step_size or test_size
    windows = []
    start = 0
    n = len(df)
    while start + train_size + test_size <= n:
        train = df.iloc[start : start + train_size]
        test = df.iloc[start + train_size : start + train_size + test_size]
        windows.append((train, test))
        start += step_size
    return windows


def evaluate_strategy(strategy, df: pd.DataFrame, config: dict, initial_capital: float = 1_000_000.0) -> dict:
    result = run_single_backtest(strategy, df, config, initial_capital)
    return compute_metrics(result.equity_curve, result.trades)


def grid_search(
    strategy_cls,
    param_grid: dict[str, list],
    df: pd.DataFrame,
    config: dict,
    initial_capital: float = 1_000_000.0,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """Every combination in the cartesian product of `param_grid`, backtested
    on `df`, sorted best-`metric`-first.
    """
    keys = list(param_grid.keys())
    rows = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        metrics = evaluate_strategy(strategy_cls(params), df, config, initial_capital)
        rows.append({**params, **metrics})

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df
    return result_df.sort_values(metric, ascending=False).reset_index(drop=True)


def parameter_sensitivity_sweep(
    strategy_cls,
    param_grid: dict[str, list],
    df: pd.DataFrame,
    config: dict,
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """Same grid as grid_search but ordered by parameter value, not
    performance — the shape to look at for cliff-edge sensitivity (does
    `metric` collapse for a small parameter nudge, rather than degrading
    smoothly).
    """
    results = grid_search(strategy_cls, param_grid, df, config, initial_capital, metric="sharpe")
    if results.empty:
        return results
    return results.sort_values(list(param_grid.keys())).reset_index(drop=True)


def walk_forward_validation(
    strategy_cls,
    param_grid: dict[str, list],
    df: pd.DataFrame,
    config: dict,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    metric: str = "sharpe",
    initial_capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """For each rolling window: grid-search `param_grid` on the train slice,
    take the best-`metric` params, then score those (unmodified) params on
    the test slice. One row per window with train and test metrics side by
    side — a large train/test gap is the overfitting signal to watch for.
    """
    windows = walk_forward_windows(df, train_size, test_size, step_size)
    rows = []

    for i, (train_df, test_df) in enumerate(windows):
        grid_results = grid_search(strategy_cls, param_grid, train_df, config, initial_capital, metric)
        if grid_results.empty:
            continue

        best_row = grid_results.iloc[0]
        # best_row is a Series spanning both int params and float metrics, so
        # pandas upcasts it to a single dtype (e.g. int lookback -> float64);
        # cast each param back to its original grid type before reuse.
        best_params = {k: type(param_grid[k][0])(best_row[k]) for k in param_grid.keys()}
        test_metrics = evaluate_strategy(strategy_cls(best_params), test_df, config, initial_capital)

        rows.append(
            {
                "window": i,
                "train_start": train_df.index[0],
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0],
                "test_end": test_df.index[-1],
                **{f"param_{k}": v for k, v in best_params.items()},
                f"train_{metric}": best_row[metric],
                **{f"test_{m}": v for m, v in test_metrics.items()},
            }
        )

    return pd.DataFrame(rows)

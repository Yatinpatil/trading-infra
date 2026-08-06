import numpy as np
import pandas as pd
import pytest

from strategies.mean_reversion import MeanReversionStrategy
from validation.walk_forward import (
    grid_search,
    parameter_sensitivity_sweep,
    train_test_split,
    walk_forward_validation,
    walk_forward_windows,
)

NO_COST_CONFIG = {"costs": {}, "risk": {"position_size_pct": 0.5}, "params": {}}


def _mean_reverting_ohlcv(n=300, seed=0):
    rng = np.random.default_rng(seed)
    # Ornstein-Uhlenbeck-ish process so mean reversion actually has something to trade
    close = np.zeros(n)
    close[0] = 100
    for i in range(1, n):
        close[i] = close[i - 1] + 0.15 * (100 - close[i - 1]) + rng.normal(0, 2)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(close, index=dates)
    return pd.DataFrame(
        {"OPEN": close, "HIGH": close + 1, "LOW": close - 1, "CLOSE": close, "VOLUME": 1000}
    )


def test_train_test_split_is_chronological_and_non_overlapping():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"CLOSE": range(10)}, index=dates)

    train, test = train_test_split(df, train_frac=0.7)

    assert len(train) == 7
    assert len(test) == 3
    assert train.index[-1] < test.index[0]
    pd.testing.assert_index_equal(train.index.append(test.index), df.index)


def test_walk_forward_windows_boundaries():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"CLOSE": range(100)}, index=dates)

    windows = walk_forward_windows(df, train_size=40, test_size=20, step_size=20)

    assert len(windows) == 3
    for train_df, test_df in windows:
        assert len(train_df) == 40
        assert len(test_df) == 20
        # test window starts exactly one bar after train window ends
        train_end_pos = df.index.get_loc(train_df.index[-1])
        test_start_pos = df.index.get_loc(test_df.index[0])
        assert test_start_pos == train_end_pos + 1


def test_walk_forward_windows_defaults_step_to_test_size():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"CLOSE": range(60)}, index=dates)

    windows = walk_forward_windows(df, train_size=30, test_size=15)  # step defaults to 15

    assert len(windows) == 2  # non-overlapping test windows


def test_grid_search_returns_one_row_per_combination_sorted_by_metric():
    df = _mean_reverting_ohlcv()
    param_grid = {"lookback": [10, 20], "entry_zscore": [-1.5, -2.0]}

    results = grid_search(MeanReversionStrategy, param_grid, df, NO_COST_CONFIG, metric="sharpe")

    assert len(results) == 4  # 2 x 2 cartesian product
    assert list(results.columns[:2]) == ["lookback", "entry_zscore"]
    assert (results["sharpe"].diff().dropna() <= 1e-9).all()  # descending


def test_parameter_sensitivity_sweep_ordered_by_params_not_performance():
    df = _mean_reverting_ohlcv()
    param_grid = {"lookback": [10, 20], "entry_zscore": [-1.5, -2.0]}

    results = parameter_sensitivity_sweep(MeanReversionStrategy, param_grid, df, NO_COST_CONFIG)

    assert len(results) == 4
    assert results["lookback"].tolist() == sorted(results["lookback"].tolist())


def test_walk_forward_validation_produces_train_and_test_metrics_per_window():
    df = _mean_reverting_ohlcv(n=260)
    param_grid = {"lookback": [10, 20], "entry_zscore": [-1.5, -2.0]}

    results = walk_forward_validation(
        MeanReversionStrategy, param_grid, df, NO_COST_CONFIG,
        train_size=100, test_size=50, metric="sharpe",
    )

    assert len(results) == 3  # floor((260-100-50)/50) + 1 = 3 windows
    assert "train_sharpe" in results.columns
    assert "test_sharpe" in results.columns
    assert "param_lookback" in results.columns
    for _, row in results.iterrows():
        assert row["train_end"] < row["test_start"]

import pandas as pd
import pytest

from analytics.metrics import (
    avg_trade_duration,
    cagr,
    compute_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def _equity(values):
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates, dtype="float64")


def test_cagr_exact_doubling_over_one_year_worth_of_periods():
    equity = _equity([100.0] * 251 + [200.0])  # 252 points -> years = 252/252 = 1
    assert cagr(equity, periods_per_year=252) == pytest.approx(1.0)


def test_cagr_zero_for_flat_or_insufficient_data():
    assert cagr(_equity([100.0])) == 0.0
    assert cagr(_equity([100.0, 100.0])) == pytest.approx(0.0)


def test_sharpe_ratio_zero_when_returns_constant():
    equity = _equity([100 * 1.01**i for i in range(20)])  # constant daily return -> zero std
    returns = equity.pct_change().dropna()
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_upward_noisy_series():
    returns = pd.Series([0.02, -0.01, 0.03, 0.01, -0.005, 0.02])
    assert sharpe_ratio(returns) > 0


def test_sortino_ignores_upside_volatility():
    # same mean/upside as a volatile-but-no-downside series -> should not penalize
    all_upside = pd.Series([0.01, 0.05, 0.02, 0.03])
    assert sortino_ratio(all_upside) == 0.0  # no downside deviation -> guarded to 0.0


def test_max_drawdown_hand_computed():
    equity = _equity([100, 120, 90, 110])
    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_win_rate():
    trades = pd.DataFrame({"pnl": [10, -5, 20, -1]})
    assert win_rate(trades) == 0.5


def test_win_rate_empty():
    assert win_rate(pd.DataFrame(columns=["pnl"])) == 0.0


def test_profit_factor():
    trades = pd.DataFrame({"pnl": [10, -5, 20, -1]})
    assert profit_factor(trades) == pytest.approx(30 / 6)


def test_profit_factor_no_losses_is_infinite():
    trades = pd.DataFrame({"pnl": [10, 20]})
    assert profit_factor(trades) == float("inf")


def test_avg_trade_duration():
    trades = pd.DataFrame({"holding_days": [1, 3, 5]})
    assert avg_trade_duration(trades) == pytest.approx(3.0)


def test_compute_metrics_bundles_expected_keys():
    equity = _equity([100, 105, 103, 110])
    trades = pd.DataFrame({"pnl": [5, -2], "holding_days": [2, 3]})

    metrics = compute_metrics(equity, trades)

    expected_keys = {
        "cagr", "sharpe", "sortino", "max_drawdown", "win_rate",
        "profit_factor", "avg_trade_duration_days", "num_trades", "total_return",
    }
    assert expected_keys <= metrics.keys()
    assert metrics["num_trades"] == 2

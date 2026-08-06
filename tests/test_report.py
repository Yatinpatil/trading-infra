import pandas as pd
import pytest

from analytics.report import (
    compare_strategies,
    generate_report,
    generate_tearsheet,
    plot_drawdown,
    plot_equity_curve,
)


def _equity():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    values = [100 + i + (5 if i % 7 == 0 else 0) for i in range(20)]
    return pd.Series(values, index=dates, dtype="float64")


def _trades():
    return pd.DataFrame(
        {
            "entry_date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "exit_date": pd.date_range("2024-01-05", periods=3, freq="D"),
            "entry_price": [100.0, 105.0, 102.0],
            "exit_price": [110.0, 100.0, 108.0],
            "pnl": [10.0, -5.0, 6.0],
            "pnl_pct": [0.1, -0.048, 0.059],
            "holding_days": [4, 4, 4],
        }
    )


def test_plot_equity_curve_returns_figure_with_data():
    fig = plot_equity_curve(_equity(), title="Test Equity")
    ax = fig.axes[0]
    assert ax.get_title() == "Test Equity"
    assert len(ax.lines[0].get_xdata()) == 20


def test_plot_drawdown_returns_figure():
    fig = plot_drawdown(_equity(), title="Test Drawdown")
    ax = fig.axes[0]
    assert ax.get_title() == "Test Drawdown"


def test_compare_strategies_builds_one_row_per_strategy():
    results = {
        "mean_reversion": {"cagr": 0.10, "sharpe": 1.2},
        "momentum": {"cagr": 0.05, "sharpe": 0.8},
    }
    comparison = compare_strategies(results)

    assert list(comparison.index) == ["mean_reversion", "momentum"]
    assert list(comparison.columns) == ["cagr", "sharpe"]
    assert comparison.loc["mean_reversion", "sharpe"] == 1.2


def test_generate_tearsheet_writes_self_contained_html(tmp_path):
    output_path = tmp_path / "reports" / "mean_reversion.html"

    result_path = generate_tearsheet("mean_reversion", _equity(), _trades(), output_path)

    assert result_path == output_path
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "mean_reversion" in content
    assert "Equity Curve" in content
    assert "Drawdown" in content
    assert "data:image/png;base64," in content  # charts embedded, no external files
    assert "Trade Log (3 trades)" in content


def test_generate_tearsheet_handles_no_trades(tmp_path):
    output_path = tmp_path / "empty.html"
    empty_trades = pd.DataFrame(columns=["pnl", "holding_days"])

    generate_tearsheet("empty_strategy", _equity(), empty_trades, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "No trades." in content


class _FakeBacktestResult:
    def __init__(self, equity_curve, trades):
        self.equity_curve = equity_curve
        self.trades = trades


def test_generate_report_works_with_duck_typed_backtest_result(tmp_path):
    fake_result = _FakeBacktestResult(_equity(), _trades())

    output_path = generate_report("momentum", fake_result, tmp_path)

    assert output_path == tmp_path / "momentum.html"
    assert output_path.exists()

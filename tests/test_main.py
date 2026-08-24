import numpy as np
import pandas as pd
import pytest

import main
from engine.portfolio import PortfolioBacktestResult
from engine.single_stock import BacktestResult
from strategies import get_strategy_class
from strategies.mean_reversion import MeanReversionStrategy


def _fake_ohlcv(symbol, start, end, adjust=True, use_cache=True):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    dates = pd.bdate_range(start, end)
    closes = 100 + np.abs(rng.normal(0, 1, len(dates)).cumsum())
    return pd.DataFrame(
        {
            "OPEN": closes,
            "HIGH": closes * 1.01,
            "LOW": closes * 0.99,
            "CLOSE": closes,
            "VOLUME": rng.integers(10_000, 50_000, len(dates)),
        },
        index=dates,
    )


CONFIG = {
    "strategy": "mean_reversion",
    "params": {"lookback": 10, "entry_zscore": -2.0, "exit_zscore": 0.0, "stop_loss_pct": 0.05},
    "universe": {"index": "NIFTY500", "min_avg_daily_value": 0},
    "costs": {"brokerage_pct": 0.0003},
    "risk": {"max_concurrent_positions": 5, "position_size_pct": 0.1},
}


def test_get_strategy_class_resolves_all_registered_strategies():
    assert get_strategy_class("mean_reversion") is MeanReversionStrategy


def test_get_strategy_class_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy_class("does_not_exist")


def test_build_strategy_applies_config_params():
    strategy = main.build_strategy(CONFIG)
    assert isinstance(strategy, MeanReversionStrategy)
    assert strategy.params["lookback"] == 10


def test_run_single_produces_backtest_result(monkeypatch):
    monkeypatch.setattr(main, "get_ohlcv", _fake_ohlcv)
    result = main.run_single(CONFIG, "RELIANCE", "2023-01-01", "2023-12-31", 1_000_000.0)
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) > 0


def test_run_portfolio_produces_backtest_result(monkeypatch):
    monkeypatch.setattr(main, "get_ohlcv", _fake_ohlcv)
    monkeypatch.setattr(main, "get_universe", lambda index_name, as_of_date: ["AAA", "BBB", "CCC"])
    result = main.run_portfolio(CONFIG, "2023-01-01", "2023-12-31", 1_000_000.0)
    assert isinstance(result, PortfolioBacktestResult)
    assert len(result.equity_curve) > 0


def test_run_portfolio_raises_when_nothing_survives_liquidity_filter(monkeypatch):
    monkeypatch.setattr(main, "get_ohlcv", _fake_ohlcv)
    monkeypatch.setattr(main, "get_universe", lambda index_name, as_of_date: ["AAA"])
    config = {**CONFIG, "universe": {**CONFIG["universe"], "min_avg_daily_value": 1e18}}
    with pytest.raises(ValueError, match="liquidity filter"):
        main.run_portfolio(config, "2023-01-01", "2023-12-31", 1_000_000.0)


def test_main_end_to_end_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_ohlcv", _fake_ohlcv)
    monkeypatch.setattr(main, "load_config", lambda name: CONFIG)
    monkeypatch.setattr(main, "get_universe", lambda index_name, as_of_date: ["AAA", "BBB"])

    result = main.main(
        [
            "--config", "mean_reversion",
            "--start", "2023-01-01",
            "--end", "2023-12-31",
            "--output-dir", str(tmp_path),
        ]
    )

    assert isinstance(result, PortfolioBacktestResult)
    assert (tmp_path / "mean_reversion_portfolio.html").exists()


def test_main_single_symbol_mode_writes_report(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_ohlcv", _fake_ohlcv)
    monkeypatch.setattr(main, "load_config", lambda name: CONFIG)

    result = main.main(
        [
            "--config", "mean_reversion",
            "--symbol", "RELIANCE",
            "--start", "2023-01-01",
            "--end", "2023-12-31",
            "--output-dir", str(tmp_path),
        ]
    )

    assert isinstance(result, BacktestResult)
    assert (tmp_path / "mean_reversion_RELIANCE.html").exists()

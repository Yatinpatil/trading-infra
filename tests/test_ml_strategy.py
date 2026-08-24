import numpy as np
import pandas as pd
import pytest

from engine.portfolio import run_portfolio_backtest
from ml.features import FEATURE_COLUMNS, build_features
from strategies.ml_strategy import MLStrategy


def _random_walk_ohlcv(n=80, seed=1, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    close = 100 + rng.normal(0, 1, n).cumsum()
    close = np.abs(close) + 50
    return pd.DataFrame(
        {
            "OPEN": close * (1 + rng.normal(0, 0.001, n)),
            "HIGH": close * 1.01,
            "LOW": close * 0.99,
            "CLOSE": close,
            "VOLUME": rng.integers(10_000, 50_000, n),
        },
        index=dates,
    )


def _dip_and_recover_ohlcv(n=600, seed=1, dip_every=40, start="2021-01-01"):
    """A synthetic series with a deliberately learnable, repeated pattern:
    every `dip_every` bars the price is knocked down sharply, then recovers
    over the next few days -- so a low z-score reliably precedes a positive
    forward return, and periods in between (flat/noisy) don't. Enough
    repetitions across a few "symbols" for a tree to actually learn the
    relationship, not just memorize noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    close = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + rng.normal(0, 0.002))
        if i % dip_every == 0:
            close[i] *= 0.85  # sharp dip
        elif i % dip_every in (1, 2, 3, 4, 5):
            close[i] *= 1.035  # steady recovery over the following days
    return pd.DataFrame(
        {
            "OPEN": close,
            "HIGH": close * 1.005,
            "LOW": close * 0.995,
            "CLOSE": close,
            "VOLUME": rng.integers(10_000, 50_000, n),
        },
        index=dates,
    )


def test_generate_signals_before_fit_raises():
    strategy = MLStrategy()
    df = _random_walk_ohlcv()
    with pytest.raises(RuntimeError, match="before fit"):
        strategy.generate_signals(df)


def test_fit_raises_on_symbols_all_shorter_than_horizon():
    strategy = MLStrategy({"horizon": 100})
    short_df = _random_walk_ohlcv(n=10)
    with pytest.raises(ValueError, match="No training rows"):
        strategy.fit({"AAA": short_df})


def test_with_params_returns_a_fresh_unfitted_instance():
    strategy = MLStrategy()
    strategy.fit({"AAA": _dip_and_recover_ohlcv(n=300, seed=1)})
    variant = strategy.with_params(entry_threshold=0.7)

    assert variant.model is None
    assert variant.params["entry_threshold"] == 0.7
    assert strategy.model is not None  # original untouched


def test_model_learns_the_dip_and_recover_pattern():
    train = {
        "AAA": _dip_and_recover_ohlcv(n=600, seed=1),
        "BBB": _dip_and_recover_ohlcv(n=600, seed=2),
        "CCC": _dip_and_recover_ohlcv(n=600, seed=3),
    }
    strategy = MLStrategy({"horizon": 5, "min_samples_leaf": 15})
    strategy.fit(train)

    # a fresh, held-out symbol with the same pattern but a different seed
    test_df = _dip_and_recover_ohlcv(n=200, seed=99, start="2023-01-01")
    features = build_features(test_df)
    proba = strategy.model.predict_proba(features[FEATURE_COLUMNS])[:, 1]

    # the dip day itself (sharp drop -> low z-score, about to recover) should
    # get a much higher predicted probability than a quiet day deep in the
    # flat stretch between dips -- confirming the model actually learned the
    # mechanism rather than some spurious correlation.
    dip_days = [i for i in range(40, len(test_df), 40)]
    quiet_days = [i for i in range(20, len(test_df), 40)]  # midway between dips

    assert proba[dip_days].mean() > proba[quiet_days].mean() + 0.15


def test_fitted_strategy_runs_end_to_end_in_the_portfolio_engine():
    train = {"AAA": _dip_and_recover_ohlcv(n=400, seed=1), "BBB": _dip_and_recover_ohlcv(n=400, seed=2)}
    strategy = MLStrategy({"horizon": 5, "min_samples_leaf": 15})
    strategy.fit(train)

    test_data = {
        "AAA": _dip_and_recover_ohlcv(n=150, seed=11, start="2022-06-01"),
        "BBB": _dip_and_recover_ohlcv(n=150, seed=12, start="2022-06-01"),
    }
    config = {"costs": {"brokerage_pct": 0.0003}, "risk": {"position_size_pct": 0.1}, "params": {}}
    result = run_portfolio_backtest(strategy, test_data, config, initial_capital=100_000)

    assert len(result.equity_curve) > 0
    assert result.final_equity > 0


def test_save_before_fit_raises(tmp_path):
    strategy = MLStrategy()
    with pytest.raises(RuntimeError, match="before fit"):
        strategy.save(tmp_path / "model.joblib")


def test_save_and_load_round_trips_params_model_and_fitted_at(tmp_path):
    strategy = MLStrategy({"min_samples_leaf": 15})
    strategy.fit({"AAA": _dip_and_recover_ohlcv(n=300, seed=1)})
    strategy.fitted_at = "2024-06-15"
    path = tmp_path / "model.joblib"
    strategy.save(path)

    reloaded = MLStrategy.load(path)

    assert reloaded.params == strategy.params
    assert reloaded.fitted_at == "2024-06-15"
    test_df = _dip_and_recover_ohlcv(n=100, seed=2, start="2023-01-01")
    pd.testing.assert_frame_equal(reloaded.generate_signals(test_df), strategy.generate_signals(test_df))

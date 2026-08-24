import pandas as pd

from engine.portfolio import run_portfolio_backtest
from strategies.buy_and_hold import BuyAndHoldStrategy


def _ohlcv(dates):
    return pd.DataFrame({"OPEN": 100, "HIGH": 101, "LOW": 99, "CLOSE": 100, "VOLUME": 1000}, index=dates)


def test_entry_signaled_on_every_bar_no_exits():
    # every bar carries entry_long=True -- the engine's "not already
    # holding" check is what makes this enter exactly once, not this signal.
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    signals = BuyAndHoldStrategy().generate_signals(_ohlcv(dates))

    assert signals["entry_long"].all()
    assert not signals["exit_long"].any()


def test_backtest_still_enters_exactly_once_on_day_one():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    config = {"costs": {}, "risk": {"position_size_pct": 1.0}, "params": {}}

    result = run_portfolio_backtest(BuyAndHoldStrategy(), {"AAA": _ohlcv(dates)}, config, initial_capital=100_000)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["entry_date"] == dates[1]  # next-bar-open fill after day-one's signal


def test_a_rolling_window_that_no_longer_starts_at_the_true_first_bar_still_signals_entry_today():
    # this is exactly PaperTradingEngine's shape: a trailing window ending
    # "today", whose own first row is far in the past, not today. The old
    # "signal only on df.index[0]" strategy would never fire here again
    # once enough history had accumulated -- a live account would sit in
    # cash forever.
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    rolling_window = _ohlcv(dates).iloc[-400:]  # "today" is rolling_window.index[-1], not [0]

    signals = BuyAndHoldStrategy().generate_signals(rolling_window)

    assert bool(signals["entry_long"].iloc[-1])  # today's row must carry the entry signal

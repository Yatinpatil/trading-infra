from datetime import date, datetime

import pandas as pd
import pytest

import scripts.poll_and_run_paper_trading as poll


@pytest.fixture(autouse=True)
def _stub_ran_state(monkeypatch):
    monkeypatch.setattr(poll, "load_account_state", lambda meta: {"last_run_date": None})


def test_already_ran_today_is_a_no_op(monkeypatch):
    monkeypatch.setattr(poll, "load_account_state", lambda meta: {"last_run_date": "2026-08-25"})
    monkeypatch.setattr(poll, "_eod_data_is_out", lambda today: pytest.fail("should not check for data"))
    monkeypatch.setattr(poll, "run_daily_paper_trading", lambda argv: pytest.fail("should not run"))

    assert poll.main(today=date(2026, 8, 25)) == 0


def test_data_not_out_yet_before_fallback_hour_is_a_no_op(monkeypatch):
    monkeypatch.setattr(poll, "_eod_data_is_out", lambda today: False)
    monkeypatch.setattr(poll, "run_daily_paper_trading", lambda argv: pytest.fail("should not run"))

    result = poll.main(today=date(2026, 8, 25), now=datetime(2026, 8, 25, 17, 0))
    assert result == 0


def test_data_out_triggers_the_daily_run(monkeypatch):
    monkeypatch.setattr(poll, "_eod_data_is_out", lambda today: True)
    calls = []
    monkeypatch.setattr(poll, "run_daily_paper_trading", lambda argv: calls.append(argv) or 0)

    result = poll.main(today=date(2026, 8, 25), now=datetime(2026, 8, 25, 16, 0))
    assert result == 0
    assert calls == [[]]


def test_past_fallback_hour_runs_even_if_data_still_looks_missing(monkeypatch):
    monkeypatch.setattr(poll, "_eod_data_is_out", lambda today: False)
    calls = []
    monkeypatch.setattr(poll, "run_daily_paper_trading", lambda argv: calls.append(argv) or 0)

    result = poll.main(today=date(2026, 8, 25), now=datetime(2026, 8, 25, 19, 5))
    assert result == 0
    assert calls == [[]]


def test_eod_data_is_out_reads_the_canary_symbol(monkeypatch):
    today = date(2026, 8, 25)

    def fake_get_raw_ohlcv(symbol, start, end, use_cache):
        assert symbol == poll.CANARY_SYMBOL
        assert (start, end, use_cache) == (today, today, False)
        return pd.DataFrame({"OPEN": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp(today)]))

    monkeypatch.setattr(poll, "get_raw_ohlcv", fake_get_raw_ohlcv)
    assert poll._eod_data_is_out(today) is True


def test_eod_data_is_out_treats_a_fetch_error_as_not_out(monkeypatch):
    def raises(*args, **kwargs):
        raise RuntimeError("NSE hiccup")

    monkeypatch.setattr(poll, "get_raw_ohlcv", raises)
    assert poll._eod_data_is_out(date(2026, 8, 25)) is False

import pytest

from data.retry import with_retries


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("data.retry.time.sleep", lambda seconds: None)


def test_returns_result_on_first_success():
    assert with_retries(lambda: 42) == 42


def test_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("NSE hiccup")
        return "ok"

    assert with_retries(flaky, attempts=3) == "ok"
    assert calls["n"] == 3


def test_raises_the_last_exception_once_attempts_are_exhausted():
    def always_fails():
        raise ConnectionError("NSE is down")

    with pytest.raises(ConnectionError, match="NSE is down"):
        with_retries(always_fails, attempts=3)

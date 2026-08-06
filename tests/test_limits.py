import pandas as pd

from risk.limits import has_room_for_position, within_correlation_limit, within_sector_limit


def test_has_room_for_position_no_limit_configured():
    assert has_room_for_position(open_count=100, max_concurrent_positions=None)


def test_has_room_for_position_under_and_at_limit():
    assert has_room_for_position(open_count=2, max_concurrent_positions=3)
    assert not has_room_for_position(open_count=3, max_concurrent_positions=3)


def test_within_sector_limit_no_limit_configured():
    assert within_sector_limit(current_sector_value=1_000_000, new_position_value=1_000_000, total_equity=100, max_exposure_per_sector_pct=None)


def test_within_sector_limit_under_and_over_cap():
    assert within_sector_limit(current_sector_value=10_000, new_position_value=5_000, total_equity=100_000, max_exposure_per_sector_pct=0.20)
    assert not within_sector_limit(current_sector_value=10_000, new_position_value=15_000, total_equity=100_000, max_exposure_per_sector_pct=0.20)


def _series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))


def test_within_correlation_limit_no_held_positions():
    candidate = _series([0.01, 0.02, -0.01])
    assert within_correlation_limit(candidate, {}, max_correlation=0.5)


def test_within_correlation_limit_no_limit_configured():
    candidate = _series([0.01, 0.02, -0.01])
    held = {"AAA": _series([0.01, 0.02, -0.01])}
    assert within_correlation_limit(candidate, held, max_correlation=None)


def test_within_correlation_limit_rejects_highly_correlated_candidate():
    base = [0.01, 0.02, -0.01, 0.03, -0.02, 0.015, 0.005]
    candidate = _series(base)
    held = {"AAA": _series(base)}  # identical -> correlation 1.0
    assert not within_correlation_limit(candidate, held, max_correlation=0.8, lookback=7)


def test_within_correlation_limit_accepts_uncorrelated_candidate():
    candidate = _series([0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.005])
    held = {"AAA": _series([-0.01, 0.02, -0.015, 0.01, -0.02, 0.015, -0.005])}
    assert within_correlation_limit(candidate, held, max_correlation=0.5, lookback=7)

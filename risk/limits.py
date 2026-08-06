"""Portfolio-level risk limits. Pure functions the portfolio engine consults
before accepting a new entry — each answers "is this one constraint OK",
composed together in engine/portfolio.py.
"""
import pandas as pd


def has_room_for_position(open_count: int, max_concurrent_positions: int | None) -> bool:
    if max_concurrent_positions is None:
        return True
    return open_count < max_concurrent_positions


def within_sector_limit(
    current_sector_value: float,
    new_position_value: float,
    total_equity: float,
    max_exposure_per_sector_pct: float | None,
) -> bool:
    if max_exposure_per_sector_pct is None or total_equity <= 0:
        return True
    projected_exposure = (current_sector_value + new_position_value) / total_equity
    return projected_exposure <= max_exposure_per_sector_pct


def within_correlation_limit(
    candidate_returns: pd.Series,
    held_returns: dict[str, pd.Series],
    max_correlation: float | None,
    lookback: int = 20,
) -> bool:
    """Reject a candidate whose trailing return correlation with any currently
    held position exceeds `max_correlation` — a crude concentration guard
    against adding several stocks that all move together.
    """
    if max_correlation is None or not held_returns:
        return True

    candidate_tail = candidate_returns.tail(lookback)
    for held_series in held_returns.values():
        aligned_candidate, aligned_held = candidate_tail.align(held_series.tail(lookback), join="inner")
        if len(aligned_candidate) < 2:
            continue
        corr = aligned_candidate.corr(aligned_held)
        if pd.notna(corr) and corr > max_correlation:
            return False
    return True

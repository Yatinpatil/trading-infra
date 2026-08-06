"""Data quality checks for OHLCV series.

These don't fix anything — they flag rows so a caller (or a strategy) can
decide to skip a symbol/date rather than silently trade on bad data.
"""
import numpy as np
import pandas as pd


def check_gaps(df: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    """Flag index gaps wider than `max_gap_days` calendar days (missed listings, halts, bad fetch)."""
    if df.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_days"])

    gap_days = df.index.to_series().diff().dt.days.to_numpy()
    flagged_positions = np.flatnonzero(gap_days > max_gap_days)
    if len(flagged_positions) == 0:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_days"])

    return pd.DataFrame(
        {
            "gap_start": df.index[flagged_positions - 1],
            "gap_end": df.index[flagged_positions],
            "gap_days": gap_days[flagged_positions],
        }
    ).reset_index(drop=True)


def check_stale_prices(df: pd.DataFrame, min_repeat_days: int = 5) -> pd.DataFrame:
    """Flag runs of `min_repeat_days`+ consecutive identical closes (halted/illiquid stock, not a real move)."""
    if df.empty:
        return pd.DataFrame(columns=["start", "end", "run_length", "price"])

    close = df["CLOSE"]
    is_same_as_prev = close.eq(close.shift())
    run_id = (~is_same_as_prev).cumsum()

    rows = []
    for _, group in df.groupby(run_id):
        if len(group) >= min_repeat_days:
            rows.append(
                {
                    "start": group.index[0],
                    "end": group.index[-1],
                    "run_length": len(group),
                    "price": group["CLOSE"].iloc[0],
                }
            )
    return pd.DataFrame(rows, columns=["start", "end", "run_length", "price"])


def check_outliers(df: pd.DataFrame, max_daily_move_pct: float = 0.20) -> pd.DataFrame:
    """Flag single-day close-to-close moves larger than `max_daily_move_pct` (likely an unadjusted corporate action or bad tick)."""
    if df.empty:
        return pd.DataFrame(columns=["date", "prev_close", "close", "pct_change"])

    pct_change = df["CLOSE"].pct_change()
    flagged = pct_change[pct_change.abs() > max_daily_move_pct]
    if flagged.empty:
        return pd.DataFrame(columns=["date", "prev_close", "close", "pct_change"])

    prev_close = df["CLOSE"].shift().loc[flagged.index]
    return pd.DataFrame(
        {
            "date": flagged.index,
            "prev_close": prev_close.values,
            "close": df.loc[flagged.index, "CLOSE"].values,
            "pct_change": flagged.values,
        }
    ).reset_index(drop=True)


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Convenience wrapper running all checks; returns a dict of the three flag DataFrames."""
    return {
        "gaps": check_gaps(df),
        "stale_prices": check_stale_prices(df),
        "outliers": check_outliers(df),
    }

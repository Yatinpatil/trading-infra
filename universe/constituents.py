"""Point-in-time index constituent membership.

NSE does not publish a free historical index-membership API — only "who's in
the index today". Building true point-in-time membership requires
accumulating daily/periodic snapshots over time (or sourcing a paid history).

This module stores those snapshots in `universe/data/constituents_history.csv`
(columns: index_name, symbol, start_date, end_date) and reads from it. Until
enough history has been accumulated, `get_universe()` for a past date falls
back to the oldest known snapshot and raises a UserWarning, so callers are
never silently exposed to survivorship bias — they know when a backtest
predates the point-in-time record and can decide whether that's acceptable.
"""
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd

HISTORY_PATH = Path(__file__).parent / "data" / "constituents_history.csv"
HISTORY_COLUMNS = ["index_name", "symbol", "start_date", "end_date"]


def _to_date(d) -> date:
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    return d


def get_current_constituents(index_name: str) -> list[str]:
    """Live constituent list for `index_name` (e.g. 'NIFTY 50', 'NIFTY 500') as of today, from NSE."""
    from jugaad_data.nse import NSELive

    n = NSELive()
    data = n.live_index(index_name)
    return sorted(
        row["symbol"] for row in data["data"] if row["symbol"] != index_name
    )


def _load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(HISTORY_PATH, parse_dates=["start_date", "end_date"])
    return df


def snapshot_constituents(index_name: str, as_of: date | None = None) -> None:
    """Fetch today's constituents and append/extend a point-in-time snapshot in the history file.

    Call this periodically (e.g. daily via a scheduled job) to build up real
    point-in-time coverage over time. Existing open-ended memberships
    (end_date is null) for symbols still present are left untouched; symbols
    no longer in the index get their end_date closed off; new symbols get a
    fresh row starting `as_of`.
    """
    as_of = as_of or date.today()
    current = set(get_current_constituents(index_name))
    history = _load_history()

    idx_mask = history["index_name"] == index_name
    open_mask = idx_mask & history["end_date"].isna()
    open_symbols = set(history.loc[open_mask, "symbol"])

    dropped = open_symbols - current
    if dropped:
        drop_mask = open_mask & history["symbol"].isin(dropped)
        history.loc[drop_mask, "end_date"] = pd.Timestamp(as_of)

    added = current - open_symbols
    new_rows = pd.DataFrame(
        {
            "index_name": index_name,
            "symbol": sorted(added),
            "start_date": pd.Timestamp(as_of),
            "end_date": pd.NaT,
        }
    )
    history = pd.concat([history, new_rows], ignore_index=True)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(HISTORY_PATH, index=False)


def get_universe(index_name: str, as_of_date) -> list[str]:
    """Point-in-time constituents of `index_name` on `as_of_date`.

    Warns and falls back to the earliest available snapshot if `as_of_date`
    predates the recorded history (survivorship-bias risk — the returned list
    was NOT actually the membership on that date).
    """
    as_of_date = _to_date(as_of_date)
    history = _load_history()
    idx_history = history[history["index_name"] == index_name]

    if idx_history.empty:
        warnings.warn(
            f"No point-in-time history for '{index_name}'; falling back to current "
            f"constituents. Run snapshot_constituents('{index_name}') periodically to "
            f"build real point-in-time coverage.",
            UserWarning,
        )
        return get_current_constituents(index_name)

    earliest_start = idx_history["start_date"].min().date()
    if as_of_date < earliest_start:
        warnings.warn(
            f"Requested as_of_date {as_of_date} predates earliest recorded snapshot "
            f"({earliest_start}) for '{index_name}'. Returning earliest known "
            f"membership instead — this backtest period is exposed to survivorship bias.",
            UserWarning,
        )
        as_of_date = earliest_start

    as_of_ts = pd.Timestamp(as_of_date)
    active = idx_history[
        (idx_history["start_date"] <= as_of_ts)
        & (idx_history["end_date"].isna() | (idx_history["end_date"] > as_of_ts))
    ]
    return sorted(active["symbol"].tolist())

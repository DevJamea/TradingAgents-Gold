"""Gold market calendar and timeframe grid helpers (UTC, deterministic).

Session model for gold (XAUUSD / GC=F), approximated in UTC for v1:

* Opens Sunday 22:00 UTC, closes Friday 21:00 UTC.
* Daily maintenance break 21:00–22:00 UTC Monday–Thursday.

DST is not modelled (documented limitation): the model is a stable UTC grid,
which keeps every timestamp rule reproducible across machines and test runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tradingagents.gold.types import TIMEFRAME_MINUTES, TimeFrame


def ensure_utc(ts: datetime) -> datetime:
    """Return ``ts`` as tz-aware UTC; naive timestamps are assumed UTC.

    The gold subsystem stores *every* timestamp in UTC.  A naive datetime is
    accepted on input (assumed UTC) and normalised here so datasets are
    internally consistent.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def is_gold_market_open(ts: datetime) -> bool:
    """Whether the gold market is trading at ``ts`` (UTC model above)."""
    ts = ensure_utc(ts)
    if ts.hour == 21:
        return False                      # daily maintenance break 21–22 UTC
    weekday = ts.weekday()                # Mon=0 .. Sun=6
    if weekday == 5:
        return False                      # Saturday
    if weekday == 6:
        return ts.hour >= 22              # Sunday opens 22:00
    if weekday == 4:
        return ts.hour < 21               # Friday closes 21:00
    return True


def timeframe_minutes(timeframe: TimeFrame | str) -> int:
    """Bar length of ``timeframe`` in minutes."""
    key = timeframe.value if isinstance(timeframe, TimeFrame) else str(timeframe)
    return TIMEFRAME_MINUTES[key]


def on_grid(ts: datetime, interval_minutes: int) -> bool:
    """Whether ``ts`` sits on the timeframe grid (whole-bar boundary)."""
    ts = ensure_utc(ts)
    return (ts.hour * 60 + ts.minute) % interval_minutes == 0 and ts.second == 0


def expected_bar_times(start: datetime, end: datetime, interval_minutes: int) -> list[datetime]:
    """Every grid timestamp in ``[start, end]`` at which a bar should exist.

    Only times when the gold market is open are returned, so the weekend close
    and the daily 21–22 UTC break never look like missing data.
    """
    start, end = ensure_utc(start), ensure_utc(end)
    # Align the cursor down to the grid so iteration is exact.
    anchor = start.replace(second=0, microsecond=0)
    offset = (anchor.hour * 60 + anchor.minute) % interval_minutes
    cursor = anchor - timedelta(minutes=offset)
    out: list[datetime] = []
    step = timedelta(minutes=interval_minutes)
    while cursor <= end:
        if cursor >= start and is_gold_market_open(cursor):
            out.append(cursor)
        cursor += step
    return out

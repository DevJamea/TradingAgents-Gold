"""Gold session classification on UTC timestamps (spec §8).

Approximate UTC session windows (DST not modelled in v1):

* ASIAN    00:00–08:00
* LONDON   07:00–16:00
* NEW_YORK 12:00–21:00 (the daily 21–22 UTC break follows)

Overlaps (Asia–London 07–08, London–NY 12–16) are reported explicitly because
liquidity/spread conditions differ in them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tradingagents.gold.data.calendar import ensure_utc
from tradingagents.gold.types import SessionName

_ASIA = (0, 8)
_LONDON = (7, 16)
_NEW_YORK = (12, 21)


@dataclass(frozen=True)
class SessionContext:
    """Which sessions are open at one timestamp and the primary label."""

    name: SessionName
    in_asian: bool
    in_london: bool
    in_new_york: bool
    overlaps: tuple[str, ...]


def _in(hour: int, window: tuple[int, int]) -> bool:
    return window[0] <= hour < window[1]


def session_at(ts: datetime) -> SessionContext:
    """Classify a timestamp into the gold session map."""
    ts = ensure_utc(ts)
    hour = ts.hour
    asia, london, ny = _in(hour, _ASIA), _in(hour, _LONDON), _in(hour, _NEW_YORK)
    overlaps: list[str] = []
    if asia and london:
        overlaps.append("ASIA_LONDON")
    if london and ny:
        overlaps.append("LONDON_NEW_YORK")
    if asia and ny:
        overlaps.append("ASIA_NEW_YORK")

    if ny:
        primary = SessionName.NEW_YORK
    elif london:
        primary = SessionName.LONDON
    elif asia:
        primary = SessionName.ASIAN
    else:
        primary = SessionName.OFF
    return SessionContext(
        name=primary,
        in_asian=asia,
        in_london=london,
        in_new_york=ny,
        overlaps=tuple(overlaps),
    )

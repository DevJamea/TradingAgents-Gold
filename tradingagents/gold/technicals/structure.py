"""Market-structure analysis: swings, HH/HL/LH/LL, S/R, breakout state.

Deterministic fractal swing detection (a bar whose high exceeds ``left`` bars
before and ``right`` bars after is a swing high, mirrored for lows) feeding
structure classification and support/resistance levels.  All rules are exact
and re-runnable — the agents *interpret* structure, they never compute it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from tradingagents.gold.types import BreakoutState, StructureBias


@dataclass(frozen=True)
class Swing:
    timestamp: datetime
    price: float


def find_swings(
    frame: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> tuple[list[Swing], list[Swing]]:
    """Fractal swing highs and swing lows of an OHLCV frame (newest last)."""
    highs = frame["High"].to_numpy()
    lows = frame["Low"].to_numpy()
    stamps = list(frame["Date"])
    swing_highs: list[Swing] = []
    swing_lows: list[Swing] = []
    for i in range(left, len(frame) - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]
        if highs[i] == max(window_h) and (window_h == highs[i]).sum() == 1:
            swing_highs.append(Swing(timestamp=stamps[i], price=float(highs[i])))
        if lows[i] == min(window_l) and (window_l == lows[i]).sum() == 1:
            swing_lows.append(Swing(timestamp=stamps[i], price=float(lows[i])))
    return swing_highs, swing_lows


def classify_structure(
    swing_highs: list[Swing],
    swing_lows: list[Swing],
) -> StructureBias:
    """HH+HL ⇒ uptrend, LH+LL ⇒ downtrend, otherwise range/mixed.

    Uses the last two confirmed swings of each kind; fewer than two of either
    leaves the bias UNKNOWN.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return StructureBias.UNKNOWN
    hh = swing_highs[-1].price > swing_highs[-2].price
    hl = swing_lows[-1].price > swing_lows[-2].price
    lh = swing_highs[-1].price < swing_highs[-2].price
    ll = swing_lows[-1].price < swing_lows[-2].price
    if hh and hl:
        return StructureBias.UPTREND
    if lh and ll:
        return StructureBias.DOWNTREND
    return StructureBias.RANGE


def support_resistance(
    swing_lows: list[Swing],
    swing_highs: list[Swing],
    n: int = 3,
) -> tuple[list[float], list[float]]:
    """Nearest ``n`` support (swing-low) and resistance (swing-high) levels."""
    supports = sorted((s.price for s in swing_lows[-n:]), reverse=True)
    resistances = sorted(s.price for s in swing_highs[-n:])
    return supports, resistances


def breakout_state(
    close: float,
    swing_highs: list[Swing],
    swing_lows: list[Swing],
    n: int = 3,
) -> BreakoutState:
    """Close above the last ``n`` swing highs ⇒ UP; below swing lows ⇒ DOWN."""
    recent_highs = [s.price for s in swing_highs[-n:]]
    recent_lows = [s.price for s in swing_lows[-n:]]
    if recent_highs and close > max(recent_highs):
        return BreakoutState.UP
    if recent_lows and close < min(recent_lows):
        return BreakoutState.DOWN
    return BreakoutState.NONE

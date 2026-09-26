"""Shared enums and small value helpers for the Gold (XAUUSD) subsystem.

Every dataset and decision in the gold system references these types so that
"source of truth" questions — spot vs futures proxy, which timeframe, which
session, which regime — always have a machine-readable answer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum


class TimeFrame(str, Enum):
    """Primary analytical timeframes for the gold system.

    H4 = macro/trend context, H1 = structure/context, M15 = primary decision
    timeframe.  M1/M5 scalping is intentionally out of scope for v1.
    """

    M15 = "M15"
    H1 = "H1"
    H4 = "H4"


#: Canonical bar length of each timeframe in minutes.
TIMEFRAME_MINUTES: dict[str, int] = {"M15": 15, "H1": 60, "H4": 240}


class AssetKind(str, Enum):
    """What a price series actually is.  Never silently mixed."""

    GOLD_SPOT = "gold_spot"
    GOLD_FUTURES_PROXY = "gold_futures_proxy"
    #: Real broker spot via a read-only MT5 terminal connection.
    GOLD_SPOT_MT5 = "gold_spot_mt5"


class DataKind(str, Enum):
    """Category of a recorded dataset / snapshot."""

    OHLCV = "ohlcv"
    QUOTE = "quote"
    MACRO = "macro"
    NEWS = "news"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"


class DecisionAction(str, Enum):
    """Actions of the final gold decision.  REVIEW is never tradeable."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REVIEW = "REVIEW"


class MarketRegime(str, Enum):
    """Deterministic market-regime classification (agents interpret it; they
    do not invent it)."""

    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class SessionName(str, Enum):
    """Gold trading sessions, classified on UTC timestamps."""

    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OFF = "OFF"


class TrendDirection(str, Enum):
    """Per-timeframe trend direction from EMA alignment + slope."""

    UP = "UP"
    DOWN = "DOWN"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class StructureBias(str, Enum):
    """HH/HL vs LH/LL classification of recent swings."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"
    UNKNOWN = "UNKNOWN"


class BreakoutState(str, Enum):
    """Whether the close has broken the recent swing envelope."""

    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class VolatilityRegime(str, Enum):
    """Normalised-ATR classification of one timeframe."""

    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class CostMode(str, Enum):
    """Configurable analyst/debate depth (LLM cost control)."""

    FAST = "FAST"
    STANDARD = "STANDARD"
    DEEP = "DEEP"


def utc_now() -> datetime:
    """Timezone-aware current time in UTC.  The single clock helper so every
    timestamp rule in the gold subsystem stays consistent."""
    return datetime.now(timezone.utc)

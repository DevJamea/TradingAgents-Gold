"""Core value types of the gold data layer.

Every dataset records: source, symbol, timestamp, timezone, asset type and
data type (spec §6).  Spot XAUUSD and the GC=F futures proxy are distinct
``AssetKind`` values and must never be silently mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from tradingagents.gold.data.calendar import ensure_utc
from tradingagents.gold.types import AssetKind, DataKind, TimeFrame


@dataclass(frozen=True)
class DatasetMeta:
    """Provenance recorded on every gold dataset."""

    source: str                      # e.g. "yfinance:yahoo", "fixture", "mt5"
    symbol: str                      # requested symbol, e.g. "XAUUSD"
    source_symbol: str               # symbol actually queried, e.g. "GC=F"
    asset_kind: AssetKind            # gold_spot | gold_futures_proxy
    data_type: DataKind              # ohlcv | quote | macro | news | sentiment
    timezone: str = "UTC"
    timeframe: TimeFrame | None = None
    is_proxy: bool = False
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    disclaimer: str | None = None    # mandatory for proxies (see validation)

    def describe(self) -> str:
        """One-line provenance string for audit records and report headers."""
        tf = f" {self.timeframe.value}" if self.timeframe else ""
        proxy = " [PROXY]" if self.is_proxy else ""
        return (
            f"{self.data_type.value}{tf} {self.symbol} (queried {self.source_symbol}) "
            f"from {self.source} · {self.asset_kind.value} · tz={self.timezone}{proxy}"
        )


@dataclass(frozen=True)
class GoldBar:
    """One OHLCV bar.  Timestamps are stored tz-aware UTC.

    ``bid``/``ask``/``spread`` are optional: most sources are mid-only.  All
    market-quality rules (impossible OHLC, bad spread, …) live in
    ``validation.py`` so the quality gate classifies problems instead of the
    constructor silently dropping rows.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def spread(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


@dataclass(frozen=True)
class GoldQuote:
    """A point-in-time bid/ask quote (when a source supplies one)."""

    symbol: str
    source: str
    asset_kind: AssetKind
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def spread(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None


@dataclass(frozen=True)
class GoldDataset:
    """An immutable, provenance-carrying OHLCV series."""

    meta: DatasetMeta
    bars: tuple[GoldBar, ...]

    @property
    def timeframe(self) -> TimeFrame | None:
        return self.meta.timeframe

    def frame(self) -> pd.DataFrame:
        """DataFrame with UTC ``Date`` column and capitalized OHLCV columns.

        Column naming matches the upstream dataflows convention so indicator
        code and snapshots can share formatting utilities.
        """
        data = {
            "Date": [b.timestamp for b in self.bars],
            "Open": [b.open for b in self.bars],
            "High": [b.high for b in self.bars],
            "Low": [b.low for b in self.bars],
            "Close": [b.close for b in self.bars],
            "Volume": [b.volume for b in self.bars],
        }
        return pd.DataFrame(data)

    def slice_(self, start: datetime | None, end: datetime | None) -> GoldDataset:
        """Sub-series with timestamps in ``[start, end]`` (UTC, inclusive)."""
        start = ensure_utc(start) if start else None
        end = ensure_utc(end) if end else None
        kept = tuple(
            b for b in self.bars
            if (start is None or b.timestamp >= start) and (end is None or b.timestamp <= end)
        )
        return GoldDataset(meta=self.meta, bars=kept)

    def __len__(self) -> int:
        return len(self.bars)

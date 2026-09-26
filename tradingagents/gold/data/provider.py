"""Gold market data provider abstraction.

Agents must never contain raw API logic: they consume datasets produced by a
``GoldMarketDataProvider`` (or the tools built on top of one).  Providers are
explicit about what they are — a spot feed or the GC=F futures proxy — and
about which fields (bid/ask/spread, timeframes) they can actually supply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from tradingagents.gold.data.models import GoldDataset, GoldQuote
from tradingagents.gold.types import AssetKind, TimeFrame


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can honestly deliver."""

    asset_kind: AssetKind
    is_proxy: bool
    supported_timeframes: frozenset[str]
    supports_bid_ask: bool
    #: True when the provider itself guarantees ``get_bars`` returns only bars
    #: whose close (open + timeframe) is at or before the requested ``end`` —
    #: a still-forming candle is never handed to analysis as if closed.
    closed_bars_enforced: bool = False


class GoldMarketDataProvider(ABC):
    """Market-data interface for the gold system (historical + recent)."""

    source_name: str

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> GoldDataset:
        """Bars for ``symbol`` in ``[start, end]`` (UTC, inclusive).

        Implementations must label the returned dataset's provenance and must
        not silently substitute one asset kind for another.
        """

    @abstractmethod
    def get_quote(self, symbol: str) -> GoldQuote | None:
        """A current bid/ask quote, or None when the source has none."""


class ProviderRegistry:
    """Name → provider registry so runs can select data sources explicitly."""

    def __init__(self) -> None:
        self._providers: dict[str, GoldMarketDataProvider] = {}

    def register(self, name: str, provider: GoldMarketDataProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> GoldMarketDataProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(
                f"unknown gold data provider {name!r}; "
                f"registered: {sorted(self._providers)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._providers)

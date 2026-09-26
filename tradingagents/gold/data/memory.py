"""In-memory gold provider: fixture data and cached historical replay.

Used by tests and by the historical paper engine (replaying stored bars).  Not
a network client — everything comes from datasets the caller already holds.
"""

from __future__ import annotations

from datetime import datetime

from tradingagents.gold.data.calendar import ensure_utc
from tradingagents.gold.data.models import GoldDataset, GoldQuote
from tradingagents.gold.data.provider import GoldMarketDataProvider, ProviderCapabilities
from tradingagents.gold.types import AssetKind, TimeFrame


class InMemoryGoldProvider(GoldMarketDataProvider):
    """Serves pre-built datasets keyed by ``(symbol, timeframe)``."""

    source_name = "in-memory"

    def __init__(
        self,
        datasets: dict[tuple[str, TimeFrame], GoldDataset] | None = None,
        quotes: dict[str, GoldQuote] | None = None,
        *,
        asset_kind: AssetKind = AssetKind.GOLD_SPOT,
    ) -> None:
        self._datasets = dict(datasets or {})
        self._quotes = dict(quotes or {})
        self._asset_kind = asset_kind

    def add_dataset(self, symbol: str, dataset: GoldDataset) -> None:
        tf = dataset.meta.timeframe
        if tf is None:
            raise ValueError("an OHLCV dataset must carry a timeframe")
        self._datasets[(symbol, tf)] = dataset

    def set_quote(self, quote: GoldQuote) -> None:
        self._quotes[quote.symbol] = quote

    @property
    def capabilities(self) -> ProviderCapabilities:
        kinds = {d.meta.asset_kind for d in self._datasets.values()}
        kind = kinds.pop() if len(kinds) == 1 else self._asset_kind
        return ProviderCapabilities(
            asset_kind=kind,
            is_proxy=kind == AssetKind.GOLD_FUTURES_PROXY,
            supported_timeframes=frozenset(tf.value for _, tf in self._datasets),
            supports_bid_ask=bool(self._quotes),
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> GoldDataset:
        key = (symbol, timeframe)
        if key not in self._datasets:
            raise KeyError(f"no in-memory dataset for {symbol} {timeframe.value}")
        return self._datasets[key].slice_(ensure_utc(start), ensure_utc(end))

    def get_quote(self, symbol: str) -> GoldQuote | None:
        return self._quotes.get(symbol)

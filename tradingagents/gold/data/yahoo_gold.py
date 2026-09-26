"""Yahoo Finance GC=F futures proxy provider for gold (clearly labelled).

IMPORTANT (spec §6): Yahoo has no XAUUSD spot symbol.  The upstream symbol
layer resolves XAUUSD → ``GC=F`` (COMEX front-month future).  This provider
uses that series strictly as a RESEARCH PROXY: every dataset it returns is
``AssetKind.GOLD_FUTURES_PROXY`` with ``is_proxy=True`` and the configured
disclaimer.  Futures basis/roll/session differences apply; results must never
be presented as broker XAUUSD spot results.

Timeframe mapping: M15 → yfinance ``15m``, H1 → ``1h``, H4 → resampled from
``1h`` bars (Open first, High max, Low min, Close last, Volume sum) on the
00/04/08/… UTC grid.  yfinance intraday retention is limited (~60 days for
15m, longer for 1h) — callers must treat missing history as a data gap, not
fabricate it.  Bid/ask is not supplied (``supports_bid_ask=False``).

The raw yfinance call lives in ``dataflows.vendors.yahoo.gold_bars`` so vendor
libraries stay behind the data layer (``tests/test_layering.py``).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.gold_bars import fetch_history_bars
from tradingagents.gold.config import GoldDataConfig
from tradingagents.gold.data.models import DatasetMeta, GoldBar, GoldDataset, GoldQuote
from tradingagents.gold.data.provider import GoldMarketDataProvider, ProviderCapabilities
from tradingagents.gold.types import AssetKind, DataKind, TimeFrame, utc_now

_YF_INTERVALS = {"M15": "15m", "H1": "1h", "H4": "1h"}  # H4 resampled from H1
_H4_ANCHOR_MINUTES = 240


class YahooGoldProxyProvider(GoldMarketDataProvider):
    """GC=F futures bars via yfinance, labelled as a gold spot proxy."""

    source_name = "yfinance:yahoo"

    def __init__(self, data_config: GoldDataConfig | None = None) -> None:
        self._cfg = data_config or GoldDataConfig()

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            asset_kind=AssetKind.GOLD_FUTURES_PROXY,
            is_proxy=True,
            supported_timeframes=frozenset(_YF_INTERVALS),
            supports_bid_ask=False,
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> GoldDataset:
        canonical, frame = fetch_history_bars(
            symbol, _YF_INTERVALS[timeframe.value], start, end,
        )
        if timeframe is TimeFrame.H4:
            frame = self._resample_h4(frame)

        bars = tuple(
            GoldBar(
                timestamp=row["Date"].to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]) if pd.notna(row.get("Volume")) else None,
            )
            for _, row in frame.iterrows()
        )
        meta = DatasetMeta(
            source=self.source_name,
            symbol=symbol,
            source_symbol=canonical,
            asset_kind=AssetKind.GOLD_FUTURES_PROXY,
            data_type=DataKind.OHLCV,
            timezone="UTC",
            timeframe=timeframe,
            is_proxy=True,
            retrieved_at=utc_now(),
            disclaimer=self._cfg.proxy_disclaimer,
        )
        return GoldDataset(meta=meta, bars=bars)

    @staticmethod
    def _resample_h4(frame: pd.DataFrame) -> pd.DataFrame:
        """H1 rows → H4 bars on the 00/04/08/… UTC grid."""
        indexed = frame.set_index("Date")
        ohlc = indexed.resample(f"{_H4_ANCHOR_MINUTES}min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        )
        return ohlc.dropna(subset=["Open"]).reset_index()

    def get_quote(self, symbol: str) -> GoldQuote | None:
        """Yahoo mid bars carry no bid/ask; a spot/MT5 provider fills this."""
        return None

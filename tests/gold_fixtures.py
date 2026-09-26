"""Shared builders for gold subsystem tests (deterministic fixture data)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tradingagents.gold.data.models import DatasetMeta, GoldBar, GoldDataset
from tradingagents.gold.types import AssetKind, DataKind, TimeFrame

MONDAY = datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)   # a plain Monday
FRIDAY = datetime(2026, 3, 6, 0, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 3, 7, 0, 0, tzinfo=timezone.utc)
SUNDAY = datetime(2026, 3, 8, 0, 0, tzinfo=timezone.utc)


def make_bar(ts: datetime, price: float = 3000.0, *, spread: float | None = None,
             volume: float | None = 100.0) -> GoldBar:
    bid = ask = None
    if spread is not None:
        bid, ask = price - spread / 2, price + spread / 2
    return GoldBar(
        timestamp=ts, open=price, high=price + 1.0, low=price - 1.0,
        close=price + 0.5, volume=volume, bid=bid, ask=ask,
    )


def make_series(start: datetime, periods: int, interval_minutes: int,
                price: float = 3000.0) -> list[GoldBar]:
    """Bars on the grid from ``start`` (skipping closed-market slots)."""
    from tradingagents.gold.data.calendar import is_gold_market_open

    bars: list[GoldBar] = []
    cursor = start
    while len(bars) < periods:
        if is_gold_market_open(cursor):
            bars.append(make_bar(cursor, price=price + len(bars) * 0.1))
        cursor += timedelta(minutes=interval_minutes)
    return bars


def make_dataset(
    bars: list[GoldBar] | tuple[GoldBar, ...],
    *,
    symbol: str = "XAUUSD",
    timeframe: TimeFrame = TimeFrame.M15,
    proxy: bool = False,
    source: str = "fixture",
    disclaimer: str | None = None,
) -> GoldDataset:
    meta = DatasetMeta(
        source=source,
        symbol=symbol,
        source_symbol="GC=F" if proxy else symbol,
        asset_kind=AssetKind.GOLD_FUTURES_PROXY if proxy else AssetKind.GOLD_SPOT,
        data_type=DataKind.OHLCV,
        timezone="UTC",
        timeframe=timeframe,
        is_proxy=proxy,
        disclaimer=disclaimer if disclaimer is not None else (
            "RESEARCH PROXY — not broker XAUUSD spot" if proxy else None
        ),
    )
    return GoldDataset(meta=meta, bars=tuple(bars))

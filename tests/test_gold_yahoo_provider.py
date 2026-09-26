"""Phase 2 — Yahoo GC=F proxy provider (mocked yfinance; no network)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import yfinance as yf

from tests.gold_fixtures import MONDAY, make_dataset, make_series
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.gold.config import default_gold_config
from tradingagents.gold.data.snapshot import build_market_snapshot
from tradingagents.gold.data.yahoo_gold import YahooGoldProxyProvider
from tradingagents.gold.types import AssetKind, TimeFrame


def _fake_ticker(captured, frame):
    class FakeTicker:
        def __init__(self, canonical):
            captured["symbol"] = canonical

        def history(self, **kwargs):
            captured["kwargs"] = kwargs
            return frame.copy()

    return FakeTicker


def _hourly_frame():
    idx = pd.date_range("2026-03-02 00:00", periods=8, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(8)],
            "High": [110 + i for i in range(8)],
            "Low": [90 + i for i in range(8)],
            "Close": [105 + i for i in range(8)],
            "Volume": [10 + i for i in range(8)],
        },
        index=idx,
    )


class TestSymbolAndProvenance:
    def test_xauusd_resolved_to_gc_f_proxy(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(yf, "Ticker", _fake_ticker(captured, _hourly_frame()))
        ds = YahooGoldProxyProvider().get_bars(
            "XAUUSD", TimeFrame.H1,
            datetime(2026, 3, 1, tzinfo=timezone.utc), datetime(2026, 3, 8, tzinfo=timezone.utc),
        )
        assert captured["symbol"] == "GC=F"
        assert ds.meta.symbol == "XAUUSD"
        assert ds.meta.source_symbol == "GC=F"
        assert ds.meta.asset_kind is AssetKind.GOLD_FUTURES_PROXY
        assert ds.meta.is_proxy is True
        assert "PROXY" in ds.meta.disclaimer
        assert "not broker XAUUSD spot" in ds.meta.disclaimer

    def test_capabilities_are_honest(self):
        caps = YahooGoldProxyProvider().capabilities
        assert caps.is_proxy is True
        assert caps.supports_bid_ask is False
        assert caps.supported_timeframes == frozenset({"M15", "H1", "H4"})

    def test_get_quote_returns_none(self):
        assert YahooGoldProxyProvider().get_quote("XAUUSD") is None

    def test_empty_history_raises_no_market_data(self, monkeypatch):
        monkeypatch.setattr(yf, "Ticker", _fake_ticker({}, pd.DataFrame()))
        with pytest.raises(NoMarketDataError):
            YahooGoldProxyProvider().get_bars(
                "XAUUSD", TimeFrame.M15,
                datetime(2026, 3, 1, tzinfo=timezone.utc), datetime(2026, 3, 8, tzinfo=timezone.utc),
            )


class TestTimeframes:
    def test_h1_bars_come_through_with_utc_timestamps(self, monkeypatch):
        monkeypatch.setattr(yf, "Ticker", _fake_ticker({}, _hourly_frame()))
        ds = YahooGoldProxyProvider().get_bars(
            "XAUUSD", TimeFrame.H1,
            datetime(2026, 3, 1, tzinfo=timezone.utc), datetime(2026, 3, 8, tzinfo=timezone.utc),
        )
        assert len(ds) == 8
        assert all(b.timestamp.tzinfo is not None for b in ds.bars)
        assert ds.bars[0].timestamp == datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)

    def test_h4_is_resampled_from_h1(self, monkeypatch):
        monkeypatch.setattr(yf, "Ticker", _fake_ticker({}, _hourly_frame()))
        ds = YahooGoldProxyProvider().get_bars(
            "XAUUSD", TimeFrame.H4,
            datetime(2026, 3, 1, tzinfo=timezone.utc), datetime(2026, 3, 8, tzinfo=timezone.utc),
        )
        assert len(ds) == 2
        first, second = ds.bars
        assert first.timestamp == datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc)
        assert (first.open, first.high, first.low, first.close) == (100, 113, 90, 108)
        assert first.volume == pytest.approx(46)
        assert second.timestamp == datetime(2026, 3, 2, 4, 0, tzinfo=timezone.utc)
        assert (second.open, second.high, second.low, second.close) == (104, 117, 94, 112)

    def test_exchange_tz_is_converted_to_utc(self, monkeypatch):
        # yfinance serves wall-clock exchange times (America/New_York); the
        # provider must convert them to UTC instants (00:00 ET == 05:00 UTC).
        idx = pd.date_range("2026-03-02 00:00", periods=8, freq="1h", tz="America/New_York")
        frame = pd.DataFrame(
            {
                "Open": [100 + i for i in range(8)],
                "High": [110 + i for i in range(8)],
                "Low": [90 + i for i in range(8)],
                "Close": [105 + i for i in range(8)],
                "Volume": [10 + i for i in range(8)],
            },
            index=idx,
        )
        monkeypatch.setattr(yf, "Ticker", _fake_ticker({}, frame))
        ds = YahooGoldProxyProvider().get_bars(
            "XAUUSD", TimeFrame.H1,
            datetime(2026, 3, 1, tzinfo=timezone.utc), datetime(2026, 3, 8, tzinfo=timezone.utc),
        )
        assert ds.bars[0].timestamp == datetime(2026, 3, 2, 5, 0, tzinfo=timezone.utc)


class TestSnapshotPassCondition:
    """Phase 2 pass condition: a valid XAUUSD market snapshot can be generated."""

    def test_valid_multi_timeframe_snapshot(self):
        from tradingagents.gold.data.memory import InMemoryGoldProvider

        provider = InMemoryGoldProvider()
        provider.add_dataset("XAUUSD", make_dataset(
            make_series(MONDAY, 48, 15), timeframe=TimeFrame.M15))
        provider.add_dataset("XAUUSD", make_dataset(
            make_series(MONDAY, 12, 60), timeframe=TimeFrame.H1))
        provider.add_dataset("XAUUSD", make_dataset(
            make_series(MONDAY, 3, 240), timeframe=TimeFrame.H4))

        snap = build_market_snapshot(
            provider, default_gold_config(), as_of=MONDAY + timedelta(hours=20),
        )
        assert snap.ok
        assert set(snap.datasets) == {"M15", "H1", "H4"}
        assert snap.symbol == "XAUUSD"
        assert len(snap.provenance) == 3
        assert snap.errors == []

    def test_invalid_series_makes_snapshot_not_ok(self):
        from tradingagents.gold.data.memory import InMemoryGoldProvider

        provider = InMemoryGoldProvider()
        bars = make_series(MONDAY, 8, 15)
        bars.append(bars[0])  # duplicate
        provider.add_dataset("XAUUSD", make_dataset(bars, timeframe=TimeFrame.M15))
        provider.add_dataset("XAUUSD", make_dataset(
            make_series(MONDAY, 6, 60), timeframe=TimeFrame.H1))
        provider.add_dataset("XAUUSD", make_dataset(
            make_series(MONDAY, 3, 240), timeframe=TimeFrame.H4))

        snap = build_market_snapshot(
            provider, default_gold_config(), as_of=MONDAY + timedelta(hours=20),
        )
        assert not snap.ok
        assert any("duplicate_timestamp" in e for e in snap.errors)

    def test_provider_failure_is_data_error(self):
        from tradingagents.gold.data.memory import InMemoryGoldProvider

        snap = build_market_snapshot(
            InMemoryGoldProvider(), default_gold_config(),
            as_of=MONDAY + timedelta(hours=20),
        )
        assert not snap.ok
        assert any("provider_failure" in e for e in snap.errors)

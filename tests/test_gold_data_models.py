"""Phase 2 — gold data models and market calendar."""

from datetime import datetime, timedelta, timezone

import pytest

from tests.gold_fixtures import (
    FRIDAY,
    MONDAY,
    SATURDAY,
    SUNDAY,
    make_bar,
    make_dataset,
    make_series,
)
from tradingagents.gold.data.calendar import (
    expected_bar_times,
    is_gold_market_open,
    on_grid,
    timeframe_minutes,
)
from tradingagents.gold.data.models import GoldBar, GoldQuote
from tradingagents.gold.types import AssetKind, TimeFrame

#: The Monday *after* FRIDAY/SUNDAY above (Mar 2 → Mar 9 week boundary).
NEXT_MONDAY = MONDAY.replace(day=MONDAY.day + 7)


class TestGoldBarAndDataset:
    def test_naive_timestamps_are_assumed_utc(self):
        naive = datetime(2026, 3, 2, 12, 0)
        bar = GoldBar(naive, 1.0, 2.0, 0.5, 1.5)
        assert bar.timestamp == naive.replace(tzinfo=timezone.utc)

    def test_bar_spread_computed_from_bid_ask(self):
        bar = make_bar(MONDAY, price=3000.0, spread=0.4)
        assert bar.spread == pytest.approx(0.4)
        assert make_bar(MONDAY).spread is None

    def test_dataset_frame_has_upstream_column_convention(self):
        ds = make_dataset(make_series(MONDAY, 3, 15))
        frame = ds.frame()
        assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert len(frame) == 3

    def test_slice_is_inclusive_and_preserves_meta(self):
        ds = make_dataset(make_series(MONDAY, 4, 15))
        start = MONDAY + timedelta(minutes=15)
        end = MONDAY + timedelta(minutes=30)
        cut = ds.slice_(start, end)
        assert [b.timestamp for b in cut.bars] == [start, end]
        assert cut.meta is ds.meta

    def test_quote_spread_and_mid(self):
        quote = GoldQuote(
            symbol="XAUUSD", source="fixture", asset_kind=AssetKind.GOLD_SPOT,
            timestamp=MONDAY, bid=2999.8, ask=3000.2,
        )
        assert quote.spread == pytest.approx(0.4)
        assert quote.mid == pytest.approx(3000.0)


class TestProvenance:
    def test_every_dataset_records_required_provenance(self):
        ds = make_dataset(make_series(MONDAY, 2, 15), proxy=True)
        m = ds.meta
        assert m.source and m.symbol and m.source_symbol
        assert m.timezone == "UTC"
        assert m.asset_kind == AssetKind.GOLD_FUTURES_PROXY
        assert m.data_type.value == "ohlcv"
        assert m.timeframe is TimeFrame.M15
        assert m.retrieved_at is not None

    def test_describe_marks_proxies(self):
        proxy = make_dataset(make_series(MONDAY, 1, 15), proxy=True)
        spot = make_dataset(make_series(MONDAY, 1, 15))
        assert "[PROXY]" in proxy.meta.describe()
        assert "[PROXY]" not in spot.meta.describe()


class TestGoldCalendar:
    @pytest.mark.parametrize("day,hour,open_", [
        (MONDAY, 12, True), (MONDAY, 21, False), (MONDAY, 22, True),
        (FRIDAY, 20, True), (FRIDAY, 21, False), (FRIDAY, 23, False),
        (SATURDAY, 12, False),
        (SUNDAY, 12, False), (SUNDAY, 22, True),
    ])
    def test_market_open_model(self, day, hour, open_):
        assert is_gold_market_open(day + timedelta(hours=hour)) is open_

    def test_expected_bars_skip_weekend_and_daily_break(self):
        # Friday 18:00 → next Monday 01:00: only Fri 18–20:45, Sun 22:00–23:45
        # and Mon 00:00–01:00 slots are expected.
        slots = expected_bar_times(
            FRIDAY + timedelta(hours=18), NEXT_MONDAY + timedelta(hours=1), 15,
        )
        assert slots[0] == FRIDAY + timedelta(hours=18)
        assert slots[-1] == NEXT_MONDAY + timedelta(hours=1)
        assert all(s < FRIDAY + timedelta(hours=21) or s >= SUNDAY + timedelta(hours=22)
                   for s in slots)
        assert not any(s.day == SATURDAY.day for s in slots)

    def test_weekday_bar_count_excludes_maintenance_hour(self):
        slots = expected_bar_times(MONDAY, MONDAY + timedelta(hours=23, minutes=59), 60)
        # 23 open hours: 00–20 and 22–23 (hour 21 is the maintenance break).
        assert len(slots) == 23

    def test_grid_alignment(self):
        assert on_grid(MONDAY + timedelta(minutes=15), 15)
        assert not on_grid(MONDAY + timedelta(minutes=16), 15)
        assert on_grid(MONDAY + timedelta(hours=4), 240)
        assert not on_grid(MONDAY + timedelta(hours=5), 240)

    def test_timeframe_minutes(self):
        assert timeframe_minutes(TimeFrame.H4) == 240
        assert timeframe_minutes("M15") == 15

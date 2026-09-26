"""Phase 2 — deterministic data-quality gate (spec §21)."""

from datetime import timedelta

from tests.gold_fixtures import (
    FRIDAY,
    MONDAY,
    SUNDAY,
    make_bar,
    make_dataset,
    make_series,
)
from tradingagents.gold.data.models import DatasetMeta, GoldBar, GoldDataset
from tradingagents.gold.data.validation import validate_dataset
from tradingagents.gold.types import AssetKind, DataKind, TimeFrame


def _codes(report, level=None):
    return [i.code for i in report.issues if level is None or i.level == level]


class TestCleanDataset:
    def test_clean_monday_series_is_ok(self):
        ds = make_dataset(make_series(MONDAY, 16, 15))  # 4h of open market
        report = validate_dataset(ds, expected_symbol="XAUUSD")
        assert report.ok
        assert not report.errors

    def test_weekend_gap_is_not_missing_data(self):
        # Fri 18:00–20:45, Sun 22:00–23:45, next Mon 00:00–00:45 — a ~49h wall
        # gap across the market close must NOT be flagged as missing bars.
        next_monday = MONDAY.replace(day=MONDAY.day + 7)
        bars = (
            [make_bar(FRIDAY + timedelta(hours=18, minutes=15 * i)) for i in range(12)]
            + [make_bar(SUNDAY + timedelta(hours=22, minutes=15 * i)) for i in range(8)]
            + [make_bar(next_monday + timedelta(minutes=15 * i)) for i in range(4)]
        )
        report = validate_dataset(make_dataset(bars), expected_symbol="XAUUSD")
        assert "missing_bars" not in _codes(report)
        assert "unordered_timestamps" not in _codes(report)
        assert report.ok


class TestStructuralErrors:
    def test_duplicate_timestamp_is_error(self):
        bars = make_series(MONDAY, 4, 15)
        bars.append(bars[0])
        report = validate_dataset(make_dataset(bars), expected_symbol="XAUUSD")
        assert "duplicate_timestamp" in _codes(report, "error")

    def test_unordered_timestamps_is_error(self):
        bars = make_series(MONDAY, 4, 15)
        bars[0], bars[1] = bars[1], bars[0]
        report = validate_dataset(make_dataset(bars), expected_symbol="XAUUSD")
        assert "unordered_timestamps" in _codes(report, "error")

    def test_non_positive_price_is_error(self):
        bars = make_series(MONDAY, 3, 15)
        bars[1] = GoldBar(
            timestamp=bars[1].timestamp, open=0.0, high=1.0, low=0.5, close=0.8,
        )
        report = validate_dataset(make_dataset(bars), expected_symbol="XAUUSD")
        assert "non_positive_price" in _codes(report, "error")

    def test_impossible_ohlc_is_error(self):
        bars = make_series(MONDAY, 3, 15)
        bars[1] = GoldBar(
            timestamp=bars[1].timestamp, open=3000.0, high=2990.0, low=3010.0, close=3000.0,
        )
        report = validate_dataset(make_dataset(bars), expected_symbol="XAUUSD")
        assert "impossible_ohlc" in _codes(report, "error")

    def test_mid_session_hole_is_missing_bars_error(self):
        bars = make_series(MONDAY, 16, 15)
        del bars[4:8]  # four consecutive expected bars gone (≥3 ⇒ error)
        report = validate_dataset(make_dataset(bars), expected_symbol="XAUUSD")
        assert "missing_bars" in _codes(report, "error")

    def test_two_missing_bars_are_only_a_warning(self):
        bars = make_series(MONDAY, 16, 15)
        del bars[4:6]
        report = validate_dataset(make_dataset(bars), expected_symbol="XAUUSD")
        assert "missing_bars" not in _codes(report, "error")
        assert "missing_bar" in _codes(report, "warning")


class TestSpreadAndStale:
    def test_negative_spread_is_error(self):
        bar = GoldBar(
            timestamp=MONDAY, open=3000, high=3001, low=2999, close=3000,
            bid=3000.5, ask=2999.5,
        )
        ds = make_dataset([make_bar(MONDAY), bar, make_bar(MONDAY + timedelta(minutes=15))])
        report = validate_dataset(ds, expected_symbol="XAUUSD")
        assert "invalid_spread" in _codes(report, "error")

    def test_spread_over_budget_is_error(self):
        bars = [make_bar(MONDAY, spread=1.5), make_bar(MONDAY + timedelta(minutes=15), spread=1.5),
                make_bar(MONDAY + timedelta(minutes=30), spread=1.5)]
        report = validate_dataset(
            make_dataset(bars), expected_symbol="XAUUSD", max_spread_price=0.8,
        )
        assert "invalid_spread" in _codes(report, "error")

    def test_stale_data_flagged_only_against_now(self):
        ds = make_dataset(make_series(MONDAY, 8, 15))
        later = MONDAY + timedelta(hours=8)
        live = validate_dataset(ds, expected_symbol="XAUUSD", now=later)
        historical = validate_dataset(ds, expected_symbol="XAUUSD", now=None)
        assert "stale_data" in _codes(live, "error")
        assert "stale_data" not in _codes(historical)
        assert historical.ok


class TestIdentityAndProxy:
    def test_symbol_mismatch_is_error(self):
        ds = make_dataset(make_series(MONDAY, 3, 15), symbol="XAUUSD")
        report = validate_dataset(ds, expected_symbol="GC=F")
        assert "symbol_mismatch" in _codes(report, "error")

    def test_unlabelled_proxy_is_error(self):
        bars = make_series(MONDAY, 3, 15)
        meta = DatasetMeta(
            source="fixture", symbol="XAUUSD", source_symbol="GC=F",
            asset_kind=AssetKind.GOLD_FUTURES_PROXY, data_type=DataKind.OHLCV,
            timeframe=TimeFrame.M15, is_proxy=False, disclaimer=None,
        )
        report = validate_dataset(
            GoldDataset(meta=meta, bars=tuple(bars)), expected_symbol="XAUUSD",
        )
        assert "proxy_label_missing" in _codes(report, "error")

    def test_labelled_proxy_passes(self):
        ds = make_dataset(make_series(MONDAY, 3, 15), proxy=True)
        assert validate_dataset(ds, expected_symbol="XAUUSD").ok

    def test_empty_dataset_warns(self):
        report = validate_dataset(make_dataset([]), expected_symbol="XAUUSD")
        assert report.ok  # warnings only
        assert "empty_dataset" in _codes(report, "warning")

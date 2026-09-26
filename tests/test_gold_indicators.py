"""Phase 3 — deterministic indicator math (hand-checkable)."""

import pandas as pd
import pytest

from tradingagents.gold.technicals import indicators


def _series(values):
    return pd.Series([float(v) for v in values])


class TestEma:
    def test_constant_series_stays_constant(self):
        out = indicators.ema(_series([5.0] * 10), 3)
        assert out.iloc[-1] == pytest.approx(5.0)

    def test_known_three_point_values(self):
        # alpha = 2/(3+1) = 0.5: [1, 1.5, 2.25]
        out = indicators.ema(_series([1, 2, 3]), 3)
        assert list(out.round(6)) == [1.0, 1.5, 2.25]

    def test_faster_period_reacts_sooner(self):
        rising = _series(range(1, 31))
        fast = indicators.ema(rising, 5).iloc[-1]
        slow = indicators.ema(rising, 20).iloc[-1]
        assert fast > slow  # rising series: shorter EMA sits higher


class TestRsi:
    def test_all_gains_is_100(self):
        out = indicators.rsi(_series(range(1, 30)), 14)
        assert out.iloc[-1] == pytest.approx(100.0)

    def test_all_losses_is_0(self):
        out = indicators.rsi(_series(range(30, 1, -1)), 14)
        assert out.iloc[-1] == pytest.approx(0.0)

    def test_range_is_bounded(self):
        zigzag = [100, 101, 100, 102, 101, 103, 100, 104, 101, 105] * 3
        out = indicators.rsi(_series(zigzag), 14).dropna()
        assert ((out >= 0) & (out <= 100)).all()


class TestAtrAndRoc:
    def test_constant_range_atr_equals_range(self):
        n = 40
        high = _series([101.0] * n)
        low = _series([99.0] * n)
        close = _series([100.0] * n)
        out = indicators.atr(high, low, close, 14)
        assert out.iloc[-1] == pytest.approx(2.0)

    def test_roc_matches_definition(self):
        close = _series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120])
        out = indicators.roc(close, 10)
        assert out.iloc[-1] == pytest.approx((120 - 100) / 100 * 100)

    def test_true_range_uses_previous_close(self):
        # h-l is small; the gap vs previous close dominates TR.
        high = _series([10.0, 12.0])
        low = _series([9.0, 11.0])
        close = _series([9.5, 11.5])
        tr = indicators.true_range(high, low, close)
        assert tr.iloc[0] == pytest.approx(1.0)          # h - l
        assert tr.iloc[1] == pytest.approx(2.5)          # |high - prev close|


class TestSlope:
    def test_rising_line_positive_slope(self):
        line = _series([100, 101, 102, 103, 104, 105])
        slope = indicators.slope_pct(line, lookback=5)
        assert slope == pytest.approx((105 - 100) / 5 / 105 * 100)

    def test_flat_line_zero_slope(self):
        assert indicators.slope_pct(_series([50.0] * 10), 5) == pytest.approx(0.0)

    def test_short_series_is_none(self):
        assert indicators.slope_pct(_series([1, 2]), 5) is None

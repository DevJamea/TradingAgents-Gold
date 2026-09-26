"""Phase 3 — structured technical snapshot: regime, MTF alignment, determinism.

Phase 3 PASS condition: the same input data always produces the same
technical output (``test_snapshot_is_deterministic``).
"""

from datetime import datetime, timedelta, timezone

from tests.gold_fixtures import MONDAY, make_dataset
from tradingagents.gold.data.models import GoldBar
from tradingagents.gold.technicals.snapshot import TechnicalEngine, TechnicalsConfig
from tradingagents.gold.types import (
    MarketRegime,
    SessionName,
    TimeFrame,
    TrendDirection,
    VolatilityRegime,
)

PINNED = datetime(2026, 3, 4, 12, 0, tzinfo=timezone.utc)


def _bars_from_closes(closes, start=MONDAY, interval=60, half_range=0.4, start_hour=8):
    bars = []
    for i, close in enumerate(closes):
        ts = start + timedelta(hours=start_hour) + timedelta(minutes=interval * i)
        bars.append(GoldBar(
            timestamp=ts,
            open=close - 0.05,
            high=close + half_range,
            low=close - half_range,
            close=close,
            volume=10.0,
        ))
    return bars


def _datasets(closes_by_tf: dict[str, list[float]], half_ranges=None) -> dict:
    half_ranges = half_ranges or {}
    out = {}
    tf_map = {"M15": TimeFrame.M15, "H1": TimeFrame.H1, "H4": TimeFrame.H4}
    intervals = {"M15": 15, "H1": 60, "H4": 240}
    for tf, closes in closes_by_tf.items():
        bars = _bars_from_closes(
            closes, interval=intervals[tf], half_range=half_ranges.get(tf, 0.4),
        )
        out[tf] = make_dataset(bars, timeframe=tf_map[tf])
    return out


def _up(n=120, base=3000.0, step=0.8):
    return [base + step * i for i in range(n)]


def _down(n=120, base=3400.0, step=0.8):
    return [base - step * i for i in range(n)]


def _flat(n=120, base=3000.0):
    return [base] * n


class TestTrendAndRegime:
    def test_uptrend_on_all_timeframes(self):
        snap = TechnicalEngine().compute(
            "XAUUSD", _datasets({"M15": _up(), "H1": _up(), "H4": _up()}), now=PINNED,
        )
        assert snap.multi_timeframe.aligned
        assert snap.multi_timeframe.conflicts == ()
        assert snap.market_regime is MarketRegime.TREND_UP
        for tf in ("M15", "H1", "H4"):
            assert snap.per_timeframe[tf].trend.direction is TrendDirection.UP

    def test_downtrend_regime(self):
        snap = TechnicalEngine().compute(
            "XAUUSD", _datasets({"M15": _down(), "H1": _down(), "H4": _down()}), now=PINNED,
        )
        assert snap.market_regime is MarketRegime.TREND_DOWN

    def test_flat_normal_vol_is_range(self):
        snap = TechnicalEngine().compute(
            "XAUUSD",
            _datasets(
                {"M15": _flat(), "H1": _flat(), "H4": _flat()},
                half_ranges={"H1": 2.2},   # ATR/close ≈ 0.0015 → NORMAL band
            ),
            now=PINNED,
        )
        assert snap.per_timeframe["H1"].volatility.regime is VolatilityRegime.NORMAL
        assert snap.market_regime is MarketRegime.RANGE

    def test_flat_low_vol_regime(self):
        snap = TechnicalEngine().compute(
            "XAUUSD",
            _datasets({"M15": _flat(), "H1": _flat(), "H4": _flat()}),  # ±0.4 → tiny ATR
            now=PINNED,
        )
        assert snap.per_timeframe["H1"].volatility.regime is VolatilityRegime.LOW
        assert snap.market_regime is MarketRegime.LOW_VOLATILITY

    def test_flat_high_vol_regime(self):
        snap = TechnicalEngine().compute(
            "XAUUSD",
            _datasets(
                {"M15": _flat(), "H1": _flat(), "H4": _flat()},
                half_ranges={"H1": 9.0},   # ATR/close ≈ 0.006 → HIGH
            ),
            now=PINNED,
        )
        assert snap.per_timeframe["H1"].volatility.regime is VolatilityRegime.HIGH
        assert snap.market_regime is MarketRegime.HIGH_VOLATILITY

    def test_short_history_is_unknown(self):
        snap = TechnicalEngine().compute(
            "XAUUSD",
            _datasets({"M15": _up(5), "H1": _up(5), "H4": _up(5)}),
            now=PINNED,
        )
        assert snap.market_regime is MarketRegime.UNKNOWN

    def test_recent_cross_is_transition(self):
        closes = _up(100, step=0.05) + [3010.0, 2995.0, 2975.0, 2955.0]
        snap = TechnicalEngine().compute(
            "XAUUSD",
            _datasets({"M15": closes, "H1": closes, "H4": closes}),
            now=PINNED,
        )
        assert snap.market_regime is MarketRegime.TRANSITION


class TestMultiTimeframeConflicts:
    def test_h4_up_h1_down_is_reported_not_forced(self):
        snap = TechnicalEngine().compute(
            "XAUUSD",
            _datasets({"M15": _down(), "H1": _down(), "H4": _up()}),
            now=PINNED,
        )
        mtf = snap.multi_timeframe
        assert not mtf.aligned
        assert mtf.conflicts and "conflict" in mtf.conflicts[0]
        assert snap.market_regime is MarketRegime.TRANSITION  # disagreement
        assert mtf.h4_trend is TrendDirection.UP
        assert mtf.h1_trend is TrendDirection.DOWN


class TestDeterminism:
    def test_snapshot_is_deterministic(self):
        datasets = _datasets(
            {"M15": _up(), "H1": _up(80, step=0.4), "H4": _flat(80)},
            half_ranges={"H1": 2.0},
        )
        engine = TechnicalEngine()
        first = engine.compute("XAUUSD", datasets, now=PINNED)
        second = engine.compute("XAUUSD", datasets, now=PINNED)
        assert first == second  # identical dataclasses field-by-field


class TestSnapshotContent:
    def test_snapshot_carries_regime_session_and_provenance(self):
        # Last M15 bar at 13:30 UTC → NEW_YORK session (London overlap).
        closes = _up(60)
        bars = _bars_from_closes(closes, start=MONDAY + timedelta(days=1), interval=15)
        datasets = {
            "M15": make_dataset(bars, timeframe=TimeFrame.M15),
            "H1": _datasets({"H1": _up()})["H1"],
            "H4": _datasets({"H4": _up(80)})["H4"],
        }
        snap = TechnicalEngine().compute("XAUUSD", datasets, now=PINNED)
        assert snap.symbol == "XAUUSD"
        assert snap.session.name is SessionName.NEW_YORK
        assert len(snap.provenance) == 3
        assert snap.per_timeframe["H1"].momentum.rsi is not None
        assert snap.market_regime is MarketRegime.TREND_UP

    def test_custom_config_thresholds_respected(self):
        # low_vol_atr_pct = 0.0 ⇒ nothing can classify as LOW → NORMAL band.
        cfg = TechnicalsConfig(low_vol_atr_pct=0.0)
        snap = TechnicalEngine(cfg).compute(
            "XAUUSD",
            _datasets(
                {"M15": _flat(), "H1": _flat(), "H4": _flat()},
                half_ranges={"H1": 2.2},
            ),
            now=PINNED,
        )
        assert snap.per_timeframe["H1"].volatility.regime is VolatilityRegime.NORMAL

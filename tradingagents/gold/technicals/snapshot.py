"""Structured technical snapshot engine (Phase 3 pass condition).

``TechnicalEngine.compute`` turns M15/H1/H4 datasets into one
``GoldTechnicalSnapshot``: per-timeframe trend/momentum/volatility/structure,
an explicit multi-timeframe alignment report (conflicts are reported, never
forced into agreement), a deterministic market regime, and session context.

Determinism contract: the snapshot is a pure function of its bar inputs and
the ``TechnicalsConfig`` — no clock, no randomness, no LLM.  The same input
data always produces the same technical output.

Market-regime priority (exact, documented):
1. UNKNOWN        – H1 or H4 trend cannot be computed (insufficient data).
2. TRANSITION     – EMA20/EMA50 crossed on H1 within the last CROSS_WINDOW
                    bars (a fresh flip; direction not yet established).
3. TREND_UP       – H1 trend UP and H4 trend not DOWN.
4. TREND_DOWN     – H1 trend DOWN and H4 trend not UP.
5. HIGH_VOLATILITY / LOW_VOLATILITY – no direction established and H1
                    normalised ATR beyond the configured thresholds.
6. RANGE          – no direction, volatility normal.
(H1 directional but H4 conflicting falls through to TRANSITION: the multi-tf
report carries the explicit conflict text either way.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from tradingagents.gold.data.models import GoldDataset
from tradingagents.gold.technicals import indicators, structure
from tradingagents.gold.technicals.sessions import SessionContext, session_at
from tradingagents.gold.types import (
    BreakoutState,
    MarketRegime,
    StructureBias,
    TrendDirection,
    VolatilityRegime,
    utc_now,
)

DEFAULT_EMA_PERIODS = (20, 50, 200)
DEFAULT_SLOPE_LOOKBACK = 5


@dataclass(frozen=True)
class TechnicalsConfig:
    """Engine parameters.  Every knob has one analytical purpose."""

    ema_periods: tuple[int, int, int] = DEFAULT_EMA_PERIODS
    rsi_period: int = 14          # momentum context
    atr_period: int = 14          # volatility + stop context
    roc_period: int = 10          # momentum confirmation
    slope_lookback: int = DEFAULT_SLOPE_LOOKBACK   # EMA slope measurement
    swing_left: int = 2           # fractal swing confirmation bars
    swing_right: int = 2
    swing_levels: int = 3         # S/R + breakout envelope depth
    cross_window: int = 5         # bars an EMA cross marks a TRANSITION
    high_vol_atr_pct: float = 0.0025   # H1 ATR/close ≥ this is HIGH vol
    low_vol_atr_pct: float = 0.0006    # H1 ATR/close ≤ this is LOW vol


@dataclass(frozen=True)
class TrendBlock:
    ema20: float | None
    ema50: float | None
    ema200: float | None
    ema20_slope_pct: float | None
    ema50_slope_pct: float | None
    direction: TrendDirection


@dataclass(frozen=True)
class MomentumBlock:
    rsi: float | None
    roc_pct: float | None


@dataclass(frozen=True)
class VolatilityBlock:
    atr: float | None
    atr_pct: float | None
    regime: VolatilityRegime


@dataclass(frozen=True)
class StructureBlock:
    swing_highs: tuple[tuple[str, float], ...]   # (ISO timestamp, price)
    swing_lows: tuple[tuple[str, float], ...]
    bias: StructureBias
    support: tuple[float, ...]
    resistance: tuple[float, ...]
    breakout: BreakoutState


@dataclass(frozen=True)
class TimeframeTechnicals:
    trend: TrendBlock
    momentum: MomentumBlock
    volatility: VolatilityBlock
    structure: StructureBlock


@dataclass(frozen=True)
class MultiTimeframeReport:
    h4_trend: TrendDirection
    h1_trend: TrendDirection
    m15_trend: TrendDirection
    aligned: bool
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class GoldTechnicalSnapshot:
    symbol: str
    computed_at: str
    as_of: str                     # latest bar timestamp (ISO) across timeframes
    per_timeframe: dict[str, TimeframeTechnicals]
    multi_timeframe: MultiTimeframeReport
    market_regime: MarketRegime
    session: SessionContext
    provenance: tuple[str, ...] = field(default_factory=tuple)


def _trend_block(frame: pd.DataFrame, cfg: TechnicalsConfig) -> TrendBlock:
    p_fast, p_slow, p_base = cfg.ema_periods
    close = frame["Close"]
    e_fast, e_slow = ema_or_none(close, p_fast), ema_or_none(close, p_slow)
    e_base = ema_or_none(close, p_base)
    s_fast = indicators.slope_pct(e_fast, cfg.slope_lookback) if e_fast is not None else None
    s_slow = indicators.slope_pct(e_slow, cfg.slope_lookback) if e_slow is not None else None

    if e_fast is None or e_slow is None or e_fast.empty or e_slow.empty:
        return TrendBlock(None, None, None, None, None, TrendDirection.UNKNOWN)
    f, sl = float(e_fast.iloc[-1]), float(e_slow.iloc[-1])
    b = float(e_base.iloc[-1]) if e_base is not None and not e_base.empty else None
    # A slow EMA needs a full window of bars before "trend" means anything.
    if len(close) < p_slow:
        return TrendBlock(f, sl, b, s_fast, s_slow, TrendDirection.UNKNOWN)
    direction = _trend_direction(f, sl, s_slow)
    return TrendBlock(f, sl, b, s_fast, s_slow, direction)


def ema_or_none(close: pd.Series, period: int) -> pd.Series | None:
    if len(close) < 2:
        return None
    return indicators.ema(close, period)


def _trend_direction(fast: float, slow: float, slow_slope: float | None) -> TrendDirection:
    if slow_slope is None:
        return TrendDirection.UNKNOWN
    if fast > slow and slow_slope > 0:
        return TrendDirection.UP
    if fast < slow and slow_slope < 0:
        return TrendDirection.DOWN
    return TrendDirection.MIXED


def _vol_block(frame: pd.DataFrame, cfg: TechnicalsConfig) -> VolatilityBlock:
    if len(frame) < 2:
        return VolatilityBlock(None, None, VolatilityRegime.UNKNOWN)
    atr = indicators.atr(frame["High"], frame["Low"], frame["Close"], cfg.atr_period)
    value = float(atr.iloc[-1])
    atr_pct = value / float(frame["Close"].iloc[-1])
    if atr_pct >= cfg.high_vol_atr_pct:
        regime = VolatilityRegime.HIGH
    elif atr_pct <= cfg.low_vol_atr_pct:
        regime = VolatilityRegime.LOW
    else:
        regime = VolatilityRegime.NORMAL
    return VolatilityBlock(value, atr_pct, regime)


def _structure_block(frame: pd.DataFrame, cfg: TechnicalsConfig) -> StructureBlock:
    highs, lows = structure.find_swings(frame, cfg.swing_left, cfg.swing_right)
    bias = structure.classify_structure(highs, lows)
    support, resistance = structure.support_resistance(lows, highs, cfg.swing_levels)
    close = float(frame["Close"].iloc[-1])
    breakout = structure.breakout_state(close, highs, lows, cfg.swing_levels)

    def _iso(swings):
        return tuple((s.timestamp.isoformat(), s.price) for s in swings[-5:])

    return StructureBlock(
        swing_highs=_iso(highs),
        swing_lows=_iso(lows),
        bias=bias,
        support=tuple(support),
        resistance=tuple(resistance),
        breakout=breakout,
    )


def _recent_cross_up_or_down(frame: pd.DataFrame, cfg: TechnicalsConfig) -> bool:
    """Whether EMA20/EMA50 crossed within the last ``cross_window`` bars."""
    p_fast, p_slow, _ = cfg.ema_periods
    if len(frame) < p_slow:
        return False
    close = frame["Close"]
    fast = indicators.ema(close, p_fast)
    slow = indicators.ema(close, p_slow)
    diff = (fast - slow).apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
    flips = diff.ne(diff.shift(1)) & diff.shift(1).notna() & diff.ne(0)
    tail = flips.iloc[-cfg.cross_window:]
    return bool(tail.any())


class TechnicalEngine:
    """Deterministic technical engine over multi-timeframe gold datasets."""

    def __init__(self, config: TechnicalsConfig | None = None) -> None:
        self.config = config or TechnicalsConfig()

    def compute(
        self,
        symbol: str,
        datasets: dict[str, GoldDataset],
        now: datetime | None = None,
    ) -> GoldTechnicalSnapshot:
        """Pure function of ``datasets`` + config (``now`` only stamps the
        record's ``computed_at`` so determinism tests can pin it)."""
        cfg = self.config
        frames = {}
        per_tf: dict[str, TimeframeTechnicals] = {}
        for key, ds in datasets.items():
            frame = ds.frame()
            if frame.empty:
                continue  # an empty series is "no data", not zero prices
            frames[key] = frame
            close = frame["Close"]
            momentum = MomentumBlock(
                rsi=float(indicators.rsi(close, cfg.rsi_period).iloc[-1]) if len(close) >= 2 else None,
                roc_pct=float(indicators.roc(close, cfg.roc_period).iloc[-1]) if len(close) > cfg.roc_period else None,
            )
            per_tf[key] = TimeframeTechnicals(
                trend=_trend_block(frame, cfg),
                momentum=momentum,
                volatility=_vol_block(frame, cfg),
                structure=_structure_block(frame, cfg),
            )

        def _dir(tf: str) -> TrendDirection:
            block = per_tf.get(tf)
            return block.trend.direction if block else TrendDirection.UNKNOWN

        h4, h1, m15 = _dir("H4"), _dir("H1"), _dir("M15")
        directional = {h4, h1, m15} - {TrendDirection.UNKNOWN, TrendDirection.MIXED}
        aligned = len(directional) == 1 and TrendDirection.MIXED not in {h4, h1, m15}
        conflicts: list[str] = []
        if TrendDirection.UP in directional and TrendDirection.DOWN in directional:
            conflicts.append(
                f"timeframe conflict: H4={h4.value} H1={h1.value} M15={m15.value} — "
                "directions disagree; do not force a single view"
            )
        if not conflicts and TrendDirection.MIXED in {h4, h1, m15} and directional:
            conflicts.append(
                f"partial alignment: H4={h4.value} H1={h1.value} M15={m15.value}"
            )

        mtf = MultiTimeframeReport(h4, h1, m15, aligned, tuple(conflicts))
        regime = self._regime(per_tf, frames, h1, h4)
        as_of = max(
            (ds.bars[-1].timestamp for ds in datasets.values() if ds.bars),
            default=None,
        )
        session = session_at(as_of) if as_of else session_at(utc_now())
        stamp = (now or utc_now()).isoformat()
        return GoldTechnicalSnapshot(
            symbol=symbol,
            computed_at=stamp,
            as_of=as_of.isoformat() if as_of else "",
            per_timeframe=per_tf,
            multi_timeframe=mtf,
            market_regime=regime,
            session=session,
            provenance=tuple(ds.meta.describe() for ds in datasets.values()),
        )

    def _regime(
        self,
        per_tf: dict[str, TimeframeTechnicals],
        frames: dict[str, pd.DataFrame],
        h1: TrendDirection,
        h4: TrendDirection,
    ) -> MarketRegime:
        cfg = self.config
        if h1 is TrendDirection.UNKNOWN or h4 is TrendDirection.UNKNOWN:
            return MarketRegime.UNKNOWN
        h1_frame = frames.get("H1")
        if h1_frame is not None and _recent_cross_up_or_down(h1_frame, cfg):
            return MarketRegime.TRANSITION
        if h1 is TrendDirection.UP and h4 is not TrendDirection.DOWN:
            return MarketRegime.TREND_UP
        if h1 is TrendDirection.DOWN and h4 is not TrendDirection.UP:
            return MarketRegime.TREND_DOWN
        if h1 is TrendDirection.MIXED:
            vol = per_tf.get("H1")
            if vol is not None and vol.volatility.regime is VolatilityRegime.HIGH:
                return MarketRegime.HIGH_VOLATILITY
            if vol is not None and vol.volatility.regime is VolatilityRegime.LOW:
                return MarketRegime.LOW_VOLATILITY
            return MarketRegime.RANGE
        # H1 directional but H4 conflicting: a fresh disagreement ⇒ transition.
        return MarketRegime.TRANSITION

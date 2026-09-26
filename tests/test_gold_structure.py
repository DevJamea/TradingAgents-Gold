"""Phase 3 — market structure, support/resistance, breakout, sessions."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tradingagents.gold.technicals.sessions import session_at
from tradingagents.gold.technicals.structure import (
    Swing,
    breakout_state,
    classify_structure,
    find_swings,
    support_resistance,
)
from tradingagents.gold.types import BreakoutState, SessionName, StructureBias


def _frame(highs, lows, closes=None):
    n = len(highs)
    closes = closes or highs
    stamps = [datetime(2026, 3, 2, tzinfo=timezone.utc) + timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame({
        "Date": stamps,
        "Open": closes,
        "High": [float(h) for h in highs],
        "Low": [float(v) for v in lows],
        "Close": [float(c) for c in closes],
        "Volume": [1.0] * n,
    })


class TestSwings:
    def test_zigzag_finds_interior_swings(self):
        frame = _frame(
            highs=[1, 2, 3, 2, 1, 2, 3, 2, 1],
            lows=[3, 2, 1, 2, 3, 2, 1, 2, 3],
        )
        highs, lows = find_swings(frame, left=2, right=2)
        assert [s.price for s in highs] == [3.0, 3.0]
        assert [s.price for s in lows] == [1.0, 1.0]
        assert highs[0].timestamp < highs[1].timestamp

    def test_monotonic_series_has_no_interior_swings(self):
        frame = _frame(highs=list(range(1, 21)), lows=list(range(0, 20)))
        highs, lows = find_swings(frame, left=2, right=2)
        assert highs == [] and lows == []


class TestStructureClassification:
    def test_hh_hl_is_uptrend(self):
        highs = [Swing(None, 10), Swing(None, 12)]      # higher highs
        lows = [Swing(None, 8), Swing(None, 9)]         # higher lows
        assert classify_structure(highs, lows) is StructureBias.UPTREND

    def test_lh_ll_is_downtrend(self):
        highs = [Swing(None, 12), Swing(None, 10)]
        lows = [Swing(None, 9), Swing(None, 8)]
        assert classify_structure(highs, lows) is StructureBias.DOWNTREND

    def test_mixed_is_range(self):
        highs = [Swing(None, 10), Swing(None, 12)]
        lows = [Swing(None, 9), Swing(None, 8)]
        assert classify_structure(highs, lows) is StructureBias.RANGE

    def test_insufficient_swings_unknown(self):
        assert classify_structure([Swing(None, 10)], []) is StructureBias.UNKNOWN


class TestSupportResistanceAndBreakout:
    def test_levels_derived_from_recent_swings(self):
        lows = [Swing(None, p) for p in (8.0, 9.0, 8.5)]
        highs = [Swing(None, p) for p in (11.0, 12.0, 11.5)]
        supports, resistances = support_resistance(lows, highs, n=3)
        assert supports == [9.0, 8.5, 8.0]              # nearest first
        assert resistances == [11.0, 11.5, 12.0]        # nearest first

    def test_breakout_up_and_down_and_none(self):
        highs = [Swing(None, p) for p in (10.0, 11.0, 11.5)]
        lows = [Swing(None, p) for p in (9.0, 9.5, 9.2)]
        assert breakout_state(12.0, highs, lows) is BreakoutState.UP
        assert breakout_state(8.0, highs, lows) is BreakoutState.DOWN
        assert breakout_state(10.5, highs, lows) is BreakoutState.NONE


class TestSessions:
    @pytest.mark.parametrize("hour,primary,overlaps", [
        (2, SessionName.ASIAN, ()),
        (7, SessionName.LONDON, ("ASIA_LONDON",)),
        (13, SessionName.NEW_YORK, ("LONDON_NEW_YORK",)),
        (10, SessionName.LONDON, ()),
        (22, SessionName.OFF, ()),
    ])
    def test_session_map(self, hour, primary, overlaps):
        ctx = session_at(datetime(2026, 3, 4, hour, 30, tzinfo=timezone.utc))
        assert ctx.name is primary
        assert ctx.overlaps == overlaps

    def test_flags_are_consistent(self):
        ctx = session_at(datetime(2026, 3, 4, 13, 0, tzinfo=timezone.utc))
        assert ctx.in_new_york and ctx.in_london and not ctx.in_asian

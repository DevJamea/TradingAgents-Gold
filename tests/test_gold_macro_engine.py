"""Phase 4 — deterministic gold macro engine (FRED/ALFRED, point-in-time).

The vendor fetcher is injected with canned reports in the exact upstream
format, so these tests exercise the engine's parsing, containment and PIT
bookkeeping — not the network.
"""

import pytest

from tradingagents.gold.macro.engine import (
    DXY_LIMITATION_NOTE,
    GOLD_MACRO_SERIES,
    MacroEngine,
    parse_latest,
)

REPORT_TEMPLATE = (
    "## FRED: Some Series ({sid})\n"
    "- Units: index\n"
    "- Frequency: Monthly\n"
    "- Window: 2026-01-01 to {date}\n"
    "\n**Latest:** {value} ({date}) | **Change over window:** +1.00 from 99.0\n"
)


def canned_fetch(indicator, curr_date):
    sid = GOLD_MACRO_SERIES.get(indicator, "CUSTOM")
    return REPORT_TEMPLATE.format(sid=sid, value="123.4", date=curr_date)


class TestSeriesTable:
    def test_gold_series_all_resolve_to_real_fred_ids(self):
        # Everything maps to an actual FRED ID (letters/digits only) — no
        # invented series.
        for sid in GOLD_MACRO_SERIES.values():
            assert sid.replace("_", "").isalnum() and sid.isupper()

    def test_gold_essentials_present(self):
        for alias in (
            "dollar_index", "fed_funds_rate", "yield_10y", "real_yield_10y",
            "cpi", "core_pce", "nonfarm_payrolls", "unemployment_rate", "vix",
        ):
            assert alias in GOLD_MACRO_SERIES, alias


class TestParsing:
    def test_parse_latest_extracts_value_and_date(self):
        report = REPORT_TEMPLATE.format(sid="X", value="4.33", date="2026-03-01")
        value, obs = parse_latest(report)
        assert value == pytest.approx(4.33)
        assert obs == "2026-03-01"

    def test_parse_latest_tolerates_garbage(self):
        assert parse_latest("no data here") == (None, None)
        assert parse_latest("") == (None, None)


class TestMacroEngine:
    def test_snapshot_parses_all_series_with_vintage(self):
        snap = MacroEngine(fetcher=canned_fetch).snapshot("2026-03-20")
        assert snap.ok
        for alias in GOLD_MACRO_SERIES:
            point = snap.points[alias]
            assert point.ok
            assert point.value == pytest.approx(123.4)
            assert point.observation_date == "2026-03-20"
            assert point.vintage_date == "2026-03-20"   # pinned to as-of

    def test_fetcher_receives_indicator_and_as_of_date(self):
        seen = []

        def spy(indicator, curr_date):
            seen.append((indicator, curr_date))
            return REPORT_TEMPLATE.format(sid="X", value="1", date=curr_date)

        MacroEngine(fetcher=spy).snapshot("2026-03-20")
        assert ("dollar_index", "2026-03-20") in seen
        assert len(seen) == len(GOLD_MACRO_SERIES)

    def test_single_series_failure_is_contained(self):
        def flaky(indicator, curr_date):
            if indicator == "vix":
                raise RuntimeError("boom")
            return canned_fetch(indicator, curr_date)

        snap = MacroEngine(fetcher=flaky).snapshot("2026-03-20")
        assert not snap.points["vix"].ok
        assert "RuntimeError" in snap.points["vix"].error
        assert snap.points["cpi"].ok and snap.ok   # others survive

    def test_unparsed_report_is_recorded_not_crashed(self):
        snap = MacroEngine(fetcher=lambda i, d: "FRED series 'X' not found.").snapshot("2026-03-20")
        assert all(not p.ok for p in snap.points.values())
        assert not snap.ok

    def test_render_contains_dxy_limitation_and_vintage_note(self):
        snap = MacroEngine(fetcher=canned_fetch).snapshot("2026-03-20")
        text = snap.render()
        assert "PROXY for the ICE DXY" in text
        assert "realtime_start=realtime_end=2026-03-20" in text
        assert DXY_LIMITATION_NOTE in text

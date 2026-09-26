"""Phase 4 — gold news engine: window clamping + deterministic future filter.

Phase 4 pass condition (partial): a historical news collection never contains
future information — the window is clamped to the decision date AND future
stamped lines are filtered out deterministically.
"""

from tradingagents.gold.config import GoldNewsConfig
from tradingagents.gold.news.engine import (
    NEWS_SYMBOL,
    GoldNewsEngine,
    filter_out_of_window,
)

BLOCK = "\n".join([
    "News for GC=F between 2026-03-10 and 2026-03-20:",
    "- [2026-03-12 09:00 UTC] Fed holds rates steady [Source: wires]",
    "- [2026-03-25 08:00 UTC] FUTURE ARTICLE must never appear [Source: wires]",
    "- [2026-03-18 14:30 UTC] Gold rallies on CPI miss [Source: wires]",
    "",
])


class TestFutureFilter:
    def test_future_and_old_lines_are_dropped(self):
        kept = filter_out_of_window(BLOCK, "2026-03-10", "2026-03-20")
        assert "FUTURE ARTICLE" not in kept
        assert "Gold rallies on CPI miss" in kept
        assert "Fed holds rates steady" in kept

    def test_header_lines_survive(self):
        kept = filter_out_of_window(BLOCK, "2026-03-10", "2026-03-20")
        assert "News for GC=F" in kept

    def test_no_stamp_line_survives(self):
        kept = filter_out_of_window("plain line without stamp", "2026-03-10", "2026-03-20")
        assert kept == "plain line without stamp"


class TestGoldNewsEngine:
    def test_window_is_clamped_to_trade_date(self):
        seen = {}

        def company(symbol, start, end):
            seen["company"] = (symbol, start, end)
            return ""

        def glob(end, lookback, limit):
            seen["global"] = (end, lookback, limit)
            return ""

        cfg = GoldNewsConfig(lookback_days=7)
        engine = GoldNewsEngine(cfg, company_fetch=company, global_fetch=glob)
        report = engine.collect("2026-03-20")
        assert seen["company"][0] == NEWS_SYMBOL == "GC=F"
        assert seen["company"][1] == "2026-03-13"
        assert seen["company"][2] == "2026-03-20"
        assert seen["global"][0] == "2026-03-20"
        assert report.window_start == "2026-03-13"
        assert report.window_end == "2026-03-20"
        assert report.queries and "gold" in report.queries[0]

    def test_future_headlines_filtered_from_blocks(self):
        engine = GoldNewsEngine(
            GoldNewsConfig(lookback_days=5),
            company_fetch=lambda s, a, b: BLOCK,
            global_fetch=lambda e, lb, lim: BLOCK,
        )
        report = engine.collect("2026-03-20")
        assert "FUTURE ARTICLE" not in report.company_block
        assert "FUTURE ARTICLE" not in report.global_block

    def test_fetch_failure_is_recorded_not_raised(self):
        def boom(*args, **kwargs):
            raise RuntimeError("down")

        engine = GoldNewsEngine(
            GoldNewsConfig(), company_fetch=boom, global_fetch=lambda *a, **k: "",
        )
        report = engine.collect("2026-03-20")
        assert any("instrument news unavailable" in n for n in report.notes)
        assert "as of 2026-03-20" in report.render()

    def test_render_states_pit_rule(self):
        engine = GoldNewsEngine(
            GoldNewsConfig(), company_fetch=lambda *a: "", global_fetch=lambda *a: "",
        )
        text = engine.collect("2026-03-20").render()
        assert "future-dated items are filtered" in text

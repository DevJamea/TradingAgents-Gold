"""Phase 4 — gold sentiment engine: symbol mapping + window + screen wiring."""

from tradingagents.gold.sentiment.engine import (
    GOLD_TWITS_SYMBOLS,
    GoldSentimentEngine,
)


def fake_twits(blocks_by_symbol):
    calls = []

    def fetch(ticker, *args, **kwargs):
        calls.append(ticker)
        return blocks_by_symbol.get(ticker, "(no messages)")

    return fetch, calls


def fake_reddit(blocks_by_query):
    calls = []

    def fetch(query, *args, **kwargs):
        calls.append(query)
        return blocks_by_query.get(query, "(no posts)")

    return fetch, calls


class TestSymbolMapping:
    def test_twits_uses_gold_streams_not_xauusd(self):
        fetch, calls = fake_twits({"GOLD": "bullish chatter", "GLD": "etl flow"})
        reddit, reddit_calls = fake_reddit({"gold": "reddit gold thread"})
        report = GoldSentimentEngine(twits_fetch=fetch, reddit_fetch=reddit).collect("2026-03-20")
        assert set(calls) == {"GOLD", "GLD"}
        assert "XAUUSD" not in calls
        assert "gold" in reddit_calls
        assert "bullish chatter" in report.stocktwits_block
        assert report.twits_symbols == GOLD_TWITS_SYMBOLS

    def test_windows_are_passed_to_vendors(self):
        seen = {}

        def twits(ticker, *args, **kwargs):
            seen.setdefault("twits", []).append(kwargs)
            return ""

        def reddit(query, *args, **kwargs):
            seen["reddit"] = kwargs
            return ""

        GoldSentimentEngine(twits_fetch=twits, reddit_fetch=reddit).collect(
            "2026-03-20", lookback_days=3,
        )
        assert all(kw["start_date"] == "2026-03-17" for kw in seen["twits"])
        assert all(kw["end_date"] == "2026-03-20" for kw in seen["twits"])
        assert seen["reddit"]["start_date"] == "2026-03-17"

    def test_screen_provider_wired(self):
        marker = object()

        def screen_provider(ticker):
            assert ticker == "XAUUSD"
            return marker

        seen = {}

        def twits(ticker, *args, **kwargs):
            seen["screen"] = kwargs.get("screen")
            return ""

        GoldSentimentEngine(
            twits_fetch=twits, reddit_fetch=lambda *a, **k: "",
            screen_provider=screen_provider,
        ).collect("2026-03-20")
        assert seen["screen"] is marker

    def test_stream_failure_is_contained(self):
        def twits(ticker, *args, **kwargs):
            if ticker == "GLD":
                raise RuntimeError("down")
            return "ok stream"

        report = GoldSentimentEngine(
            twits_fetch=twits, reddit_fetch=lambda *a, **k: "posts",
        ).collect("2026-03-20")
        assert "ok stream" in report.stocktwits_block
        assert any("GLD unavailable" in n for n in report.notes)
        assert "supporting evidence only" in report.render()

    def test_render_flags_sentiment_as_evidence_only(self):
        report = GoldSentimentEngine(
            twits_fetch=lambda *a, **k: "", reddit_fetch=lambda *a, **k: "",
        ).collect("2026-03-20")
        assert "must not by itself produce a BUY/SELL signal" in report.render()

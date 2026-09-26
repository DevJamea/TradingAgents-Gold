"""Phase 5 — gold run context builder (deterministic blocks per trade date)."""

from datetime import datetime, timedelta, timezone

from tests.gold_fixtures import make_dataset, make_series
from tradingagents.gold.agents.context import (
    GoldContextBuilder,
    as_of_for_trade_date,
)
from tradingagents.gold.config import default_gold_config
from tradingagents.gold.data.memory import InMemoryGoldProvider
from tradingagents.gold.macro.engine import MacroEngine
from tradingagents.gold.news.engine import GoldNewsEngine
from tradingagents.gold.sentiment.engine import GoldSentimentEngine
from tradingagents.gold.types import TimeFrame

SEPT_MONDAY = datetime(2026, 9, 14, 0, 0, tzinfo=timezone.utc)
TRADE_DATE = "2026-09-18"


def offline_provider() -> InMemoryGoldProvider:
    provider = InMemoryGoldProvider()
    provider.add_dataset("XAUUSD", make_dataset(make_series(SEPT_MONDAY, 480, 15), timeframe=TimeFrame.M15))
    provider.add_dataset("XAUUSD", make_dataset(make_series(SEPT_MONDAY, 120, 60), timeframe=TimeFrame.H1))
    # 60 H4 bars: the trend rule needs >= EMA-slow (50) bars before "trend" is defined.
    # H4 history starts earlier so >= EMA-slow (50) bars exist BY the analysis
    # date (slicing at as_of leaves only ~30 bars from Sep 14 itself).
    aug_monday = SEPT_MONDAY - timedelta(days=21)
    provider.add_dataset("XAUUSD", make_dataset(make_series(aug_monday, 200, 240), timeframe=TimeFrame.H4))
    return provider


def offline_builder(**overrides) -> GoldContextBuilder:
    kwargs = {
        "config": default_gold_config(),
        "provider": offline_provider(),
        "macro_engine": MacroEngine(fetcher=lambda i, d: "no data"),
        "news_engine": GoldNewsEngine(
            company_fetch=lambda *a: "", global_fetch=lambda *a: "",
        ),
        "sentiment_engine": GoldSentimentEngine(
            twits_fetch=lambda *a, **k: "", reddit_fetch=lambda *a, **k: "",
        ),
        "now_fn": lambda: datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
    }
    kwargs.update(overrides)
    return GoldContextBuilder(**kwargs)


class TestAsOfRule:
    def test_historical_trade_date_uses_market_close(self):
        as_of = as_of_for_trade_date(
            "2026-09-18", datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
        )
        assert as_of == datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)

    def test_today_uses_now(self):
        now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        assert as_of_for_trade_date("2026-09-24", now) == now


class TestBuildContext:
    def test_blocks_are_built_and_carry_disclaimer(self):
        ctx = offline_builder().build(TRADE_DATE)
        assert ctx.market_ok
        assert not ctx.has_data_error
        assert "DETERMINISTIC TECHNICAL SNAPSHOT" in ctx.technical_block
        assert "MARKET DATA" in ctx.market_data_block
        assert "PROXY" in ctx.market_data_block        # futures-proxy labelling
        assert "GOLD MACRO SNAPSHOT" in ctx.macro_block
        assert ctx.proxy_disclaimer.startswith("GC=F")

    def test_as_of_limits_visible_bars(self):
        ctx = offline_builder().build(TRADE_DATE)
        assert ctx.as_of == datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
        for ds in ctx.snapshot.datasets.values():
            assert all(b.timestamp <= ctx.as_of for b in ds.bars)

    def test_regime_computed_from_uptrend_fixtures(self):
        ctx = offline_builder().build(TRADE_DATE)
        assert ctx.technical is not None
        assert "regime: TREND_UP" in ctx.technical_block

    def test_builder_caches_per_trade_date(self):
        builder = offline_builder()
        assert builder.build(TRADE_DATE) is builder.build(TRADE_DATE)

    def test_missing_provider_data_is_a_data_error(self):
        ctx = GoldContextBuilder(
            provider=InMemoryGoldProvider(),
            macro_engine=MacroEngine(fetcher=lambda i, d: "x"),
            news_engine=GoldNewsEngine(company_fetch=lambda *a: "", global_fetch=lambda *a: ""),
            sentiment_engine=GoldSentimentEngine(
                twits_fetch=lambda *a, **k: "", reddit_fetch=lambda *a, **k: "",
            ),
        ).build(TRADE_DATE)
        assert ctx.has_data_error
        assert "DATA ERROR" in ctx.technical_block
        assert ctx.snapshot.errors  # provider_failure recorded per timeframe

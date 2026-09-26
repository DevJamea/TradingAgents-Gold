"""Per-run gold context: deterministic blocks injected into analyst prompts.

``GoldContextBuilder`` assembles, once per trade date, everything the gold
analysts interpret: the multi-timeframe market snapshot (quality-gated), the
deterministic technical snapshot, the macro snapshot, and the news and
sentiment collections.  Agents receive rendered blocks — they never call raw
APIs and never compute indicators (spec §7, §8).

Point-in-time rule: a run dated today uses "now" as ``as_of`` (and live
staleness checks apply); a historical run uses that date's 21:00 UTC (the gold
market close), so only bars published by the end of the analysis date are
visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from tradingagents.gold.config import GoldConfig, default_gold_config
from tradingagents.gold.data.memory import InMemoryGoldProvider
from tradingagents.gold.data.provider import GoldMarketDataProvider
from tradingagents.gold.data.snapshot import MarketSnapshot, build_market_snapshot
from tradingagents.gold.macro.engine import GoldMacroSnapshot, MacroEngine
from tradingagents.gold.news.engine import GoldNewsEngine, GoldNewsReport
from tradingagents.gold.sentiment.engine import (
    GoldSentimentEngine,
    GoldSentimentReport,
)
from tradingagents.gold.technicals.render import render_technical_snapshot
from tradingagents.gold.technicals.snapshot import (
    GoldTechnicalSnapshot,
    TechnicalEngine,
)
from tradingagents.gold.types import utc_now

#: Hour (UTC) of the gold market close used as the historical as-of instant.
_MARKET_CLOSE_HOUR = 21


def as_of_for_trade_date(trade_date: str, now: datetime | None = None) -> datetime:
    """The latest instant usable for a ``trade_date`` analysis."""
    now = now or utc_now()
    if trade_date >= now.date().isoformat():
        return now
    naive = datetime.fromisoformat(trade_date)
    return naive.replace(hour=_MARKET_CLOSE_HOUR, minute=0, tzinfo=timezone.utc)


def render_market_data_block(snapshot: MarketSnapshot, max_bars: int = 10) -> str:
    """Recent closes per timeframe plus the mandatory provenance block."""
    lines = ["## MARKET DATA (pre-fetched, deterministic)"]
    for tf, dataset in snapshot.datasets.items():
        bars = dataset.bars[-max_bars:]
        closes = ", ".join(f"{b.close:.2f}" for b in bars)
        latest = bars[-1].timestamp.strftime("%Y-%m-%d %H:%M UTC") if bars else "n/a"
        lines.append(f"### {tf} — last {len(bars)} closes (latest {latest})\n{closes}")
    lines.append("### Provenance")
    lines.extend(f"- {item}" for item in snapshot.provenance)
    lines.append(f"- {snapshot.symbol} is analysed as XAUUSD spot; the price source is a "
                 "labelled futures proxy wherever provenance says [PROXY].")
    return "\n".join(lines)


@dataclass
class GoldRunContext:
    """Everything a gold analyst node needs, built once per trade date."""

    trade_date: str
    as_of: datetime
    snapshot: MarketSnapshot | None
    technical: GoldTechnicalSnapshot | None
    macro: GoldMacroSnapshot | None
    news: GoldNewsReport | None
    sentiment: GoldSentimentReport | None
    market_data_block: str = ""
    technical_block: str = ""
    macro_block: str = ""
    news_block: str = ""
    sentiment_block: str = ""
    data_errors: list[str] = field(default_factory=list)
    proxy_disclaimer: str = ""

    @property
    def market_ok(self) -> bool:
        return self.snapshot is not None and self.snapshot.ok

    @property
    def has_data_error(self) -> bool:
        return bool(self.data_errors) or not self.market_ok


class GoldContextBuilder:
    """Builds :class:`GoldRunContext` with injectable engines and provider."""

    def __init__(
        self,
        config: GoldConfig | None = None,
        provider: GoldMarketDataProvider | None = None,
        technical_engine: TechnicalEngine | None = None,
        macro_engine: MacroEngine | None = None,
        news_engine: GoldNewsEngine | None = None,
        sentiment_engine: GoldSentimentEngine | None = None,
        now_fn=None,
    ) -> None:
        self.config = config or default_gold_config()
        # Default to an empty in-memory provider rather than hitting the
        # network implicitly: live wiring passes YahooGoldProxyProvider (or an
        # MT5 provider later) explicitly.
        self.provider = provider or InMemoryGoldProvider()
        self.technical_engine = technical_engine or TechnicalEngine()
        self.macro_engine = macro_engine or MacroEngine()
        self.news_engine = news_engine or GoldNewsEngine(self.config.news)
        self.sentiment_engine = sentiment_engine or GoldSentimentEngine()
        self.now_fn = now_fn or utc_now
        self._cache: dict[str, GoldRunContext] = {}

    def build(self, trade_date: str) -> GoldRunContext:
        """Context for ``trade_date`` (cached: one build per run/date)."""
        cached = self._cache.get(trade_date)
        if cached is not None:
            return cached

        now = self.now_fn()
        as_of = as_of_for_trade_date(trade_date, now)
        live = trade_date >= now.date().isoformat()

        snapshot = build_market_snapshot(
            self.provider, self.config, as_of=as_of, now=now if live else None,
        )
        datasets = {}
        for tf in self.config.data.timeframes:
            ds = snapshot.dataset(tf)
            if ds is not None and len(ds) > 0:
                datasets[tf.value] = ds   # string keys: the engine indexes by "M15"/"H1"/"H4"
        technical = (
            self.technical_engine.compute(self.config.data.symbol, datasets, now=as_of)
            if datasets else None
        )
        macro = self.macro_engine.snapshot(trade_date)
        news = self.news_engine.collect(trade_date)
        sentiment = self.sentiment_engine.collect(trade_date)

        ctx = GoldRunContext(
            trade_date=trade_date,
            as_of=as_of,
            snapshot=snapshot,
            technical=technical,
            macro=macro,
            news=news,
            sentiment=sentiment,
            market_data_block=render_market_data_block(snapshot),
            technical_block=(
                render_technical_snapshot(technical)
                if technical is not None
                else "## DETERMINISTIC TECHNICAL SNAPSHOT\n- DATA ERROR: no valid market data; "
                     "no technical interpretation is possible."
            ),
            macro_block=macro.render() if macro is not None else "(macro unavailable)",
            news_block=news.render() if news is not None else "(news unavailable)",
            sentiment_block=sentiment.render() if sentiment is not None else "(sentiment unavailable)",
            data_errors=list(snapshot.errors),
            proxy_disclaimer=self.config.data.proxy_disclaimer,
        )
        self._cache[trade_date] = ctx
        return ctx

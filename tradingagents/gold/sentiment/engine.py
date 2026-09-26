"""Gold/macro sentiment collection (spec §11).

Adapts the upstream social layer to gold: StockTwits streams are mapped from
XAUUSD to the actively-posted gold tickers (``GOLD``, ``GLD``) and Reddit is
searched for gold/macro terms.  The upstream windowing (#1220), the Jev
TypeSafe screen (reused as-is — it classifies relevance/stance for the
instrument) and the unarchived-coverage caveat all carry over.

Sentiment is SUPPORTING EVIDENCE ONLY: it never produces a BUY/SELL signal by
itself — the final decision schema treats it as one analyst input among many.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from tradingagents.agents.post_screen import jev_screen
from tradingagents.dataflows.vendors.reddit import fetch_reddit_posts
from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages

#: StockTwits streams that actually carry gold chatter (XAUUSD itself is not
#: a StockTwits symbol).  Ordered by volume of gold-related posts.
GOLD_TWITS_SYMBOLS: tuple[str, ...] = ("GOLD", "GLD")

#: Reddit search terms for gold/macro chatter.
GOLD_REDDIT_QUERIES: tuple[str, ...] = ("gold", "XAUUSD")

TwitsFetch = Callable[..., str]
RedditFetch = Callable[..., str]
ScreenProvider = Callable[[str], object | None]


@dataclass
class GoldSentimentReport:
    trade_date: str
    window_start: str | None
    window_end: str | None
    twits_symbols: tuple[str, ...] = field(default_factory=tuple)
    reddit_queries: tuple[str, ...] = field(default_factory=tuple)
    stocktwits_block: str = ""
    reddit_block: str = ""
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            f"## GOLD SENTIMENT — as of {self.trade_date}",
            "### StockTwits (gold streams)",
            self.stocktwits_block or "(unavailable)",
            "### Reddit (gold/macro chatter)",
            self.reddit_block or "(unavailable)",
            "Sentiment is supporting evidence only — it must not by itself "
            "produce a BUY/SELL signal.",
        ]
        parts.extend(self.notes)
        return "\n".join(parts)


class GoldSentimentEngine:
    """Collects gold sentiment through the upstream social vendors."""

    source_name = "gold-sentiment"

    def __init__(
        self,
        twits_fetch: TwitsFetch | None = None,
        reddit_fetch: RedditFetch | None = None,
        screen_provider: ScreenProvider | None = None,
    ) -> None:
        self._twits_fetch = twits_fetch or fetch_stocktwits_messages
        self._reddit_fetch = reddit_fetch or fetch_reddit_posts
        self._screen_provider = screen_provider or jev_screen

    def collect(
        self,
        trade_date: str,
        lookback_days: int = 7,
    ) -> GoldSentimentReport:
        """Collect sentiment as of ``trade_date`` with the given lookback.

        The window is passed straight through to the vendors, which trim
        messages to it (#1220) and report unreachable history as unavailable
        rather than silence.
        """
        start = (date.fromisoformat(trade_date) - timedelta(days=lookback_days)).isoformat()
        report = GoldSentimentReport(
            trade_date=trade_date, window_start=start, window_end=trade_date,
            twits_symbols=GOLD_TWITS_SYMBOLS, reddit_queries=GOLD_REDDIT_QUERIES,
        )

        screen = self._screen_provider("XAUUSD")
        twits_parts: list[str] = []
        for symbol in GOLD_TWITS_SYMBOLS:
            try:
                block = self._twits_fetch(
                    symbol, start_date=start, end_date=trade_date, screen=screen,
                )
                twits_parts.append(f"[{symbol}]\n{block}")
            except Exception as exc:  # noqa: BLE001 — one dead stream ≠ dead report
                report.notes.append(f"stocktwits {symbol} unavailable: {type(exc).__name__}: {exc}")
        report.stocktwits_block = "\n\n".join(twits_parts).strip()

        for query in GOLD_REDDIT_QUERIES:
            try:
                block = self._reddit_fetch(
                    query, start_date=start, end_date=trade_date, screen=screen,
                )
                report.reddit_block = (
                    f"{report.reddit_block}\n{block}".strip() if report.reddit_block else block
                )
            except Exception as exc:  # noqa: BLE001
                report.notes.append(f"reddit {query!r} unavailable: {type(exc).__name__}: {exc}")
        return report

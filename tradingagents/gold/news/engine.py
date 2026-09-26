"""Gold & macro news collection (spec §10).

Wraps the upstream news vendors with gold-relevant queries and enforces the
point-in-time window: a historical run may only see articles published inside
``[start, end]`` — future-dated lines are deterministically filtered out of the
rendered blocks, not merely hidden by the vendor.

Category coverage: gold, XAUUSD, USD, Federal Reserve, Treasury yields,
inflation, employment, central banks, major economic releases and geopolitical
events (see ``GoldNewsConfig.gold_queries``).  Each rendered headline retains
its publication timestamp and source as produced by the vendor.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from tradingagents.dataflows.date_window import as_of_window
from tradingagents.dataflows.vendors.yahoo.news import (
    get_global_news_yfinance,
    get_news_yfinance,
)
from tradingagents.gold.config import GoldNewsConfig

#: Canonical query symbol for the news vendors (GC=F carries a gold news feed).
NEWS_SYMBOL = "GC=F"

CompanyFetch = Callable[[str, str, str], str]   # (symbol, start, end) -> block
GlobalFetch = Callable[[str, int | None, int | None], str]  # (end, lookback, limit)

# Headline lines carry their publication stamp as "[YYYY-MM-DD ...]".
_STAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})")


def filter_out_of_window(block: str, start: str, end: str) -> str:
    """Drop headline lines stamped outside ``[start, end]`` (ISO dates).

    Deterministic second line of defence for point-in-time integrity: even if
    a vendor ignores the window, a future article cannot reach an analyst.
    Non-headline lines (headers, blank separators) are kept.
    """
    kept: list[str] = []
    for line in block.splitlines():
        match = _STAMP_RE.search(line)
        if match and not (start <= match.group(1) <= end):
            continue
        kept.append(line)
    return "\n".join(kept)


@dataclass
class GoldNewsReport:
    trade_date: str
    window_start: str
    window_end: str
    queries: tuple[str, ...] = field(default_factory=tuple)
    company_block: str = ""
    global_block: str = ""
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = [
            f"## GOLD & MACRO NEWS — window {self.window_start} .. {self.window_end} "
            f"(as of {self.trade_date})",
            "### Instrument news",
            self.company_block or "(no instrument news)",
            "### Global / macro news",
            self.global_block or "(no global news)",
            "Point-in-time: only articles published inside the window are included; "
            "future-dated items are filtered out deterministically.",
        ]
        parts.extend(self.notes)
        return "\n".join(parts)


class GoldNewsEngine:
    """Collects gold/macro news inside a clamped, auditable window."""

    source_name = "gold-news"

    def __init__(
        self,
        config: GoldNewsConfig | None = None,
        company_fetch: CompanyFetch | None = None,
        global_fetch: GlobalFetch | None = None,
    ) -> None:
        self._config = config or GoldNewsConfig()
        self._company_fetch = company_fetch or get_news_yfinance
        self._global_fetch = global_fetch or get_global_news_yfinance

    def collect(self, trade_date: str) -> GoldNewsReport:
        """Collect news as of ``trade_date`` (yyyy-mm-dd)."""
        end = trade_date
        start = (date.fromisoformat(trade_date) - timedelta(days=self._config.lookback_days)).isoformat()
        # Clamp so the end can never exceed the decision date (PIT rule).
        start, end = as_of_window(start, end, trade_date)

        report = GoldNewsReport(
            trade_date=trade_date, window_start=start, window_end=end,
            queries=tuple(self._config.gold_queries),
        )
        try:
            raw = self._company_fetch(NEWS_SYMBOL, start, end)
            report.company_block = filter_out_of_window(raw or "", start, end)
        except Exception as exc:  # noqa: BLE001 — news outage ≠ failed analysis
            report.notes.append(f"instrument news unavailable: {type(exc).__name__}: {exc}")
        try:
            raw = self._global_fetch(end, self._config.lookback_days, self._config.global_article_limit)
            report.global_block = filter_out_of_window(raw or "", start, end)
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"global news unavailable: {type(exc).__name__}: {exc}")
        return report

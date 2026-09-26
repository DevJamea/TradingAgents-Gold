"""Multi-timeframe market snapshot builder (Phase 2 pass condition).

``build_market_snapshot`` fetches M15/H1/H4 through a provider, runs the
deterministic data-quality gate on every series, attaches a quote when one
exists, and returns a ``MarketSnapshot`` whose ``ok`` flag is the AND of the
per-timeframe reports.  Invalid data flows out as DATA ERROR — downstream
(the risk gate) must answer NO TRADE on ``not snapshot.ok``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.gold.config import GoldConfig
from tradingagents.gold.data.calendar import ensure_utc, timeframe_minutes
from tradingagents.gold.data.models import GoldDataset, GoldQuote
from tradingagents.gold.data.provider import GoldMarketDataProvider
from tradingagents.gold.data.validation import ValidationReport, validate_dataset
from tradingagents.gold.types import TimeFrame, utc_now


@dataclass
class MarketSnapshot:
    """Multi-timeframe XAUUSD market state at one moment."""

    symbol: str
    built_at: datetime
    as_of: datetime
    datasets: dict[str, GoldDataset] = field(default_factory=dict)
    reports: dict[str, ValidationReport] = field(default_factory=dict)
    quote: GoldQuote | None = None
    provenance: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.reports) and all(r.ok for r in self.reports.values())

    @property
    def errors(self) -> list[str]:
        return [
            f"{tf}/{issue.code}: {issue.message}"
            for tf, report in self.reports.items()
            for issue in report.errors
        ]

    def dataset(self, timeframe: TimeFrame) -> GoldDataset | None:
        return self.datasets.get(timeframe.value)


def build_market_snapshot(
    provider: GoldMarketDataProvider,
    config: GoldConfig,
    *,
    as_of: datetime,
    now: datetime | None = None,
) -> MarketSnapshot:
    """Assemble + validate the multi-timeframe snapshot as of ``as_of``.

    ``now`` (when given) enables live staleness checks; historical builds pass
    ``None`` so a past snapshot is never judged against today's clock.
    """
    as_of = ensure_utc(as_of)
    snapshot = MarketSnapshot(
        symbol=config.data.symbol,
        built_at=utc_now(),
        as_of=as_of,
    )
    max_spread = config.risk.max_spread_price
    for timeframe in config.data.timeframes:
        minutes = timeframe_minutes(timeframe)
        bars_wanted = config.data.history_bars.get(timeframe.value, 300)
        # Calendar slack: weekends + breaks shrink available bars, so ask for
        # a wider window than the raw bar count.
        span = timedelta(minutes=minutes * bars_wanted * 3)
        start = as_of - span
        try:
            dataset = provider.get_bars(config.data.symbol, timeframe, start, as_of)
        except Exception as exc:  # noqa: BLE001 — a dead source is a data error
            report = ValidationReport(symbol=config.data.symbol, timeframe=timeframe.value)
            report.error("provider_failure", f"{provider.source_name}: {exc}")
            snapshot.reports[timeframe.value] = report
            continue
        report = validate_dataset(
            dataset,
            expected_symbol=config.data.symbol,
            now=now,
            max_spread_price=max_spread,
        )
        snapshot.datasets[timeframe.value] = dataset
        snapshot.reports[timeframe.value] = report
        snapshot.provenance.append(dataset.meta.describe())

    try:
        snapshot.quote = provider.get_quote(config.data.symbol)
    except Exception as exc:  # noqa: BLE001 — quote is optional enrichment
        logger_quote_failure(snapshot, exc)
    # A provider that CLAIMS bid/ask support must actually deliver a valid
    # quote: unavailable or invalid bid/ask is a data error (DATA ERROR →
    # NO TRADE downstream), never a silently-ignored gap.
    if provider.capabilities.supports_bid_ask:
        quote_report = ValidationReport(symbol=config.data.symbol, timeframe=None)
        for code, message in quote_issues(snapshot.quote):
            quote_report.error(code, message)
        snapshot.reports["quote"] = quote_report
    return snapshot


def quote_issues(quote: GoldQuote | None) -> list[tuple[str, str]]:
    """Deterministic bid/ask validation: ``(code, message)`` issues.

    Invalid or missing bid/ask must surface as data errors — the spread is
    never fabricated (spec: DATA ERROR → NO TRADE).
    """
    if quote is None:
        return [("quote_unavailable", "provider claims bid/ask support but returned no quote")]
    issues: list[tuple[str, str]] = []
    if quote.bid is None or not math.isfinite(quote.bid) or quote.bid <= 0:
        issues.append(("invalid_bid", f"bid {quote.bid!r} must be a positive finite price"))
    if quote.ask is None or not math.isfinite(quote.ask) or quote.ask <= 0:
        issues.append(("invalid_ask", f"ask {quote.ask!r} must be a positive finite price"))
    if quote.bid is not None and quote.ask is not None and quote.bid > 0 and quote.ask > 0:
        if quote.ask < quote.bid:
            issues.append(("invalid_bid_ask", f"ask {quote.ask} is below bid {quote.bid}"))
        elif quote.ask == quote.bid:
            issues.append(("invalid_spread", "zero spread is not credible market data for gold"))
    return issues


def logger_quote_failure(snapshot: MarketSnapshot, exc: Exception) -> None:
    snapshot.provenance.append(f"quote unavailable: {type(exc).__name__}: {exc}")

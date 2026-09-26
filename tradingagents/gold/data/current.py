"""Read-only current gold snapshot (spec: closed bars + live tick).

``build_current_snapshot`` assembles, at one instant:

* the current bid/ask tick and its spread (validated — invalid bid/ask is a
  DATA ERROR, never a fabricated spread);
* for each of M15/H1/H4 the latest **CLOSED** bar, with any currently-forming
  bar explicitly identified and EXCLUDED from the analysis datasets — a
  forming candle is never silently used as a closed candle;
* per-timeframe deterministic data quality (reusing the existing validator:
  OHLC sanity, duplicates, monotonic timestamps, calendar-aware gaps,
  staleness) plus an ``insufficient_bars`` gate for thin history;
* full provenance (requested symbol vs broker symbol, source, asset kind).

The ``datasets`` it returns feed the existing deterministic technical engine
and the multi-agent context unchanged.  READ-ONLY: nothing here executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.gold.config import GoldConfig, default_gold_config
from tradingagents.gold.data.calendar import ensure_utc, timeframe_minutes
from tradingagents.gold.data.models import GoldBar, GoldDataset, GoldQuote
from tradingagents.gold.data.provider import GoldMarketDataProvider
from tradingagents.gold.data.snapshot import quote_issues, validate_dataset
from tradingagents.gold.data.validation import ValidationReport
from tradingagents.gold.types import TimeFrame, utc_now


@dataclass(frozen=True)
class BarStatus:
    """Closed/forming state of one timeframe at the snapshot instant."""

    timeframe: str
    bar: GoldBar | None            # latest CLOSED bar (analysis input)
    forming_bar: GoldBar | None    # latest bar still forming (excluded)
    closed_count: int

    @property
    def is_closed(self) -> bool:
        return self.bar is not None

    @property
    def has_forming(self) -> bool:
        return self.forming_bar is not None


@dataclass
class GoldCurrentSnapshot:
    """The current read-only gold market state at one instant."""

    symbol: str
    broker_symbol: str | None
    timestamp: datetime
    as_of: datetime
    quote: GoldQuote | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    bars: dict[str, BarStatus] = field(default_factory=dict)
    datasets: dict[str, GoldDataset] = field(default_factory=dict)  # CLOSED bars only
    reports: dict[str, ValidationReport] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.reports) and all(r.ok for r in self.reports.values())

    @property
    def errors(self) -> list[str]:
        return [
            f"{name}/{issue.code}: {issue.message}"
            for name, report in self.reports.items()
            for issue in report.errors
        ]

    def dataset(self, timeframe: TimeFrame) -> GoldDataset | None:
        return self.datasets.get(timeframe.value)


def _split_closed(
    bars: tuple[GoldBar, ...], timeframe: TimeFrame, as_of: datetime
) -> tuple[list[GoldBar], list[GoldBar]]:
    """A bar with open time t covers [t, t + tf): closed iff t + tf <= as_of."""
    minutes = timedelta(minutes=timeframe_minutes(timeframe))
    closed = [b for b in bars if b.timestamp + minutes <= as_of]
    forming = [b for b in bars if b.timestamp + minutes > as_of]
    return closed, forming


def build_current_snapshot(
    provider: GoldMarketDataProvider,
    config: GoldConfig | None = None,
    *,
    now: datetime,
    as_of: datetime | None = None,
    min_bars: int = 50,
) -> GoldCurrentSnapshot:
    """Assemble + validate the current multi-timeframe snapshot.

    ``now`` is the wall clock (staleness checks); ``as_of`` (default ``now``)
    is the data cutoff — no bar after it enters any dataset, and bars still
    forming at it are reported as forming rather than used as closed.
    """
    config = config or default_gold_config()
    now = ensure_utc(now)
    as_of = ensure_utc(as_of) if as_of is not None else now
    snap = GoldCurrentSnapshot(
        symbol=config.data.symbol,
        broker_symbol=getattr(provider, "discovered_symbol", None),
        timestamp=utc_now(),
        as_of=as_of,
    )

    for timeframe in config.data.timeframes:
        minutes = timeframe_minutes(timeframe)
        wanted = config.data.history_bars.get(timeframe.value, 300)
        span = timedelta(minutes=minutes * wanted * 3)
        report = ValidationReport(symbol=config.data.symbol, timeframe=timeframe.value)
        status = BarStatus(timeframe.value, None, None, 0)
        try:
            if provider.capabilities.closed_bars_enforced:
                # The provider excludes forming bars from its normal output;
                # ask for them explicitly so they can be LABELLED as forming
                # here (and still excluded from the analysis datasets below).
                dataset = provider.get_bars(
                    config.data.symbol, timeframe, as_of - span, as_of, include_forming=True,
                )
            else:
                dataset = provider.get_bars(config.data.symbol, timeframe, as_of - span, as_of)
        except Exception as exc:  # noqa: BLE001 — a dead/unavailable feed is a data error
            report.error("provider_failure", f"{provider.source_name}: {exc}")
            snap.reports[timeframe.value] = report
            snap.bars[timeframe.value] = status
            continue

        closed, forming = _split_closed(dataset.bars, timeframe, as_of)
        closed_ds = GoldDataset(meta=dataset.meta, bars=tuple(closed))
        quality = validate_dataset(
            closed_ds,
            expected_symbol=config.data.symbol,
            now=now,
            max_spread_price=config.risk.max_spread_price,
        )
        report.issues.extend(quality.issues)
        if not closed:
            report.error(
                "insufficient_bars",
                f"no closed {timeframe.value} bars at or before {as_of.isoformat()}",
            )
        elif len(closed) < min_bars:
            report.error(
                "insufficient_bars",
                f"{len(closed)} closed {timeframe.value} bars < required minimum {min_bars}",
            )
        status = BarStatus(
            timeframe.value,
            closed[-1] if closed else None,
            forming[-1] if forming else None,
            len(closed),
        )
        snap.datasets[timeframe.value] = closed_ds
        snap.bars[timeframe.value] = status
        snap.reports[timeframe.value] = report
        snap.provenance.append(dataset.meta.describe())

    # Provider connects lazily; resolve the broker symbol once it is known.
    snap.broker_symbol = getattr(provider, "discovered_symbol", None) or snap.broker_symbol

    # --- current tick: bid / ask / spread ----------------------------------
    try:
        snap.quote = provider.get_quote(config.data.symbol)
    except Exception as exc:  # noqa: BLE001 — quote fetch failure is recorded, not raised
        snap.quote = None
        snap.provenance.append(f"quote fetch failed: {type(exc).__name__}: {exc}")
    quote_report = ValidationReport(symbol=config.data.symbol, timeframe=None)
    severity = quote_report.error if provider.capabilities.supports_bid_ask else quote_report.warn
    for code, message in quote_issues(snap.quote):
        severity(code, message)
    if snap.quote is not None:
        age_minutes = (as_of - snap.quote.timestamp).total_seconds() / 60
        budget = config.data.stale_after_minutes.get("M15", 120)
        if age_minutes > budget:
            quote_report.error(
                "stale_quote",
                f"tick is {age_minutes:.0f} min old (budget {budget} min)",
            )
        snap.bid = snap.quote.bid
        snap.ask = snap.quote.ask
        snap.spread = snap.quote.spread
        snap.provenance.append(
            f"quote {snap.quote.source} {snap.quote.symbol} "
            f"bid={snap.bid} ask={snap.ask} spread={snap.spread} "
            f"@ {snap.quote.timestamp.isoformat()}"
        )
    snap.reports["quote"] = quote_report
    return snap

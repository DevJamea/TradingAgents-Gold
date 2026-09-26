"""Deterministic data-quality gate for gold datasets (spec §21).

Before any agent sees data, the dataset is checked for: timestamp sanity,
duplicates, missing bars, impossible OHLC, non-positive prices, invalid
spreads, stale data, timezone consistency and symbol identity.  Errors mean
the downstream decision pipeline must answer NO TRADE / DATA ERROR — never an
invented decision.  Warnings are recorded but do not block.

Missing-bar detection walks the *expected* grid (market calendar aware), so
the weekend close and the daily 21–22 UTC break are not false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from tradingagents.gold.data.calendar import (
    ensure_utc,
    expected_bar_times,
    timeframe_minutes,
)
from tradingagents.gold.data.models import GoldDataset
from tradingagents.gold.types import AssetKind

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    level: str          # "error" | "warning"
    code: str
    message: str


@dataclass
class ValidationReport:
    symbol: str
    timeframe: str | None
    checked_bars: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(ERROR, code, message))

    def warn(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(WARNING, code, message))


def validate_dataset(
    dataset: GoldDataset,
    *,
    expected_symbol: str,
    now: datetime | None = None,
    max_spread_price: float | None = None,
) -> ValidationReport:
    """Full quality check of ``dataset``; ``now`` enables the staleness check.

    ``expected_symbol`` is the symbol the caller asked for; a mismatch is a
    symbol-identity error (never silently swap instruments).
    """
    tf = dataset.meta.timeframe
    report = ValidationReport(
        symbol=dataset.meta.symbol,
        timeframe=tf.value if tf else None,
        checked_bars=len(dataset.bars),
    )

    # --- identity & provenance -------------------------------------------
    if dataset.meta.symbol != expected_symbol:
        report.error(
            "symbol_mismatch",
            f"dataset is for {dataset.meta.symbol!r} but {expected_symbol!r} was requested",
        )
    if (
        dataset.meta.asset_kind == AssetKind.GOLD_FUTURES_PROXY
        and (not dataset.meta.is_proxy or not dataset.meta.disclaimer)
    ):
        report.error(
            "proxy_label_missing",
            "a futures-proxy dataset must set is_proxy=True and carry a disclaimer",
        )
    if dataset.meta.timezone.upper() != "UTC":
        report.error("timezone_inconsistent", f"meta timezone must be UTC, got {dataset.meta.timezone!r}")

    bars = dataset.bars
    if not bars:
        report.warn("empty_dataset", "dataset contains no bars")
        return report

    seen: dict[datetime, int] = {}
    for bar in bars:
        # --- timezone consistency ---------------------------------------
        if bar.timestamp.tzinfo is None:
            report.error("timezone_inconsistent", f"naive timestamp {bar.timestamp}")
        elif bar.timestamp.utcoffset() != timedelta(0):
            report.error("timezone_inconsistent", f"non-UTC timestamp {bar.timestamp}")
        # --- duplicates / order ------------------------------------------
        if bar.timestamp in seen:
            report.error("duplicate_timestamp", f"duplicate bar at {bar.timestamp}")
        seen[bar.timestamp] = 1
        # --- price sanity ------------------------------------------------
        prices = (bar.open, bar.high, bar.low, bar.close)
        if any(p is None or p != p or p <= 0 for p in prices):  # NaN or <= 0
            report.error("non_positive_price", f"invalid price at {bar.timestamp}: {prices}")
            continue
        if bar.high < bar.low:
            report.error("impossible_ohlc", f"high < low at {bar.timestamp}")
        if bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
            report.error("impossible_ohlc", f"OHLC inconsistent at {bar.timestamp}")
        if bar.volume is not None and bar.volume < 0:
            report.error("negative_volume", f"negative volume at {bar.timestamp}")
        # --- spread sanity ------------------------------------------------
        spread = bar.spread
        if spread is not None:
            if spread < 0:
                report.error("invalid_spread", f"negative spread at {bar.timestamp}")
            elif max_spread_price is not None and spread > max_spread_price:
                report.error(
                    "invalid_spread",
                    f"spread {spread:.2f} exceeds max {max_spread_price:.2f} at {bar.timestamp}",
                )

    # --- ordering (separate pass so duplicates don't hide disorder) -------
    stamps = [b.timestamp for b in bars]
    if any(a >= b for a, b in zip(stamps, stamps[1:], strict=False)):
        report.error("unordered_timestamps", "bars are not strictly increasing in time")

    # --- missing bars (market-calendar aware) ----------------------------
    if tf is not None and stamps == sorted(stamps) and len(bars) >= 2:
        interval = timeframe_minutes(tf)
        expected = expected_bar_times(min(stamps), max(stamps), interval)
        present = set(stamps)
        missing = [t for t in expected if t not in present]
        run = 0
        for t in missing:
            run += 1
            if run >= 3:
                report.error(
                    "missing_bars",
                    f"{run or 'several'} consecutive expected {tf.value} bars missing near {t}",
                )
        if 0 < run < 3:
            pass  # individual stragglers are warnings, emitted below
        if missing and run < 3:
            report.warn(
                "missing_bar",
                f"{len(missing)} expected {tf.value} bar(s) missing (weekend/holiday gaps excluded)",
            )
        off_grid = [t for t in stamps if (t.hour * 60 + t.minute) % interval != 0]
        if off_grid:
            report.warn("off_grid_bar", f"{len(off_grid)} bar(s) off the {tf.value} grid")

    # --- staleness (live mode only) --------------------------------------
    if now is not None and tf is not None:
        now = ensure_utc(now)
        age_minutes = (now - max(stamps)).total_seconds() / 60
        budget = {"M15": 120, "H1": 480, "H4": 1500}.get(tf.value, 480)
        if age_minutes > budget:
            report.error(
                "stale_data",
                f"latest bar is {age_minutes:.0f} min old (budget {budget} min for {tf.value})",
            )

    return report

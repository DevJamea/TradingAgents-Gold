"""Deterministic gold macro engine over FRED/ALFRED (spec §9).

Replaces the stock fundamentals vertical for gold: DXY proxy, Treasury and
real yields, Fed policy, inflation (CPI/PCE), labor (NFP/unemployment/claims),
growth and stress gauges — every series pulled through the upstream FRED
vendor with its ALFRED vintage pin (``realtime_start = realtime_end = as-of``),
so a historical decision can never see a later revision (#1275 preserved).

Dollar-index limitation (documented, not hidden): FRED publishes the Fed's
broad trade-weighted index ``DTWEXBGS``, which is NOT the ICE ``DXY``.  The
engine labels it explicitly as a DXY proxy in every report.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.vendors.fred import (
    FredNotConfiguredError,
    get_macro_data,
)

#: Gold-relevant macro series: engine alias -> FRED series ID.
#: Every ID already exists upstream; nothing new is invented here.
GOLD_MACRO_SERIES: dict[str, str] = {
    "dollar_index": "DTWEXBGS",      # DXY proxy (Fed broad trade-weighted USD)
    "fed_funds_rate": "FEDFUNDS",    # policy rate
    "yield_2y": "DGS2",              # rate-sensitive short end
    "yield_10y": "DGS10",            # nominal benchmark yield
    "real_yield_10y": "DFII10",      # 10y TIPS: gold's opportunity cost
    "yield_30y": "DGS30",
    "yield_curve_10y2y": "T10Y2Y",   # growth-expectations signal
    "breakeven_10y": "T10YIE",       # inflation expectations
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",          # the Fed's inflation target gauge
    "nonfarm_payrolls": "PAYEMS",
    "unemployment_rate": "UNRATE",
    "initial_claims": "ICSA",        # highest-frequency labor gauge
    "real_gdp": "GDPC1",
    "vix": "VIXCLS",                 # risk sentiment / safe-haven demand
}

DXY_LIMITATION_NOTE = (
    "dollar_index uses FRED DTWEXBGS (Fed broad trade-weighted USD index), "
    "which is a PROXY for the ICE DXY, not the DXY itself."
)

_LATEST_RE = re.compile(r"\*\*Latest:\*\*\s*([-\d.]+)\s*\((\d{4}-\d{2}-\d{2})\)")

#: Fetcher signature so tests can inject canned FRED reports.
MacroFetcher = Callable[[str, str], str]


@dataclass
class MacroPoint:
    indicator: str
    series_id: str
    as_of_date: str
    value: float | None = None
    observation_date: str | None = None
    vintage_date: str | None = None    # the ALFRED realtime pin actually used
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None

    def line(self) -> str:
        if self.ok:
            return f"{self.indicator} ({self.series_id}): {self.value} @ {self.observation_date}"
        return f"{self.indicator} ({self.series_id}): unavailable ({self.error})"


@dataclass
class GoldMacroSnapshot:
    curr_date: str
    points: dict[str, MacroPoint] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return any(p.ok for p in self.points.values())

    @property
    def vintage_note(self) -> str:
        pins = sorted({p.vintage_date for p in self.points.values() if p.vintage_date})
        pin = pins[0] if pins else self.curr_date
        return (
            "Point-in-time: FRED/ALFRED vintages pinned to realtime_start="
            f"realtime_end={pin}; no observation published after that date is visible."
        )

    def render(self) -> str:
        lines = [f"## GOLD MACRO SNAPSHOT — as of {self.curr_date}"]
        for name in GOLD_MACRO_SERIES:
            point = self.points.get(name)
            if point is not None:
                lines.append(f"- {point.line()}")
        lines.append(f"- NOTE: {DXY_LIMITATION_NOTE}")
        lines.append(f"- {self.vintage_note}")
        return "\n".join(lines)


def parse_latest(report: str) -> tuple[float | None, str | None]:
    """Extract ``(value, observation_date)`` from a FRED vendor report."""
    match = _LATEST_RE.search(report or "")
    if not match:
        return None, None
    try:
        return float(match.group(1)), match.group(2)
    except ValueError:
        return None, None


class MacroEngine:
    """Builds a point-in-time gold macro snapshot from FRED/ALFRED."""

    def __init__(self, fetcher: MacroFetcher | None = None) -> None:
        self._fetch: MacroFetcher = fetcher or self._vendor_fetch

    @staticmethod
    def _vendor_fetch(indicator: str, curr_date: str) -> str:
        try:
            return get_macro_data(indicator, curr_date)
        except FredNotConfiguredError as exc:
            return f"FRED unavailable: {exc}"
        except VendorError as exc:
            return f"FRED vendor error: {exc}"

    def snapshot(self, curr_date: str) -> GoldMacroSnapshot:
        """Snapshot as of ``curr_date`` (yyyy-mm-dd).  One failing series is
        recorded as an unavailable point — it never aborts the whole snapshot."""
        snap = GoldMacroSnapshot(curr_date=curr_date)
        for alias, series_id in GOLD_MACRO_SERIES.items():
            try:
                report = self._fetch(alias, curr_date)
            except Exception as exc:  # noqa: BLE001 — one bad series ≠ dead snapshot
                snap.points[alias] = MacroPoint(
                    indicator=alias, series_id=series_id, as_of_date=curr_date,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            if report.startswith("FRED unavailable:") or report.startswith("FRED vendor error:"):
                snap.points[alias] = MacroPoint(
                    indicator=alias, series_id=series_id, as_of_date=curr_date,
                    error=report,
                )
                continue
            value, obs_date = parse_latest(report)
            snap.points[alias] = MacroPoint(
                indicator=alias,
                series_id=series_id,
                as_of_date=curr_date,
                value=value,
                observation_date=obs_date,
                vintage_date=curr_date,
                error=None if value is not None else "no parseable observation",
            )
        return snap

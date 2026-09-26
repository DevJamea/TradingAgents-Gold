"""Live market observation for XAUUSD (spec §31 Phase 9) — NO execution.

Runs the gold pipeline on a cadence against a live-capable provider and logs
one auditable record per cycle (spec §26): market snapshot + provenance +
data-quality errors, analyst outputs and decision when an LLM graph is
configured, the deterministic gate outcome, and the hypothetical paper
trade.

Safety posture (unchanged, absolute): this module contains no order, broker
or MT5 code path.  A cycle with failing data quality records DATA ERROR and
NO TRADE — it never calls the LLM on broken data and never invents a
decision (spec §21).

Modes:
* Deterministic-only (default, no API keys): snapshot + quality gate + NO
  TRADE records.
* Full analysis (``graph_factory`` supplied, LLM keys configured): the
  complete multi-agent run per cycle.  REQUIRES API KEY.
* Real feeds: pass a live provider (e.g. ``YahooGoldProxyProvider``).
  REQUIRES LIVE DATA; the test-suite never touches the network.

Run with: ``python -m tradingagents.gold.observer`` (see ``observer.py``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from tradingagents.gold.agents.context import GoldContextBuilder
from tradingagents.gold.config import GoldConfig, default_gold_config
from tradingagents.gold.data.provider import GoldMarketDataProvider
from tradingagents.gold.data.snapshot import build_market_snapshot
from tradingagents.gold.decision import GoldDecision, build_gold_decision
from tradingagents.gold.paper.engine import PaperTradingEngine
from tradingagents.gold.risk_gate import AccountState, evaluate_decision
from tradingagents.gold.types import MarketRegime, utc_now

logger = logging.getLogger(__name__)


@dataclass
class ObservationRecord:
    """One auditable observation cycle (spec §26 record contract)."""

    cycle: int
    observed_at: datetime
    as_of: datetime
    snapshot_ok: bool
    data_errors: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    quote_spread: float | None = None
    analysis_available: bool = False
    decision: dict | None = None
    gate_approved: bool = False
    gate_summary: str = ""
    hypothetical_position_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["observed_at"] = self.observed_at.isoformat()
        payload["as_of"] = self.as_of.isoformat()
        return json.dumps(payload, sort_keys=True)


class ObservationLog:
    """Append-only JSONL audit log of observation cycles."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ObservationRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records


class LiveObserver:
    """Observation loop: live data in, auditable hypothetical records out."""

    def __init__(
        self,
        provider: GoldMarketDataProvider,
        *,
        gold_config: GoldConfig | None = None,
        log_path: Path | None = None,
        graph_factory=None,           # () -> GoldTradingAgentsGraph (LLM mode)
        paper_engine: PaperTradingEngine | None = None,
        account_equity: float = 10_000.0,
        clock=None,                   # () -> datetime (injectable for tests)
        sleeper=time.sleep,           # injectable for tests
    ) -> None:
        self.provider = provider
        self.config = gold_config or default_gold_config()
        self.graph_factory = graph_factory
        self.paper_engine = paper_engine or PaperTradingEngine(self.config.execution)
        self.account_equity = account_equity
        self.clock = clock or utc_now
        self.sleeper = sleeper
        if log_path is None:
            from tradingagents.default_config import DEFAULT_CONFIG

            log_path = Path(DEFAULT_CONFIG["results_dir"]) / "observations" / "gold_live.jsonl"
        self.log = ObservationLog(log_path)
        self._graph = None
        self._context_builder = GoldContextBuilder(self.config, provider=provider)

    # ------------------------------------------------------------------

    def run_once(self, cycle: int) -> ObservationRecord:
        """One observation cycle: data → (analysis) → gate → hypothetical."""
        now = self.clock()
        record = ObservationRecord(
            cycle=cycle, observed_at=now, as_of=now, snapshot_ok=False,
        )
        snapshot = build_market_snapshot(self.provider, self.config, as_of=now, now=now)
        record.snapshot_ok = snapshot.ok
        record.data_errors = list(snapshot.errors)
        record.provenance = list(snapshot.provenance)
        record.quote_spread = snapshot.quote.spread if snapshot.quote else None

        if not snapshot.ok:
            # Data-integrity failure: record DATA ERROR + NO TRADE.  No LLM,
            # no decision, no paper action (spec §21).
            record.notes.append("NO TRADE / DATA ERROR — cycle skipped analysis")
            self.log.append(record)
            return record

        trade_date = now.date().isoformat()
        decision = self._analyze(record, trade_date, snapshot)

        if decision is None:
            record.notes.append("NO TRADE — no decision available")
            self.log.append(record)
            return record

        record.decision = decision.to_record()
        context = self._context_builder.build(trade_date)
        gate = evaluate_decision(
            decision, context,
            AccountState(
                equity=self.account_equity,
                open_positions=len(self.paper_engine.open_positions),
            ),
            limits=self.config.risk,
            spread=record.quote_spread,
        )
        record.gate_approved = gate.approved
        record.gate_summary = gate.summary()
        if gate.approved:
            position = self.paper_engine.open_from_gate(decision, gate, fill_time=now)
            record.hypothetical_position_id = position.position_id if position else None
            record.notes.append("hypothetical paper position opened")
        else:
            record.notes.append("NO TRADE — risk gate rejected")
        self.log.append(record)
        return record

    def _analyze(self, record: ObservationRecord, trade_date: str, snapshot) -> GoldDecision | None:
        """Full multi-agent analysis when configured; else None."""
        if self.graph_factory is None:
            record.notes.append("deterministic-only mode (no LLM graph configured)")
            return None
        if self._graph is None:
            self._graph = self.graph_factory()
        try:
            final_state, _signal = self._graph.propagate("XAUUSD", trade_date)
        except Exception as exc:  # noqa: BLE001 — observation must survive provider/LLM outages
            record.notes.append(f"analysis failed: {type(exc).__name__}: {exc}")
            return None
        record.analysis_available = True
        context = self._context_builder.build(trade_date)
        return build_gold_decision(
            final_state,
            symbol=self.config.data.symbol,
            trade_date=trade_date,
            market_regime=(
                context.technical.market_regime
                if context.technical is not None else MarketRegime.UNKNOWN
            ),
            data_sources=list(snapshot.provenance),
            now=self.clock(),
            asset_kind=self.provider.capabilities.asset_kind,
        )

    # ------------------------------------------------------------------

    def run(self, cycles: int, interval_seconds: float) -> list[ObservationRecord]:
        """Run ``cycles`` observations ``interval_seconds`` apart (no sleep
        after the last cycle).  Returns this run's records."""
        records = []
        for cycle in range(cycles):
            records.append(self.run_once(cycle))
            if cycle < cycles - 1 and interval_seconds > 0:
                self.sleeper(interval_seconds)
        return records

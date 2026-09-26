"""Deterministic paper-trading engine (spec §18, §24).

Receives gate-APPROVED decisions only (the deterministic risk gate is the sole
admission path), simulates fills with explicit spread and slippage models,
tracks stops/targets bar by bar, and reports P&L, R-multiples and portfolio
metrics.  Every position carries the ``decision_id`` of the agent decision
that created it (full traceability, spec §18) and the assumption set used.

Modes (spec §24): the engine is mode-agnostic — *historical* replay feeds it
bars from stored datasets; *paper-live* feeds it current bars; neither touches
a broker.  There is deliberately no code path here that could reach MT5.

Fill model (deterministic, all values USD/oz):
* Buy fills at  entry + spread/2 + slippage; sells at entry − spread/2 − slippage
  (crossing the spread and slipping adversely, as in reality).
* Stop exits fill at stop − slippage for longs (stop + slippage for shorts).
* Target exits fill AT the target (a resting limit; no slippage).
* If one bar touches both stop and target, the STOP is assumed first — the
  pessimistic convention, so results are never flattered by ambiguity.

R-multiple: pnl / initial_risk, where initial_risk = units × |entry − stop|
of the original decision (the gate-approved risk budget).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from tradingagents.gold.config import ExecutionAssumptions
from tradingagents.gold.data.models import GoldBar
from tradingagents.gold.decision import GoldDecision
from tradingagents.gold.risk_gate import GateResult
from tradingagents.gold.types import DecisionAction, utc_now

_OPEN = "OPEN"
_CLOSED = "CLOSED"

EXIT_STOP = "STOP_LOSS"
EXIT_TARGET = "TAKE_PROFIT"
EXIT_END_OF_DATA = "END_OF_DATA"


@dataclass
class PaperPosition:
    """One simulated position, traceable to its agent decision."""

    position_id: str
    decision_id: str
    symbol: str
    side: str                       # "BUY" | "SELL"
    units: float                    # troy ounces
    requested_entry: float
    entry_fill: float               # modeled fill (spread + slippage applied)
    stop_loss: float
    take_profit: float
    opened_at: datetime
    initial_risk: float             # units × |entry − stop| (gate-approved budget)
    assumptions_spread: float       # assumption set in force at open
    assumptions_slippage: float
    status: str = _OPEN
    closed_at: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl: float = 0.0
    r_multiple: float | None = None
    total_cost: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.status == _OPEN

    @property
    def is_closed(self) -> bool:
        return self.status == _CLOSED


class PaperTradingEngine:
    """Deterministic simulated execution for gate-approved gold decisions."""

    def __init__(self, assumptions: ExecutionAssumptions | None = None) -> None:
        self.assumptions = assumptions or ExecutionAssumptions()
        self.positions: list[PaperPosition] = []

    # -- opening -----------------------------------------------------------

    def open_from_gate(
        self,
        decision: GoldDecision,
        gate_result: GateResult,
        fill_time: datetime | None = None,
    ) -> PaperPosition | None:
        """Open a position from an APPROVED gate result; None otherwise.

        Refuses anything the gate did not approve — this engine has no other
        admission path (spec §17).
        """
        if not gate_result.approved or gate_result.is_no_trade:
            return None
        spread = self.assumptions.spread_price
        slip = self.assumptions.slippage_price
        if decision.action is DecisionAction.BUY:
            entry_fill = decision.entry + spread / 2 + slip
        else:
            entry_fill = decision.entry - spread / 2 - slip
        position = PaperPosition(
            position_id=uuid.uuid5(
                uuid.NAMESPACE_URL, f"{decision.decision_id}:{gate_result.final_action}",
            ).hex,
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            side=decision.action.value,
            units=gate_result.position_units,
            requested_entry=decision.entry,
            entry_fill=entry_fill,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            opened_at=fill_time or utc_now(),
            initial_risk=gate_result.position_units * abs(decision.entry - decision.stop_loss),
            assumptions_spread=spread,
            assumptions_slippage=slip,
            total_cost=gate_result.position_units * (spread / 2 + slip),
        )
        self.positions.append(position)
        return position

    # -- bar-by-bar simulation ----------------------------------------------

    def process_bar(self, bar: GoldBar) -> list[PaperPosition]:
        """Advance every open position with one bar; returns positions closed."""
        closed: list[PaperPosition] = []
        for position in self.positions:
            if not position.is_open:
                continue
            exit_price, reason = self._check_exit(position, bar)
            if reason is not None:
                self._close(position, exit_price, reason, bar.timestamp)
                closed.append(position)
        return closed

    @staticmethod
    def _check_exit(position: PaperPosition, bar: GoldBar):
        # Touch detection at raw levels; stop slippage is applied in _close.
        if position.side == "BUY":
            # Pessimistic convention: check the stop first when both hit.
            if bar.low <= position.stop_loss:
                return position.stop_loss, EXIT_STOP
            if bar.high >= position.take_profit:
                return position.take_profit, EXIT_TARGET
        else:
            if bar.high >= position.stop_loss:
                return position.stop_loss, EXIT_STOP
            if bar.low <= position.take_profit:
                return position.take_profit, EXIT_TARGET
        return None, None

    def _close(self, position: PaperPosition, exit_price: float, reason: str,
               closed_at: datetime) -> None:
        slip = self.assumptions.slippage_price
        if reason == EXIT_STOP:   # stops slip adversely; targets are resting limits
            exit_price = exit_price - slip if position.side == "BUY" else exit_price + slip
            position.total_cost += position.units * slip
        direction = 1 if position.side == "BUY" else -1
        position.status = _CLOSED
        position.closed_at = closed_at
        position.exit_price = exit_price
        position.exit_reason = reason
        position.pnl = (exit_price - position.entry_fill) * position.units * direction
        position.r_multiple = (
            position.pnl / position.initial_risk if position.initial_risk > 0 else None
        )

    def close_all_at(self, bar: GoldBar) -> list[PaperPosition]:
        """Mark every open position to the bar's close (end-of-data / audit)."""
        closed: list[PaperPosition] = []
        for position in self.positions:
            if position.is_open:
                self._close(position, bar.close, EXIT_END_OF_DATA, bar.timestamp)
                closed.append(position)
        return closed

    # -- accessors -----------------------------------------------------------

    @property
    def open_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions if p.is_open]

    @property
    def closed_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions if p.is_closed]

    def position_for_decision(self, decision_id: str) -> list[PaperPosition]:
        return [p for p in self.positions if p.decision_id == decision_id]

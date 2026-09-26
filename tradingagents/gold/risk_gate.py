"""Deterministic risk gate (spec §16) — code, never the LLM.

Pipeline position (absolute rule, spec §17):

    LLM → Analysis → Structured Decision → **Deterministic Risk Gate** → Paper
    Execution → (later) MT5 Demo Adapter

The gate re-validates the decision against the configured numeric limits and
the run's data-quality state.  It can only output APPROVED (with a
deterministically sized position) or NO TRADE — it never modifies the
decision's direction or levels.  Anything unsafe, missing, stale or
contradictory becomes NO TRADE / DATA ERROR.

Data-quality gates reuse the Phase 2 ``ValidationReport`` results carried on
the run context: stale-data and missing-data errors arrive pre-computed and
are treated as hard rejections here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tradingagents.gold.agents.context import GoldRunContext
from tradingagents.gold.config import RiskLimits
from tradingagents.gold.decision import GoldDecision
from tradingagents.gold.types import DecisionAction

NO_TRADE = "NO_TRADE"


@dataclass
class AccountState:
    """The (paper) account state the gate enforces portfolio limits against."""

    equity: float
    open_positions: int = 0
    daily_loss_fraction: float = 0.0    # realized loss today as a fraction of equity

    def __post_init__(self) -> None:
        if self.equity <= 0:
            raise ValueError("equity must be > 0")
        if self.open_positions < 0:
            raise ValueError("open_positions must be >= 0")


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    """Outcome of the deterministic gate.  ``approved`` decisions carry a
    deterministic position size; everything else is NO TRADE."""

    approved: bool
    final_action: str                 # DecisionAction value or NO_TRADE
    checks: list[GateCheck] = field(default_factory=list)
    position_units: float = 0.0       # troy ounces when approved
    risk_fraction: float = 0.0        # fraction of equity actually risked
    risk_amount: float = 0.0          # quote currency when approved
    reasons: list[str] = field(default_factory=list)

    @property
    def is_no_trade(self) -> bool:
        return self.final_action == NO_TRADE

    def summary(self) -> str:
        lines = [f"GATE: {'APPROVED' if self.approved else NO_TRADE}"]
        lines += [f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}" for c in self.checks]
        if self.reasons:
            lines.append("  reasons: " + "; ".join(self.reasons))
        return "\n".join(lines)


class RiskGate:
    """Deterministic gate: decisions in, APPROVED/NO_TRADE out."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def evaluate(
        self,
        decision: GoldDecision,
        context: GoldRunContext,
        account: AccountState,
        *,
        spread: float | None = None,
    ) -> GateResult:
        """Gate one decision.  Pure function of its inputs.

        ``spread`` overrides the observed spread (USD/oz) when the caller has
        a quote; otherwise the latest bar spread in the context is used.
        """
        limits = self.limits
        checks: list[GateCheck] = []
        reasons: list[str] = []

        def check(name: str, passed: bool, detail: str, *, fatal: bool = False) -> bool:
            checks.append(GateCheck(name, passed, detail))
            if not passed:
                reasons.append(f"{name}: {detail}")
                if fatal:
                    raise _GateShortCircuit(checks, reasons)
            return passed

        # --- 1. data quality (hard rejections) ---------------------------
        data_ok = not context.has_data_error
        check(
            "data_quality",
            data_ok,
            "no data errors" if data_ok else "; ".join(context.data_errors[:3]),
            fatal=not data_ok,
        )

        # --- 2. action admissibility -------------------------------------
        if decision.action in (DecisionAction.HOLD, DecisionAction.REVIEW):
            return GateResult(
                approved=False,
                final_action=NO_TRADE,
                checks=checks,
                reasons=[f"action {decision.action.value} requires no trade"],
            )
        if not decision.is_tradeable:
            return GateResult(
                approved=False, final_action=NO_TRADE, checks=checks,
                reasons=[f"action {decision.action.value} is not tradeable"],
            )

        # --- 3. completeness of price structure --------------------------
        levels_ok = None not in (decision.entry, decision.stop_loss, decision.take_profit)
        check(
            "complete_levels", levels_ok,
            "entry/stop_loss/take_profit all present" if levels_ok
            else "missing levels: "
            + ", ".join(
                name for name, v in (
                    ("entry", decision.entry), ("stop_loss", decision.stop_loss),
                    ("take_profit", decision.take_profit),
                ) if v is None
            ),
            fatal=not levels_ok,
        )

        entry, stop, take = decision.entry, decision.stop_loss, decision.take_profit

        # --- 4. level geometry --------------------------------------------
        if decision.action is DecisionAction.BUY:
            geometry_ok = stop < entry < take
        else:
            geometry_ok = take < entry < stop
        check(
            "level_geometry", geometry_ok,
            f"{decision.action.value} stop<entry<take respected" if geometry_ok
            else f"levels violate {decision.action.value} ordering "
                 f"(stop={stop}, entry={entry}, take={take})",
            fatal=not geometry_ok,
        )

        stop_distance = abs(entry - stop)
        check(
            "stop_distance",
            limits.min_stop_distance_price <= stop_distance <= limits.max_stop_distance_price,
            f"stop distance {stop_distance:.2f} within "
            f"[{limits.min_stop_distance_price}, {limits.max_stop_distance_price}]",
            fatal=stop_distance < limits.min_stop_distance_price,
        )

        # --- 5. reward:risk ------------------------------------------------
        reward = abs(take - entry)
        rr = reward / stop_distance if stop_distance else 0.0
        check(
            "min_reward_risk", rr >= limits.min_reward_risk,
            f"R:R {rr:.2f} >= {limits.min_reward_risk}",
            fatal=rr < limits.min_reward_risk,
        )

        # --- 6. spread -----------------------------------------------------
        observed = spread if spread is not None else self._latest_spread(context)
        if observed is not None:
            check(
                "max_spread", observed <= limits.max_spread_price,
                f"spread {observed:.2f} <= {limits.max_spread_price}",
                fatal=observed > limits.max_spread_price,
            )

        # --- 7. portfolio limits -------------------------------------------
        check(
            "max_open_positions", account.open_positions < limits.max_open_positions,
            f"{account.open_positions} open < {limits.max_open_positions}",
            fatal=account.open_positions >= limits.max_open_positions,
        )
        check(
            "max_daily_loss", account.daily_loss_fraction < limits.max_daily_loss_fraction,
            f"daily loss {account.daily_loss_fraction:.2%} < {limits.max_daily_loss_fraction:.2%}",
            fatal=account.daily_loss_fraction >= limits.max_daily_loss_fraction,
        )

        # --- 8. deterministic position sizing ------------------------------
        risk_amount = account.equity * limits.max_risk_fraction_per_trade
        units = risk_amount / stop_distance                     # ounces
        capped = min(units, limits.max_position_units)
        if capped < units:
            reasons.append(
                f"position size capped at {limits.max_position_units} oz by max_position_units"
            )
        actual_risk = capped * stop_distance
        actual_fraction = actual_risk / account.equity
        check(
            "max_risk_fraction", actual_fraction <= limits.max_risk_fraction_per_trade + 1e-9,
            f"risk {actual_fraction:.4%} <= {limits.max_risk_fraction_per_trade:.4%} "
            f"({capped:.2f} oz × {stop_distance:.2f} stop)",
        )
        check(
            "max_position_size", capped <= limits.max_position_units,
            f"{capped:.2f} oz <= {limits.max_position_units} oz",
        )

        return GateResult(
            approved=True,
            final_action=decision.action.value,
            checks=checks,
            position_units=capped,
            risk_fraction=actual_fraction,
            risk_amount=actual_risk,
            reasons=reasons,
        )

    @staticmethod
    def _latest_spread(context: GoldRunContext) -> float | None:
        quote = context.snapshot.quote if context.snapshot is not None else None
        return quote.spread if quote is not None else None


class _GateShortCircuit(Exception):
    """Internal: carries the checks/reasons gathered before a fatal failure."""

    def __init__(self, checks, reasons):
        super().__init__(reasons[0] if reasons else "gate rejected")
        self.checks = checks
        self.reasons = reasons


def evaluate_decision(
    decision: GoldDecision,
    context: GoldRunContext,
    account: AccountState,
    limits: RiskLimits | None = None,
    *,
    spread: float | None = None,
) -> GateResult:
    """Evaluate, converting fatal short-circuits into a NO_TRADE result."""
    gate = RiskGate(limits)
    try:
        return gate.evaluate(decision, context, account, spread=spread)
    except _GateShortCircuit as exc:
        return GateResult(
            approved=False, final_action=NO_TRADE,
            checks=exc.checks, reasons=exc.reasons,
        )

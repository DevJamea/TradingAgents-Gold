"""The structured gold decision model (spec §15) and its deterministic builder.

Field provenance, explicitly (the spec requires existing/adapted/new to be
distinguishable):

* ``action`` — EXISTING vocabulary, adapted: upstream TraderProposal has
  Buy/Hold/Sell; gold adds REVIEW (never tradeable).
* ``entry`` / ``stop_loss`` — EXISTING (trader proposal entry_price/stop_loss).
* ``timestamp``, ``symbol``, ``asset_class``, ``data_sources`` — EXISTING
  upstream concepts (run date, ticker, provenance), now first-class fields.
* ``take_profit``, ``risk_fraction``, ``holding_horizon``, ``market_regime``,
  ``key_reasons``, ``bull_case``, ``bear_case``, ``invalidation_conditions``,
  ``analyst_consensus``, ``risk_flags``, ``confidence``, ``decision_id`` — NEW.

Confidence definition (binding): the trader's self-reported conviction that
its action correctly reads the supplied evidence, on 0–1.  It is NOT a
probability of profit, is never computed by code, and is never an input to
the risk gate (deterministic limits own risk).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tradingagents.gold.agents.schemas import GoldTraderAction
from tradingagents.gold.types import AssetKind, DecisionAction, MarketRegime, utc_now

_ACTION_RE = re.compile(r"FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|SELL|HOLD|REVIEW)\*\*", re.IGNORECASE)
_FIELD_RE = {
    "confidence": re.compile(r"\*\*Confidence\*\*:\s*([0-9.]+)", re.IGNORECASE),
    "entry": re.compile(r"\*\*Entry Price\*\*:\s*([0-9.]+)", re.IGNORECASE),
    "stop_loss": re.compile(r"\*\*Stop Loss\*\*:\s*([0-9.]+)", re.IGNORECASE),
    "take_profit": re.compile(r"\*\*Take Profit\*\*:\s*([0-9.]+)", re.IGNORECASE),
    "invalidation": re.compile(r"\*\*Invalidation\*\*:\s*(.+)", re.IGNORECASE),
}

#: Default holding horizon for the M15-decision / H1-H4-context system.
DEFAULT_HOLDING_HORIZON = "short_swing"   # ~1–5 trading days


def _parse_action(text: str) -> DecisionAction:
    match = _ACTION_RE.search(text or "")
    if not match:
        return DecisionAction.REVIEW   # unparseable ⇒ never tradeable
    return DecisionAction(match.group(1).upper())


def _parse_field(text: str, name: str) -> float | None:
    match = _FIELD_RE[name].search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def make_decision_id(symbol: str, timestamp: datetime, action: DecisionAction,
                     entry: float | None, salt: str = "") -> str:
    """Stable content hash identifying one decision (auditable, reproducible)."""
    payload = "|".join([
        symbol, timestamp.strftime("%Y-%m-%dT%H:%M:%S%z"), action.value,
        "" if entry is None else f"{entry:.2f}", salt,
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class GoldDecision:
    """The final structured decision of one gold run (spec §15)."""

    symbol: str
    asset_class: str                       # "gold" — this system is gold-only
    timestamp: datetime
    action: DecisionAction
    confidence: float | None = None        # LLM-stated conviction (see module doc)
    entry: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_fraction: float | None = None     # filled deterministically by the risk gate
    holding_horizon: str = DEFAULT_HOLDING_HORIZON
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    key_reasons: str = ""
    bull_case: str = ""
    bear_case: str = ""
    invalidation_conditions: str = ""
    analyst_consensus: dict = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    decision_id: str = ""
    proxy_labelled: bool = False           # True when data sources carry the proxy disclaimer

    def __post_init__(self) -> None:
        if isinstance(self.action, str):
            self.action = DecisionAction(self.action.upper())
        if isinstance(self.market_regime, str):
            self.market_regime = MarketRegime(self.market_regime)
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        if not self.decision_id:
            self.decision_id = make_decision_id(
                self.symbol, self.timestamp, self.action, self.entry,
            )

    @property
    def is_tradeable(self) -> bool:
        return self.action in (DecisionAction.BUY, DecisionAction.SELL)

    def to_record(self) -> dict:
        """Serialisable audit record (spec §26)."""
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "confidence": self.confidence,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_fraction": self.risk_fraction,
            "holding_horizon": self.holding_horizon,
            "market_regime": self.market_regime.value,
            "key_reasons": self.key_reasons,
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "invalidation_conditions": self.invalidation_conditions,
            "analyst_consensus": self.analyst_consensus,
            "risk_flags": list(self.risk_flags),
            "data_sources": list(self.data_sources),
            "proxy_labelled": self.proxy_labelled,
        }


def build_gold_decision(
    final_state: dict,
    *,
    symbol: str = "XAUUSD",
    trade_date: str,
    market_regime: MarketRegime = MarketRegime.UNKNOWN,
    data_sources: list[str] | None = None,
    now: datetime | None = None,
    asset_kind: AssetKind = AssetKind.GOLD_SPOT,
) -> GoldDecision:
    """Deterministically assemble the GoldDecision from a finished run state.

    Everything is parsed from the agents' rendered outputs with the fixed
    parse contracts (no LLM in this path).  A missing/foreign value becomes
    None/REVIEW — never an invention.
    """
    trader_plan = final_state.get("trader_investment_plan", "") or ""
    action = _parse_action(trader_plan)
    timestamp = (
        now or utc_now()
    )
    bull_history = (final_state.get("investment_debate_state") or {}).get("bull_history", "")
    bear_history = (final_state.get("investment_debate_state") or {}).get("bear_history", "")

    decision = GoldDecision(
        symbol=symbol,
        asset_class="gold",
        timestamp=timestamp,
        action=action,
        confidence=_parse_field(trader_plan, "confidence"),
        entry=_parse_field(trader_plan, "entry"),
        stop_loss=_parse_field(trader_plan, "stop_loss"),
        take_profit=_parse_field(trader_plan, "take_profit"),
        market_regime=market_regime,
        key_reasons=(trader_plan.split("FINAL TRANSACTION PROPOSAL")[0]).strip(),
        bull_case=bull_history.strip()[-2000:],
        bear_case=bear_history.strip()[-2000:],
        invalidation_conditions=_parse_invalidation(trader_plan),
        analyst_consensus={
            "research_manager": _first_line(final_state.get("investment_plan", "")),
            "trader_action": action.value,
            "market_regime": market_regime.value,
        },
        data_sources=list(data_sources or []),
        proxy_labelled=asset_kind == AssetKind.GOLD_FUTURES_PROXY
        or any("[PROXY]" in s for s in (data_sources or [])),
    )
    decision.decision_id = make_decision_id(
        symbol, timestamp, action, decision.entry, salt=trade_date,
    )
    return decision


def _parse_invalidation(text: str) -> str:
    match = _FIELD_RE["invalidation"].search(text or "")
    return match.group(1).strip() if match else ""


def _first_line(text: str, limit: int = 200) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""


__all__ = [
    "DEFAULT_HOLDING_HORIZON",
    "GoldDecision",
    "GoldTraderAction",
    "build_gold_decision",
    "make_decision_id",
]

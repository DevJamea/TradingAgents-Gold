"""Structured schemas for gold-specific agent outputs (Phase 5 slice).

The gold Trader keeps the upstream contract shape (action + reasoning +
concrete levels rendered with the same ``FINAL TRANSACTION PROPOSAL`` parse
line) but extends the action vocabulary with REVIEW (insufficient evidence)
and the fields the gold decision schema needs (take-profit, invalidation,
explicitly-defined confidence).

Confidence definition (deliberate, per the gold spec): the trader's stated
conviction that THIS action is the correct reading of the supplied evidence.
It is the LLM's self-reported confidence in its own reasoning — NOT a
probability of profit, NOT computed from data, and never used by code as a
risk input (the deterministic risk gate applies its own numeric limits).
"""

from enum import Enum

from pydantic import BaseModel, Field


class GoldTraderAction(str, Enum):
    """Gold trader actions.  REVIEW is never tradeable."""

    BUY = "Buy"
    SELL = "Sell"
    HOLD = "Hold"
    REVIEW = "Review"


class GoldTraderProposal(BaseModel):
    """Structured gold transaction proposal produced by the gold Trader."""

    action: GoldTraderAction = Field(
        description=(
            "The transaction direction. Exactly one of Buy / Sell / Hold / Review. "
            "Use Review only when the supplied evidence is genuinely insufficient "
            "to commit to any direction (missing or contradictory data)."
        ),
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the deterministic technical "
            "snapshot and the other analysts' reports. Two to four sentences."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "Your stated conviction that this action is the correct reading of "
            "the supplied evidence, from 0 (guessing) to 1 (fully supported). "
            "This is self-reported confidence in the analysis, not a probability "
            "of profit and not a risk input."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price as an absolute number in USD per troy ounce "
            "(e.g. 2648.5), never a percentage or a range. Omit unless the "
            "action is Buy or Sell and the technical snapshot supports a level."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute USD price per troy ounce. Must be "
            "below entry for Buy and above entry for Sell if stated."
        ),
    )
    take_profit: float | None = Field(
        default=None,
        description=(
            "Optional take-profit as an absolute USD price per troy ounce. Must "
            "be above entry for Buy and below entry for Sell if stated."
        ),
    )
    invalidation: str = Field(
        default="",
        description=(
            "The specific condition that would invalidate this trade idea "
            "(price level, regime change, or macro event)."
        ),
    )


def render_gold_trader_proposal(proposal: GoldTraderProposal) -> str:
    """Render the proposal with the same parse contract as upstream."""
    lines = [
        f"**Action**: {proposal.action.value}",
        f"**Reasoning**: {proposal.reasoning}",
        f"**Confidence**: {proposal.confidence:.2f}",
    ]
    if proposal.entry_price is not None:
        lines.append(f"**Entry Price**: {proposal.entry_price}")
    if proposal.stop_loss is not None:
        lines.append(f"**Stop Loss**: {proposal.stop_loss}")
    if proposal.take_profit is not None:
        lines.append(f"**Take Profit**: {proposal.take_profit}")
    if proposal.invalidation:
        lines.append(f"**Invalidation**: {proposal.invalidation}")
    lines.append(f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**")
    return "\n".join(lines)

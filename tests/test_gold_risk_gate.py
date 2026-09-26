"""Phase 6 — deterministic risk gate (PASS: unsafe/invalid => NO TRADE)."""

from dataclasses import replace

import pytest

from tests.test_gold_context_builder import TRADE_DATE, offline_builder
from tradingagents.gold.config import RiskLimits
from tradingagents.gold.data.models import GoldQuote
from tradingagents.gold.decision import build_gold_decision
from tradingagents.gold.risk_gate import NO_TRADE, AccountState, evaluate_decision
from tradingagents.gold.types import AssetKind, DecisionAction

BUY_PLAN = """**Action**: Buy
**Reasoning**: trend continuation.
**Confidence**: 0.7
**Entry Price**: 2650.0
**Stop Loss**: 2640.0
**Take Profit**: 2665.0
FINAL TRANSACTION PROPOSAL: **BUY**"""


def make_decision(plan: str = BUY_PLAN, **kwargs):
    return build_gold_decision(
        {"trader_investment_plan": plan}, trade_date=TRADE_DATE, **kwargs,
    )


def gate_context(errors: list[str] | None = None, spread: float | None = None):
    ctx = offline_builder().build(TRADE_DATE)
    if errors is not None:
        # Simulate a failed data-quality state without rebuilding fixtures.
        ctx.data_errors.extend(errors)
        object.__setattr__(ctx, "market_ok_override", True)
        ctx.__dict__["snapshot"] = _snapshot_with_errors(ctx.snapshot, errors) if ctx.snapshot else ctx.snapshot
    if spread is not None:
        quote = GoldQuote(
            symbol="XAUUSD", source="fixture", asset_kind=AssetKind.GOLD_SPOT,
            timestamp=ctx.as_of, bid=3000.0 - spread / 2, ask=3000.0 + spread / 2,
        )
        ctx.snapshot.quote = quote
    return ctx


def _snapshot_with_errors(snapshot, errors):
    snapshot.reports["M15"].issues.extend(
        __import__("tradingagents.gold.data.validation", fromlist=["ValidationIssue"]).ValidationIssue(
            "error", "injected", e,
        )
        for e in errors
    )
    return snapshot


ACCOUNT = AccountState(equity=10_000.0)


class TestApprovals:
    def test_valid_buy_is_approved_with_deterministic_sizing(self):
        decision = make_decision()
        result = evaluate_decision(decision, gate_context(), ACCOUNT)
        assert result.approved and result.final_action == DecisionAction.BUY.value
        # risk 1% of 10k = $100 over a $10 stop ⇒ 10 oz (at the cap by default)
        assert result.position_units == pytest.approx(10.0)
        assert result.risk_amount == pytest.approx(100.0)
        assert result.risk_fraction == pytest.approx(0.01)

    def test_sizing_scales_with_stop_distance(self):
        # With the size cap lifted, $100 risk over a $5 stop = 20 oz.
        plan = BUY_PLAN.replace("**Stop Loss**: 2640.0", "**Stop Loss**: 2645.0")
        limits = RiskLimits(max_position_units=1000.0)
        result = evaluate_decision(make_decision(plan), gate_context(), ACCOUNT, limits=limits)
        assert result.position_units == pytest.approx(20.0)

    def test_size_cap_binds_before_risk_grows(self):
        # $100 risk over a $1 stop would be 100 oz; the 10 oz cap binds and the
        # actual risked fraction drops below the limit.
        plan = BUY_PLAN.replace("**Stop Loss**: 2640.0", "**Stop Loss**: 2649.0")
        result = evaluate_decision(make_decision(plan), gate_context(), ACCOUNT)
        assert result.position_units == pytest.approx(10.0)
        assert result.risk_fraction < 0.01

    def test_sell_geometry_validated(self):
        plan = (BUY_PLAN
                .replace("**Action**: Buy", "**Action**: Sell")
                .replace("**Entry Price**: 2650.0", "**Entry Price**: 2650.0")
                .replace("**Stop Loss**: 2640.0", "**Stop Loss**: 2660.0")
                .replace("**Take Profit**: 2665.0", "**Take Profit**: 2635.0")
                .replace("**BUY**", "**SELL**"))
        result = evaluate_decision(make_decision(plan), gate_context(), ACCOUNT)
        assert result.approved and result.final_action == DecisionAction.SELL.value


class TestRejections:
    def test_poor_reward_risk_is_rejected(self):
        # R:R = 5/10 = 0.5 < 1.5 ⇒ deterministic rejection
        plan = BUY_PLAN.replace("**Take Profit**: 2665.0", "**Take Profit**: 2655.0")
        result = evaluate_decision(make_decision(plan), gate_context(), ACCOUNT)
        assert not result.approved and result.is_no_trade
        assert any("min_reward_risk" in r for r in result.reasons)

    def test_data_errors_are_hard_rejections(self):
        decision = make_decision()
        result = evaluate_decision(decision, gate_context(errors=["stale_data: too old"]), ACCOUNT)
        assert not result.approved
        assert any("data_quality" in r for r in result.reasons)

    def test_wide_spread_rejects(self):
        result = evaluate_decision(make_decision(), gate_context(spread=1.5), ACCOUNT)
        assert not result.approved
        assert any("max_spread" in r for r in result.reasons)

    def test_too_many_open_positions_rejects(self):
        result = evaluate_decision(
            make_decision(), gate_context(), replace(ACCOUNT, open_positions=2),
        )
        assert not result.approved
        assert any("max_open_positions" in r for r in result.reasons)

    def test_daily_loss_limit_rejects(self):
        result = evaluate_decision(
            make_decision(), gate_context(), replace(ACCOUNT, daily_loss_fraction=0.05),
        )
        assert not result.approved
        assert any("max_daily_loss" in r for r in result.reasons)

    def test_inverted_levels_reject(self):
        plan = BUY_PLAN.replace("**Stop Loss**: 2640.0", "**Stop Loss**: 2655.0")
        result = evaluate_decision(make_decision(plan), gate_context(), ACCOUNT)
        assert not result.approved
        assert any("level_geometry" in r for r in result.reasons)

    def test_missing_levels_reject(self):
        plan = "\n".join(
            line for line in BUY_PLAN.splitlines() if "Take Profit" not in line
        )
        result = evaluate_decision(make_decision(plan), gate_context(), ACCOUNT)
        assert not result.approved
        assert any("complete_levels" in r for r in result.reasons)

    def test_hold_and_review_are_no_trade_without_fault(self):
        for action in ("**HOLD**", "**REVIEW**"):
            plan = BUY_PLAN.replace("**BUY**", action)
            result = evaluate_decision(make_decision(plan), gate_context(), ACCOUNT)
            assert not result.approved
            assert result.final_action == NO_TRADE
            # nothing was wrong: every recorded check passed
            assert all(c.passed for c in result.checks)
            assert any("requires no trade" in r for r in result.reasons)


class TestDeterminism:
    def test_same_inputs_same_outcome(self):
        decision = make_decision()
        ctx = gate_context()
        one = evaluate_decision(decision, ctx, ACCOUNT)
        two = evaluate_decision(decision, ctx, ACCOUNT)
        assert one == two

    def test_gate_never_modifies_the_decision(self):
        decision = make_decision()
        evaluate_decision(decision, gate_context(), ACCOUNT)
        assert decision.action is DecisionAction.BUY
        assert decision.entry == pytest.approx(2650.0)
        assert decision.risk_fraction is None   # gate result owns sizing, not the decision

"""Phase 6 — the structured gold decision model and its deterministic builder."""

import pytest

from tradingagents.gold.decision import (
    DEFAULT_HOLDING_HORIZON,
    GoldDecision,
    build_gold_decision,
    make_decision_id,
)
from tradingagents.gold.types import DecisionAction, MarketRegime

TRADER_PLAN = """**Action**: Buy
**Reasoning**: Regime TREND_UP with M15 pullback into support; macro real yields falling.
**Confidence**: 0.62
**Entry Price**: 2650.5
**Stop Loss**: 2645.5
**Take Profit**: 2668.0
**Invalidation**: H1 close below 2644 breaks the higher-low structure.
FINAL TRANSACTION PROPOSAL: **BUY**"""


def run_state(trader_plan: str = TRADER_PLAN) -> dict:
    return {
        "trader_investment_plan": trader_plan,
        "investment_plan": "**Recommendation**: Buy — the bull case wins on real yields.",
        "investment_debate_state": {
            "bull_history": "Bull Researcher: gold breaks out",
            "bear_history": "Bear Researcher: real yields still rising",
        },
    }


class TestParsing:
    def test_full_plan_parses_every_field(self):
        decision = build_gold_decision(
            run_state(), symbol="XAUUSD", trade_date="2026-09-18",
            market_regime=MarketRegime.TREND_UP,
            data_sources=["ohlcv M15 XAUUSD (queried GC=F) from yfinance:yahoo [PROXY]"],
        )
        assert decision.action is DecisionAction.BUY
        assert decision.confidence == pytest.approx(0.62)
        assert decision.entry == pytest.approx(2650.5)
        assert decision.stop_loss == pytest.approx(2645.5)
        assert decision.take_profit == pytest.approx(2668.0)
        assert "2644" in decision.invalidation_conditions
        assert decision.market_regime is MarketRegime.TREND_UP
        assert decision.holding_horizon == DEFAULT_HOLDING_HORIZON
        assert "Bull Researcher" in decision.bull_case
        assert "Bear Researcher" in decision.bear_case
        assert decision.analyst_consensus["trader_action"] == "BUY"
        assert decision.analyst_consensus["market_regime"] == "TREND_UP"
        assert decision.proxy_labelled is True
        assert decision.asset_class == "gold"

    def test_unparseable_plan_is_review_never_tradeable(self):
        decision = build_gold_decision(
            {"trader_investment_plan": "the model rambled without a proposal line"},
            trade_date="2026-09-18",
        )
        assert decision.action is DecisionAction.REVIEW
        assert decision.is_tradeable is False
        assert decision.entry is None and decision.confidence is None

    def test_review_action_never_tradeable(self):
        decision = build_gold_decision(
            {"trader_investment_plan": "insufficient evidence\nFINAL TRANSACTION PROPOSAL: **REVIEW**"},
            trade_date="2026-09-18",
        )
        assert decision.action is DecisionAction.REVIEW
        assert decision.is_tradeable is False


class TestDecisionId:
    def test_id_is_stable_for_identical_inputs(self):
        from datetime import datetime, timezone

        ts = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
        one = make_decision_id("XAUUSD", ts, DecisionAction.BUY, 2650.5, salt="2026-09-18")
        two = make_decision_id("XAUUSD", ts, DecisionAction.BUY, 2650.5, salt="2026-09-18")
        assert one == two
        assert len(one) == 16

    def test_id_changes_with_action(self):
        from datetime import datetime, timezone

        ts = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
        assert make_decision_id("XAUUSD", ts, DecisionAction.BUY, 2650.5) != \
            make_decision_id("XAUUSD", ts, DecisionAction.SELL, 2650.5)

    def test_to_record_round_trips_the_audit_fields(self):
        decision = build_gold_decision(run_state(), trade_date="2026-09-18")
        record = decision.to_record()
        for key in ("decision_id", "symbol", "asset_class", "timestamp", "action",
                    "confidence", "entry", "stop_loss", "take_profit", "risk_fraction",
                    "holding_horizon", "market_regime", "key_reasons", "bull_case",
                    "bear_case", "invalidation_conditions", "analyst_consensus",
                    "risk_flags", "data_sources"):
            assert key in record


class TestDefaults:
    def test_naive_timestamp_gets_utc(self):
        from datetime import datetime

        d = GoldDecision(
            symbol="XAUUSD", asset_class="gold", timestamp=datetime(2026, 9, 18, 21, 0),
            action=DecisionAction.HOLD,
        )
        assert d.timestamp.tzinfo is not None

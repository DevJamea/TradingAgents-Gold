"""Phase 8 — research & validation: splits, robustness, stress, reproducibility."""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.gold.paper.engine import PaperTradingEngine
from tradingagents.gold.research.pipeline import run_research
from tradingagents.gold.research.robustness import (
    bootstrap_mean_r_ci,
    monte_carlo_max_drawdown_r,
    permutation_test_mean_r,
)
from tradingagents.gold.research.splits import chronological_splits, segment

T0 = datetime(2026, 9, 21, 0, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def synthetic_trades(n: int = 24) -> list:
    """A deterministic mixed sample: alternating +1.5R / −1R closes."""
    from tradingagents.gold.config import ExecutionAssumptions, RiskLimits
    from tradingagents.gold.data.models import GoldBar
    from tradingagents.gold.decision import build_gold_decision
    from tradingagents.gold.risk_gate import AccountState, evaluate_decision

    plan = """**Action**: Buy
**Reasoning**: test.
**Confidence**: 0.7
**Entry Price**: 2650.0
**Stop Loss**: 2640.0
**Take Profit**: 2665.0
FINAL TRANSACTION PROPOSAL: **BUY**"""
    engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.0))
    ctx = __import__(
        "tests.test_gold_context_builder", fromlist=["offline_builder"],
    ).offline_builder().build("2026-09-18")
    trades = []
    for i in range(n):
        decision = build_gold_decision(
            {"trader_investment_plan": plan}, trade_date="2026-09-18",
            now=T0 + timedelta(hours=i),
        )
        gate = evaluate_decision(
            decision, ctx, AccountState(equity=10_000.0),
            limits=RiskLimits(max_position_units=1000.0),
        )
        pos = engine.open_from_gate(decision, gate, fill_time=T0 + timedelta(hours=i))
        if i % 2 == 0:   # winners touch the target
            engine.process_bar(GoldBar(
                timestamp=T0 + timedelta(hours=i, minutes=30),
                open=2651, high=2666.0, low=2649.0, close=2660.0, volume=1.0,
            ))
        else:            # losers hit the stop
            engine.process_bar(GoldBar(
                timestamp=T0 + timedelta(hours=i, minutes=30),
                open=2649, high=2650.0, low=2639.0, close=2641.0, volume=1.0,
            ))
        trades.append(pos)
    return trades


class TestSplits:
    def test_chronological_segments_are_ordered_and_complete(self):
        trades = synthetic_trades()
        splits = chronological_splits(trades)
        dev = segment(trades, splits.development)
        val = segment(trades, splits.validation)
        oos = segment(trades, splits.out_of_sample)
        assert dev and val and oos
        assert len(dev) + len(val) + len(oos) == len(trades)
        assert dev[-1].closed_at <= val[0].closed_at <= oos[-1].closed_at

    def test_no_trade_is_split_in_two(self):
        trades = synthetic_trades(9)
        splits = chronological_splits(trades, fractions=(0.5, 0.25, 0.25))
        assigned = [s for s in splits.segments for t in segment(trades, s)]
        assert len(assigned) == len(trades)


class TestRobustness:
    def test_bootstrap_ci_reproducible_and_sensible(self):
        trades = synthetic_trades(24)
        one = bootstrap_mean_r_ci(trades, seed=7, iterations=500)
        two = bootstrap_mean_r_ci(trades, seed=7, iterations=500)
        assert one == two                       # same seed -> identical CI
        assert one.lower < one.estimate < one.upper

    def test_bootstrap_changes_with_seed(self):
        trades = synthetic_trades(24)
        assert bootstrap_mean_r_ci(trades, seed=7).lower != \
            bootstrap_mean_r_ci(trades, seed=8).lower

    def test_monte_carlo_drawdown_reported(self):
        trades = synthetic_trades(24)
        mc = monte_carlo_max_drawdown_r(trades, seed=11, iterations=200)
        assert mc["max_drawdown_r_worst"] >= mc["max_drawdown_r_p95"] >= 0
        assert monte_carlo_max_drawdown_r(trades, seed=11, iterations=200) == mc

    def test_permutation_null_test_is_deterministic(self):
        trades = synthetic_trades(24)
        one = permutation_test_mean_r(trades, seed=13, iterations=500)
        two = permutation_test_mean_r(trades, seed=13, iterations=500)
        assert one == two
        assert 0.0 <= one["p_value"] <= 1.0


class TestPipeline:
    def test_full_report_reproducible_and_sectioned(self):
        trades = synthetic_trades(24)
        one = run_research(trades, now=NOW)
        two = run_research(trades, now=NOW)
        assert one.render() == two.render()     # Phase 8 PASS: reproducible
        assert set(one.segment_reports) == {"development", "validation", "out_of_sample"}
        # OOS is a distinct, clearly-labelled section.
        assert "out-of-sample was NOT used during development" in one.render()
        assert one.bootstrap is not None and one.permutation is not None
        assert any("not a claim of profitability" in lim for lim in one.limitations)

    def test_no_trades_raises_a_clear_error(self):
        with pytest.raises(ValueError):
            chronological_splits([])

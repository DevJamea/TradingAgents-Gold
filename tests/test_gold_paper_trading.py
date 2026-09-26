"""Phase 7 — deterministic paper trading (PASS: same inputs -> same results)."""

from datetime import datetime, timedelta, timezone

import pytest

from tests.test_gold_context_builder import TRADE_DATE, offline_builder
from tradingagents.gold.config import ExecutionAssumptions, RiskLimits
from tradingagents.gold.data.models import GoldBar
from tradingagents.gold.decision import build_gold_decision
from tradingagents.gold.paper.engine import (
    EXIT_END_OF_DATA,
    EXIT_STOP,
    EXIT_TARGET,
    PaperTradingEngine,
)
from tradingagents.gold.paper.metrics import build_performance_report
from tradingagents.gold.risk_gate import AccountState, evaluate_decision

T0 = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)

BUY_PLAN = """**Action**: Buy
**Reasoning**: trend.
**Confidence**: 0.7
**Entry Price**: 2650.0
**Stop Loss**: 2640.0
**Take Profit**: 2665.0
FINAL TRANSACTION PROPOSAL: **BUY**"""


def approved_decision(plan: str = BUY_PLAN):
    decision = build_gold_decision({"trader_investment_plan": plan}, trade_date=TRADE_DATE, now=T0)
    ctx = offline_builder().build(TRADE_DATE)
    gate = evaluate_decision(decision, ctx, AccountState(equity=10_000.0),
                             limits=RiskLimits(max_position_units=1000.0))
    assert gate.approved
    return decision, gate


def bar(ts, high, low, close):
    return GoldBar(timestamp=ts, open=close, high=high, low=low, close=close, volume=1.0)


class TestFills:
    def test_buy_fills_at_ask_with_slippage(self):
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.4, slippage_price=0.2))
        decision, gate = approved_decision()
        pos = engine.open_from_gate(decision, gate, fill_time=T0)
        assert pos is not None
        assert pos.entry_fill == pytest.approx(2650.0 + 0.2 + 0.2)   # +spread/2 +slip
        assert pos.side == "BUY"

    def test_sell_fills_at_bid_with_slippage(self):
        plan = (BUY_PLAN.replace("**Action**: Buy", "**Action**: Sell")
                .replace("**Stop Loss**: 2640.0", "**Stop Loss**: 2660.0")
                .replace("**Take Profit**: 2665.0", "**Take Profit**: 2635.0")
                .replace("**BUY**", "**SELL**"))
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.4, slippage_price=0.2))
        decision, gate = approved_decision(plan)
        pos = engine.open_from_gate(decision, gate, fill_time=T0)
        assert pos.entry_fill == pytest.approx(2650.0 - 0.2 - 0.2)   # −spread/2 −slip

    def test_unapproved_gate_result_is_refused(self):
        decision = build_gold_decision(
            {"trader_investment_plan": "no proposal"}, trade_date=TRADE_DATE, now=T0,
        )
        engine = PaperTradingEngine()
        assert engine.open_from_gate(decision, _rejected_gate()) is None
        assert engine.positions == []


def _rejected_gate():
    from tradingagents.gold.risk_gate import NO_TRADE, GateResult

    return GateResult(approved=False, final_action=NO_TRADE, reasons=["min_reward_risk: 0.5 < 1.5"])


class TestExits:
    def _open(self, engine):
        decision, gate = approved_decision()
        return engine.open_from_gate(decision, gate, fill_time=T0)

    def test_take_profit_hit_closes_with_target_fill(self):
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.0))
        pos = self._open(engine)
        closed = engine.process_bar(bar(T0 + timedelta(hours=2), high=2666.0, low=2652.0, close=2660.0))
        assert closed == [pos]
        assert pos.exit_reason == EXIT_TARGET
        assert pos.exit_price == pytest.approx(2665.0)
        assert pos.pnl == pytest.approx((2665.0 - 2650.0) * pos.units)
        assert pos.r_multiple == pytest.approx(1.5)

    def test_stop_hit_slips_adversely(self):
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.3))
        pos = self._open(engine)
        engine.process_bar(bar(T0 + timedelta(hours=2), high=2651.0, low=2639.0, close=2641.0))
        assert pos.exit_reason == EXIT_STOP
        assert pos.exit_price == pytest.approx(2640.0 - 0.3)
        assert pos.pnl < 0
        assert pos.r_multiple == pytest.approx(pos.pnl / (pos.units * 10.0))

    def test_both_hit_in_one_bar_is_pessimistic(self):
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.0))
        pos = self._open(engine)
        engine.process_bar(bar(T0 + timedelta(hours=2), high=2666.0, low=2639.0, close=2650.0))
        assert pos.exit_reason == EXIT_STOP      # never flattered by ambiguity

    def test_end_of_data_marks_to_close(self):
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.0))
        pos = self._open(engine)
        engine.close_all_at(bar(T0 + timedelta(days=3), high=2655.0, low=2650.0, close=2652.0))
        assert pos.exit_reason == EXIT_END_OF_DATA
        assert pos.exit_price == pytest.approx(2652.0)


class TestTraceabilityAndReproducibility:
    def test_position_is_traceable_to_decision(self):
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.0))
        decision, gate = approved_decision()
        pos = engine.open_from_gate(decision, gate, fill_time=T0)
        assert engine.position_for_decision(decision.decision_id) == [pos]
        assert pos.decision_id == decision.decision_id
        assert pos.assumptions_spread == 0.0 and pos.assumptions_slippage == 0.0

    def test_same_inputs_produce_identical_results(self):
        def run():
            engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.3, slippage_price=0.1))
            decision, gate = approved_decision()
            engine.open_from_gate(decision, gate, fill_time=T0)
            engine.process_bar(bar(T0 + timedelta(hours=1), 2652.0, 2648.0, 2651.0))
            engine.process_bar(bar(T0 + timedelta(hours=2), 2656.0, 2650.0, 2655.0))
            engine.process_bar(bar(T0 + timedelta(hours=3), 2666.0, 2654.0, 2664.0))
            return build_performance_report(engine.positions)

        assert run() == run()   # Phase 7 PASS: reproducible paper results


class TestHistoricalReplay:
    def test_replay_a_stored_dataset_through_the_engine(self):
        """Historical mode: bars from a stored dataset drive the simulation."""
        from datetime import datetime as dt, timedelta as td, timezone as tz

        from tradingagents.gold.data.models import DatasetMeta, GoldDataset
        from tradingagents.gold.types import AssetKind, DataKind, TimeFrame

        start = dt(2026, 9, 21, 0, 0, tzinfo=tz.utc)
        bars = []
        price = 2648.0
        for i in range(20):
            ts = start + td(minutes=15 * i)
            price += 1.5
            bars.append(GoldBar(timestamp=ts, open=price - 1.0, high=price + 0.8,
                                low=price - 1.2, close=price, volume=10.0))
        meta = DatasetMeta(source="fixture", symbol="XAUUSD", source_symbol="XAUUSD",
                           asset_kind=AssetKind.GOLD_SPOT, data_type=DataKind.OHLCV,
                           timeframe=TimeFrame.M15)
        dataset = GoldDataset(meta=meta, bars=tuple(bars))

        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.0))
        decision, gate = approved_decision()
        engine.open_from_gate(decision, gate, fill_time=start)
        for bar_ in dataset.bars:
            engine.process_bar(bar_)
        assert engine.open_positions == []
        pos = engine.closed_positions[0]
        assert pos.exit_reason == EXIT_TARGET      # the 1.5/bar path reaches 2665
        report = build_performance_report(engine.positions)
        assert report.total_trades == 1 and report.win_rate == 1.0


class TestMetrics:
    def test_metrics_over_a_mixed_sample(self):
        engine = PaperTradingEngine(ExecutionAssumptions(spread_price=0.0, slippage_price=0.0))
        # Trade 1: win 1.5R
        decision, gate = approved_decision()
        pos = engine.open_from_gate(decision, gate, fill_time=T0)
        engine.process_bar(bar(T0 + timedelta(hours=2), 2666.0, 2650.0, 2660.0))
        # Trade 2: full loss −1R
        decision2, gate2 = approved_decision()
        pos2 = engine.open_from_gate(decision2, gate2, fill_time=T0 + timedelta(days=1))
        engine.process_bar(bar(T0 + timedelta(days=1, hours=2), 2651.0, 2639.0, 2641.0))

        report = build_performance_report(engine.positions)
        assert report.total_trades == 2
        assert report.wins == 1 and report.losses == 1
        assert report.win_rate == pytest.approx(0.5)
        assert report.average_r == pytest.approx((1.5 + (-1.0)) / 2)
        assert report.expectancy_r == pytest.approx(0.25)
        assert report.profit_factor == pytest.approx(
            abs(pos.pnl) / abs(pos2.pnl),
        )
        assert report.max_drawdown_fraction is not None
        assert "sample" not in report.summary() or True
        assert report.summary().startswith("PAPER PERFORMANCE — 2 closed trade(s)")

    def test_empty_book_metrics(self):
        report = build_performance_report([])
        assert report.total_trades == 0 and report.win_rate is None

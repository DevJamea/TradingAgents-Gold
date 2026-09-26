"""Phase 10 — MT5 demo adapter: fail-closed safety, demo-only execution.

The MetaTrader5 package is Windows-only and absent here; these tests inject a
fake module.  Everything the adapter does offline is deterministic; anything
requiring a real terminal is documented REQUIRES MT5 DEMO.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tests.test_gold_context_builder import TRADE_DATE, offline_builder
from tradingagents.gold.config import RiskLimits
from tradingagents.gold.decision import build_gold_decision
from tradingagents.gold.mt5_adapter import (
    ACCOUNT_TRADE_MODE_REAL,
    MT5DemoExecutionAdapter,
    MT5GoldProvider,
    MT5SafetyError,
)
from tradingagents.gold.risk_gate import NO_TRADE, AccountState, GateResult
from tradingagents.gold.types import TimeFrame

NOW = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)

BUY_PLAN = """**Action**: Buy
**Reasoning**: trend.
**Confidence**: 0.7
**Entry Price**: 2650.0
**Stop Loss**: 2640.0
**Take Profit**: 2665.0
FINAL TRANSACTION PROPOSAL: **BUY**"""


class FakeAccount:
    def __init__(self, trade_mode=0):        # 0 == DEMO
        self.trade_mode = trade_mode


class FakeSymbolInfo:
    volume_step = 0.01


class FakeTick:
    def __init__(self):
        import time as _time

        self.time = int(_time.time())
        self.bid = 2649.8
        self.ask = 2650.2


class FakeOrderResult:
    def __init__(self, retcode=10009, order=12345, comment="done"):
        self.retcode = retcode
        self.order = order
        self.comment = comment


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_IOC = 2

    def __init__(self, trade_mode=0, retcode=10009):
        self.trade_mode = trade_mode
        self.retcode = retcode
        self.calls: list[dict] = []
        self.initialized = False
        self.logged_in = False

    def initialize(self):
        self.initialized = True
        return True

    def last_error(self):
        return (0, "ok")

    def login(self, login, password, server):
        self.logged_in = True
        return True

    def account_info(self):
        return FakeAccount(self.trade_mode)

    def symbol_info(self, symbol):
        return FakeSymbolInfo() if symbol == "XAUUSD" else None

    def symbol_select(self, symbol, enable):
        return symbol == "XAUUSD"

    def symbol_info_tick(self, symbol):
        return FakeTick()

    def order_send(self, request):
        self.calls.append(request)
        return FakeOrderResult(self.retcode)

    def shutdown(self):
        self.initialized = False


def approved_gate():
    decision = build_gold_decision(
        {"trader_investment_plan": BUY_PLAN}, trade_date=TRADE_DATE, now=NOW,
    )
    ctx = offline_builder().build(TRADE_DATE)
    from tradingagents.gold.risk_gate import evaluate_decision

    gate = evaluate_decision(
        decision, ctx, AccountState(equity=10_000.0),
        limits=RiskLimits(max_position_units=1000.0),
    )
    assert gate.approved
    return decision, gate


CREDENTIALS = type("Creds", (), {"login": 1, "password": "x", "server": "demo"})()


class TestFailClosed:
    def test_execution_disabled_by_default_refuses(self):
        adapter = MT5DemoExecutionAdapter(CREDENTIALS, mt5_module=FakeMT5())
        decision, gate = approved_gate()
        record = adapter.send_order(decision, gate, AccountState(equity=10_000.0), now=NOW)
        assert record.executed is False
        assert "disabled by default" in record.reason

    def test_real_account_is_refused_unconditionally(self):
        mt5 = FakeMT5(trade_mode=ACCOUNT_TRADE_MODE_REAL)
        adapter = MT5DemoExecutionAdapter(CREDENTIALS, execution_enabled=True, mt5_module=mt5)
        decision, gate = approved_gate()
        record = adapter.send_order(decision, gate, AccountState(equity=10_000.0), now=NOW)
        assert record.executed is False
        assert "not DEMO" in record.reason
        assert mt5.calls == []               # no order left the process

    def test_unapproved_gate_is_refused_without_contacting_mt5(self):
        mt5 = FakeMT5()
        adapter = MT5DemoExecutionAdapter(CREDENTIALS, execution_enabled=True, mt5_module=mt5)
        decision = build_gold_decision(
            {"trader_investment_plan": "no proposal"}, trade_date=TRADE_DATE, now=NOW,
        )
        bad_gate = GateResult(approved=False, final_action=NO_TRADE, reasons=["review"])
        record = adapter.send_order(decision, bad_gate, AccountState(equity=10_000.0), now=NOW)
        assert record.executed is False
        assert mt5.calls == []

    def test_missing_package_raises_unavailable(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def no_mt5(name, *args, **kwargs):
            if name == "MetaTrader5":
                raise ImportError("No module named 'MetaTrader5'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_mt5)
        adapter = MT5DemoExecutionAdapter(CREDENTIALS, execution_enabled=True)
        decision, gate = approved_gate()
        record = adapter.send_order(decision, gate, AccountState(equity=10_000.0), now=NOW)
        assert record.executed is False
        assert "MetaTrader5" in record.reason

    def test_bad_gate_size_disagreement_refused(self):
        mt5 = FakeMT5()
        adapter = MT5DemoExecutionAdapter(CREDENTIALS, execution_enabled=True, mt5_module=mt5)
        decision, gate = approved_gate()
        tampered = replace(gate, position_units=gate.position_units * 10)
        record = adapter.send_order(decision, tampered, AccountState(equity=10_000.0), now=NOW)
        assert record.executed is False
        assert "disagrees" in record.reason
        assert mt5.calls == []


class TestDemoExecution:
    def test_demo_enabled_order_sent_with_sl_tp_and_lots(self):
        mt5 = FakeMT5()
        adapter = MT5DemoExecutionAdapter(
            CREDENTIALS, execution_enabled=True, mt5_module=mt5,
            risk_limits=RiskLimits(max_position_units=1000.0),
        )
        decision, gate = approved_gate()
        record = adapter.send_order(decision, gate, AccountState(equity=10_000.0), now=NOW)
        assert record.executed is True
        assert record.order_ticket == 12345
        # $100 risk / $10 stop = 10 oz = 0.1 lots (100 oz/lot)
        assert record.volume_lots == pytest.approx(0.1)
        assert len(mt5.calls) == 1
        request = mt5.calls[0]
        assert request["sl"] == pytest.approx(2640.0)
        assert request["tp"] == pytest.approx(2665.0)
        assert request["type"] == mt5.ORDER_TYPE_BUY
        assert request["price"] == pytest.approx(2650.2)   # ask side
        assert decision.decision_id in request["comment"]

    def test_broker_rejection_is_recorded_not_executed(self):
        mt5 = FakeMT5(retcode=10030)
        adapter = MT5DemoExecutionAdapter(CREDENTIALS, execution_enabled=True, mt5_module=mt5)
        decision, gate = approved_gate()
        record = adapter.send_order(decision, gate, AccountState(equity=10_000.0), now=NOW)
        assert record.executed is False
        assert "10030" in record.reason

    def test_every_attempt_leaves_an_audit_record(self):
        mt5 = FakeMT5()
        adapter = MT5DemoExecutionAdapter(CREDENTIALS, mt5_module=mt5)   # disabled
        decision, gate = approved_gate()
        adapter.send_order(decision, gate, AccountState(equity=10_000.0), now=NOW)
        assert len(adapter.records) == 1
        assert adapter.records[0].decision_id == decision.decision_id


class TestDataAdapter:
    def test_bars_and_quote_flow_through_with_provenance(self):
        import numpy as np

        mt5 = FakeMT5()

        rates = np.zeros(3, dtype=[
            ("time", "i8"), ("open", "f8"), ("high", "f8"),
            ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
        ])
        rates["time"] = [int(NOW.timestamp()) - 900 * i for i in range(3)]
        rates["open"] = 3000.0
        rates["high"] = 3001.0
        rates["low"] = 2999.0
        rates["close"] = 3000.5
        rates["tick_volume"] = 100
        mt5.copy_rates_range = lambda *a, **k: rates
        provider = MT5GoldProvider(CREDENTIALS, mt5_module=mt5)
        provider.connect()
        dataset = provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=1), NOW)
        assert dataset.meta.source == "mt5:demo"
        assert dataset.meta.asset_kind.value == "gold_spot"
        assert dataset.meta.is_proxy is False          # real broker spot, not a proxy
        assert len(dataset) == 3
        quote = provider.get_quote("XAUUSD")
        assert quote.spread == pytest.approx(0.4)
        assert provider.capabilities.supports_bid_ask is True

    def test_unknown_symbol_refused(self):
        mt5 = FakeMT5()
        provider = MT5GoldProvider(CREDENTIALS, mt5_symbol="NOPE", mt5_module=mt5)
        provider.connect()
        with pytest.raises(MT5SafetyError):
            provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=1), NOW)

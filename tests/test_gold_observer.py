"""Phase 9 — live observation loop (offline; NO execution anywhere)."""

import json
from datetime import datetime, timezone

import pytest

from tests.test_gold_context_builder import offline_provider
from tradingagents.gold.data.memory import InMemoryGoldProvider
from tradingagents.gold.observation import LiveObserver

#: Inside the offline fixture window (provider bars run to 2026-09-18 20:45 UTC).
PinnedClock = lambda: datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)  # noqa: E731


class FakeSleeper:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


@pytest.fixture
def observer(tmp_path):
    sleeper = FakeSleeper()
    obs = LiveObserver(
        offline_provider(),
        log_path=tmp_path / "obs" / "gold_live.jsonl",
        clock=PinnedClock,
        sleeper=sleeper,
    )
    obs._sleeper = sleeper
    return obs


class TestDeterministicMode:
    def test_cycle_records_snapshot_and_no_trade(self, observer, tmp_path):
        record = observer.run_once(0)
        assert record.snapshot_ok
        assert not record.analysis_available
        assert record.decision is None
        assert record.gate_approved is False
        assert any("NO TRADE" in note for note in record.notes)
        assert any("deterministic-only mode" in note for note in record.notes)

    def test_records_are_appended_to_the_jsonl_log(self, observer):
        observer.run(3, interval_seconds=5)
        entries = observer.log.read_all()
        assert len(entries) == 3
        assert [e["cycle"] for e in entries] == [0, 1, 2]

    def test_interval_sleeps_between_but_not_after_last(self, observer):
        observer.run(3, interval_seconds=5)
        assert observer._sleeper.calls == [5, 5]


class TestDataIntegrityFirst:
    def test_bad_data_cycles_skip_analysis_and_trade(self, tmp_path):
        sleeper = FakeSleeper()
        observer = LiveObserver(
            InMemoryGoldProvider(),          # no datasets at all -> provider_failure
            log_path=tmp_path / "bad" / "obs.jsonl",
            clock=PinnedClock,
            sleeper=sleeper,
        )
        record = observer.run_once(0)
        assert not record.snapshot_ok
        assert record.data_errors
        assert any("DATA ERROR" in note for note in record.notes)
        assert record.decision is None and record.gate_approved is False


class TestNoExecution:
    def test_observer_has_no_broker_or_mt5_code_path(self):
        # Guard the absolute rule at the source level (spec §17): the
        # observation module must never reference execution functions.
        import inspect

        import tradingagents.gold.observation as module

        source = inspect.getsource(module)
        for forbidden in ("order_send", "order_check", "MetaTrader5", "PositionsTotal", "trade_api"):
            assert forbidden not in source, forbidden


class TestLogFormat:
    def test_log_lines_are_valid_json_with_required_fields(self, observer):
        observer.run_once(0)
        line = observer.log.path.read_text(encoding="utf-8").strip().splitlines()[0]
        payload = json.loads(line)
        for key in ("cycle", "observed_at", "as_of", "snapshot_ok", "data_errors",
                    "provenance", "analysis_available", "decision", "gate_approved",
                    "gate_summary", "hypothetical_position_id", "notes"):
            assert key in payload

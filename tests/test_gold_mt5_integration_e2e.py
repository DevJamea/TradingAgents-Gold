"""INTEGRATION: MT5 read-only XAUUSD data through the complete gold pipeline.

Pipeline under test (the gold orchestration is REAL; only external services
are faked — LLM, MT5 terminal, news/macro/sentiment feeds):

    MT5 Read-Only Provider
        ↓
XAUUSD Market Snapshot
        ↓
M15 / H1 / H4 Gold Market Data
        ↓
Gold Technical Engine
        ↓
Gold Analysts            (scripted LLM — no API key required)
        ↓
GoldDecision
        ↓
Deterministic RiskGate
        ↓
Paper Trading

FAKE MT5 TEST vs REAL MT5 ENVIRONMENT TEST: every test in the first class
uses the in-repo FakeMT5 double (deterministic symbol list, M15/H1/H4 bars,
bid/ask tick).  They prove CODE correctness ONLY — never real connectivity.
The final class is the REAL MT5 ENVIRONMENT TEST; it skips with
``REQUIRES MT5 TERMINAL`` when the Windows-only MetaTrader5 package is
absent, and no fake result is ever labelled real.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tests.test_gold_mt5_readonly import RATE_DTYPE, FakeMT5, make_rates  # noqa: F401
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.gold import CostMode  # noqa: F401 — config parity with phase-5 E2E
from tradingagents.gold.config import default_gold_config
from tradingagents.gold.data.mt5 import MT5ReadOnlyGoldProvider
from tradingagents.gold.data.validation import validate_dataset
from tradingagents.gold.decision import build_gold_decision
from tradingagents.gold.graph import GoldTradingAgentsGraph
from tradingagents.gold.macro.engine import MacroEngine
from tradingagents.gold.news.engine import GoldNewsEngine
from tradingagents.gold.paper.engine import PaperTradingEngine
from tradingagents.gold.risk_gate import AccountState, evaluate_decision
from tradingagents.gold.sentiment.engine import GoldSentimentEngine
from tradingagents.gold.types import AssetKind, TimeFrame
from tradingagents.graph import trading_graph

TRADE_DATE = "2026-09-18"                      # a Friday; historical → as_of 21:00 UTC close
AS_OF = datetime(2026, 9, 18, 21, 0, tzinfo=timezone.utc)
E2E_END = AS_OF                                # bars are generated up to this cutoff
READ_ONLY_APIS = {
    "initialize", "login", "shutdown", "symbols_get",
    "symbol_info", "symbol_select", "symbol_info_tick", "copy_rates_range",
}

BUY_TEXT = (
    "Gold H1 breakout continuation with H4 trend support.\n\n"
    "**Rating**: Buy\n\n"
    "**Action**: BUY\n"
    "**Confidence**: 0.72\n"
    "**Entry Price**: 2650.00\n"
    "**Stop Loss**: 2646.00\n"
    "**Take Profit**: 2662.00\n"
    "**Invalidation**: H1 close back below 2646.00\n\n"
    "FINAL TRANSACTION PROPOSAL: **BUY**"
)


def e2e_rates(symbol: str = "XAUUSD") -> dict:
    """M15/H1/H4 up to the 2026-09-18 21:00 UTC close.

    Deliberate boundary: the last H1 bar is stamped 20:00 (closes exactly at
    21:00 ✓) and the H4 bar stamped 20:00 (covers 20:00–24:00) is FORMING at
    the as_of — the provider must exclude it; the last closed H4 is 16:00.
    """
    return {
        (symbol, 15): make_rates(AS_OF, 480, 15),
        (symbol, 16385): make_rates(AS_OF, 240, 60),
        (symbol, 16388): make_rates(AS_OF, 120, 240),
    }


def e2e_tick(bid=2649.90, ask=2650.20):
    """Fresh tick at 20:57 UTC — inside the session, before the close."""
    return SimpleNamespace(
        time=int(datetime(2026, 9, 18, 20, 57, tzinfo=timezone.utc).timestamp()),
        bid=bid, ask=ask,
    )


def make_fake_mt5(*, symbols=("EURUSD", "XAUUSD", "XAGUSD", "GBPUSD"), tick=None,
                  symbol="XAUUSD", rates=None):
    """Fake terminal exposing gold among unrelated FX (discovery exercised)."""
    return FakeMT5(
        symbols=symbols,
        rates=rates if rates is not None else e2e_rates(symbol),
        tick=tick if tick is not None else e2e_tick(),
    )


class ScriptedLLM(BaseChatModel):
    """Records every prompt; answers the deterministic BUY proposal."""

    prompts: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-mt5-e2e"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=BUY_TEXT))])

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("free-text only")

    def bind_tools(self, tools, **kwargs):
        return self


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


def build_mt5_context_builder(gold_config, fake_mt5):
    """The real context builder wired to the read-only MT5 provider.

    Only EXTERNAL services are mocked: macro/news/sentiment feeds return
    empty blocks; the market-data path (provider → snapshot → technical
    engine) is entirely real.
    """
    from tradingagents.gold.agents.context import GoldContextBuilder

    provider = MT5ReadOnlyGoldProvider(mt5_module=fake_mt5)
    return GoldContextBuilder(
        config=gold_config,
        provider=provider,
        macro_engine=MacroEngine(fetcher=lambda i, d: "no data"),
        news_engine=GoldNewsEngine(company_fetch=lambda *a: "", global_fetch=lambda *a: ""),
        sentiment_engine=GoldSentimentEngine(
            twits_fetch=lambda *a, **k: "", reddit_fetch=lambda *a, **k: "",
        ),
    )


def build_gold_graph(tmp_path, monkeypatch, fake_mt5, llm: ScriptedLLM):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        results_dir=str(tmp_path / "results"),
        data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "log.md"),
    )
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: _Client(llm))
    gold_config = default_gold_config()
    builder = build_mt5_context_builder(gold_config, fake_mt5)
    graph = GoldTradingAgentsGraph(config=cfg, gold_config=gold_config, context_builder=builder)
    return graph, builder, gold_config


# ---------------------------------------------------------------------------
# The pipeline (FAKE MT5 TEST)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mt5_pipeline_snapshot_to_paper_trade(tmp_path, monkeypatch):
    """MT5 provider → snapshot → technicals → analysts → decision → gate → paper."""
    llm = ScriptedLLM()
    fake = make_fake_mt5()
    graph, builder, gold_config = build_gold_graph(tmp_path, monkeypatch, fake, llm)

    # -- MT5 Read-Only Provider → discovery ---------------------------------
    state, signal = graph.propagate("XAUUSD", TRADE_DATE)
    provider = builder.provider
    assert provider.discovered_symbol == "XAUUSD"        # broker symbol discovered

    # -- XAUUSD Market Snapshot ---------------------------------------------
    ctx = builder.build(TRADE_DATE)
    snap = ctx.snapshot
    assert snap is not None and snap.ok, snap.errors
    assert snap.as_of == AS_OF                           # historical PIT cutoff
    assert any("mt5:readonly" in line for line in snap.provenance)
    assert any("gold_spot_mt5" in line for line in snap.provenance)

    # -- M15/H1/H4 data: closed bars only, nothing after as_of --------------
    spans = {TimeFrame.M15: 15, TimeFrame.H1: 60, TimeFrame.H4: 240}
    for tf_value, minutes in spans.items():
        ds = snap.dataset(TimeFrame(tf_value))
        assert ds is not None and len(ds) > 50, tf_value
        latest = ds.bars[-1]
        assert latest.timestamp + timedelta(minutes=minutes) <= snap.as_of, tf_value
        assert all(
            bar.bid is None and bar.ask is None and bar.spread is None
            for bar in ds.bars
        ), "historical bid/ask must not be fabricated"
    # The H4 bar stamped 20:00 (still forming at 21:00) is NOT in the dataset:
    h4_stamps = [b.timestamp for b in snap.dataset(TimeFrame.H4).bars]
    assert datetime(2026, 9, 18, 20, 0, tzinfo=timezone.utc) not in h4_stamps
    assert h4_stamps[-1] == datetime(2026, 9, 18, 16, 0, tzinfo=timezone.utc)

    # datasets re-validate through the deterministic quality gate untouched
    for tf_value in spans:
        ds = snap.dataset(TimeFrame(tf_value))
        report = validate_dataset(ds, expected_symbol="XAUUSD")
        assert report.ok, (tf_value, [i.code for i in report.errors])

    # -- Gold Technical Engine ----------------------------------------------
    assert ctx.technical is not None
    assert ctx.technical_block.startswith("## DETERMINISTIC TECHNICAL SNAPSHOT")

    # -- Gold Analysts: gold nodes ran, MT5 data reached their prompts ------
    assert llm.prompts, "the scripted LLM must have been invoked"
    joined = "\n".join(str(m.content) for prompt in llm.prompts for m in prompt)
    assert "DETERMINISTIC TECHNICAL SNAPSHOT" in joined   # gold market analyst node
    assert "mt5:readonly" in joined                       # MT5 snapshot in analysis layer
    assert "gold_spot_mt5" in joined
    assert "GC=F" not in "\n".join(                       # no futures-proxy data substitution
        line for line in joined.splitlines() if "queried" in line
    )
    assert len(llm.prompts) <= 25                         # ≤ 25 LLM calls per STANDARD run

    # -- GoldDecision --------------------------------------------------------
    decision = build_gold_decision(
        state,
        symbol="XAUUSD",
        trade_date=TRADE_DATE,
        data_sources=list(snap.provenance),
        now=AS_OF,
        asset_kind=AssetKind.GOLD_SPOT_MT5,
    )
    assert decision.action.value == "BUY"
    assert decision.entry == 2650.00 and decision.stop_loss == 2646.00
    assert decision.take_profit == 2662.00 and decision.confidence == 0.72
    assert decision.proxy_labelled is False               # MT5 spot is NOT the proxy
    assert any("mt5:readonly" in s for s in decision.data_sources)

    # -- Deterministic RiskGate (spread from the ACTUAL MT5 bid/ask) --------
    spread = snap.quote.spread
    assert spread == pytest.approx(0.30)
    account = AccountState(equity=10_000.0, open_positions=0, daily_loss_fraction=0.0)
    gate = evaluate_decision(decision, ctx, account, spread=spread)
    assert gate.approved, gate.summary()
    assert not gate.is_no_trade
    checks = {c.name: c.passed for c in gate.checks}
    assert checks["data_quality"] is True
    assert checks.get("max_spread") is True               # 0.30 ≤ 0.80
    assert checks.get("min_reward_risk") is True          # 12.0/4.0 = 3.0 ≥ 1.5
    assert gate.position_units == pytest.approx(10.0)     # risk-derived 25 oz, capped at 10

    # gate determinism: identical inputs → identical outcome
    gate2 = evaluate_decision(decision, ctx, account, spread=spread)
    assert gate2.approved == gate.approved
    assert gate2.position_units == gate.position_units
    assert [c.name for c in gate2.checks] == [c.name for c in gate.checks]

    # -- Paper Trading (simulated execution ONLY — no real orders) ----------
    engine = PaperTradingEngine()
    position = engine.open_from_gate(decision, gate, fill_time=AS_OF)
    assert position is not None
    assert position.side == "BUY" and position.units == pytest.approx(10.0)
    assert position.entry_fill == pytest.approx(2650.00 + 0.30 / 2 + 0.10)  # 2650.25
    assert position.decision_id == decision.decision_id

    # mark-to-target on the next session's bar (Sunday open → Monday range)
    target_bar = make_rates(
        datetime(2026, 9, 21, 22, 0, tzinfo=timezone.utc), 1, 15,
        base=2662.0, step=0.0,
    )
    closed = engine.process_bar(_mk_bar(target_bar[-1]))
    assert closed and closed[0].exit_reason == "TAKE_PROFIT"
    assert closed[0].pnl == pytest.approx(10.0 * (2662.00 - 2650.25))       # 117.50
    assert closed[0].r_multiple == pytest.approx(117.50 / 40.0)             # R = pnl / risk

    # -- runtime read-only guarantee on the fake terminal --------------------
    for call in fake.calls:
        name = call if isinstance(call, str) else call[0]
        assert name in READ_ONLY_APIS, call


def _mk_bar(row) -> object:
    from datetime import datetime as _dt

    from tradingagents.gold.data.models import GoldBar

    return GoldBar(
        timestamp=_dt.fromtimestamp(int(row["time"]), tz=timezone.utc),
        open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
        close=float(row["close"]), volume=float(row["tick_volume"]),
    )


@pytest.mark.unit
def test_mt5_data_error_blocks_gate_despite_llm_buy(tmp_path, monkeypatch):
    """The LLM says BUY; inverted MT5 bid/ask makes the gate say NO_TRADE.

    Proves the architecture LLM → GoldDecision → RiskGate (NOT LLM → MT5):
    a corrupted quote is a DATA ERROR → NO TRADE → no paper position.
    """
    llm = ScriptedLLM()
    fake = make_fake_mt5(tick=e2e_tick(bid=2650.50, ask=2649.90))  # ask < bid
    graph, builder, gold_config = build_gold_graph(tmp_path, monkeypatch, fake, llm)

    state, _signal = graph.propagate("XAUUSD", TRADE_DATE)
    ctx = builder.build(TRADE_DATE)
    assert not ctx.market_ok
    assert any("invalid_bid_ask" in e for e in ctx.data_errors)

    decision = build_gold_decision(
        state, symbol="XAUUSD", trade_date=TRADE_DATE,
        data_sources=list(ctx.snapshot.provenance),
        now=AS_OF, asset_kind=AssetKind.GOLD_SPOT_MT5,
    )
    assert decision.action.value == "BUY"                 # the LLM still proposes BUY

    gate = evaluate_decision(
        decision, ctx, AccountState(equity=10_000.0),
        spread=ctx.snapshot.quote.spread,                 # negative — invalid
    )
    assert not gate.approved and gate.is_no_trade
    assert any("data_quality" in r for r in gate.reasons)

    engine = PaperTradingEngine()
    assert engine.open_from_gate(decision, gate, fill_time=AS_OF) is None
    assert engine.positions == []


@pytest.mark.unit
def test_mt5_identity_gold_spot_mt5_distinct_from_futures_proxy(tmp_path, monkeypatch):
    """GOLD_SPOT_MT5 (is_proxy=False) vs GC=F GOLD_FUTURES_PROXY (is_proxy=True).

    Broker-decorated discovery (XAUUSDm) keeps the GOLD_SPOT_MT5 identity;
    nothing silently converts MT5 XAUUSD into the Yahoo futures proxy.
    """
    from tradingagents.gold.data.yahoo_gold import YahooGoldProxyProvider

    fake = make_fake_mt5(symbols=("XAUUSDm",), symbol="XAUUSDm")
    llm = ScriptedLLM()
    graph, builder, _cfg = build_gold_graph(tmp_path, monkeypatch, fake, llm)
    graph.propagate("XAUUSD", TRADE_DATE)
    ctx = builder.build(TRADE_DATE)

    assert builder.provider.discovered_symbol == "XAUUSDm"     # actual broker symbol
    assert builder.provider.capabilities.asset_kind is AssetKind.GOLD_SPOT_MT5
    assert builder.provider.capabilities.is_proxy is False
    for tf in (TimeFrame.M15, TimeFrame.H1, TimeFrame.H4):
        meta = ctx.snapshot.dataset(tf).meta
        assert meta.asset_kind is AssetKind.GOLD_SPOT_MT5, tf
        assert meta.is_proxy is False, tf
        assert meta.source_symbol == "XAUUSDm"                 # broker identity preserved
        assert meta.symbol == "XAUUSD"                         # requested identity preserved
        assert meta.source == "mt5:readonly"
        assert "[PROXY]" not in meta.describe()

    quote = ctx.snapshot.quote
    assert quote.asset_kind is AssetKind.GOLD_SPOT_MT5 and quote.source == "mt5:readonly"

    # the Yahoo provider remains, explicitly, the labelled futures proxy
    yahoo_caps = YahooGoldProxyProvider().capabilities
    assert yahoo_caps.asset_kind is AssetKind.GOLD_FUTURES_PROXY
    assert yahoo_caps.is_proxy is True


@pytest.mark.unit
def test_mt5_forming_candle_excluded_from_current_snapshot():
    """A forming candle in the MT5 response is labelled, never used as closed."""
    from tradingagents.gold.data.current import build_current_snapshot

    fake = FakeMT5(
        symbols=("XAUUSD",),
        rates={
            ("XAUUSD", 15): make_rates(AS_OF, 480, 15, forming=True),
            ("XAUUSD", 16385): make_rates(AS_OF, 240, 60),
            ("XAUUSD", 16388): make_rates(AS_OF, 120, 240),
        },
        tick=e2e_tick(),
    )
    provider = MT5ReadOnlyGoldProvider(mt5_module=fake)
    snap = build_current_snapshot(
        provider, default_gold_config(),
        now=datetime(2026, 9, 18, 20, 59, tzinfo=timezone.utc),
        as_of=datetime(2026, 9, 18, 20, 45, tzinfo=timezone.utc),
    )
    m15 = snap.bars["M15"]
    assert m15.has_forming                                    # explicitly identified
    assert m15.forming_bar.timestamp == datetime(2026, 9, 18, 20, 45, tzinfo=timezone.utc)
    assert m15.forming_bar.timestamp not in {
        b.timestamp for b in snap.datasets["M15"].bars        # excluded from analysis
    }
    assert m15.bar.timestamp == datetime(2026, 9, 18, 20, 30, tzinfo=timezone.utc)
    for ds in snap.datasets.values():
        assert max(b.timestamp for b in ds.bars) <= snap.as_of   # PIT: no future leak


# ---------------------------------------------------------------------------
# REAL MT5 ENVIRONMENT TEST (distinct from the FAKE MT5 TESTS above)
# ---------------------------------------------------------------------------


class TestRealMT5Environment:
    """REAL MT5 ENVIRONMENT TEST — requires an actual terminal connection.

    Skipped with ``REQUIRES MT5 TERMINAL`` unless the Windows-only
    MetaTrader5 package is importable AND a terminal initializes.  The FAKE
    MT5 TESTS above never substitute for this.
    """

    def test_real_terminal_read_only_probe(self):
        pytest.importorskip(
            "MetaTrader5",
            reason=(
                "REQUIRES MT5 TERMINAL: the MetaTrader5 Python package "
                "(Windows + a running MT5 terminal) is not available in this "
                "environment. All other MT5 tests here are FAKE MT5 TESTS "
                "proving code correctness only — no real broker connection, "
                "no real XAUUSD data, and no claim of real MT5 validation."
            ),
        )
        import MetaTrader5 as mt5

        from tradingagents.gold.data.mt5 import discover_gold_symbol

        try:
            assert mt5.initialize(), f"MT5 initialize failed: {mt5.last_error()}"
            symbol = discover_gold_symbol(mt5)
            assert mt5.symbol_select(symbol, True)
            for tf_attr in ("TIMEFRAME_M15", "TIMEFRAME_H1", "TIMEFRAME_H4"):
                rates = mt5.copy_rates_range(
                    symbol, getattr(mt5, tf_attr),
                    datetime.now(timezone.utc) - timedelta(days=10),
                    datetime.now(timezone.utc),
                )
                assert rates is not None and len(rates) > 0, tf_attr
            tick = mt5.symbol_info_tick(symbol)
            assert tick is not None and tick.bid > 0 and tick.ask >= tick.bid
        finally:
            mt5.shutdown()

"""Regression: gold factory overrides must actually reach the graph.

Guards the two integration defects found by the post-transformation audit:

1. ``TradingAgentsGraph`` accepted ``analyst_factories``/``core_factories``
   but never forwarded them to ``GraphSetup`` — the overrides were silently
   dropped and upstream nodes ran in their place.
2. ``gold_analyst_factories`` shipped zero-argument lambdas hardcoding
   ``llm=None`` instead of following the one-LLM-argument factory contract.

These tests pin implementation IDENTITY (which node implementation actually
executes and which LLM object it receives), not merely state-key
compatibility.
"""

from __future__ import annotations

import copy

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.gold.agents.context import GoldContextBuilder
from tradingagents.gold.config import default_gold_config
from tradingagents.gold.graph import GoldTradingAgentsGraph, gold_analyst_factories
from tradingagents.gold.macro.engine import MacroEngine
from tradingagents.gold.news.engine import GoldNewsEngine
from tradingagents.gold.sentiment.engine import GoldSentimentEngine
from tradingagents.graph import trading_graph
from tradingagents.graph.trading_graph import TradingAgentsGraph

TRADE_DATE = "2026-09-18"
HOLD_TEXT = "Gold analysis.\n\n**Rating**: Hold\n\nFINAL TRANSACTION PROPOSAL: **HOLD**"


class ScriptedModel(BaseChatModel):
    """Records every prompt it is invoked with; answers HOLD_TEXT."""

    prompts: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-wiring"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.prompts.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=HOLD_TEXT))])

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("free-text only")

    def bind_tools(self, tools, **kwargs):
        return self


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


def _config(tmp_path):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        results_dir=str(tmp_path / "results"),
        data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "log.md"),
    )
    return cfg


def _gold_builder(gold_cfg, twits_seen=None):
    def twits(ticker, *args, **kwargs):
        if twits_seen is not None:
            twits_seen.append(ticker)
        return "gold-sentiment-block"

    return GoldContextBuilder(
        config=gold_cfg,
        provider=__import__("tests.test_gold_context_builder", fromlist=["offline_provider"]).offline_provider(),
        macro_engine=MacroEngine(fetcher=lambda i, d: "no data"),
        news_engine=GoldNewsEngine(company_fetch=lambda *a: "", global_fetch=lambda *a: ""),
        sentiment_engine=GoldSentimentEngine(
            twits_fetch=twits, reddit_fetch=lambda *a, **k: "gold-reddit-block",
        ),
    )


# ---------------------------------------------------------------------------
# Defect 1: the forwarding path itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_factory_overrides_are_forwarded_to_graph_setup(tmp_path, monkeypatch):
    """The exact regression: overrides must survive TradingAgentsGraph init."""
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: _Client(ScriptedModel()))

    def market_factory(llm):
        def node(state):
            return {"market_report": "sentinel"}

        return node

    graph = TradingAgentsGraph(
        config=_config(tmp_path),
        selected_analysts=("market", "social", "news", "fundamentals"),
        analyst_factories={"market": market_factory},
    )
    assert graph.graph_setup.analyst_factories["market"] is market_factory


@pytest.mark.unit
def test_core_factory_overrides_are_forwarded_to_graph_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: _Client(ScriptedModel()))

    def bull_factory(llm):
        def node(state):
            return {"investment_debate_state": {}}

        return node

    graph = TradingAgentsGraph(config=_config(tmp_path), core_factories={"bull": bull_factory})
    assert graph.graph_setup.core_factories["bull"] is bull_factory


# ---------------------------------------------------------------------------
# Defect 1 + 2 end to end: distinctive gold nodes execute, upstream does not,
# and the real LLM object reaches the factory.
# ---------------------------------------------------------------------------


_REPORT_KEYS = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def _distinctive(slot, captured):
    report_key = _REPORT_KEYS[slot]

    def factory(llm):
        captured.append((slot, llm))

        def node(state):
            return {
                report_key: f"DISTINCTIVE-{slot.upper()}",
                "messages": [AIMessage(content=f"DISTINCTIVE-{slot.upper()}")],
            }

        return node

    return factory


@pytest.mark.unit
def test_distinctive_override_nodes_execute_and_upstream_is_not_substituted(
    tmp_path, monkeypatch
):
    """Upstream analyst factories are rigged to raise: if ANY of them runs in
    place of an override, this test fails loudly (implementation identity)."""
    import tradingagents.graph.setup as setup_module

    def _boom(name):
        return lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(f"upstream {name} implementation was silently substituted")
        )

    monkeypatch.setattr(setup_module, "create_market_analyst", _boom("market"))
    monkeypatch.setattr(setup_module, "create_sentiment_analyst", _boom("social"))
    monkeypatch.setattr(setup_module, "create_news_analyst", _boom("news"))
    monkeypatch.setattr(setup_module, "create_fundamentals_analyst", _boom("fundamentals"))

    quick_model = ScriptedModel()
    deep_model = ScriptedModel()
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        trading_graph,
        "create_llm_client",
        lambda **k: _Client(quick_model if k.get("model") == cfg["quick_think_llm"] else deep_model),
    )

    captured: list[tuple[str, object]] = []
    overrides = {
        "market": _distinctive("market", captured),
        "social": _distinctive("social", captured),
        "news": _distinctive("news", captured),
        "fundamentals": _distinctive("fundamentals", captured),
    }
    graph = TradingAgentsGraph(
        config=cfg,
        selected_analysts=("market", "social", "news", "fundamentals"),
        analyst_factories=overrides,
    )
    state, _signal = graph.propagate("XAUUSD", TRADE_DATE)

    # The distinctive nodes ran and their outputs reached the state.
    assert state["market_report"] == "DISTINCTIVE-MARKET"
    assert state["sentiment_report"] == "DISTINCTIVE-SOCIAL"
    assert state["news_report"] == "DISTINCTIVE-NEWS"
    assert state["fundamentals_report"] == "DISTINCTIVE-FUNDAMENTALS"

    # Every override was created through the real wiring and received the
    # actual quick-thinking LLM instance (the GraphSetup factory contract).
    assert {slot for slot, _ in captured} == {"market", "social", "news", "fundamentals"}
    for _slot, llm in captured:
        assert llm is graph.quick_thinking_llm


# ---------------------------------------------------------------------------
# Defect 2 contract at the helper level + the real gold graph identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gold_analyst_factories_accept_exactly_one_llm_argument():
    """The broken build shipped zero-arg lambdas; the contract is one LLM."""
    factories = gold_analyst_factories(lambda trade_date: None)
    assert set(factories) == {"market", "fundamentals", "news", "social"}
    for factory in factories.values():
        node = factory(object())  # must accept the LLM as its only argument
        assert callable(node)


@pytest.mark.unit
def test_gold_graph_runs_gold_implementations_with_gold_context_and_mapping(
    tmp_path, monkeypatch
):
    """The full GoldTradingAgentsGraph: gold node prompts reach the LLM, the
    gold sentiment mapping (GOLD/GLD) is used, and the upstream social
    fetchers (XAUUSD mapping) are never touched."""
    import tradingagents.agents.analysts.sentiment_analyst as upstream_social

    upstream_calls: list[str] = []
    monkeypatch.setattr(
        upstream_social,
        "fetch_stocktwits_messages",
        lambda *a, **k: upstream_calls.append("stocktwits") or "",
    )
    monkeypatch.setattr(
        upstream_social,
        "fetch_reddit_posts",
        lambda *a, **k: upstream_calls.append("reddit") or "",
    )

    model = ScriptedModel()
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: _Client(model))

    gold_cfg = default_gold_config()
    twits_seen: list[str] = []
    graph = GoldTradingAgentsGraph(
        config=_config(tmp_path),
        gold_config=gold_cfg,
        context_builder=_gold_builder(gold_cfg, twits_seen),
    )
    state, signal = graph.propagate("XAUUSD", TRADE_DATE, asset_type="gold")

    assert signal == "Hold"
    prompt_text = "\n".join(
        getattr(message, "content", "") for messages in model.prompts for message in messages
    )
    # Gold node implementations produced these prompts (identity proof).
    assert "DETERMINISTIC TECHNICAL SNAPSHOT" in prompt_text
    assert "GOLD MACRO ANALYST" in prompt_text
    assert "GOLD & MACRO NEWS ANALYST" in prompt_text
    assert "GOLD SENTIMENT" in prompt_text
    assert "NEVER invent market data" in prompt_text
    # The upstream social mapping (StockTwits for XAUUSD) was NOT used; the
    # gold mapping (GOLD/GLD streams) was.
    assert upstream_calls == []
    assert set(twits_seen) == {"GOLD", "GLD"}
    # Gold-specific outputs present in the final state.
    assert state["asset_type"] == "gold"
    assert "PROXY" in state["instrument_context"]
    assert state["final_trade_decision"].strip()

"""Phase 5 — the complete gold multi-agent graph, end to end, offline.

Phase 5 PASS condition: a complete XAUUSD multi-agent run produces a
structured final decision.  Same scripted-model technique as
``tests/test_graph_end_to_end.py`` — no network, no API keys.
"""

from __future__ import annotations

import copy

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.gold import CostMode
from tradingagents.gold.config import default_gold_config
from tradingagents.gold.graph import GoldTradingAgentsGraph
from tradingagents.gold.macro.engine import MacroEngine
from tradingagents.gold.news.engine import GoldNewsEngine
from tradingagents.gold.sentiment.engine import GoldSentimentEngine
from tradingagents.graph import trading_graph

TRADE_DATE = "2026-09-18"   # a Friday, before "today" (2026-09-24)
TEXT = (
    "Gold report.\n\n**Rating**: Hold\n\n"
    "FINAL TRANSACTION PROPOSAL: **HOLD**"
)


class FreeTextModel(BaseChatModel):
    """Answers every call with TEXT; no structured output (free-text path)."""

    @property
    def _llm_type(self) -> str:
        return "scripted-gold"

    def invoke(self, messages, *args, **kwargs):
        return AIMessage(content=TEXT)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=TEXT))])

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("free-text only")

    def bind_tools(self, tools, **kwargs):
        return self


class _Client:
    def __init__(self, model):
        self.model = model

    def get_llm(self):
        return self.model


def _gold_graph(tmp_path, monkeypatch, **kwargs):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        results_dir=str(tmp_path / "results"),
        data_cache_dir=str(tmp_path / "cache"),
        memory_log_path=str(tmp_path / "log.md"),
    )
    monkeypatch.setattr(trading_graph, "create_llm_client", lambda **k: _Client(FreeTextModel()))

    from tests.test_gold_context_builder import offline_provider
    from tradingagents.gold.agents.context import GoldContextBuilder

    gold_config = kwargs.pop("gold_config", None) or default_gold_config()
    builder = GoldContextBuilder(
        config=gold_config,
        provider=offline_provider(),
        macro_engine=MacroEngine(fetcher=lambda i, d: "no data"),
        news_engine=GoldNewsEngine(company_fetch=lambda *a: "", global_fetch=lambda *a: ""),
        sentiment_engine=GoldSentimentEngine(
            twits_fetch=lambda *a, **k: "", reddit_fetch=lambda *a, **k: "",
        ),
    )
    return GoldTradingAgentsGraph(
        config=cfg, gold_config=gold_config, context_builder=builder, **kwargs,
    )


@pytest.mark.unit
def test_full_gold_run_produces_structured_decision(tmp_path, monkeypatch):
    graph = _gold_graph(tmp_path, monkeypatch)

    state, signal = graph.propagate("XAUUSD", TRADE_DATE)

    assert signal == "Hold"
    for key in ("market_report", "sentiment_report", "news_report", "fundamentals_report",
                "investment_plan", "trader_investment_plan", "final_trade_decision"):
        assert state[key].strip(), key
    assert state["asset_type"] == "gold"
    assert "PROXY" in state["instrument_context"]       # proxy disclaimer reached the graph
    assert [e["rating"] for e in graph.memory_log.load_entries()] == ["Hold"]


@pytest.mark.unit
def test_gold_run_is_deterministic_in_reports_structure(tmp_path, monkeypatch):
    """Two identical offline runs produce the same pipeline shape (reports all
    present, same signal) — the deterministic layer does not drift."""
    first = _gold_graph(tmp_path, monkeypatch).propagate("XAUUSD", TRADE_DATE)
    second = _gold_graph(tmp_path, monkeypatch).propagate("XAUUSD", TRADE_DATE)
    assert first[1] == second[1] == "Hold"
    for key in ("market_report", "fundamentals_report", "investment_plan"):
        assert bool(first[0][key].strip()) == bool(second[0][key].strip())


@pytest.mark.unit
def test_cost_modes_change_graph_shape(tmp_path, monkeypatch):
    standard = _gold_graph(tmp_path, monkeypatch)
    fast = _gold_graph(
        tmp_path, monkeypatch,
        gold_config=default_gold_config(cost_mode=CostMode.FAST),
    )
    standard_nodes = set(standard.graph.get_graph().nodes)
    fast_nodes = set(fast.graph.get_graph().nodes)
    assert {"Market Analyst", "Fundamentals Analyst", "News Analyst", "Sentiment Analyst"} <= standard_nodes
    assert "Sentiment Analyst" not in fast_nodes
    assert {"Market Analyst", "News Analyst"} <= fast_nodes
    # The shared pipeline is always present, unchanged.
    for node in ("Bull Researcher", "Bear Researcher", "Research Manager", "Trader",
                 "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst",
                 "Portfolio Manager"):
        assert node in fast_nodes and node in standard_nodes


@pytest.mark.unit
def test_default_selected_analysts_untouched(tmp_path, monkeypatch):
    """The gold graph must not mutate upstream defaults (additivity rule)."""
    _gold_graph(tmp_path, monkeypatch)
    from tradingagents.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["llm_provider"] == "openai"
    assert DEFAULT_CONFIG["deep_think_llm"] and DEFAULT_CONFIG["quick_think_llm"]
    assert "gold" not in str(DEFAULT_CONFIG.get("data_vendors", {}))

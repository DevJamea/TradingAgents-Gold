"""Phase 5 — gold node factories preserve the upstream state contracts."""

from datetime import datetime, timezone

from langchain_core.messages import AIMessage

from tests.test_gold_context_builder import offline_provider
from tradingagents.gold.agents.context import GoldContextBuilder
from tradingagents.gold.agents.nodes import (
    create_gold_aggressive_debator,
    create_gold_bear_researcher,
    create_gold_bull_researcher,
    create_gold_conservative_debator,
    create_gold_macro_analyst,
    create_gold_market_analyst,
    create_gold_neutral_debator,
    create_gold_news_analyst,
    create_gold_portfolio_manager,
    create_gold_research_manager,
    create_gold_trader,
)
from tradingagents.gold.data.memory import InMemoryGoldProvider
from tradingagents.gold.macro.engine import MacroEngine
from tradingagents.gold.news.engine import GoldNewsEngine
from tradingagents.gold.sentiment.engine import GoldSentimentEngine

TRADE_DATE = "2026-09-18"
REPORT = "GOLD ANALYSIS REPORT"


class FakeLLM:
    """Records prompts; answers free text; no structured output support."""

    def __init__(self, text: str = REPORT):
        self.text = text
        self.prompts: list = []

    def invoke(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return AIMessage(content=self.text)

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("no structured output in fake")

    def bind_tools(self, tools, **kwargs):
        return self


def make_state(**extra):
    state = {
        "company_of_interest": "XAUUSD",
        "trade_date": TRADE_DATE,
        "messages": [("human", "XAUUSD")],
        "instrument_context": "The instrument to analyze is `XAUUSD` (GOLD / XAUUSD spot).",
        "investment_debate_state": {
            "history": "", "bull_history": "", "bear_history": "",
            "current_response": "", "count": 0,
        },
        "risk_debate_state": {
            "history": "", "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "latest_speaker": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 0,
        },
        "investment_plan": "PLAN",
        "trader_investment_plan": "PROPOSAL",
        "past_context": "",
        "portfolio_context": "",
    }
    state.update(extra)
    return state


def make_context_provider(**builder_overrides):
    from tradingagents.gold.config import default_gold_config

    kwargs = {
        "config": default_gold_config(),
        "provider": offline_provider(),
        "macro_engine": MacroEngine(fetcher=lambda i, d: "no data"),
        "news_engine": GoldNewsEngine(company_fetch=lambda *a: "", global_fetch=lambda *a: ""),
        "sentiment_engine": GoldSentimentEngine(
            twits_fetch=lambda *a, **k: "", reddit_fetch=lambda *a, **k: "",
        ),
        "now_fn": lambda: datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
    }
    kwargs.update(builder_overrides)
    return GoldContextBuilder(**kwargs).build


class TestAnalystNodes:
    def test_market_analyst_writes_market_report_and_prompt_has_snapshot(self):
        llm = FakeLLM()
        node = create_gold_market_analyst(llm, make_context_provider())
        out = node(make_state())
        assert out["market_report"] == REPORT
        assert out["messages"][0].content == REPORT
        prompt_text = str(llm.prompts[0])
        assert "DETERMINISTIC TECHNICAL SNAPSHOT" in prompt_text
        assert "do not recompute" in prompt_text

    def test_macro_analyst_writes_fundamentals_report(self):
        llm = FakeLLM()
        node = create_gold_macro_analyst(llm, make_context_provider())
        out = node(make_state())
        assert out["fundamentals_report"] == REPORT
        assert "GOLD MACRO ANALYST" in str(llm.prompts[0])

    def test_news_analyst_writes_news_report(self):
        llm = FakeLLM()
        node = create_gold_news_analyst(llm, make_context_provider())
        out = node(make_state())
        assert out["news_report"] == REPORT
        assert "GOLD & MACRO NEWS ANALYST" in str(llm.prompts[0])

    def test_data_errors_reach_the_prompt(self):
        llm = FakeLLM()
        node = create_gold_market_analyst(
            llm, make_context_provider(provider=InMemoryGoldProvider()),
        )
        node(make_state())
        assert "DATA QUALITY ERRORS" in str(llm.prompts[0])


class TestDebateNodes:
    def test_bull_updates_debate_state(self):
        llm = FakeLLM("Bullish on gold.")
        node = create_gold_bull_researcher(llm)
        out = node(make_state())
        debate = out["investment_debate_state"]
        assert "Bull Researcher: Bullish on gold." in debate["history"]
        assert "Bull Researcher: Bullish on gold." in debate["bull_history"]
        assert debate["count"] == 1
        assert "invalidation" in str(llm.prompts[0]).lower()

    def test_bear_updates_debate_state(self):
        llm = FakeLLM("Bearish on gold.")
        node = create_gold_bear_researcher(llm)
        out = node(make_state())
        debate = out["investment_debate_state"]
        assert "Bear Researcher: Bearish on gold." in debate["bear_history"]
        assert debate["count"] == 1

    def test_bull_sees_bear_argument(self):
        llm = FakeLLM()
        state = make_state()
        state["investment_debate_state"]["current_response"] = "BEAR SAID THIS"
        create_gold_bull_researcher(llm)(state)
        assert "BEAR SAID THIS" in str(llm.prompts[0])


class TestManagerNodes:
    def test_research_manager_writes_investment_plan(self):
        llm = FakeLLM("Plan: **Recommendation**: Hold")
        out = create_gold_research_manager(llm)(make_state())
        assert out["investment_plan"] == "Plan: **Recommendation**: Hold"
        assert out["investment_debate_state"]["judge_decision"] == out["investment_plan"]
        assert "FACTS" in str(llm.prompts[0])  # facts/interpretations/uncertainties

    def test_trader_writes_trader_plan_with_sender(self):
        llm = FakeLLM("FINAL TRANSACTION PROPOSAL: **HOLD**")
        out = create_gold_trader(llm)(make_state(), name="Trader")
        assert out["trader_investment_plan"].endswith("**HOLD**")
        assert out["sender"] == "Trader"
        assert "NEVER invent market data" in str(llm.prompts[0])

    def test_trader_prompt_has_review_rule(self):
        llm = FakeLLM()
        create_gold_trader(llm)(make_state(), name="Trader")
        assert "Review" in str(llm.prompts[0])


class TestRiskNodes:
    def test_aggressive_risk_debator_updates_state(self):
        llm = FakeLLM("Take the trade.")
        out = create_gold_aggressive_debator(llm)(make_state())
        debate = out["risk_debate_state"]
        assert debate["latest_speaker"] == "Aggressive"
        assert "Aggressive Risk Analyst: Take the trade." in debate["aggressive_history"]
        assert debate["count"] == 1

    def test_conservative_flags_spread_and_proxy_risks(self):
        llm = FakeLLM("Too risky.")
        out = create_gold_conservative_debator(llm)(make_state())
        debate = out["risk_debate_state"]
        assert debate["latest_speaker"] == "Conservative"
        prompt = str(llm.prompts[0])
        assert "spread" in prompt and "futures-proxy" in prompt

    def test_neutral_advocates_no_direction(self):
        llm = FakeLLM("Balanced view.")
        out = create_gold_neutral_debator(llm)(make_state())
        assert out["risk_debate_state"]["latest_speaker"] == "Neutral"


class TestPortfolioManager:
    def test_pm_writes_final_decision_and_judge(self):
        llm = FakeLLM("**Rating**: Hold\nExecutive: no action")
        out = create_gold_portfolio_manager(llm)(make_state())
        assert "**Rating**: Hold" in out["final_trade_decision"]
        assert out["risk_debate_state"]["latest_speaker"] == "Judge"
        prompt = str(llm.prompts[0])
        assert "deterministic risk gate" in prompt
        assert "DATA ERROR" in prompt  # the gate-aware instruction exists

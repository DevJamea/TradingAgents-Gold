from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.state import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


def _tools_or_clear(spec):
    """Route an analyst's turn: run its tool calls, or finish its report."""
    def route(state) -> str:
        return spec.tool_node if state["messages"][-1].tool_calls else spec.clear_node
    return route


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        conditional_logic: ConditionalLogic,
        analyst_factories: dict[str, Any] | None = None,
        core_factories: dict[str, Any] | None = None,
    ):
        """Initialize with required components.

        ``analyst_factories`` (optional) overrides per-slot analyst node
        factories, keyed by the analyst slot key (``market`` / ``social`` /
        ``news`` / ``fundamentals``).  ``core_factories`` (optional) overrides
        the post-analyst node factories by role (``bull``, ``bear``,
        ``research_manager``, ``trader``, ``aggressive``, ``conservative``,
        ``neutral``, ``portfolio_manager``).  Both default to ``None`` (stock /
        crypto behavior); they exist so asset specializations (e.g. gold) can
        swap node implementations without forking the graph wiring.
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.conditional_logic = conditional_logic
        self.analyst_factories = dict(analyst_factories or {})
        self.core_factories = dict(core_factories or {})

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Sentiment analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        # Slot factories: defaults first, so a specialization overrides only
        # the slots it implements.  Every factory is invoked with the
        # quick-thinking LLM at node-creation time (defaults ignore it).
        analyst_factories: dict[str, Any] = {
            "market": lambda llm: create_market_analyst(llm),
            "social": lambda llm: create_sentiment_analyst(llm),
            "news": lambda llm: create_news_analyst(llm),
            "fundamentals": lambda llm: create_fundamentals_analyst(llm),
        }
        analyst_factories.update(self.analyst_factories)

        # Role factories for the shared part of the graph (debate, managers).
        core_factories: dict[str, Any] = {
            "bull": create_bull_researcher,
            "bear": create_bear_researcher,
            "research_manager": create_research_manager,
            "trader": create_trader,
            "aggressive": create_aggressive_debator,
            "conservative": create_conservative_debator,
            "neutral": create_neutral_debator,
            "portfolio_manager": create_portfolio_manager,
        }
        core_factories.update(self.core_factories)

        bull_researcher_node = core_factories["bull"](self.quick_thinking_llm)
        bear_researcher_node = core_factories["bear"](self.quick_thinking_llm)
        research_manager_node = core_factories["research_manager"](self.deep_thinking_llm)
        trader_node = core_factories["trader"](self.quick_thinking_llm)

        aggressive_analyst = core_factories["aggressive"](self.quick_thinking_llm)
        neutral_analyst = core_factories["neutral"](self.quick_thinking_llm)
        conservative_analyst = core_factories["conservative"](self.quick_thinking_llm)
        portfolio_manager_node = core_factories["portfolio_manager"](self.deep_thinking_llm)

        workflow = StateGraph(AgentState)

        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key](self.quick_thinking_llm))
            workflow.add_node(spec.clear_node, create_msg_delete())
            if spec.tools:
                workflow.add_node(spec.tool_node, ToolNode(list(spec.tools)))

        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        workflow.add_edge(START, plan.specs[0].agent_node)

        for i, spec in enumerate(plan.specs):
            if spec.tools:
                workflow.add_conditional_edges(
                    spec.agent_node, _tools_or_clear(spec), [spec.tool_node, spec.clear_node]
                )
                workflow.add_edge(spec.tool_node, spec.agent_node)
            else:
                workflow.add_edge(spec.agent_node, spec.clear_node)

            # The last analyst hands over to the research debate.
            following = plan.specs[i + 1].agent_node if i < len(plan.specs) - 1 else "Bull Researcher"
            workflow.add_edge(spec.clear_node, following)

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # All three risk edges share the complete RISK_ANALYSIS_PATH_MAP (#1088).
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow

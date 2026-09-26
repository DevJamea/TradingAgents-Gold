"""GoldTradingAgentsGraph: the XAUUSD specialization of the upstream graph.

Reuses the complete upstream LangGraph architecture — topology, conditional
logic, checkpoints, memory log, settlement, reports — and swaps only the node
implementations for gold ones via the graph's factory-override hooks:

* analyst slots: market→gold technical analyst, fundamentals→GOLD MACRO
  analyst, news→gold news analyst, social→gold sentiment analyst;
* core roles: gold bull/bear, gold research manager, gold trader, gold risk
  debators, gold portfolio manager.

The LLM still only analyses; execution (risk gate → paper trading → later MT5
demo) is deterministic and lives in the remaining ``tradingagents.gold``
modules.  Nothing here can place an order.
"""

from __future__ import annotations

from typing import Any

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
    create_gold_sentiment_analyst,
    create_gold_trader,
)
from tradingagents.gold.config import GoldConfig, default_gold_config
from tradingagents.graph.trading_graph import TradingAgentsGraph

#: Gold analyst vocabulary → the graph's analyst slot keys.  ``macro`` rides
#: the ``fundamentals`` slot (the gold macro analyst replaces equity
#: fundamentals); everything else maps 1:1.
GOLD_ANALYST_SLOTS: dict[str, str] = {
    "market": "market",
    "macro": "fundamentals",
    "news": "news",
    "social": "social",
}


def gold_analyst_factories(context_provider) -> dict[str, Any]:
    """Slot-keyed factory overrides for the gold analysts.

    Each factory receives the quick-thinking LLM at node-creation time (the
    ``GraphSetup`` factory contract) so the actual LLM instance reaches the
    gold node — never a hardcoded ``None``.
    """
    return {
        "market": lambda llm: create_gold_market_analyst(llm, context_provider),
        "fundamentals": lambda llm: create_gold_macro_analyst(llm, context_provider),
        "news": lambda llm: create_gold_news_analyst(llm, context_provider),
        "social": lambda llm: create_gold_sentiment_analyst(llm, context_provider),
    }


def gold_core_factories() -> dict[str, Any]:
    """Role-keyed factory overrides for the shared part of the graph."""
    return {
        "bull": create_gold_bull_researcher,
        "bear": create_gold_bear_researcher,
        "research_manager": create_gold_research_manager,
        "trader": create_gold_trader,
        "aggressive": create_gold_aggressive_debator,
        "conservative": create_gold_conservative_debator,
        "neutral": create_gold_neutral_debator,
        "portfolio_manager": create_gold_portfolio_manager,
    }


class GoldTradingAgentsGraph(TradingAgentsGraph):
    """Run the multi-agent research graph specialized for XAUUSD."""

    def __init__(
        self,
        gold_config: GoldConfig | None = None,
        debug: bool = False,
        config: dict[str, Any] | None = None,
        callbacks: list | None = None,
        context_builder: GoldContextBuilder | None = None,
    ):
        self.gold_config = gold_config or default_gold_config()
        self.context_builder = context_builder or GoldContextBuilder(self.gold_config)
        provider = self._cached_context_provider(self.context_builder)

        # Gold analyst vocabulary → graph slot keys for the execution plan.
        slots = [GOLD_ANALYST_SLOTS[a] for a in self.gold_config.cost_profile.analysts]

        super().__init__(
            selected_analysts=slots,
            debug=debug,
            config=config,
            callbacks=callbacks,
            analyst_factories=gold_analyst_factories(provider),
            core_factories=gold_core_factories(),
        )

    @staticmethod
    def _cached_context_provider(builder: GoldContextBuilder):
        def provider(trade_date: str):
            return builder.build(trade_date)

        return provider

    def resolve_instrument_context(self, ticker: str, asset_type: str = "gold",
                                   curr_date: str | None = None) -> str:
        """Deterministic gold instrument identity for every agent.

        Replaces the stock company-identity lookup: gold runs anchor to XAUUSD
        spot with the GC=F futures-proxy disclaimer attached, so no agent can
        mistake the proxy for broker spot (spec §6).
        """
        return (
            f"The instrument to analyze is `{ticker}` (GOLD / XAUUSD spot). "
            "This is a GOLD run: every price reference is USD per troy ounce. "
            f"{self.gold_config.data.proxy_disclaimer}"
        )

    def propagate(self, company_name, trade_date, asset_type: str = "gold", portfolio=None):
        """Run the gold graph; ``asset_type`` defaults to ``gold``."""
        return super().propagate(company_name, trade_date, asset_type=asset_type, portfolio=portfolio)

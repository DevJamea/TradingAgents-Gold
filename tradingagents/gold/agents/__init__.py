"""Gold agent nodes and run context."""

from tradingagents.gold.agents.context import GoldContextBuilder, GoldRunContext
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
from tradingagents.gold.agents.schemas import (
    GoldTraderAction,
    GoldTraderProposal,
    render_gold_trader_proposal,
)

__all__ = [
    "GoldContextBuilder",
    "GoldRunContext",
    "GoldTraderAction",
    "GoldTraderProposal",
    "create_gold_aggressive_debator",
    "create_gold_bear_researcher",
    "create_gold_bull_researcher",
    "create_gold_conservative_debator",
    "create_gold_macro_analyst",
    "create_gold_market_analyst",
    "create_gold_neutral_debator",
    "create_gold_news_analyst",
    "create_gold_portfolio_manager",
    "create_gold_research_manager",
    "create_gold_sentiment_analyst",
    "create_gold_trader",
    "render_gold_trader_proposal",
]

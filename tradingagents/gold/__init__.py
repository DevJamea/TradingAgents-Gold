"""GoldTradingAgents — the XAUUSD / GOLD specialization of TradingAgents.

This subpackage is the deterministic side of the gold system: configuration,
data abstractions, technical/macro/news/sentiment engines, the structured gold
decision model, the deterministic risk gate, paper trading, research tooling,
live observation, and (later) the MT5 demo adapter.

Design rule (absolute): LLMs analyse; deterministic code validates, gates and
executes.  Nothing in this package may place a real order.  The LLM never
calls broker / MT5 execution functions.
"""

from tradingagents.gold.config import GoldConfig, default_gold_config
from tradingagents.gold.types import (
    AssetKind,
    CostMode,
    DataKind,
    DecisionAction,
    MarketRegime,
    SessionName,
    TimeFrame,
)

__all__ = [
    "AssetKind",
    "CostMode",
    "DataKind",
    "DecisionAction",
    "GoldConfig",
    "MarketRegime",
    "SessionName",
    "TimeFrame",
    "default_gold_config",
]

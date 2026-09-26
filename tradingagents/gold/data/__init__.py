"""Gold data layer: providers, provenance-carrying datasets, quality gate."""

from tradingagents.gold.data.current import (
    BarStatus,
    GoldCurrentSnapshot,
    build_current_snapshot,
)
from tradingagents.gold.data.memory import InMemoryGoldProvider
from tradingagents.gold.data.models import DatasetMeta, GoldBar, GoldDataset, GoldQuote
from tradingagents.gold.data.mt5 import (
    MT5ReadOnlyGoldProvider,
    MT5SymbolAmbiguityError,
    MT5SymbolDiscoveryError,
    discover_gold_symbol,
)
from tradingagents.gold.data.provider import (
    GoldMarketDataProvider,
    ProviderCapabilities,
    ProviderRegistry,
)
from tradingagents.gold.data.snapshot import MarketSnapshot, build_market_snapshot, quote_issues
from tradingagents.gold.data.validation import ValidationIssue, ValidationReport, validate_dataset
from tradingagents.gold.data.yahoo_gold import YahooGoldProxyProvider

__all__ = [
    "BarStatus",
    "DatasetMeta",
    "GoldBar",
    "GoldCurrentSnapshot",
    "GoldDataset",
    "GoldMarketDataProvider",
    "GoldQuote",
    "InMemoryGoldProvider",
    "MT5ReadOnlyGoldProvider",
    "MT5SymbolAmbiguityError",
    "MT5SymbolDiscoveryError",
    "MarketSnapshot",
    "ProviderCapabilities",
    "ProviderRegistry",
    "ValidationIssue",
    "ValidationReport",
    "YahooGoldProxyProvider",
    "build_current_snapshot",
    "build_market_snapshot",
    "discover_gold_symbol",
    "quote_issues",
    "validate_dataset",
]

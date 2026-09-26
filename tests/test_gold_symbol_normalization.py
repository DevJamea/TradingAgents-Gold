"""Regression: the gold symbol vocabulary normalizes deterministically.

Covers the ``XAU/USD`` edge case found by the integration audit: the gold
layer accepted it as a gold spelling while the data-layer normalizer passed
it through untouched.  Every user-facing gold spelling must normalize to the
same Yahoo futures proxy, ``GC=F``.

The distinction is preserved end to end: normalization targets the PROXY
(Yahoo has no spot gold), while dataset metadata keeps ``XAUUSD`` as the spot
identity and labels ``GC=F`` data as ``GOLD_FUTURES_PROXY`` — a futures proxy
is never silently presented as spot.
"""

import pytest

from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.gold.config import is_gold_symbol
from tradingagents.gold.data.yahoo_gold import YahooGoldProxyProvider
from tradingagents.gold.types import AssetKind, TimeFrame


@pytest.mark.parametrize("raw,expected", [
    ("XAUUSD", "GC=F"),
    ("xauusd", "GC=F"),
    ("XAUUSD+", "GC=F"),
    ("XAU/USD", "GC=F"),
    ("xau/usd", "GC=F"),
    ("XAU", "GC=F"),
    ("GOLD", "GC=F"),
    ("gold", "GC=F"),
    ("GC=F", "GC=F"),
])
def test_gold_vocabulary_normalizes_to_futures_proxy(raw, expected):
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("XAGUSD", "SI=F"),      # silver stays silver, never gold
    ("AAPL", "AAPL"),        # equities untouched
    ("BTCUSD", "BTC-USD"),   # crypto untouched
    ("EURUSD", "EURUSD=X"),  # forex untouched
])
def test_non_gold_symbols_untouched(raw, expected):
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize("spelling", ["XAUUSD", "XAU/USD", "XAU", "GOLD", "GC=F"])
def test_gold_layer_and_data_layer_agree(spelling):
    """Both layers recognize exactly the same gold vocabulary, deterministically."""
    assert is_gold_symbol(spelling)
    assert normalize_symbol(spelling) == "GC=F"
    assert normalize_symbol(spelling) == normalize_symbol(spelling.lower())


def test_normalized_proxy_symbol_is_path_safe():
    assert safe_ticker_component(normalize_symbol("XAU/USD")) == "GC=F"


def test_proxy_labeling_distinction_preserved():
    """Normalization resolves to the futures proxy; the dataset layer keeps
    the spot identity as ``symbol`` and marks the data as a labelled proxy."""
    from datetime import datetime, timezone

    provider = YahooGoldProxyProvider()
    assert provider.capabilities.asset_kind is AssetKind.GOLD_FUTURES_PROXY
    assert provider.capabilities.is_proxy is True

    from unittest.mock import patch

    import pandas as pd

    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=pd.date_range("2026-03-02", periods=1, freq="1h", tz="UTC"),
    )

    class _Ticker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return frame.copy()

    with patch("yfinance.Ticker", _Ticker):
        dataset = provider.get_bars(
            "XAU/USD", TimeFrame.H1,
            datetime(2026, 3, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 8, tzinfo=timezone.utc),
        )
    assert dataset.meta.symbol == "XAU/USD"          # caller's spelling preserved
    assert dataset.meta.source_symbol == "GC=F"      # proxy actually queried
    assert dataset.meta.asset_kind is AssetKind.GOLD_FUTURES_PROXY
    assert dataset.meta.is_proxy is True
    assert "not broker XAUUSD spot" in dataset.meta.disclaimer

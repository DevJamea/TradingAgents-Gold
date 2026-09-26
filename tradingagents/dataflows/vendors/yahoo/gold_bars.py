"""Raw price-history fetch from Yahoo Finance (vendor layer).

Lives under ``dataflows`` on purpose: vendor libraries are imported only by the
data layer, where failures are raised as ``VendorError`` subclasses (see
``tests/test_layering.py``).  Gold-domain semantics (proxy labelling, H4
resampling, ``GoldBar`` construction) belong to ``tradingagents.gold.data`` and
build on this thin fetcher.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component


def _as_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC")
    return ts.tz_localize("UTC")


def fetch_history_bars(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> tuple[str, pd.DataFrame]:
    """Daily/intraday bars for ``symbol`` over ``[start, end]`` (inclusive).

    Returns ``(canonical_symbol, frame)`` where ``frame`` carries a UTC
    ``DatetimeIndex`` and Open/High/Low/Close/Volume columns.  ``symbol`` is
    resolved through the shared symbol table (XAUUSD → GC=F).  Empty history
    raises ``NoMarketDataError`` so the routing/gold layers treat it as a data
    condition, not an outage-shaped surprise.
    """
    canonical = normalize_symbol(symbol)
    safe_ticker_component(canonical)

    start_ts = _as_utc_timestamp(start)
    end_ts = _as_utc_timestamp(end)
    # yfinance `end` is exclusive; add one day so `end`'s bars are included.
    end_exclusive = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.Ticker(canonical).history(
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_exclusive,
        interval=interval,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        raise NoMarketDataError(symbol, canonical, f"no {interval} bars from Yahoo")

    frame = raw.copy()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC")
    else:
        frame.index = frame.index.tz_localize("UTC")
    frame = frame.rename_axis("Date").reset_index()
    return canonical, frame

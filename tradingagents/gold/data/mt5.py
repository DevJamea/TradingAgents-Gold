"""READ-ONLY MT5 market-data integration for XAUUSD (spec: MT5 data phase).

ABSOLUTE SAFETY RULE: this module is market-data/account-read only.  It must
never place/modify/close orders, modify positions, or call any MT5 trading or
execution API.  A source-level guard test
(``tests/test_gold_mt5_readonly.py``) fails the suite if an execution API
reference is introduced here.

MT5 APIs USED (reads only): ``initialize``, ``login`` (optional),
``shutdown``, ``symbols_get``, ``symbol_info``, ``symbol_select``,
``symbol_info_tick``, ``copy_rates_range``.
MT5 APIs NEVER USED: ``order_send``, ``order_check``, any ``ORDER_TYPE_*`` /
``TRADE_ACTION_*`` constant, ``positions_*``, ``orders_*``, ``history_*``.

Symbol discovery: the broker's actual Gold symbol is discovered from the
terminal using the gold vocabulary (XAUUSD / XAU / GOLD, with broker
decorations like ``XAUUSD.a``).  Silver (XAG/SILVER), futures-style (GC/FUT)
and unrelated instruments are excluded.  With zero candidates the discovery
fails; with MULTIPLE candidates it fails safely (``MT5SymbolAmbiguityError``)
requiring an explicit ``mt5_symbol`` — it never guesses.

Identity/provenance: every dataset records the CALLER's requested symbol
(``meta.symbol``, e.g. ``XAUUSD``) and the BROKER's actual symbol
(``meta.source_symbol``, e.g. ``XAUUSD.a``) with
``asset_kind=GOLD_SPOT_MT5`` and ``is_proxy=False`` — real broker spot, never
a futures proxy, never silently re-labelled.

Timestamps: MT5 reports bar/tick times on the broker server clock; they are
labelled UTC exactly as returned by the terminal API (no fabricated
conversion).  Cross-check your broker's server timezone for session-sensitive
analysis.

Bid/ask: only the CURRENT tick provides bid/ask (and spread = ask − bid).
Historical bars carry OHLC + tick volume; they are never padded with
historical bid/ask the source does not provide.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.gold.config import GoldDataConfig
from tradingagents.gold.data.calendar import ensure_utc, timeframe_minutes
from tradingagents.gold.data.models import DatasetMeta, GoldBar, GoldDataset, GoldQuote
from tradingagents.gold.data.provider import GoldMarketDataProvider, ProviderCapabilities
from tradingagents.gold.mt5_adapter import MT5SafetyError, MT5UnavailableError
from tradingagents.gold.types import AssetKind, DataKind, TimeFrame, utc_now

logger = logging.getLogger(__name__)

#: Gold-relevant MT5 timeframes ONLY (no M1/M5 scalping timeframes).
_TIMEFRAME_ATTRS = {
    TimeFrame.M15: "TIMEFRAME_M15",
    TimeFrame.H1: "TIMEFRAME_H1",
    TimeFrame.H4: "TIMEFRAME_H4",
}


class MT5SymbolDiscoveryError(RuntimeError):
    """No usable Gold symbol could be identified on this broker."""


class MT5SymbolAmbiguityError(MT5SymbolDiscoveryError):
    """Multiple Gold candidates exist — an explicit symbol choice is required."""


_DECORATION = re.compile(r"[^A-Z0-9]")
#: Tokens that mark a candidate as NOT the gold spot CFD we want.
_EXCLUDED_TOKENS = ("XAG", "SILVER", "GC", "FUT")


def _normalize_broker_symbol(name: str) -> str:
    """Broker spelling (``XAUUSD.a``, ``GOLD#``) → comparable base (``XAUUSDA``)."""
    return _DECORATION.sub("", str(name).upper())


def _looks_like_gold(normalized: str) -> bool:
    """Deterministic gold match on the normalized broker symbol."""
    if not normalized or any(token in normalized for token in _EXCLUDED_TOKENS):
        return False
    return (
        normalized.startswith("XAUUSD")
        or normalized.startswith("GOLD")
        or normalized == "XAU"
    )


def discover_gold_symbol(mt5, *, explicit: str | None = None) -> str:
    """Identify the broker's Gold symbol (read-only).

    ``explicit`` (when given) must be offered by the broker and is returned
    unchanged — the user's choice is never second-guessed.  Otherwise the
    terminal's symbol list is scanned; exactly one match is returned, any
    other outcome raises a clear error demanding an explicit selection.
    """
    listed = [str(s.name) for s in (mt5.symbols_get() or [])]
    if explicit is not None:
        if explicit in listed:
            return explicit
        raise MT5SymbolDiscoveryError(
            f"explicit symbol {explicit!r} is not offered by this broker "
            f"({len(listed)} symbols listed); check the terminal's symbol tree"
        )
    matched = sorted({_normalize_broker_symbol(name) and name for name in listed
                      if _looks_like_gold(_normalize_broker_symbol(name))})
    if len(matched) == 1:
        return matched[0]
    if not matched:
        raise MT5SymbolDiscoveryError(
            "no Gold symbol found on this broker (scanned "
            f"{len(listed)} symbols); pass the broker's gold symbol explicitly"
        )
    raise MT5SymbolAmbiguityError(
        "multiple Gold symbol candidates found: "
        f"{matched}. Fail-safe: pass mt5_symbol explicitly instead of guessing."
    )


class MT5ReadOnlyGoldProvider(GoldMarketDataProvider):
    """Read-only XAUUSD market data from a local MT5 terminal.

    Connects (optionally logging in), discovers or verifies the broker's gold
    symbol, and serves M15/H1/H4 bars plus the current bid/ask tick.  It has
    NO execution surface (enforced by test) and never writes to the terminal.
    """

    source_name = "mt5:readonly"

    def __init__(
        self,
        credentials=None,
        *,
        mt5_module=None,
        mt5_symbol: str | None = None,
        data_config: GoldDataConfig | None = None,
        auto_discover: bool = True,
    ) -> None:
        self._credentials = credentials
        self._mt5 = mt5_module
        self._mt5_symbol = mt5_symbol
        self._cfg = data_config or GoldDataConfig()
        self._auto_discover = auto_discover
        self._connected = False
        self._point: float | None = None

    # -- connection (reads only) ---------------------------------------------

    def _load(self):
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # noqa: PLC0415 — Windows-only package
            except ImportError as exc:
                raise MT5UnavailableError(
                    "MetaTrader5 package/terminal unavailable; the provider fails "
                    "closed (no data is fabricated)."
                ) from exc
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> str:
        """Initialize the terminal and resolve/verify the broker gold symbol."""
        mt5 = self._load()
        if not mt5.initialize():
            raise MT5UnavailableError(f"MT5 initialize failed: {mt5.last_error()}")
        if self._credentials is not None and not mt5.login(
            self._credentials.login, self._credentials.password, self._credentials.server,
        ):
            raise MT5SafetyError(f"MT5 login failed: {mt5.last_error()}")
        if self._mt5_symbol is None and self._auto_discover:
            self._mt5_symbol = discover_gold_symbol(mt5)
        if self._mt5_symbol is None:
            raise MT5SymbolDiscoveryError(
                "no MT5 gold symbol configured: enable auto_discover or pass mt5_symbol"
            )
        info = mt5.symbol_info(self._mt5_symbol)
        if info is None:
            raise MT5SafetyError(f"symbol {self._mt5_symbol!r} is not available on this broker")
        if not mt5.symbol_select(self._mt5_symbol, True):
            raise MT5SafetyError(f"symbol {self._mt5_symbol!r} could not be selected in the terminal")
        self._point = float(getattr(info, "point", 0.0) or 0.01)
        self._connected = True
        logger.info("MT5 read-only data connection ready for %s", self._mt5_symbol)
        return self._mt5_symbol

    def _ensure(self):
        if not self._connected:
            self.connect()
        return self._mt5

    def shutdown(self) -> None:
        if self._connected and self._mt5 is not None:
            self._mt5.shutdown()
        self._connected = False

    # -- GoldMarketDataProvider ----------------------------------------------

    @property
    def discovered_symbol(self) -> str | None:
        """The broker's actual gold symbol (None before first connect)."""
        return self._mt5_symbol

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            asset_kind=AssetKind.GOLD_SPOT_MT5,
            is_proxy=False,
            supported_timeframes=frozenset(_TIMEFRAME_ATTRS),
            supports_bid_ask=True,
            closed_bars_enforced=True,
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        *,
        include_forming: bool = False,
    ) -> GoldDataset:
        """Closed XAUUSD bars (spec: only completed candles enter analysis).

        Point-in-time hard guarantees, applied in this order:

        1. nothing after ``end`` is returned (``as_of`` cutoff);
        2. by default a bar still FORMING at ``end`` (open + timeframe > end)
           is excluded — its close/high/low do not exist yet and would smuggle
           post-``end`` information into the dataset.  Pass
           ``include_forming=True`` only to LABEL the currently-forming candle
           (the current-snapshot builder does this and then excludes it from
           the analysis datasets).
        """
        mt5 = self._ensure()
        attr = _TIMEFRAME_ATTRS[timeframe]
        const = getattr(mt5, attr, None)
        if const is None:
            raise MT5SafetyError(f"terminal does not expose {attr}")
        broker_symbol = self._mt5_symbol
        rates = mt5.copy_rates_range(broker_symbol, const, ensure_utc(start), ensure_utc(end))
        if rates is None or len(rates) == 0:
            raise NoMarketDataError(symbol, broker_symbol, f"no {timeframe.value} rates from MT5")

        end = ensure_utc(end)
        bar_span = timedelta(minutes=timeframe_minutes(timeframe))
        bars: list[GoldBar] = []
        for row in rates:
            ts = datetime.fromtimestamp(int(row["time"]), tz=timezone.utc)
            if ts > end:
                continue  # point-in-time hard guarantee: nothing after `end`
            if not include_forming and ts + bar_span > end:
                continue  # still forming at `end` — never used as a closed candle
            bars.append(
                GoldBar(
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["tick_volume"]),
                )
            )
        if not bars:
            raise NoMarketDataError(
                symbol, broker_symbol,
                f"no {timeframe.value} bars at or before the as_of cutoff {end.isoformat()}",
            )
        meta = DatasetMeta(
            source=self.source_name,
            symbol=symbol,                # requested identity, preserved
            source_symbol=broker_symbol,  # broker's actual symbol
            asset_kind=AssetKind.GOLD_SPOT_MT5,
            data_type=DataKind.OHLCV,
            timezone="UTC",
            timeframe=timeframe,
            is_proxy=False,               # real broker spot, not a proxy
            retrieved_at=utc_now(),
            disclaimer=None,
        )
        return GoldDataset(meta=meta, bars=tuple(bars))

    def get_quote(self, symbol: str) -> GoldQuote | None:
        """Current bid/ask tick; None when the terminal has no tick.

        Validity (positive prices, ask >= bid, non-zero spread) is enforced by
        the snapshot layer's ``quote_issues`` — invalid quotes become data
        errors, never fabricated spreads.
        """
        mt5 = self._ensure()
        tick = mt5.symbol_info_tick(self._mt5_symbol)
        if tick is None:
            return None
        return GoldQuote(
            symbol=symbol,
            source=self.source_name,
            asset_kind=AssetKind.GOLD_SPOT_MT5,
            timestamp=datetime.fromtimestamp(int(tick.time), tz=timezone.utc),
            bid=float(tick.bid),
            ask=float(tick.ask),
        )

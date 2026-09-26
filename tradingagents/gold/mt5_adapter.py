"""MT5 DEMO adapter (spec §25/§31 Phase 10) — the ONLY execution door, and it
opens for demo accounts with explicit authorization, nothing else.

Architecture rule (spec §17, absolute): agents never touch MT5.  The only
caller of :class:`MT5DemoExecutionAdapter` is the deterministic pipeline after
the risk gate, and the adapter re-verifies every safety condition itself
(fail-closed):

* the MetaTrader5 package must be importable (REQUIRES MT5 TERMINAL — the
  adapter raises :class:`MT5UnavailableError` otherwise, never fakes data);
* the account must verify as ``ACCOUNT_TRADE_MODE_DEMO`` — REAL and CONTEST
  accounts are refused unconditionally (no flag can change this in v1);
* execution must be explicitly enabled (``execution_enabled=True``), and the
  default is disabled;
* the symbol must verify (``symbol_select`` + ``symbol_info``);
* the incoming order must carry an APPROVED deterministic gate result, and the
  adapter re-checks the numeric limits itself before sending;
* the broker result must be ``TRADE_RETCODE_DONE`` — anything else is an
  error record, never a silent half-fill.

Data side: :class:`MT5GoldProvider` adapts broker bars/ticks into the same
provenance-carrying gold datasets (real bid/ask spread at last).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from tradingagents.gold.config import GoldDataConfig, RiskLimits
from tradingagents.gold.data.models import (
    DatasetMeta,
    GoldBar,
    GoldDataset,
    GoldQuote,
)
from tradingagents.gold.data.provider import GoldMarketDataProvider, ProviderCapabilities
from tradingagents.gold.decision import GoldDecision
from tradingagents.gold.risk_gate import AccountState
from tradingagents.gold.types import AssetKind, DataKind, TimeFrame, utc_now

logger = logging.getLogger(__name__)

#: MetaTrader5 account trade modes (mirrored without importing the package).
ACCOUNT_TRADE_MODE_DEMO = 0
ACCOUNT_TRADE_MODE_CONTEST = 1
ACCOUNT_TRADE_MODE_REAL = 2

TRADE_RETCODE_DONE = 10009

#: MetaTrader5 timeframe constants (same values as the package's enums).
_MT5_TIMEFRAMES = {TimeFrame.M15: 15, TimeFrame.H1: 16385, TimeFrame.H4: 16388}

#: Broker symbol for XAUUSD in MT5 (brokers differ; configurable).
DEFAULT_MT5_SYMBOL = "XAUUSD"


class MT5UnavailableError(RuntimeError):
    """The MetaTrader5 package/terminal is not available.  Fail-closed."""


class MT5SafetyError(PermissionError):
    """A safety precondition failed.  No order is ever sent."""


@dataclass(frozen=True)
class MT5Credentials:
    login: int
    password: str
    server: str


def _load_mt5():
    try:
        import MetaTrader5 as mt5  # noqa: PLC0415 — lazily, Windows-only package
    except ImportError as exc:
        raise MT5UnavailableError(
            "MetaTrader5 package/terminal unavailable; the adapter fails closed "
            "(no data is fabricated and no order is possible)."
        ) from exc
    return mt5


# ---------------------------------------------------------------------------
# Data adapter
# ---------------------------------------------------------------------------


class MT5GoldProvider(GoldMarketDataProvider):
    """Broker-side XAUUSD market data (bars with real bid/ask via tick)."""

    source_name = "mt5:demo"

    def __init__(
        self,
        credentials: MT5Credentials,
        mt5_symbol: str = DEFAULT_MT5_SYMBOL,
        data_config: GoldDataConfig | None = None,
        mt5_module=None,                # injectable for tests
    ) -> None:
        self._credentials = credentials
        self._mt5_symbol = mt5_symbol
        self._cfg = data_config or GoldDataConfig()
        self._mt5 = mt5_module
        self._connected = False

    # -- connection ---------------------------------------------------------

    def connect(self):
        mt5 = self._mt5 or _load_mt5()
        if not mt5.initialize():
            raise MT5UnavailableError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.login(self._credentials.login, self._credentials.password,
                         self._credentials.server):
            raise MT5SafetyError(f"MT5 login failed: {mt5.last_error()}")
        account = mt5.account_info()
        if account is None:
            raise MT5SafetyError("MT5 account_info unavailable")
        if account.trade_mode != ACCOUNT_TRADE_MODE_DEMO:
            raise MT5SafetyError(
                "MT5 account is not a DEMO account — refused (real accounts are "
                "unsupported by design; spec §25)."
            )
        self._connected = True
        self._mt5 = mt5
        return account

    def _ensure_connected(self):
        if not self._connected:
            self.connect()
        return self._mt5

    def shutdown(self):
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
        self._connected = False

    # -- GoldMarketDataProvider ----------------------------------------------

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            asset_kind=AssetKind.GOLD_SPOT,
            is_proxy=False,
            supported_timeframes=frozenset(tf.value for tf in _MT5_TIMEFRAMES),
            supports_bid_ask=True,
        )

    def _verify_symbol(self, mt5):
        info = mt5.symbol_info(self._mt5_symbol)
        if info is None:
            raise MT5SafetyError(f"unknown MT5 symbol {self._mt5_symbol!r}")
        if not mt5.symbol_select(self._mt5_symbol, True):
            raise MT5SafetyError(f"MT5 symbol {self._mt5_symbol!r} could not be selected")
        return info

    def get_bars(self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime):
        mt5 = self._ensure_connected()
        self._verify_symbol(mt5)
        rates = mt5.copy_rates_range(self._mt5_symbol, _MT5_TIMEFRAMES[timeframe], start, end)
        if rates is None or len(rates) == 0:
            from tradingagents.dataflows.errors import NoMarketDataError

            raise NoMarketDataError(symbol, self._mt5_symbol,
                                    f"no {timeframe.value} rates from MT5")
        bars = tuple(
            GoldBar(
                timestamp=datetime.fromtimestamp(row["time"], tz=utc_now().tzinfo),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["tick_volume"]),
            )
            for row in rates
        )
        meta = DatasetMeta(
            source=self.source_name,
            symbol=symbol,
            source_symbol=self._mt5_symbol,
            asset_kind=AssetKind.GOLD_SPOT,
            data_type=DataKind.OHLCV,
            timezone="UTC",
            timeframe=timeframe,
            is_proxy=False,
            retrieved_at=utc_now(),
            disclaimer=None,
        )
        return GoldDataset(meta=meta, bars=bars)

    def get_quote(self, symbol: str) -> GoldQuote | None:
        mt5 = self._ensure_connected()
        self._verify_symbol(mt5)
        tick = mt5.symbol_info_tick(self._mt5_symbol)
        if tick is None:
            return None
        return GoldQuote(
            symbol=symbol,
            source=self.source_name,
            asset_kind=AssetKind.GOLD_SPOT,
            timestamp=datetime.fromtimestamp(tick.time, tz=utc_now().tzinfo),
            bid=float(tick.bid),
            ask=float(tick.ask),
        )


# ---------------------------------------------------------------------------
# Execution adapter (demo only)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionRecord:
    """Auditable outcome of one MT5 demo order attempt."""

    decision_id: str
    requested_at: datetime
    executed: bool
    reason: str
    order_ticket: int | None = None
    volume_lots: float | None = None
    price_sent: float | None = None
    details: dict = field(default_factory=dict)


class MT5DemoExecutionAdapter:
    """Sends demo orders — and nothing else — after exhaustive re-checks."""

    def __init__(
        self,
        credentials: MT5Credentials,
        *,
        execution_enabled: bool = False,
        mt5_symbol: str = DEFAULT_MT5_SYMBOL,
        risk_limits: RiskLimits | None = None,
        contract_size_oz: float = 100.0,
        mt5_module=None,                # injectable for tests
        deviation: int = 20,
    ) -> None:
        self.credentials = credentials
        self.execution_enabled = execution_enabled
        self.mt5_symbol = mt5_symbol
        self.risk_limits = risk_limits or RiskLimits()
        self.contract_size_oz = contract_size_oz
        self.deviation = deviation
        self._mt5 = mt5_module
        self.records: list[ExecutionRecord] = []

    # -- safety --------------------------------------------------------------

    def _connect_and_verify_demo(self):
        if not self.execution_enabled:
            raise MT5SafetyError("execution is disabled by default; enable explicitly for demo")
        mt5 = self._mt5 or _load_mt5()
        if not mt5.initialize():
            raise MT5UnavailableError(f"MT5 initialize failed: {mt5.last_error()}")
        if not mt5.login(self.credentials.login, self.credentials.password,
                         self.credentials.server):
            raise MT5SafetyError(f"MT5 login failed: {mt5.last_error()}")
        account = mt5.account_info()
        if account is None:
            raise MT5SafetyError("MT5 account_info unavailable")
        if account.trade_mode != ACCOUNT_TRADE_MODE_DEMO:
            raise MT5SafetyError(
                "refused: MT5 account is not DEMO (real accounts are unsupported by design)"
            )
        return mt5, account

    # -- order path ------------------------------------------------------------

    def send_order(
        self,
        decision: GoldDecision,
        gate_result,
        account_state: AccountState,
        *,
        now: datetime | None = None,
    ) -> ExecutionRecord:
        """Send one demo order for a gate-approved decision.  Fail-closed."""
        now = now or utc_now()
        record = ExecutionRecord(decision_id=decision.decision_id, requested_at=now, executed=False,
                                 reason="")
        try:
            mt5, account = self._connect_and_verify_demo()
            info = mt5.symbol_info(self.mt5_symbol)
            if info is None or not mt5.symbol_select(self.mt5_symbol, True):
                raise MT5SafetyError(f"symbol {self.mt5_symbol!r} failed verification")

            # Independent re-validation (defense in depth — never trust the caller).
            from tradingagents.gold.risk_gate import evaluate_decision as _recheck

            recheck = _recheck(decision, _MT5GateContextShim(), account_state,
                               limits=self.risk_limits)
            if not recheck.approved:
                raise MT5SafetyError(f"gate re-check rejected the order: {recheck.reasons}")
            if recheck.position_units != gate_result.position_units:
                raise MT5SafetyError("position size disagrees with the gate result")

            volume_lots = recheck.position_units / self.contract_size_oz
            lot_step = getattr(info, "volume_step", 0.01) or 0.01
            volume_lots = max(lot_step, round(volume_lots / lot_step) * lot_step)

            is_buy = decision.action.value == "BUY"
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.mt5_symbol,
                "volume": float(volume_lots),
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": float(tick_price(mt5, self.mt5_symbol, is_buy)),
                "sl": float(decision.stop_loss),
                "tp": float(decision.take_profit),
                "deviation": self.deviation,
                "magic": 20260918,          # gold system tag
                "comment": f"gold:{decision.decision_id}",
                "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 0),
            }
            result = mt5.order_send(request)
            if result is None:
                raise MT5SafetyError(f"order_send returned None: {mt5.last_error()}")
            if result.retcode != TRADE_RETCODE_DONE:
                record.reason = f"order rejected by broker retcode={result.retcode}"
                record.details = {"retcode": result.retcode, "comment": result.comment}
                self.records.append(record)
                logger.warning("MT5 demo order not executed: %s", record.reason)
                return record

            record.executed = True
            record.reason = "demo order executed"
            record.order_ticket = result.order
            record.volume_lots = volume_lots
            record.price_sent = request["price"]
            record.details = {"retcode": result.retcode, "request": {k: request[k] for k in
                                                                     ("symbol", "volume", "sl", "tp", "type")}}
            logger.info("MT5 DEMO order executed: ticket=%s lots=%.2f", result.order, volume_lots)
            return record
        except Exception as exc:  # noqa: BLE001 — fail-closed with an audit record
            record.reason = f"{type(exc).__name__}: {exc}"
            self.records.append(record)
            logger.warning("MT5 demo order refused: %s", record.reason)
            return record


def tick_price(mt5, symbol: str, is_buy: bool) -> float:
    """The executable side's price: ask for buys, bid for sells."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise MT5SafetyError(f"no tick available for {symbol!r}")
    return float(tick.ask) if is_buy else float(tick.bid)


class _MT5GateContextShim:
    """Minimal context for the adapter's independent gate re-check.

    The full data-quality gate ran in the pipeline; the adapter re-checks the
    numeric risk rules (geometry, R:R, sizing, portfolio limits) without
    re-fetching market data.
    """

    def __init__(self) -> None:
        self.snapshot = None
        self.data_errors: list[str] = []

    @property
    def has_data_error(self) -> bool:
        return False

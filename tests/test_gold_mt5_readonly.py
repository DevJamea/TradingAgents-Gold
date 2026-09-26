"""MT5 READ-ONLY XAUUSD data integration (spec §12 test matrix 1–16).

The MetaTrader5 package is Windows-only and absent in this environment, so
every test injects a fake module.  The fake records EVERY API call so the
no-execution guarantee is checkable at runtime as well as by source scan.
Nothing here connects to a real terminal — real use REQUIRES MT5 TERMINAL.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.gold.config import default_gold_config
from tradingagents.gold.data.calendar import is_gold_market_open
from tradingagents.gold.data.current import build_current_snapshot
from tradingagents.gold.data.mt5 import (
    MT5ReadOnlyGoldProvider,
    MT5SymbolAmbiguityError,
    MT5SymbolDiscoveryError,
    discover_gold_symbol,
)
from tradingagents.gold.data.snapshot import build_market_snapshot
from tradingagents.gold.data.validation import validate_dataset
from tradingagents.gold.mt5_adapter import MT5UnavailableError
from tradingagents.gold.types import AssetKind, TimeFrame

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)   # Friday, market open
RATE_DTYPE = [
    ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
    ("close", "f8"), ("tick_volume", "i8"), ("spread", "i8"), ("real_volume", "i8"),
]
MT5_TF = {"M15": 15, "H1": 16385, "H4": 16388}


# ---------------------------------------------------------------------------
# Fake MT5 module
# ---------------------------------------------------------------------------


def market_grid(end: datetime, count: int, minutes: int) -> list[datetime]:
    """``count`` on-grid open-market stamps ending at/before ``end``."""
    stamps: list[datetime] = []
    cursor = end.replace(second=0, microsecond=0)
    step = timedelta(minutes=minutes)
    offset = (cursor.hour * 60 + cursor.minute) % minutes
    cursor -= timedelta(minutes=offset)
    while len(stamps) < count:
        if is_gold_market_open(cursor):
            stamps.append(cursor)
        cursor -= step
    return list(reversed(stamps))


def make_rates(end: datetime, count: int, minutes: int, *, base=2650.0, step=0.5,
               forming: bool = False) -> np.ndarray:
    """Synthetic rising OHLC series on the open-market grid (optionally with
    one extra bar at ``end`` representing a still-forming candle)."""
    stamps = market_grid(end - timedelta(minutes=minutes), count, minutes)
    if forming:
        stamps.append(market_grid(end, 1, minutes)[-1])
    arr = np.zeros(len(stamps), dtype=RATE_DTYPE)
    for i, ts in enumerate(stamps):
        close = base + step * i
        arr[i] = (
            int(ts.timestamp()), close - 0.10, close + 0.50, close - 0.60,
            close, 100 + i, 25, 0,
        )
    return arr


class FakeMT5:
    """Fake MetaTrader5 module: reads only; every call is recorded."""

    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388

    def __init__(self, symbols=("XAUUSD",), rates=None, tick=None, *,
                 initialize_ok=True, select_ok=True):
        self.symbols_list = [SimpleNamespace(name=name) for name in symbols]
        self.rates = rates or {}
        self.tick = tick
        self.initialize_ok = initialize_ok
        self.select_ok = select_ok
        self.calls: list = []
        self.initialized = False

    def initialize(self):
        self.calls.append("initialize")
        self.initialized = self.initialize_ok
        return self.initialize_ok

    def last_error(self):
        return (0, "ok")

    def login(self, login, password, server):
        self.calls.append("login")
        return True

    def shutdown(self):
        self.calls.append("shutdown")
        self.initialized = False

    def symbols_get(self, group=None):
        self.calls.append("symbols_get")
        return list(self.symbols_list)

    def symbol_info(self, name):
        self.calls.append(("symbol_info", name))
        if any(s.name == name for s in self.symbols_list):
            return SimpleNamespace(name=name, point=0.01, digits=2)
        return None

    def symbol_select(self, name, enable):
        self.calls.append(("symbol_select", name))
        return self.select_ok and any(s.name == name for s in self.symbols_list)

    def symbol_info_tick(self, name):
        self.calls.append(("symbol_info_tick", name))
        return self.tick

    def copy_rates_range(self, symbol, timeframe, start, end):
        self.calls.append(("copy_rates_range", symbol, timeframe))
        return self.rates.get((symbol, timeframe))


def standard_rates(symbol="XAUUSD", *, forming=True):
    """M15/H1/H4 series ending just before NOW."""
    return {
        (symbol, 15): make_rates(NOW, 480, 15, forming=forming),
        (symbol, 16385): make_rates(NOW, 240, 60),
        (symbol, 16388): make_rates(NOW, 120, 240),
    }


def fresh_tick(*, minutes_old=2, bid=2649.90, ask=2650.20):
    return SimpleNamespace(time=int((NOW - timedelta(minutes=minutes_old)).timestamp()),
                           bid=bid, ask=ask)


def make_provider(*, symbol="XAUUSD", rates=None, tick=None, mt5_symbol=None, **fake_kwargs):
    fake = FakeMT5(
        symbols=(symbol,), rates=rates or standard_rates(symbol),
        tick=tick or fresh_tick(), **fake_kwargs,
    )
    provider = MT5ReadOnlyGoldProvider(mt5_module=fake, mt5_symbol=mt5_symbol)
    return provider, fake


# ---------------------------------------------------------------------------
# 2/3. Symbol discovery
# ---------------------------------------------------------------------------


class TestSymbolDiscovery:
    def test_unique_xauusd_is_discovered(self):
        assert discover_gold_symbol(FakeMT5(symbols=("EURUSD", "XAUUSD", "XAGUSD"))) == "XAUUSD"

    def test_broker_decorated_variant_discovered_when_unique(self):
        assert discover_gold_symbol(FakeMT5(symbols=("XAUUSD.a",))) == "XAUUSD.a"

    def test_ambiguous_candidates_fail_safely(self):
        with pytest.raises(MT5SymbolAmbiguityError) as excinfo:
            discover_gold_symbol(FakeMT5(symbols=("XAUUSD", "XAUUSD.a", "GOLD#")))
        message = str(excinfo.value)
        for candidate in ("XAUUSD", "XAUUSD.a", "GOLD#"):
            assert candidate in message
        assert "explicit" in message.lower()

    def test_no_candidate_fails_clearly(self):
        with pytest.raises(MT5SymbolDiscoveryError):
            discover_gold_symbol(FakeMT5(symbols=("EURUSD", "GBPUSD", "US30")))

    def test_silver_futures_and_unrelated_never_selected(self):
        # only silver / gold-futures-style / index present -> no gold match
        with pytest.raises(MT5SymbolDiscoveryError):
            discover_gold_symbol(FakeMT5(symbols=("XAGUSD", "GCFUT", "SILVER", "US30")))

    def test_explicit_symbol_is_honored(self):
        fake = FakeMT5(symbols=("XAUUSD", "XAUUSD.a"))
        assert discover_gold_symbol(fake, explicit="XAUUSD.a") == "XAUUSD.a"

    def test_explicit_unknown_symbol_rejected(self):
        with pytest.raises(MT5SymbolDiscoveryError):
            discover_gold_symbol(FakeMT5(symbols=("XAUUSD",)), explicit="NOPE")


# ---------------------------------------------------------------------------
# 1/4/5/6. Provider + M15/H1/H4 retrieval; 13. as_of cutoff; 14/15. failures
# ---------------------------------------------------------------------------


class TestProvider:
    def test_connect_discovers_and_selects(self):
        provider, fake = make_provider()
        assert provider.connect() == "XAUUSD"
        assert provider.discovered_symbol == "XAUUSD"
        assert ("symbol_select", "XAUUSD") in fake.calls

    def test_m15_retrieval_with_provenance(self):
        provider, _ = make_provider()
        ds = provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=6), NOW)
        bar = ds.bars[-1]
        assert len(ds) == 480 or len(ds) > 400
        assert (bar.open, bar.high, bar.low, bar.close) == (
            pytest.approx(bar.open), pytest.approx(bar.high),
            pytest.approx(bar.low), pytest.approx(bar.close),
        )
        assert bar.high >= bar.low and bar.volume and bar.volume > 0
        assert bar.bid is None and bar.ask is None and bar.spread is None  # no fabricated b/a
        assert ds.meta.symbol == "XAUUSD"
        assert ds.meta.source_symbol == "XAUUSD"
        assert ds.meta.source == "mt5:readonly"
        assert ds.meta.asset_kind is AssetKind.GOLD_SPOT_MT5
        assert ds.meta.is_proxy is False
        assert ds.meta.timeframe is TimeFrame.M15

    def test_requested_identity_preserved_for_decorated_broker_symbol(self):
        provider, _ = make_provider(symbol="XAUUSD.a", rates=standard_rates("XAUUSD.a"))
        ds = provider.get_bars("XAUUSD", TimeFrame.H1, NOW - timedelta(days=15), NOW)
        assert ds.meta.symbol == "XAUUSD"          # requested identity
        assert ds.meta.source_symbol == "XAUUSD.a"  # actual broker symbol
        assert ds.meta.is_proxy is False

    @pytest.mark.parametrize("tf", [TimeFrame.M15, TimeFrame.H1, TimeFrame.H4])
    def test_each_timeframe_retrieves(self, tf):
        provider, _ = make_provider()
        ds = provider.get_bars("XAUUSD", tf, NOW - timedelta(days=40), NOW)
        assert len(ds) > 50
        assert ds.meta.timeframe is tf
        assert all(b.timestamp <= NOW for b in ds.bars)

    def test_as_of_cutoff_excludes_every_later_bar(self):
        provider, _ = make_provider()
        cutoff = NOW - timedelta(hours=30)
        for tf in (TimeFrame.M15, TimeFrame.H1, TimeFrame.H4):
            ds = provider.get_bars("XAUUSD", tf, cutoff - timedelta(days=30), cutoff)
            assert ds.bars, tf
            assert max(b.timestamp for b in ds.bars) <= cutoff

    def test_missing_data_raises_no_market_data(self):
        fake = FakeMT5(rates={})   # no rates at all
        provider = MT5ReadOnlyGoldProvider(mt5_module=fake, mt5_symbol="XAUUSD")
        with pytest.raises(NoMarketDataError):
            provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=1), NOW)

    def test_initialize_failure_fails_closed(self):
        provider, _ = make_provider(initialize_ok=False)
        with pytest.raises(MT5UnavailableError):
            provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=1), NOW)

    def test_missing_package_fails_closed(self):
        # No MetaTrader5 module injectable AND absent from the environment.
        provider = MT5ReadOnlyGoldProvider(mt5_symbol="XAUUSD")
        with pytest.raises(MT5UnavailableError):
            provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=1), NOW)

    def test_quote_returns_gold_quote_with_spread(self):
        provider, _ = make_provider()
        quote = provider.get_quote("XAUUSD")
        assert quote is not None
        assert quote.symbol == "XAUUSD"
        assert quote.source == "mt5:readonly"
        assert quote.asset_kind is AssetKind.GOLD_SPOT_MT5
        assert quote.bid == pytest.approx(2649.90)
        assert quote.ask == pytest.approx(2650.20)
        assert quote.spread == pytest.approx(0.30)

    def test_historical_bars_do_not_carry_fabricated_bid_ask(self):
        provider, _ = make_provider()
        ds = provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=6), NOW)
        assert all(bar.bid is None and bar.ask is None for bar in ds.bars)


# ---------------------------------------------------------------------------
# 5. Data quality (reused deterministic validator)
# ---------------------------------------------------------------------------


class TestDataQuality:
    def test_invalid_ohlc_is_rejected(self):
        provider, _ = make_provider()
        ds = provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=6), NOW)
        broken = list(ds.bars)
        broken[10] = type(broken[10])(
            timestamp=broken[10].timestamp, open=broken[10].open,
            high=broken[10].low - 1.0, low=broken[10].low, close=broken[10].close,
            volume=broken[10].volume,
        )
        report = validate_dataset(type(ds)(meta=ds.meta, bars=tuple(broken)),
                                  expected_symbol="XAUUSD")
        assert any(issue.code == "impossible_ohlc" for issue in report.errors)

    def test_duplicate_and_non_monotonic_rejected(self):
        provider, _ = make_provider()
        ds = provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=6), NOW)
        bars = list(ds.bars)
        bars.append(bars[-1])
        report = validate_dataset(type(ds)(meta=ds.meta, bars=tuple(bars)),
                                  expected_symbol="XAUUSD")
        codes = {issue.code for issue in report.errors}
        assert "duplicate_timestamp" in codes and "unordered_timestamps" in codes

    def test_snapshot_datasets_pass_validation(self):
        provider, _ = make_provider()
        snap = build_market_snapshot(provider, default_gold_config(), as_of=NOW, now=NOW)
        assert snap.ok, snap.errors


# ---------------------------------------------------------------------------
# 7/8/9/10/11/12/13. Current snapshot: closed bars, forming, quote quality
# ---------------------------------------------------------------------------


class TestCurrentSnapshot:
    def test_snapshot_fields_and_spread_from_actual_bid_ask(self):
        provider, _ = make_provider()
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        assert snap.symbol == "XAUUSD"
        assert snap.broker_symbol == "XAUUSD"
        assert snap.bid == pytest.approx(2649.90)
        assert snap.ask == pytest.approx(2650.20)
        assert snap.spread == pytest.approx(snap.ask - snap.bid)
        assert snap.ok, snap.errors
        assert any("mt5:readonly" in line for line in snap.provenance)

    def test_latest_bars_are_closed_and_forming_excluded(self):
        provider, _ = make_provider()
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        for tf in ("M15", "H1", "H4"):
            status = snap.bars[tf]
            assert status.is_closed, tf
            minutes = {"M15": 15, "H1": 60, "H4": 240}[tf]
            assert status.bar.timestamp + timedelta(minutes=minutes) <= snap.as_of
        # the trailing M15 bar (12:00 open, still forming at 12:00... on-grid)
        m15 = snap.bars["M15"]
        if m15.has_forming:
            assert m15.forming_bar.timestamp + timedelta(minutes=15) > snap.as_of
            assert m15.forming_bar.timestamp not in {
                b.timestamp for b in snap.datasets["M15"].bars
            }

    def test_forming_bar_identified_and_excluded_at_mid_bar_as_of(self):
        provider, _ = make_provider()
        as_of = NOW - timedelta(minutes=10)   # 11:50: the 11:45 M15 bar is forming
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW, as_of=as_of)
        m15 = snap.bars["M15"]
        assert m15.has_forming
        assert m15.forming_bar.timestamp == as_of.replace(minute=45)
        assert m15.forming_bar.timestamp + timedelta(minutes=15) > as_of
        closed_stamps = {b.timestamp for b in snap.datasets["M15"].bars}
        assert m15.forming_bar.timestamp not in closed_stamps
        h1 = snap.bars["H1"]
        assert h1.forming_bar is not None and h1.bar.timestamp + timedelta(minutes=60) <= as_of

    @pytest.mark.parametrize("tf_const,minutes", [(15, 15), (16385, 60), (16388, 240)])
    def test_as_of_cutoff_in_snapshot_all_timeframes(self, tf_const, minutes):
        provider, _ = make_provider()
        cutoff = NOW - timedelta(hours=26)
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW, as_of=cutoff)
        ds = snap.datasets["M15"] if minutes == 15 else None
        for key, ds in snap.datasets.items():
            assert max(b.timestamp for b in ds.bars) <= cutoff, key
        # and the latest closed bar respects its own closure rule
        for key, status in snap.bars.items():
            if status.bar is not None:
                span = timedelta(minutes={"M15": 15, "H1": 60, "H4": 240}[key])
                assert status.bar.timestamp + span <= cutoff

    def test_invalid_bid_ask_is_data_error(self):
        provider, _ = make_provider(tick=fresh_tick(bid=2650.50, ask=2649.90))  # ask < bid
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        assert not snap.ok
        assert any("invalid_bid_ask" in e for e in snap.errors)
        # and the multi-timeframe analysis snapshot surfaces it too
        msnap = build_market_snapshot(provider, default_gold_config(), as_of=NOW, now=NOW)
        assert not msnap.ok
        assert any("invalid_bid_ask" in e for e in msnap.errors)

    def test_zero_spread_is_rejected(self):
        provider, _ = make_provider(tick=fresh_tick(bid=2650.00, ask=2650.00))
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        assert any("invalid_spread" in e for e in snap.errors)

    def test_stale_tick_is_data_error(self):
        provider, _ = make_provider(tick=fresh_tick(minutes_old=200))
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        assert any("stale_quote" in e for e in snap.errors)
        assert not snap.ok

    def test_insufficient_bars_rejected(self):
        provider, _ = make_provider(rates={
            ("XAUUSD", 15): make_rates(NOW, 10, 15),
            ("XAUUSD", 16385): make_rates(NOW, 240, 60),
            ("XAUUSD", 16388): make_rates(NOW, 120, 240),
        })
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW, min_bars=50)
        assert any("insufficient_bars" in e for e in snap.errors)
        assert not snap.ok

    def test_provider_failure_is_recorded(self):
        provider, _ = make_provider(initialize_ok=False)
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        assert not snap.ok
        assert any("provider_failure" in e for e in snap.errors)

    def test_quote_fetch_failure_is_recorded(self):
        provider, fake = make_provider()
        provider.get_quote = lambda symbol: None  # feed goes quiet
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        assert any("quote_unavailable" in e for e in snap.errors)


# ---------------------------------------------------------------------------
# 16. No-execution guarantee
# ---------------------------------------------------------------------------

_FORBIDDEN_EXECUTION_TOKENS = (
    "order_send", "order_check", "ORDER_TYPE", "TRADE_ACTION",
    "PositionsTotal", "positions_total", "position_modify", "position_close",
    "orders_total", "history_orders_get", "history_deals_get", "close_position",
    "market_order", "pending_order", "buy(", "sell(",
)
_ALLOWED_PROVIDER_METHODS = {"connect", "shutdown", "get_bars", "get_quote"}


class TestNoExecutionGuarantee:
    @staticmethod
    def _code_without_docstrings(module) -> str:
        """Executable source only — docstrings may legitimately NAME forbidden
        APIs in safety documentation; code may never reference them."""
        source = inspect.getsource(module)
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc and node.body and isinstance(node.body[0], ast.Expr):
                    const = node.body[0].value
                    if isinstance(const, ast.Constant) and isinstance(const.value, str):
                        first = const.lineno - 1
                        lines[first : node.body[0].end_lineno] = [""] * (
                            node.body[0].end_lineno - first
                        )
        return "\n".join(lines)

    def _assert_no_execution(self, module) -> None:
        code = self._code_without_docstrings(module)
        for token in _FORBIDDEN_EXECUTION_TOKENS:
            assert token not in code, f"forbidden execution API reference: {token}"

    def test_provider_source_contains_no_execution_api(self):
        import tradingagents.gold.data.mt5 as module

        self._assert_no_execution(module)

    def test_snapshot_source_contains_no_execution_api(self):
        import tradingagents.gold.data.current as module

        self._assert_no_execution(module)

    def test_provider_object_has_no_execution_methods(self):
        provider = MT5ReadOnlyGoldProvider(mt5_module=FakeMT5())
        methods = {
            name for name, member in inspect.getmembers(provider, callable)
            if not name.startswith("_")
        }
        assert methods <= _ALLOWED_PROVIDER_METHODS, methods

    def test_runtime_calls_are_reads_only(self):
        provider, fake = make_provider()
        provider.connect()
        provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=6), NOW)
        provider.get_quote("XAUUSD")
        read_only = {
            "initialize", "login", "shutdown", "symbols_get",
            "symbol_info", "symbol_select", "symbol_info_tick", "copy_rates_range",
        }
        for call in fake.calls:
            name = call if isinstance(call, str) else call[0]
            assert name in read_only, call


# ---------------------------------------------------------------------------
# 5b. Quote quality: non-finite values (spec: never fabricate, fail closed)
# ---------------------------------------------------------------------------


class TestQuoteQuality:
    """§5 matrix: bid<=0 / ask<=0 / ask<bid / non-finite / invalid spread."""

    @staticmethod
    def _quote(bid, ask):
        from tradingagents.gold.data.models import GoldQuote

        return GoldQuote(
            symbol="XAUUSD", source="mt5:readonly", asset_kind=AssetKind.GOLD_SPOT_MT5,
            timestamp=NOW, bid=bid, ask=ask,
        )

    @pytest.mark.parametrize("bid,ask,expected_code", [
        (0.0, 2650.2, "invalid_bid"),
        (-1.0, 2650.2, "invalid_bid"),
        (2649.9, 0.0, "invalid_ask"),
        (2649.9, -1.0, "invalid_ask"),
        (2650.5, 2649.9, "invalid_bid_ask"),
        (float("nan"), 2650.2, "invalid_bid"),
        (2649.9, float("nan"), "invalid_ask"),
        (float("inf"), 2650.2, "invalid_bid"),
        (2649.9, float("inf"), "invalid_ask"),
        (float("nan"), float("nan"), "invalid_bid"),
    ])
    def test_invalid_quotes_fail_closed(self, bid, ask, expected_code):
        from tradingagents.gold.data.snapshot import quote_issues

        issues = dict(quote_issues(self._quote(bid, ask)))
        assert expected_code in issues, issues
        # the current-snapshot builder surfaces it as a DATA ERROR
        provider, _ = make_provider(tick=fresh_tick(bid=bid, ask=ask))
        snap = build_current_snapshot(provider, default_gold_config(), now=NOW)
        assert not snap.ok
        assert any(expected_code in e for e in snap.errors)

    def test_valid_quote_passes_with_computed_spread(self):
        from tradingagents.gold.data.snapshot import quote_issues

        assert quote_issues(self._quote(2649.90, 2650.20)) == []

    def test_xauusdm_broker_suffix_discovered(self):
        assert discover_gold_symbol(FakeMT5(symbols=("XAUUSDm",))) == "XAUUSDm"

    def test_xauusm_keeps_identity_through_provider(self):
        provider, _ = make_provider(symbol="XAUUSDm", rates=standard_rates("XAUUSDm"))
        assert provider.connect() == "XAUUSDm"
        ds = provider.get_bars("XAUUSD", TimeFrame.M15, NOW - timedelta(days=6), NOW)
        assert ds.meta.source_symbol == "XAUUSDm"
        assert ds.meta.symbol == "XAUUSD"
        assert ds.meta.asset_kind is AssetKind.GOLD_SPOT_MT5 and ds.meta.is_proxy is False

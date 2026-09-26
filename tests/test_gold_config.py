"""Phase 1 — Gold configuration layer: defaults, cost modes, risk limits.

The Gold config must be additive: it must not perturb upstream defaults or the
existing test baseline.
"""

import pytest

from tradingagents.gold.config import (
    COST_PROFILES,
    GOLD_SYMBOL_ALIASES,
    ExecutionAssumptions,
    GoldConfig,
    GoldDataConfig,
    GoldNewsConfig,
    RiskLimits,
    default_gold_config,
    is_gold_symbol,
)
from tradingagents.gold.types import CostMode, TimeFrame


class TestGoldConfigDefaults:
    def test_default_is_standard_cost_mode(self):
        cfg = default_gold_config()
        assert cfg.cost_mode is CostMode.STANDARD
        assert cfg.cost_profile.analysts == ("market", "macro", "news", "social")

    def test_instrument_identity_spots_vs_proxy(self):
        cfg = default_gold_config()
        assert cfg.data.symbol == "XAUUSD"
        assert cfg.data.proxy_symbol == "GC=F"
        assert "PROXY" in cfg.data.proxy_disclaimer
        assert "not broker XAUUSD spot" in cfg.data.proxy_disclaimer

    def test_primary_timeframes_are_m15_h1_h4(self):
        cfg = default_gold_config()
        assert cfg.data.timeframes == (TimeFrame.M15, TimeFrame.H1, TimeFrame.H4)
        assert set(cfg.data.stale_after_minutes) == {"M15", "H1", "H4"}
        assert set(cfg.data.history_bars) == {"M15", "H1", "H4"}

    def test_timezone_is_utc(self):
        assert default_gold_config().data.timezone == "UTC"


class TestCostModes:
    @pytest.mark.parametrize("mode", list(CostMode))
    def test_every_mode_has_a_profile(self, mode):
        profile = COST_PROFILES[mode]
        assert profile.analysts
        assert profile.max_debate_rounds >= 1
        assert profile.max_risk_discuss_rounds >= 1

    def test_fast_is_cheaper_than_standard(self):
        fast = COST_PROFILES[CostMode.FAST].llm_call_budget()
        standard = COST_PROFILES[CostMode.STANDARD].llm_call_budget()
        deep = COST_PROFILES[CostMode.DEEP].llm_call_budget()
        assert fast[0] < standard[0] < deep[0]
        assert fast[1] < standard[1] < deep[1]

    def test_standard_budget_is_in_target_band(self):
        floor, typical = COST_PROFILES[CostMode.STANDARD].llm_call_budget()
        # Target: ~12–25 LLM calls per STANDARD run (spec §27).
        assert 10 <= floor <= 16
        assert typical <= 25

    def test_fast_omits_fundamentals_slot(self):
        assert "fundamentals" not in COST_PROFILES[CostMode.FAST].analysts
        assert "macro" in COST_PROFILES[CostMode.STANDARD].analysts


class TestRiskLimits:
    def test_defaults_are_conservative(self):
        limits = RiskLimits()
        assert limits.max_risk_fraction_per_trade <= 0.02
        assert limits.min_reward_risk >= 1.0
        assert limits.max_open_positions <= 3

    @pytest.mark.parametrize("kwargs", [
        {"max_risk_fraction_per_trade": 0},
        {"min_reward_risk": -1},
        {"max_stop_distance_price": 0.1},
        {"max_daily_loss_fraction": 0},
    ])
    def test_invalid_limits_rejected(self, kwargs):
        with pytest.raises(ValueError):
            RiskLimits(**kwargs)

    def test_stop_distance_window_must_be_ordered(self):
        with pytest.raises(ValueError):
            RiskLimits(min_stop_distance_price=10.0, max_stop_distance_price=5.0)


class TestExecutionAssumptions:
    def test_negative_spread_rejected(self):
        with pytest.raises(ValueError):
            ExecutionAssumptions(spread_price=-0.1)

    def test_defaults_are_deterministic(self):
        a, b = ExecutionAssumptions(), ExecutionAssumptions()
        assert a == b


class TestGoldSymbolAliases:
    @pytest.mark.parametrize("raw", [
        "XAUUSD", "xauusd", "XAUUSD+", "XAU/USD", "XAU", "GOLD", "GC=F", " gold ",
    ])
    def test_gold_spellings_recognised(self, raw):
        assert is_gold_symbol(raw) is True

    @pytest.mark.parametrize("raw", [
        "XAGUSD", "AAPL", "BTCUSD", "EURUSD", "SI=F", "", None, 123,
    ])
    def test_non_gold_rejected(self, raw):
        assert is_gold_symbol(raw) is False

    def test_alias_set_is_gold_only(self):
        # Spec §4: XAUUSD only — no silver/miners/ETFs as instruments.
        assert "XAGUSD" not in GOLD_SYMBOL_ALIASES
        assert "GLD" not in GOLD_SYMBOL_ALIASES


class TestAdditivity:
    def test_upstream_default_config_untouched(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["llm_provider"] == "openai"
        assert "gold" not in DEFAULT_CONFIG.get("data_vendors", {})

    def test_gold_config_is_frozen(self):
        import dataclasses

        cfg = GoldConfig(
            data=GoldDataConfig(), risk=RiskLimits(),
            execution=ExecutionAssumptions(), news=GoldNewsConfig(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.cost_mode = CostMode.FAST  # frozen dataclass

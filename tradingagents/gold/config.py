"""Gold (XAUUSD) configuration layer.

One frozen dataclass tree owns the gold system's deterministic settings:
symbol identity (spot XAUUSD vs the GC=F futures proxy), timeframes, data
freshness budgets, risk limits, paper-execution assumptions, gold news topics
and the FAST/STANDARD/DEEP cost modes.  Nothing here is an LLM setting — those
keep living in the upstream ``DEFAULT_CONFIG``; a gold run layers this config
on top of it.

Symbol identity is explicit: the system analyses *XAUUSD spot* while the only
price source wired in v1 is the Yahoo *GC=F front-month future*, used strictly
as a labelled research proxy.  Futures results must never be presented as
broker XAUUSD spot results (see ``GoldDataConfig.proxy_disclaimer``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tradingagents.gold.types import CostMode, TimeFrame

#: User/broker spellings accepted as "gold" by the gold subsystem.  This is
#: deliberately the gold slice of ``dataflows.symbols`` — the gold system
#: supports XAUUSD only (no silver, no miners, no ETFs as instruments).
GOLD_SYMBOL_ALIASES: frozenset[str] = frozenset({
    "XAUUSD", "XAUUSD+", "XAU/USD", "XAU", "GOLD", "GC=F",
})


def is_gold_symbol(raw: str) -> bool:
    """Whether ``raw`` (user or broker spelling) is the gold instrument."""
    if not isinstance(raw, str):
        return False
    cleaned = raw.strip().upper().rstrip("+")
    return cleaned in {a.rstrip("+") for a in GOLD_SYMBOL_ALIASES}


@dataclass(frozen=True)
class CostProfile:
    """Analyst roster and debate depth for one cost mode."""

    analysts: tuple[str, ...]
    max_debate_rounds: int
    max_risk_discuss_rounds: int
    description: str

    def llm_call_budget(self) -> tuple[int, int]:
        """Rough (floor, typical) LLM-call budget for one complete run.

        Floor = one call per analyst + 2×debate + 1 RM + 1 trader + 3×risk + 1
        PM.  Typical adds one tool-loop iteration per tool-using analyst and an
        occasional structured-output retry.  Used only for cost planning.
        """
        tool_analysts = max(0, len(self.analysts) - 1)  # social pre-fetches
        floor = (
            len(self.analysts)
            + 2 * self.max_debate_rounds
            + 1 + 1
            + 3 * self.max_risk_discuss_rounds
            + 1
        )
        typical = floor + tool_analysts * 2 + 1
        return floor, typical


COST_PROFILES: dict[CostMode, CostProfile] = {
    CostMode.FAST: CostProfile(
        analysts=("market", "news"),
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        description="Minimal analysts (technical + news), single debate and risk round.",
    ),
    CostMode.STANDARD: CostProfile(
        analysts=("market", "macro", "news", "social"),
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        description=(
            "Full gold analyst team (technical, macro, news, sentiment), "
            "single debate and risk round — target ~12–25 LLM calls per run."
        ),
    ),
    CostMode.DEEP: CostProfile(
        analysts=("market", "macro", "news", "social"),
        max_debate_rounds=2,
        max_risk_discuss_rounds=2,
        description="Full analyst team, double debate and risk rounds (research only).",
    ),
}


@dataclass(frozen=True)
class GoldDataConfig:
    """Instrument identity and data-freshness budgets."""

    symbol: str = "XAUUSD"
    proxy_symbol: str = "GC=F"
    timeframes: tuple[TimeFrame, ...] = (TimeFrame.M15, TimeFrame.H1, TimeFrame.H4)
    timezone: str = "UTC"
    #: A bar older than this (vs "now") marks its timeframe stale in live mode.
    stale_after_minutes: dict[str, int] = field(default_factory=lambda: {
        "M15": 120, "H1": 480, "H4": 1500,
    })
    #: Warmup/analysis depth requested per timeframe when building snapshots.
    history_bars: dict[str, int] = field(default_factory=lambda: {
        "M15": 300, "H1": 500, "H4": 400,
    })
    #: Required metadata line on every proxied dataset.  Never remove.
    proxy_disclaimer: str = (
        "GC=F (COMEX front-month gold future) is a RESEARCH PROXY for XAUUSD "
        "spot. Futures basis, roll and session differences apply; futures "
        "results are not broker XAUUSD spot results."
    )


@dataclass(frozen=True)
class RiskLimits:
    """Deterministic limits enforced by the risk gate (code, not the LLM)."""

    max_risk_fraction_per_trade: float = 0.01   # ≤1% account equity risked per trade
    max_open_positions: int = 2
    max_daily_loss_fraction: float = 0.03       # stop trading after -3% day
    max_spread_price: float = 0.80              # reject entries when spread wider (USD/oz)
    min_reward_risk: float = 1.5                # require ≥1.5R projected reward
    max_position_units: float = 10.0            # hard cap on position size (units)
    min_stop_distance_price: float = 0.5        # stops closer than this are noise
    max_stop_distance_price: float = 120.0      # stops farther than this are un-riskable

    def __post_init__(self) -> None:
        for name in (
            "max_risk_fraction_per_trade", "max_daily_loss_fraction",
            "min_reward_risk", "max_position_units",
            "min_stop_distance_price", "max_stop_distance_price", "max_spread_price",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"RiskLimits.{name} must be > 0")
        if self.max_stop_distance_price <= self.min_stop_distance_price:
            raise ValueError("max_stop_distance_price must exceed min_stop_distance_price")
        if self.max_open_positions < 0:
            raise ValueError("max_open_positions must be >= 0")


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Paper-execution model inputs.  Deterministic and configurable so cost /
    spread / slippage stress is a re-run with different assumptions."""

    spread_price: float = 0.30       # USD/oz assumed spread when no quote feed
    slippage_price: float = 0.10     # USD/oz adverse slippage per fill
    account_equity: float = 10_000.0
    contract_size_oz: float = 100.0  # one standard lot

    def __post_init__(self) -> None:
        if self.spread_price < 0 or self.slippage_price < 0:
            raise ValueError("spread/slippage assumptions must be >= 0")
        if self.account_equity <= 0 or self.contract_size_oz <= 0:
            raise ValueError("account_equity and contract_size_oz must be > 0")


@dataclass(frozen=True)
class GoldNewsConfig:
    """Gold/macro news topics.  Fed/USD/macro/geopolitical coverage is the
    news analyst's diet; equity-earnings queries are deliberately absent."""

    gold_queries: tuple[str, ...] = (
        "gold XAUUSD spot price outlook",
        "Federal Reserve interest rates inflation",
        "US dollar DXY Treasury yields real yields",
        "central banks gold reserves",
        "geopolitical risk safe haven",
        "CPI PCE nonfarm payrolls economic releases",
    )
    article_limit: int = 12
    global_article_limit: int = 10
    lookback_days: int = 7


@dataclass(frozen=True)
class GoldConfig:
    """Complete deterministic configuration of one gold system."""

    data: GoldDataConfig = field(default_factory=GoldDataConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionAssumptions = field(default_factory=ExecutionAssumptions)
    news: GoldNewsConfig = field(default_factory=GoldNewsConfig)
    cost_mode: CostMode = CostMode.STANDARD

    @property
    def cost_profile(self) -> CostProfile:
        return COST_PROFILES[self.cost_mode]


def default_gold_config(
    cost_mode: CostMode = CostMode.STANDARD,
    *,
    data: GoldDataConfig | None = None,
    risk: RiskLimits | None = None,
    execution: ExecutionAssumptions | None = None,
    news: GoldNewsConfig | None = None,
) -> GoldConfig:
    """Build the gold config, defaulting every subtree unless overridden."""
    return GoldConfig(
        data=data or GoldDataConfig(),
        risk=risk or RiskLimits(),
        execution=execution or ExecutionAssumptions(),
        news=news or GoldNewsConfig(),
        cost_mode=cost_mode,
    )

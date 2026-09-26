"""Gold macro layer: deterministic FRED/ALFRED macro snapshots."""

from tradingagents.gold.macro.engine import (
    GOLD_MACRO_SERIES,
    GoldMacroSnapshot,
    MacroEngine,
    MacroPoint,
)

__all__ = ["GOLD_MACRO_SERIES", "GoldMacroSnapshot", "MacroEngine", "MacroPoint"]

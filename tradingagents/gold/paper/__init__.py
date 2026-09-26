"""Deterministic paper trading (spec §18/§24)."""

from tradingagents.gold.paper.engine import (
    PaperPosition,
    PaperTradingEngine,
)
from tradingagents.gold.paper.metrics import PerformanceReport, build_performance_report

__all__ = [
    "PaperPosition",
    "PaperTradingEngine",
    "PerformanceReport",
    "build_performance_report",
]

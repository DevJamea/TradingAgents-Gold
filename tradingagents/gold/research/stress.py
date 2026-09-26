"""Cost stress: re-simulate recorded decisions under harsher assumptions.

Because fills are a deterministic function of the assumption set, spread and
slippage stress is a re-run of the same decisions through the paper engine
with scaled assumptions — the honest way to stress costs (spec §19/§31: cost
paid, spread sensitivity, slippage sensitivity).
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents.gold.config import ExecutionAssumptions, RiskLimits
from tradingagents.gold.data.models import GoldDataset
from tradingagents.gold.decision import GoldDecision
from tradingagents.gold.paper.engine import PaperTradingEngine
from tradingagents.gold.paper.metrics import PerformanceReport, build_performance_report
from tradingagents.gold.risk_gate import AccountState, evaluate_decision


@dataclass(frozen=True)
class StressPoint:
    label: str
    spread_price: float
    slippage_price: float
    report: PerformanceReport


def stress_costs(
    decisions: list[tuple[GoldDecision, GoldDataset]],
    *,
    base: ExecutionAssumptions | None = None,
    equity: float = 10_000.0,
    spread_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
    slippage_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> list[StressPoint]:
    """Replay the same gate-approved decisions under assumption grids.

    ``decisions`` pairs each decision with the bar dataset used to resolve it
    (its replay window).  Returns one stress point per (spread, slippage)
    multiplier combination — deterministic and comparable.
    """
    base = base or ExecutionAssumptions()
    points: list[StressPoint] = []
    for sm in spread_multipliers:
        for lm in slippage_multipliers:
            assumptions = ExecutionAssumptions(
                spread_price=base.spread_price * sm,
                slippage_price=base.slippage_price * lm,
                account_equity=equity,
                contract_size_oz=base.contract_size_oz,
            )
            engine = PaperTradingEngine(assumptions)
            for decision, dataset in decisions:
                ctx = _context_for(dataset)
                gate = evaluate_decision(
                    decision, ctx, AccountState(equity=equity),
                    limits=RiskLimits(max_position_units=1000.0),
                )
                position = engine.open_from_gate(decision, gate, fill_time=dataset.bars[0].timestamp)
                if position is None:
                    continue
                for bar in dataset.bars:
                    if engine.open_positions:
                        engine.process_bar(bar)
                if engine.open_positions:
                    engine.close_all_at(dataset.bars[-1])
            points.append(StressPoint(
                label=f"spread×{sm:g} slippage×{lm:g}",
                spread_price=assumptions.spread_price,
                slippage_price=assumptions.slippage_price,
                report=build_performance_report(engine.positions),
            ))
    return points


def _context_for(dataset: GoldDataset):
    """Minimal gate context carrying just this dataset (offline, deterministic).

    The gate's data-quality check passes when the dataset is valid; building a
    full multi-timeframe context here would couple cost stress to fixtures.
    """
    return _StressContext(dataset)


class _StressContext:
    """Duck-typed GoldRunContext subset the risk gate actually reads."""

    def __init__(self, dataset: GoldDataset) -> None:
        from tradingagents.gold.data.validation import validate_dataset

        report = validate_dataset(dataset, expected_symbol=dataset.meta.symbol)
        self.snapshot = _SnapshotShim(dataset, report)
        self.data_errors = [
            f"{issue.code}: {issue.message}" for issue in report.errors
        ]

    @property
    def has_data_error(self) -> bool:
        return bool(self.data_errors)


class _SnapshotShim:
    def __init__(self, dataset, report) -> None:
        self.quote = None
        self._report = report
        self.errors = [f"{i.code}: {i.message}" for i in report.errors]

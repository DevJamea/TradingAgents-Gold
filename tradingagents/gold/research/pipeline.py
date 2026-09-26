"""Research pipeline orchestration (spec §31 Phase 8, §19).

``run_research`` produces one reproducible, sectioned report over the closed
trades of a research run:

* chronological Development / Validation / Out-of-Sample segments (computed
  independently; OOS returned separately);
* per-segment performance;
* cost/spread/slippage stress when decision+replay pairs are supplied;
* bootstrap CI, Monte-Carlo drawdown and permutation null test with fixed
  seeds.

The pipeline REPORTS; it never tunes, selects or optimizes (spec §32).  The
OOS section is marked explicitly so no one can mistake it for development
output (spec §34: sample size, period, assumptions, OOS status, limitations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from tradingagents.gold.paper.engine import PaperPosition
from tradingagents.gold.paper.metrics import PerformanceReport, build_performance_report
from tradingagents.gold.research.robustness import (
    BootstrapCI,
    bootstrap_mean_r_ci,
    monte_carlo_max_drawdown_r,
    permutation_test_mean_r,
)
from tradingagents.gold.research.splits import ChronologicalSplits, chronological_splits, segment
from tradingagents.gold.research.stress import StressPoint

#: Bumped when robustness methodology changes; keeps reports reproducible.
RESEARCH_METHODOLOGY_VERSION = 1


@dataclass
class ResearchReport:
    generated_at: datetime
    methodology_version: int
    splits: ChronologicalSplits
    segment_reports: dict[str, PerformanceReport]
    bootstrap: BootstrapCI | None = None
    monte_carlo: dict | None = None
    permutation: dict | None = None
    stress: list[StressPoint] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"GOLD RESEARCH REPORT (methodology v{self.methodology_version}, "
            f"generated {self.generated_at:%Y-%m-%d %H:%M} UTC)",
            "",
            "Chronological splits (development and validation are tuning-time data; "
            "out-of-sample was NOT used during development):",
        ]
        for name, split in (
            ("development", self.splits.development),
            ("validation", self.splits.validation),
            ("out_of_sample", self.splits.out_of_sample),
        ):
            report = self.segment_reports.get(name)
            window = f"{split.start} .. {split.end}"
            if report is None or report.total_trades == 0:
                lines.append(f"- {name}: {window} — no closed trades")
            else:
                lines.append(
                    f"- {name}: {window} — {report.total_trades} trade(s), "
                    f"win rate {report.win_rate:.0%}, net {report.net_pnl:+.2f}, "
                    f"expectancy {report.expectancy_r:+.2f}R"
                )
        if self.bootstrap is not None:
            lines.append(
                f"Bootstrap mean-R {self.bootstrap.confidence:.0%} CI: "
                f"[{self.bootstrap.lower:+.2f}, {self.bootstrap.upper:+.2f}] "
                f"(estimate {self.bootstrap.estimate:+.2f}, {self.bootstrap.iterations} draws, "
                f"seed {self.bootstrap.seed})"
            )
        if self.permutation is not None:
            lines.append(
                f"Permutation null test: observed mean R {self.permutation['observed_mean_r']:+.2f}, "
                f"p={self.permutation['p_value']:.3f} ({self.permutation['iterations']} flips, "
                f"seed {self.permutation['seed']})"
            )
        if self.monte_carlo is not None:
            lines.append(
                "Monte-Carlo max drawdown (R): median "
                f"{self.monte_carlo['max_drawdown_r_median']:.2f}, "
                f"p95 {self.monte_carlo['max_drawdown_r_p95']:.2f}, "
                f"worst {self.monte_carlo['max_drawdown_r_worst']:.2f}"
            )
        for point in self.stress:
            report = point.report
            lines.append(
                f"Cost stress {point.label}: {report.total_trades} trade(s), "
                f"net {report.net_pnl:+.2f}, cost paid {report.total_cost:.2f}"
            )
        lines.append("Limitations: " + ("; ".join(self.limitations) if self.limitations else "none recorded"))
        return "\n".join(lines)


def run_research(
    trades: list[PaperPosition],
    *,
    now: datetime,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    bootstrap_seed: int = 7,
    monte_carlo_seed: int = 11,
    permutation_seed: int = 13,
    decision_replays: list | None = None,
    limitations: list[str] | None = None,
) -> ResearchReport:
    """Full deterministic research report over ``trades`` (closed only)."""
    closed = [t for t in trades if t.is_closed]
    splits = chronological_splits(closed, fractions)
    segment_reports = {
        split.name: build_performance_report(segment(closed, split))
        for split in splits.segments
    }
    report = ResearchReport(
        generated_at=now,
        methodology_version=RESEARCH_METHODOLOGY_VERSION,
        splits=splits,
        segment_reports=segment_reports,
        bootstrap=bootstrap_mean_r_ci(closed, seed=bootstrap_seed),
        monte_carlo=monte_carlo_max_drawdown_r(closed, seed=monte_carlo_seed),
        permutation=permutation_test_mean_r(closed, seed=permutation_seed),
        stress=[],
        limitations=list(limitations or []),
    )
    if decision_replays:
        from tradingagents.gold.research.stress import stress_costs

        report.stress = stress_costs(decision_replays)
    report.limitations.extend([
        "results describe the specific sample and assumption set only; they are "
        "not a claim of profitability",
        "paper fills model spread/slippage by fixed assumptions, not observed "
        "order-book data",
    ])
    return report

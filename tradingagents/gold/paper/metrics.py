"""Performance metrics for closed paper trades (deterministic).

All formulas operate on closed positions only.  Nothing here predicts or
optimizes — it reports.  Every report carries its sample size so results are
never presented as more than they are (spec §34).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from tradingagents.gold.paper.engine import PaperPosition


@dataclass
class PerformanceReport:
    total_trades: int
    wins: int
    losses: int
    win_rate: float | None
    loss_rate: float | None
    average_r: float | None
    expectancy_r: float | None
    profit_factor: float | None
    net_pnl: float
    max_drawdown_fraction: float | None
    total_cost: float
    first_closed_at: datetime | None = None
    last_closed_at: datetime | None = None

    def summary(self) -> str:
        pct = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
        return (
            f"PAPER PERFORMANCE — {self.total_trades} closed trade(s) "
            f"({self.wins}W/{self.losses}L)\n"
            f"win rate {pct(self.win_rate)} | profit factor "
            f"{'n/a' if self.profit_factor is None else f'{self.profit_factor:.2f}'} | "
            f"expectancy {'n/a' if self.expectancy_r is None else f'{self.expectancy_r:+.2f}R'} | "
            f"avg R {'n/a' if self.average_r is None else f'{self.average_r:+.2f}R'}\n"
            f"net P&L {self.net_pnl:+.2f} | cost paid {self.total_cost:.2f} | "
            f"max drawdown {pct(self.max_drawdown_fraction)}"
        )


def build_performance_report(positions: list[PaperPosition]) -> PerformanceReport:
    closed = [p for p in positions if p.is_closed]
    if not closed:
        return PerformanceReport(total_trades=0, wins=0, losses=0, win_rate=None,
                                 loss_rate=None, average_r=None, expectancy_r=None,
                                 profit_factor=None, net_pnl=0.0,
                                 max_drawdown_fraction=None, total_cost=0.0)

    pnls = [p.pnl for p in closed]
    rs = [p.r_multiple for p in closed if p.r_multiple is not None]
    wins = [p for p in closed if p.pnl > 0]
    losses = [p for p in closed if p.pnl <= 0]
    gross_win = sum(p.pnl for p in wins)
    gross_loss = abs(sum(p.pnl for p in losses))
    total_cost = sum(p.total_cost for p in closed)

    # Max drawdown over the closed-trade equity curve (cumulative realized P&L).
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cum += pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    start_equity = closed[0].entry_fill * closed[0].units
    max_dd_fraction = (max_dd / start_equity) if start_equity > 0 else None

    return PerformanceReport(
        total_trades=len(closed),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(closed),
        loss_rate=len(losses) / len(closed),
        average_r=sum(rs) / len(rs) if rs else None,
        expectancy_r=sum(rs) / len(rs) if rs else None,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else (
            None if gross_win == 0 else float("inf")
        ),
        net_pnl=sum(pnls),
        max_drawdown_fraction=max_dd_fraction,
        total_cost=total_cost,
        first_closed_at=min(p.closed_at for p in closed),
        last_closed_at=max(p.closed_at for p in closed),
    )

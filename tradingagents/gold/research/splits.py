"""Chronological research splits (spec §31 Phase 8).

Development → Validation/Walk-Forward → Out-of-Sample, cut strictly by time.
There is no shuffling anywhere: the split boundaries are explicit dates, the
segments are non-overlapping and ordered, and the OOS segment is returned
separately so development tooling physically cannot tune against it.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingagents.gold.paper.engine import PaperPosition


@dataclass(frozen=True)
class Split:
    name: str
    start: object    # datetime | None (None = open start)
    end: object      # datetime | None (None = open end)

    def contains(self, timestamp) -> bool:
        """Half-open membership: ``(start, end]`` so boundaries never overlap."""
        if self.start is not None and timestamp <= self.start:
            return False
        return self.end is None or timestamp <= self.end


@dataclass(frozen=True)
class ChronologicalSplits:
    development: Split
    validation: Split
    out_of_sample: Split

    @property
    def segments(self) -> tuple[Split, ...]:
        return (self.development, self.validation, self.out_of_sample)


def chronological_splits(
    trades: list[PaperPosition],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> ChronologicalSplits:
    """Split closed trades chronologically by close time.

    Boundaries fall between trades (no trade is ever half-in/half-out), the
    segments are ordered and non-overlapping by construction.
    """
    closed = sorted((t for t in trades if t.is_closed), key=lambda t: t.closed_at)
    if not closed:
        raise ValueError("no closed trades to split")
    total = sum(fractions)
    if total <= 0:
        raise ValueError("fractions must be positive")
    n = len(closed)
    cut1 = closed[max(1, round(n * fractions[0] / total)) - 1].closed_at
    cut2 = closed[max(1, round(n * (fractions[0] + fractions[1]) / total)) - 1].closed_at
    return ChronologicalSplits(
        development=Split("development", None, cut1),
        validation=Split("validation", cut1, cut2),
        out_of_sample=Split("out_of_sample", cut2, None),
    )


def segment(trades: list[PaperPosition], split: Split) -> list[PaperPosition]:
    """Closed trades whose close time falls inside ``split``."""
    return [t for t in trades if t.is_closed and split.contains(t.closed_at)]

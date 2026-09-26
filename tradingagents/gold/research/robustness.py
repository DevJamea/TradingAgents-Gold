"""Deterministic robustness statistics for paper-trading results.

Bootstrap confidence intervals, Monte-Carlo drawdown distributions and a
permutation (null) test on mean R.  Every routine takes an explicit RNG seed
and is therefore exactly reproducible; none of them fits or optimizes
anything (spec §32: no automated parameter optimization, no curve fitting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tradingagents.gold.paper.engine import PaperPosition


@dataclass(frozen=True)
class BootstrapCI:
    statistic: str
    estimate: float
    lower: float
    upper: float
    iterations: int
    seed: int
    confidence: float

    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper


def _r_multiples(trades: list[PaperPosition]) -> np.ndarray:
    rs = [t.r_multiple for t in trades if t.is_closed and t.r_multiple is not None]
    if not rs:
        raise ValueError("no closed trades with R-multiples")
    return np.asarray(rs, dtype=float)


def bootstrap_mean_r_ci(
    trades: list[PaperPosition],
    seed: int = 7,
    iterations: int = 2000,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap CI of mean R (resampling trades with replacement)."""
    rs = _r_multiples(trades)
    rng = np.random.default_rng(seed)
    samples = rng.choice(rs, size=(iterations, rs.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapCI(
        statistic="mean_r", estimate=float(rs.mean()), lower=float(lower),
        upper=float(upper), iterations=iterations, seed=seed, confidence=confidence,
    )


def monte_carlo_max_drawdown_r(
    trades: list[PaperPosition],
    seed: int = 11,
    iterations: int = 2000,
) -> dict:
    """Distribution of max drawdown (in R) over randomized trade order."""
    rs = _r_multiples(trades)
    rng = np.random.default_rng(seed)
    drawdowns = np.empty(iterations)
    for i in range(iterations):
        order = rng.permutation(rs)
        equity = np.cumsum(order)
        peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
        drawdowns[i] = float((peak - equity).max())
    return {
        "max_drawdown_r_median": float(np.median(drawdowns)),
        "max_drawdown_r_p95": float(np.quantile(drawdowns, 0.95)),
        "max_drawdown_r_worst": float(drawdowns.max()),
        "iterations": iterations,
        "seed": seed,
    }


def permutation_test_mean_r(
    trades: list[PaperPosition],
    seed: int = 13,
    iterations: int = 5000,
) -> dict:
    """Null hypothesis: the system's mean R is zero (sign-flip permutation).

    p-value = fraction of sign-flipped samples whose mean is at least the
    observed mean (one-sided, larger-than).  Deterministic for a fixed seed.
    """
    rs = _r_multiples(trades)
    observed = float(rs.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, rs.size))
    null_means = (signs * rs).mean(axis=1)
    p_value = float((null_means >= observed).mean())
    return {
        "observed_mean_r": observed,
        "p_value": p_value,
        "iterations": iterations,
        "seed": seed,
        "null_mean_abs": float(np.abs(null_means).mean()),
    }

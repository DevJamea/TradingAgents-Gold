"""Deterministic markdown rendering of the technical snapshot.

The rendering is part of the deterministic engine: the analyst prompt receives
this exact text, so the LLM interprets numbers it did not compute.
"""

from __future__ import annotations

from tradingagents.gold.technicals.snapshot import GoldTechnicalSnapshot


def _fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _trend_line(block) -> str:
    parts = [
        f"direction={block.direction.value}",
        f"EMA20={_fmt(block.ema20)}",
        f"EMA50={_fmt(block.ema50)}",
        f"EMA200={_fmt(block.ema200)}",
        f"slope20={_fmt(block.ema20_slope_pct, 4, '%/bar')}",
        f"slope50={_fmt(block.ema50_slope_pct, 4, '%/bar')}",
    ]
    return ", ".join(parts)


def _structure_line(block) -> str:
    highs = ", ".join(f"{price:.2f}" for _, price in block.swing_highs[-3:]) or "n/a"
    lows = ", ".join(f"{price:.2f}" for _, price in block.swing_lows[-3:]) or "n/a"
    support = ", ".join(f"{v:.2f}" for v in block.support) or "n/a"
    resistance = ", ".join(f"{v:.2f}" for v in block.resistance) or "n/a"
    return (
        f"bias={block.bias.value}; recent swing highs [{highs}]; swing lows [{lows}]; "
        f"support [{support}]; resistance [{resistance}]; breakout={block.breakout.value}"
    )


def render_technical_snapshot(snap: GoldTechnicalSnapshot) -> str:
    """Render the snapshot as the analyst-facing markdown block."""
    lines = [
        "## DETERMINISTIC TECHNICAL SNAPSHOT (computed by code — do not recompute or invent values)",
        f"Symbol: {snap.symbol} | regime: {snap.market_regime.value} | "
        f"session: {snap.session.name.value}"
        + (f" (overlaps: {', '.join(snap.session.overlaps)})" if snap.session.overlaps else ""),
    ]
    label = {"H4": "H4 — macro/trend context", "H1": "H1 — structure/context",
             "M15": "M15 — primary decision timeframe"}
    for tf in ("H4", "H1", "M15"):
        block = snap.per_timeframe.get(tf)
        if block is None:
            lines.append(f"### {label.get(tf, tf)}\n- no data")
            continue
        lines.append(
            f"### {label.get(tf, tf)}\n"
            f"- Trend: {_trend_line(block.trend)}\n"
            f"- Momentum: RSI={_fmt(block.momentum.rsi, 1)}, "
            f"ROC={_fmt(block.momentum.roc_pct, 3, '%')}\n"
            f"- Volatility: ATR={_fmt(block.volatility.atr)}, "
            f"normalised ATR={_fmt(block.volatility.atr_pct, 5)} → "
            f"{block.volatility.regime.value}\n"
            f"- Structure: {_structure_line(block.structure)}"
        )
    mtf = snap.multi_timeframe
    lines.append(
        "### Multi-timeframe alignment\n"
        f"- H4={mtf.h4_trend.value} | H1={mtf.h1_trend.value} | M15={mtf.m15_trend.value} | "
        f"aligned={mtf.aligned}"
    )
    for conflict in mtf.conflicts:
        lines.append(f"- CONFLICT: {conflict}")
    lines.append(
        "Interpretation note: the regime and every number above are deterministic "
        "outputs of the technical engine. Your job is to INTERPRET them (what the "
        "regime, alignment and structure imply for gold right now), not to invent "
        "or adjust values."
    )
    return "\n".join(lines)

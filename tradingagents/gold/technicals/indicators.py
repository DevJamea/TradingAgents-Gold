"""Deterministic trend/momentum/volatility indicators for the gold system.

Pure pandas implementations — the LLM never calculates indicators (spec §8).
Every function is a pure function of its inputs: same bars in, same values out.

Formulas (documented purpose in the gold analysis docs):
* EMA(n): standard exponential average, alpha = 2/(n+1), seeded with the first
  value (pandas ``ewm(adjust=False)``).  Trend direction and dynamic S/R.
* RSI(n): Wilder's RSI (Wilder-smoothed average gains/losses via
  ``ewm(alpha=1/n)``).  Momentum / overbought-oversold context.
* ATR(n): Wilder's average true range.  Volatility + stop-distance context.
* ROC(n): close-to-close rate of change in percent.  Momentum confirmation.
* slope_pct: EMA change per bar in percent of price.  Trend strength/flatness.
"""

from __future__ import annotations

import pandas as pd


def ema(close: pd.Series, period: int) -> pd.Series:
    """EMA of ``close`` over ``period`` bars."""
    return close.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100]."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / losses
    out = 100.0 - 100.0 / (1.0 + rs)
    # Zero average loss ⇒ RSI 100 (all gains).
    return out.where(losses != 0, 100.0).where(avg_gain.notna() | (losses == 0))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR over ``period`` bars."""
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False).mean()


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change in percent over ``period`` bars."""
    return close.pct_change(periods=period) * 100.0


def slope_pct(series: pd.Series, lookback: int = 5) -> float | None:
    """Latest average change of ``series`` per bar, in percent of its value.

    Positive ⇒ rising, negative ⇒ falling, magnitude near zero ⇒ flat.  Used to
    distinguish a flat/mixed market from a directional one.
    """
    if len(series) <= lookback or pd.isna(series.iloc[-1]) or series.iloc[-1] == 0:
        return None
    change = series.iloc[-1] - series.iloc[-1 - lookback]
    return float(change / lookback / series.iloc[-1] * 100.0)

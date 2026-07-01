"""Shared helpers: safe rates, Wilson intervals, formatting, IO."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def safe_div(num, den):
    """Division that returns 0.0 when denominator is 0 (avoids NaN in KPIs)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, 0.0, num / den)
    return out if out.shape else float(out)


def pct(x, digits=2):
    """Percentage, works on scalars, numpy arrays and pandas Series."""
    if isinstance(x, pd.Series):
        return (x.astype(float) * 100).round(digits)
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return round(float(arr) * 100, digits)
    return np.round(arr * 100, digits)


def wilson_interval(successes: int, n: int, z: float = 1.645):
    """Wilson score interval for a proportion. z=1.645 -> 90% CI.

    Chosen over the normal approximation because it stays inside [0,1] and is
    reliable for the small, imbalanced counts we have (e.g. 56/4270).
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), p, min(1.0, center + half))


def save_table(df: pd.DataFrame, path: str, index: bool = True):
    df.to_csv(path, index=index)
    return path


def banner(title: str) -> str:
    line = "=" * 78
    return f"\n{line}\n{title}\n{line}"

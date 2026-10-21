"""Seeded bootstrap intervals and a least-squares line."""
from __future__ import annotations

import random
import statistics
from typing import Callable, Sequence


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float] = statistics.mean,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(estimate, lower, upper): percentile bootstrap interval of stat over values."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return stat(values), lo, hi


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """Least squares y = slope * x + intercept, with R2. Descriptive only."""
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("need at least 3 paired points")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("x has no variance")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return {"n": len(xs), "slope": slope, "intercept": intercept, "r2": 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot}


def bootstrap_ratio_ci(numerators: Sequence[float], denominators: Sequence[float], n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Interval of sum(numerators) / sum(denominators), resampling items (e.g. WER over clips)."""
    if len(numerators) != len(denominators) or not numerators:
        raise ValueError("need two non-empty sequences of equal length")
    rng = random.Random(seed)
    n = len(numerators)

    def ratio(idx):
        d = sum(denominators[i] for i in idx)
        return sum(numerators[i] for i in idx) / d if d else float("nan")

    boots = sorted(r for r in (ratio([rng.randrange(n) for _ in range(n)]) for _ in range(n_boot)) if r == r)
    point = ratio(range(n))
    return point, boots[int((alpha / 2) * len(boots))], boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)))]

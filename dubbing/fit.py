"""Duration fitting rule of the original v12.4 pipeline.

rate = (voice_ms / target_ms - 1) * 100. Above max_natural_rate the text is rewritten shorter;
otherwise the voice is sped up by min(rate, max_natural_rate). Accepted if overrun <= tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import FitConfig


def required_rate(voice_ms: float, target_ms: float) -> int:
    """Percent speed-up needed to fit the slot (0 if it already fits)."""
    if target_ms <= 0:
        raise ValueError("target_ms must be > 0")
    if voice_ms <= target_ms:
        return 0
    return round((voice_ms / target_ms - 1) * 100)


def overrun_ms(voice_ms: float, target_ms: float) -> float:
    return max(0.0, voice_ms - target_ms)


def tempo_factor(rate_percent: float) -> float:
    return 1.0 + rate_percent / 100.0


@dataclass(frozen=True)
class FitDecision:
    rate_needed: int
    needs_adaptation: bool
    speedup_rate: int
    acceptable_now: bool


def decide(voice_ms: float, target_ms: float, cfg: FitConfig) -> FitDecision:
    rate = required_rate(voice_ms, target_ms)
    return FitDecision(
        rate_needed=rate,
        needs_adaptation=rate > cfg.max_natural_rate,
        speedup_rate=min(rate, cfg.max_natural_rate),
        acceptable_now=overrun_ms(voice_ms, target_ms) <= cfg.tolerance_ms,
    )


def residual_overrun_ms(voice_ms: float, target_ms: float, applied_rate: float) -> float:
    return overrun_ms(voice_ms / tempo_factor(applied_rate), target_ms)

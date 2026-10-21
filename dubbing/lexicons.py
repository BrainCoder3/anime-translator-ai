"""Concept lexicons for the semantic guard.

LEGACY_TEST_SCENE is the list hard-coded in the original pipeline. It only fits one test scene,
is off by default and exists to reproduce old runs.
"""
from __future__ import annotations

LEGACY_TEST_SCENE: dict[str, list[str]] = {
    "JOKER": [r"\bjoker\b"],
    "RECTO_VISIBLE": [r"\brecto\b", r"\bface visible\b", r"\ble droit\b", r"\bcôté visible\b"],
    "VERSO_CACHE": [r"\bverso\b", r"\benvers\b", r"\bface cachée\b", r"\bcôté caché\b"],
    "LUMIERE": [r"\blumière\b"],
    "OMBRE": [r"\bombre\b", r"\bobscurité\b"],
}

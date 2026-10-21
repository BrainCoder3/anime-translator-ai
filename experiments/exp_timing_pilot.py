"""Duration-fit metrics and figures for the original pipeline's last run (16 segments, one scene).

Descriptive only: the TTS engine is mixed up with the character and with the adaptation loop.

    python experiments/exp_timing_pilot.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing.eval import plots, stats, timing  # noqa: E402

AUDIT = ROOT / "data" / "audit_v15_5_sanitized.json"
OUT = ROOT / "results"
TOL_MS = 200


def linear_fits(segments: list[dict], min_n: int = 5) -> dict:
    """final = slope * slot + intercept, for each engine with at least min_n segments."""
    out = {}
    for engine in sorted({s["engine"] for s in segments}):
        pts = [s for s in segments if s["engine"] == engine]
        if len(pts) >= min_n:
            out[engine] = stats.linear_fit([s["target_ms"] for s in pts], [s["final_duration_ms"] for s in pts])
    return out


def main() -> dict:
    segments = json.loads(AUDIT.read_text(encoding="utf-8"))["segments"]
    result = {
        "experiment": "timing_pilot",
        "n_segments": len(segments),
        "tolerance_ms": TOL_MS,
        "overall": timing.overflow_stats(segments, TOL_MS),
        "by_engine": timing.by_group(segments, "engine", TOL_MS),
        "stretch": timing.stretch_stats(segments),
        "linear_fit_by_engine": linear_fits(segments),
        "caveat": "one scene, 16 segments; engine confounded with character and with the adaptation loop",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "timing_pilot.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    plots.timing_scatter(segments, OUT / "figures" / "timing_pilot_scatter.png", TOL_MS,
                         "Pilot scene: dubbed duration vs source slot (n=16)")
    plots.overrun_bars(segments, OUT / "figures" / "timing_pilot_overrun.png", TOL_MS,
                       "Pilot scene: duration error per segment")
    print(json.dumps({k: result[k] for k in ("n_segments", "overall")}, indent=1))
    return result


if __name__ == "__main__":
    main()

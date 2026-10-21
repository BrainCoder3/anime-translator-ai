"""Duration-fit metrics. Segments need target_ms and final_duration_ms; overflow is recomputed from them.

    python -m dubbing.eval.timing data/audit_v15_5_sanitized.json --tol 200 --group-by engine
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Iterable

from .report import to_markdown_table


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        raise ValueError("empty list")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _check(segments: list[dict]) -> None:
    if not segments:
        raise ValueError("no segments")
    for s in segments:
        if s["target_ms"] <= 0:
            raise ValueError(f"segment {s.get('id')}: target_ms must be > 0")


def overflow_stats(segments: list[dict], tolerance_ms: float = 0.0) -> dict:
    """overrun: final > target + tolerance. fit_rate: share with |final - target| <= tolerance. ratio: final / target."""
    _check(segments)
    n = len(segments)
    diff = [s["final_duration_ms"] - s["target_ms"] for s in segments]
    ratio = [s["final_duration_ms"] / s["target_ms"] for s in segments]
    over = [d for d in diff if d > tolerance_ms]
    return {
        "n_segments": n,
        "tolerance_ms": tolerance_ms,
        "n_overrun": len(over),
        "overrun_rate": len(over) / n,
        "fit_rate": sum(1 for d in diff if abs(d) <= tolerance_ms) / n,
        "mean_overrun_ms": statistics.mean(over) if over else 0.0,
        "max_overrun_ms": max(over) if over else 0.0,
        "median_ratio": statistics.median(ratio),
        "p95_ratio": _percentile(ratio, 95),
        "mean_abs_error_ms": statistics.mean(abs(d) for d in diff),
    }


def stretch_stats(segments: Iterable[dict]) -> dict:
    """Summary of the atempo factors found in an audit."""
    factors = [float(s["atempo"]) for s in segments if s.get("atempo") is not None]
    if not factors:
        return {"n_with_atempo": 0}
    return {
        "n_with_atempo": len(factors),
        "n_stretched": sum(1 for f in factors if abs(f - 1.0) > 1e-9),
        "max_factor": max(factors),
        "mean_factor": statistics.mean(factors),
    }


def by_group(segments: list[dict], key: str, tolerance_ms: float = 0.0) -> dict[str, dict]:
    """overflow_stats per value of a segment field, e.g. engine."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        groups[str(s.get(key, "?"))].append(s)
    return {g: overflow_stats(items, tolerance_ms) for g, items in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("audit_json")
    ap.add_argument("--tol", type=float, default=0.0, help="tolerance in ms (default 0)")
    ap.add_argument("--group-by", default=None, help="segment field, e.g. engine or character_id")
    args = ap.parse_args()
    with open(args.audit_json, encoding="utf-8") as f:
        segments = json.load(f)["segments"]

    rows = {"all": overflow_stats(segments, args.tol)}
    if args.group_by:
        rows.update(by_group(segments, args.group_by, args.tol))
    cols = ["n_segments", "n_overrun", "overrun_rate", "fit_rate", "mean_overrun_ms",
            "max_overrun_ms", "median_ratio", "p95_ratio"]
    print(to_markdown_table([{"group": g, **r} for g, r in rows.items()], ["group"] + cols))
    print()
    print("time-stretch:", stretch_stats(segments))


if __name__ == "__main__":
    main()

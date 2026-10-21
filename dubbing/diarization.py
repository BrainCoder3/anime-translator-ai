"""Assign a speaker to each segment from diarization turns."""
from __future__ import annotations

from typing import Sequence

from .schema import Segment

Turn = dict   # {"start": seconds, "end": seconds, "speaker": str}


def overlap_ms(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers(segments: Sequence[Segment], turns: Sequence[Turn]) -> list[dict]:
    """One dict per segment: the speaker with the largest overlap, plus a fallback tag if none overlapped.

    Fallbacks, in order: in a two-speaker dialogue a gap between two lines of the same speaker goes to the
    other one ("dialogue_alternation"); otherwise the speaker of the nearest turn ("nearest_turn").
    Turns are in seconds, segments in milliseconds.
    """
    if not turns:
        return [{"speaker": None, "overlap_ms": 0.0} for _ in segments]
    order: list[str] = []
    for t in turns:
        if t["speaker"] not in order:
            order.append(t["speaker"])

    out: list[dict] = []
    for seg in segments:
        scores = {s: 0.0 for s in order}
        for t in turns:
            scores[t["speaker"]] += overlap_ms(seg.start_ms, seg.end_ms, t["start"] * 1000, t["end"] * 1000)
        best = max(scores, key=scores.get)
        out.append({"speaker": best if scores[best] > 0 else None, "overlap_ms": scores[best]})

    if len(order) == 2:
        first, second = order
        for i, item in enumerate(out):
            if item["speaker"] is not None:
                continue
            prev = out[i - 1]["speaker"] if i > 0 else None
            nxt = out[i + 1]["speaker"] if i < len(out) - 1 else None
            if prev and prev == nxt:
                item["speaker"] = second if prev == first else first
                item["fallback"] = "dialogue_alternation"

    for i, item in enumerate(out):
        if item["speaker"] is not None:
            continue
        mid = (segments[i].start_ms + segments[i].end_ms) / 2
        nearest = min(turns, key=lambda t: min(abs(mid - t["start"] * 1000), abs(mid - t["end"] * 1000)))
        item["speaker"] = nearest["speaker"]
        item["fallback"] = "nearest_turn"
    return out

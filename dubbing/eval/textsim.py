"""chrF character n-gram similarity (Popovic, 2015).

A surface proxy for how much text survives a rewrite, not a measure of meaning.
"""
from __future__ import annotations

import re
from collections import Counter


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _ngrams(text: str, n: int) -> Counter:
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf(reference: str, hypothesis: str, max_n: int = 4, beta: float = 2.0) -> float:
    """In [0, 1]; case and whitespace are ignored."""
    ref, hyp = _clean(reference), _clean(hypothesis)
    if not ref or not hyp:
        return 1.0 if ref == hyp else 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        r, h = _ngrams(ref, n), _ngrams(hyp, n)
        if not r or not h:
            continue
        overlap = sum((r & h).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, rc = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p == 0 and rc == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * rc / (b2 * p + rc)

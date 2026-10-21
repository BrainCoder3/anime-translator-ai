"""WER and CER with substitution, deletion and insertion counts.

The total is exact. When alignments tie (one insertion plus one deletion costs the same as two
substitutions), the S/D/I split is just one of the optimal ones.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Sequence

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lowercase, NFKC, no punctuation, single spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def _align(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """Levenshtein alignment. Returns (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            t, s, d, ins = prev[j - 1]
            if ref[i - 1] == hyp[j - 1]:
                best = (t, s, d, ins)
            else:
                best = (t + 1, s + 1, d, ins)
            t, s, d, ins = prev[j]          # deletion
            if t + 1 < best[0]:
                best = (t + 1, s, d + 1, ins)
            t, s, d, ins = cur[j - 1]       # insertion
            if t + 1 < best[0]:
                best = (t + 1, s, d, ins + 1)
            cur.append(best)
        prev = cur
    _, s, d, ins = prev[m]
    return s, d, ins


def error_counts(reference: str, hypothesis: str, level: str = "word") -> dict:
    """Edit counts and error rate at level 'word' or 'char'."""
    if level not in ("word", "char"):
        raise ValueError("level must be 'word' or 'char'")
    ref, hyp = normalize(reference), normalize(hypothesis)
    ref_units = ref.split() if level == "word" else list(ref)
    hyp_units = hyp.split() if level == "word" else list(hyp)
    if not ref_units:
        raise ValueError("empty reference")
    s, d, i = _align(ref_units, hyp_units)
    return {"n_ref": len(ref_units), "S": s, "D": d, "I": i,
            "rate": (s + d + i) / len(ref_units)}


def wer(reference: str, hypothesis: str) -> float:
    return error_counts(reference, hypothesis, "word")["rate"]


def cer(reference: str, hypothesis: str) -> float:
    return error_counts(reference, hypothesis, "char")["rate"]


def corpus_error_rate(pairs: Sequence[tuple[str, str]], level: str = "word") -> dict:
    """Micro-average: total errors over total reference units."""
    tot = {"n_ref": 0, "S": 0, "D": 0, "I": 0}
    for ref, hyp in pairs:
        c = error_counts(ref, hyp, level)
        for k in tot:
            tot[k] += c[k]
    tot["rate"] = (tot["S"] + tot["D"] + tot["I"]) / tot["n_ref"]
    return tot

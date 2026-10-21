"""Open-set speaker identification and voice similarity.

A speaker may be unknown, so accepting a stranger is an error just like misidentifying a known
character. The threshold trades these off and should be chosen on labelled data.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

Vector = Sequence[float]


def _norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        raise ValueError("zero vector")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def prototypes(embeddings: Sequence[Vector], labels: Sequence[str]) -> dict[str, list[float]]:
    """Mean of the unit-length embeddings of each label."""
    if len(embeddings) != len(labels):
        raise ValueError("embeddings and labels differ in length")
    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for emb, lab in zip(embeddings, labels):
        n = _norm(emb)
        unit = [x / n for x in emb]
        if lab not in sums:
            sums[lab], counts[lab] = [0.0] * len(unit), 0
        sums[lab] = [s + u for s, u in zip(sums[lab], unit)]
        counts[lab] += 1
    return {lab: [s / counts[lab] for s in vec] for lab, vec in sums.items()}


def identify(emb: Vector, protos: Mapping[str, Vector], threshold: float,
             margin: float = 0.0) -> tuple[str | None, float, float]:
    """(label or None, best score, runner-up score). None unless best >= threshold and it leads by margin."""
    if not protos:
        return None, float("-inf"), float("-inf")
    scores = sorted(((cosine(emb, p), lab) for lab, p in protos.items()), reverse=True)
    best, best_lab = scores[0]
    second = scores[1][0] if len(scores) > 1 else float("-inf")
    if best >= threshold and (best - second) >= margin:
        return best_lab, best, second
    return None, best, second


def open_set_eval(items: Sequence[tuple[Vector, str]], protos: Mapping[str, Vector],
                  thresholds: Sequence[float], margin: float = 0.0) -> list[dict]:
    """Sweep thresholds over (embedding, true_label) items. A label missing from protos is an unknown speaker."""
    known = [(e, l) for e, l in items if l in protos]
    unknown = [(e, l) for e, l in items if l not in protos]
    out = []
    for thr in thresholds:
        correct = wrong = rejected = 0
        for e, l in known:
            pred, _, _ = identify(e, protos, thr, margin)
            if pred is None:
                rejected += 1
            elif pred == l:
                correct += 1
            else:
                wrong += 1
        false_accept = sum(1 for e, _ in unknown if identify(e, protos, thr, margin)[0] is not None)
        out.append({
            "threshold": thr,
            "correct_id_rate": correct / len(known) if known else float("nan"),
            "misid_rate": wrong / len(known) if known else float("nan"),
            "false_reject_rate": rejected / len(known) if known else float("nan"),
            "false_accept_rate": false_accept / len(unknown) if unknown else float("nan"),
            "n_known": len(known), "n_unknown": len(unknown),
        })
    return out


def similarity_to_reference(clone_embedding: Vector, reference_embedding: Vector) -> float:
    """Cosine between a synthesised voice and its reference."""
    return cosine(clone_embedding, reference_embedding)

"""Text helpers: number preservation, concept guard, JSON extraction, word budget."""
from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

NUMBER_WORDS: dict[str, str] = {
    "zéro": "0", "zero": "0", "deux": "2", "trois": "3", "quatre": "4", "cinq": "5", "six": "6",
    "sept": "7", "huit": "8", "neuf": "9", "dix": "10", "onze": "11", "douze": "12",
    "treize": "13", "quatorze": "14", "quinze": "15", "seize": "16", "vingt": "20",
    "trente": "30", "quarante": "40", "cinquante": "50", "cent": "100",
}
# Mostly articles ("une table"), so they are not counted as numbers unless asked.
AMBIGUOUS_NUMBER_WORDS: dict[str, str] = {"un": "1", "une": "1"}

Lexicon = Mapping[str, Sequence[str]]


_UNITS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
          "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16}
_TENS = {"vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60}


def _compound_value(parts: list[str]) -> int | None:
    """Value of a hyphenated French number (dix-huit, vingt-et-un, quatre-vingt-dix), or None."""
    total, i = 0, 0
    while i < len(parts):
        w = parts[i]
        if w == "et":
            i += 1
            continue
        if w == "quatre" and i + 1 < len(parts) and parts[i + 1] in ("vingt", "vingts"):
            total += 80
            i += 2
            continue
        if w in _TENS:
            total += _TENS[w]
        elif w in _UNITS:
            total += _UNITS[w]
        else:
            return None
        i += 1
    return total


def canonical_number_tokens(text: str, include_ambiguous: bool = False) -> set[str]:
    lowered = text.lower().replace("’", "'")
    lowered = re.sub(r"\b(vingt|trente|quarante|cinquante|soixante)\s+et\s+(un|une|onze)\b", r"\1-et-\2", lowered)
    tokens = set(re.findall(r"(?<!\d)\d+(?:[.,]\d+)?(?!\d)", lowered))
    table = {**NUMBER_WORDS, **(AMBIGUOUS_NUMBER_WORDS if include_ambiguous else {})}
    for word in re.findall(r"\b[\wÀ-ÿ'-]+\b", lowered):
        parts = word.split("-")
        if len(parts) > 1 and (value := _compound_value(parts)) is not None:
            tokens.add(str(value))
        elif word in table:
            tokens.add(table[word])
    return tokens


def concepts(text: str, lexicon: Lexicon | None) -> set[str]:
    if not lexicon:
        return set()
    lowered = text.lower().replace("’", "'")
    return {name for name, patterns in lexicon.items()
            if any(re.search(p, lowered) for p in patterns)}


def basic_semantic_guard(original: str, candidate: str, lexicon: Lexicon | None = None) -> tuple[bool, str]:
    """Reject empty candidates, lost numbers and concepts that the original does not contain."""
    candidate = candidate.strip()
    if not candidate:
        return False, "empty candidate"
    lost = canonical_number_tokens(original) - canonical_number_tokens(candidate)
    if lost:
        return False, f"number lost {sorted(lost)}"
    introduced = concepts(candidate, lexicon) - concepts(original, lexicon)
    if introduced:
        return False, f"concept added {sorted(introduced)}"
    return True, "ok"


def extract_json_payload(raw_text: str):
    """Parse a model answer, tolerating ``` fences and text around the JSON object."""
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError("no valid JSON object in the model answer")


def word_budget(target_seconds: float, words_per_second: float = 3.4, minimum: int = 3) -> int:
    return max(minimum, round(target_seconds * words_per_second))

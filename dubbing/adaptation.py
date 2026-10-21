"""Ask an LLM for shorter versions of a line, vet them, keep the best one that fits the slot.

The LLM and the TTS are passed in as callables, so the logic can be tested without any model.
Among vetted candidates that need at most max_natural_rate percent speed-up, the highest semantic
score wins (ties: lower rate, then shorter text). If none fits, the original text is kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import FitConfig
from .fit import required_rate
from .textutil import Lexicon, basic_semantic_guard, extract_json_payload, word_budget


@dataclass
class Candidate:
    text: str
    valid: bool = False
    semantic_score: int = 0
    reason: str = ""


@dataclass
class AdaptationResult:
    text: str
    adapted: bool
    chosen: Candidate | None = None
    attempts: list[dict] = field(default_factory=list)   # contains no dialogue text


def build_prompt(text: str, target_seconds: float, cfg: FitConfig, *, context: str = "",
                 previous: str = "", next_: str = "") -> str:
    max_words = word_budget(target_seconds, cfg.words_per_second)
    return f"""Tu es adaptateur et contrôleur qualité de doublage français.

CONTEXTE GLOBAL : {context or "(aucun)"}
Phrase précédente : {previous or "(aucune)"}
PHRASE À ADAPTER : {text}
Phrase suivante : {next_ or "(aucune)"}

DURÉE DISPONIBLE : {target_seconds:.2f} secondes.

Propose {cfg.max_adapt_candidates} adaptations COURTES et distinctes, chacune de {max_words} mots maximum.
Pour chaque candidat : conserve strictement le sens, les nombres, les conditions et les négations ;
n'ajoute aucune information ; français oral naturel ; donne semantic_score de 0 à 100 ;
valid=true seulement si le sens est réellement conservé.

Réponds uniquement avec du JSON valide :
{{"candidates": [{{"text": "adaptation", "valid": true, "semantic_score": 95, "reason": "raison courte"}}]}}"""


def parse_candidates(payload: dict, limit: int) -> list[Candidate]:
    out: list[Candidate] = []
    for item in payload.get("candidates", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            score = int(item.get("semantic_score", 0))
        except (TypeError, ValueError):
            score = 0
        out.append(Candidate(text, bool(item.get("valid", False)), score, str(item.get("reason", ""))))
    return out[:limit]


def vet_candidates(original: str, candidates: Sequence[Candidate], cfg: FitConfig,
                   lexicon: Lexicon | None = None) -> tuple[list[Candidate], list[dict]]:
    """Split candidates into (kept, rejected); rejections record the reason and who rejected."""
    kept: list[Candidate] = []
    rejected: list[dict] = []
    for c in candidates:
        ok, reason = basic_semantic_guard(original, c.text, lexicon)
        if not ok:
            rejected.append({"semantic_score": c.semantic_score, "rejected_by": "guard", "why": reason})
        elif not c.valid or c.semantic_score < cfg.semantic_score_threshold:
            rejected.append({"semantic_score": c.semantic_score, "rejected_by": "llm_score",
                             "why": f"valid={c.valid} score={c.semantic_score}"})
        else:
            kept.append(c)
    return kept, rejected


def choose_candidate(measured: Sequence[tuple[Candidate, int]], target_ms: int,
                     cfg: FitConfig) -> Candidate | None:
    """measured: (candidate, synthesised duration in ms). Returns None if none fits naturally."""
    fitting = [(c, required_rate(ms, target_ms)) for c, ms in measured
               if required_rate(ms, target_ms) <= cfg.max_natural_rate]
    if not fitting:
        return None
    fitting.sort(key=lambda cr: (-cr[0].semantic_score, cr[1], len(cr[0].text)))
    return fitting[0][0]


def adapt(original: str, target_ms: int, *, ask: Callable[[str], str],
          measure_ms: Callable[[str], int], cfg: FitConfig, lexicon: Lexicon | None = None,
          context: str = "", previous: str = "", next_: str = "") -> AdaptationResult:
    """ask(prompt) returns the raw LLM answer; measure_ms(text) the duration of its synthesis.

    Any LLM or parsing failure leaves the original text unchanged.
    """
    prompt = build_prompt(original, target_ms / 1000, cfg, context=context, previous=previous, next_=next_)
    try:
        candidates = parse_candidates(extract_json_payload(ask(prompt)), cfg.max_adapt_candidates)
    except Exception as exc:
        return AdaptationResult(original, False, None, [{"error": f"{type(exc).__name__}: {exc}"}])

    kept, rejected = vet_candidates(original, candidates, cfg, lexicon)
    attempts: list[dict] = [{"n_proposed": len(candidates), "rejected": rejected}]
    measured: list[tuple[Candidate, int]] = []
    for c in kept:
        ms = int(measure_ms(c.text))
        measured.append((c, ms))
        attempts.append({"semantic_score": c.semantic_score, "n_words": len(c.text.split()),
                         "voice_ms": ms, "rate_needed": required_rate(ms, target_ms)})
    best = choose_candidate(measured, target_ms, cfg)
    if best is None:
        return AdaptationResult(original, False, None, attempts)
    return AdaptationResult(best.text, True, best, attempts)

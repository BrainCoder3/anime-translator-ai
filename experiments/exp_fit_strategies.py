"""Compare the five fitting strategies on 20 French sentences with shortened slots.

The slot is set to 0.9, 0.75 or 0.6 times the natural duration. Real LLM (Ollama) and real TTS.
Per trial: residual overrun, speed-up, whether the text was rewritten, chrF to the original, numbers kept.
LLM answers are cached in results/cache/llm_cache.json so a re-run gives the same results.
Limits are stored in the results file.

    python experiments/exp_fit_strategies.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing.backends.tts_edge import EdgeTTS  # noqa: E402
from dubbing.backends.translate_ollama import OllamaClient, OllamaTranslator  # noqa: E402
from dubbing.config import Config  # noqa: E402
from dubbing.eval import plots, stats, textsim  # noqa: E402
from dubbing.pipeline import STRATEGIES, DubbingPipeline  # noqa: E402
from dubbing.textutil import canonical_number_tokens  # noqa: E402

RATIOS = [0.9, 0.75, 0.6]
SEED = 20262
VOICE = "fr-FR-DeniseNeural"
SENTENCES = [
    "Je ne pensais pas que le voyage serait aussi long.",
    "Demain matin, nous irons au marché pour acheter des légumes frais.",
    "Elle a promis de rappeler dès qu'elle aurait terminé son travail.",
    "Il faut absolument que tu viennes avec nous à la montagne ce week-end.",
    "Tout le monde attendait avec impatience le résultat de l'examen.",
    "Le médecin m'a conseillé de me reposer pendant au moins trois jours.",
    "Nous avons perdu beaucoup de temps à chercher cette vieille clé.",
    "Si tu veux gagner, tu dois d'abord apprendre à écouter les autres.",
    "La boutique ferme à dix-huit heures, mais je peux te déposer avant.",
    "Je n'ai jamais vu une ville aussi calme et aussi lumineuse.",
    "Ils ont construit cette maison de leurs propres mains, pierre par pierre.",
    "Peux-tu m'expliquer pourquoi tu as refusé leur proposition hier soir ?",
    "Le concours est ouvert aux candidats de moins de vingt-cinq ans.",
    "Malgré la fatigue, il a continué à courir jusqu'à la ligne d'arrivée.",
    "On m'a dit que le dernier bus était déjà parti depuis longtemps.",
    "Ma grand-mère raconte toujours la même histoire quand nous dînons ensemble.",
    "Cette nuit, le vent a soufflé si fort que personne n'a pu dormir.",
    "Vous devez remplir ce formulaire en deux exemplaires avant de partir.",
    "Elle a ouvert la fenêtre pour laisser entrer un peu d'air frais.",
    "Il suffit de tourner la clé deux fois pour que la porte s'ouvre.",
]
CACHE = ROOT / "results" / "cache"
OUT = ROOT / "results"


class CachedLLM(OllamaTranslator):
    """Ollama client that caches JSON answers on disk."""

    def __init__(self, path: pathlib.Path, **kw) -> None:
        super().__init__(OllamaClient(**kw))
        self.path = path
        self.cache: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.calls = self.hits = 0

    def ask(self, prompt: str) -> str:
        self.calls += 1
        if prompt in self.cache:
            self.hits += 1
            return self.cache[prompt]
        answer = super().ask(prompt)
        self.cache[prompt] = answer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return answer


def run() -> dict:
    cfg = Config()
    llm = CachedLLM(CACHE / "llm_cache.json", url=cfg.ollama_url, model=cfg.ollama_model, timeout_s=cfg.ollama_timeout_s)
    pipe = DubbingPipeline(cfg, asr=None, translator=llm, tts=EdgeTTS(VOICE), log=lambda m: None)
    pipe.use_workdir(CACHE / "fit_tts")
    trials: list[dict] = []
    t0 = time.time()
    for i, text in enumerate(SENTENCES):
        natural = pipe.synth(text, VOICE, 0).ms
        for ratio in RATIOS:
            target = round(natural * ratio)
            for strategy in STRATEGIES:
                o = pipe.fit_text(text, target, VOICE, strategy=strategy)
                trials.append({
                    "sentence": i, "ratio": ratio, "strategy": strategy, "natural_ms": natural, "target_ms": target,
                    "final_ms": o.final_ms, "overrun_ms": o.overrun_ms, "rate_applied": o.rate_applied,
                    "adapted": o.adapted, "chrf": textsim.chrf(text, o.text),
                    "numbers_kept": canonical_number_tokens(text) <= canonical_number_tokens(o.text),
                    "n_words_orig": len(text.split()), "n_words_final": len(o.text.split()), "final_text": o.text})
        print(f"  sentence {i + 1}/{len(SENTENCES)}  ({time.time() - t0:.0f}s, LLM calls {llm.calls}, cache hits {llm.hits})", flush=True)
    return summarise(trials, llm)


def summarise(trials: list[dict], llm) -> dict:
    tol = Config().fit.tolerance_ms
    rows = []
    for ratio in RATIOS:
        for strategy in STRATEGIES:
            t = [x for x in trials if x["ratio"] == ratio and x["strategy"] == strategy]
            over = [x["overrun_ms"] for x in t]
            mean, lo, hi = stats.bootstrap_ci(over, seed=SEED)
            rows.append({
                "ratio": ratio, "strategy": strategy, "n": len(t),
                "overrun_mean_ms": mean, "overrun_lo": lo, "overrun_hi": hi,
                "overrun_median_ms": statistics.median(over),
                "fit_within_tol": sum(1 for v in over if v <= tol) / len(t),
                "fit_within_200ms": sum(1 for v in over if v <= 200) / len(t),
                "mean_rate_applied": statistics.mean(x["rate_applied"] for x in t),
                "share_beyond_natural_rate": sum(1 for x in t if x["rate_applied"] > Config().fit.max_natural_rate) / len(t),
                "adapted_rate": sum(1 for x in t if x["adapted"]) / len(t),
                "chrf_mean": statistics.mean(x["chrf"] for x in t),
                "numbers_kept_rate": sum(1 for x in t if x["numbers_kept"]) / len(t)})
    return {"experiment": "fit_strategies", "seed": SEED, "ratios": RATIOS, "strategies": list(STRATEGIES),
            "n_sentences": len(SENTENCES), "tolerance_ms": tol, "model": Config().ollama_model, "voice": VOICE,
            "llm_calls": llm.calls, "llm_cache_hits": llm.hits,
            "caveat": ("synthetic slots (ratio x natural duration), one voice, one small LLM; chrF is a surface proxy, not meaning "
                       "preservation; the LLM's own semantic score is not validated; no human ratings"),
            "summary": rows, "trials": trials}


def figures(result: dict) -> None:
    fig = OUT / "figures"
    ratios, strategies = result["ratios"], result["strategies"]
    box = {f"slot = {int(r * 100)}% of natural": {s: [t["overrun_ms"] for t in result["trials"]
                                                       if t["ratio"] == r and t["strategy"] == s] for s in strategies}
           for r in ratios}
    plots.strategy_boxplots(box, fig / "fit_overrun_boxplots.png", "Residual overrun by strategy (20 sentences per box)")
    get = {(r["ratio"], r["strategy"]): r for r in result["summary"]}
    names = [f"slot {int(r * 100)}%" for r in ratios]
    plots.grouped_bars({n: {s: get[(r, s)]["fit_within_200ms"] for s in strategies} for n, r in zip(names, ratios)},
                       fig / "fit_share_within_200ms.png", "Share of segments fitting within 200 ms", "share")
    plots.grouped_bars({n: {s: get[(r, s)]["chrf_mean"] for s in strategies} for n, r in zip(names, ratios)},
                       fig / "fit_text_similarity.png", "Surface similarity to the original text (chrF)", "chrF")
    plots.grouped_bars({n: {s: get[(r, s)]["mean_rate_applied"] for s in strategies} for n, r in zip(names, ratios)},
                       fig / "fit_mean_speedup.png", "Mean speed-up applied to the voice", "percent")


def save(result: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "fit_strategies.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    with (OUT / "fit_strategies_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result["summary"][0]))
        w.writeheader()
        w.writerows(result["summary"])


if __name__ == "__main__":
    res = run()
    save(res)
    figures(res)
    for r in res["summary"]:
        print(f'ratio {r["ratio"]:.2f}  {r["strategy"]:17} overrun mean {r["overrun_mean_ms"]:6.0f} ms  '
              f'fit<=200ms {r["fit_within_200ms"]:.2f}  rate {r["mean_rate_applied"]:5.1f}%  '
              f'adapted {r["adapted_rate"]:.2f}  chrF {r["chrf_mean"]:.2f}  numbers {r["numbers_kept_rate"]:.2f}')

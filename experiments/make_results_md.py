"""Write results/RESULTS.md from the experiments' JSON files.

    python experiments/make_results_md.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing.eval.report import to_markdown_table  # noqa: E402

R = ROOT / "results"


def load(name: str):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig(name: str, alt: str) -> str:
    return f"![{alt}](figures/{name})" if (R / "figures" / name).exists() else f"*(figure {name} not generated)*"


def section_timing() -> str:
    d = load("timing_pilot.json")
    if not d:
        return "## 1. Timing pilot\n\n*not run*\n"
    cols = ["group", "n_segments", "n_overrun", "overrun_rate", "mean_overrun_ms", "max_overrun_ms", "median_ratio"]
    rows = [{"group": "all", **d["overall"]}] + [{"group": g, **s} for g, s in d["by_engine"].items()]
    out = ["## 1. Timing pilot (original pipeline, one scene)", "",
           f"n = {d['n_segments']} segments; tolerance {d['tolerance_ms']} ms. *{d['caveat']}.*", "",
           to_markdown_table(rows, cols), ""]
    if d.get("linear_fit_by_engine"):
        out += ["Descriptive least-squares fit, dubbed duration = slope x slot + intercept:", "",
                to_markdown_table([{"engine": e, **f} for e, f in d["linear_fit_by_engine"].items()],
                                  ["engine", "n", "slope", "intercept", "r2"]), ""]
    out += [fig("timing_pilot_scatter.png", "duration scatter"), "", fig("timing_pilot_overrun.png", "overrun per segment"), ""]
    return "\n".join(out)


def section_asr() -> str:
    d = load("asr_noise.json")
    if not d:
        return "## 2. ASR under noise\n\n*not run*\n"
    rows = [{**r, "WER [95% CI]": f"{r['wer']:.3f} [{r['wer_lo']:.3f}, {r['wer_hi']:.3f}]",
             "CER": f"{r['cer']:.3f}"} for r in d["summary"]]
    cols = ["model", "snr", "n_clips", "WER [95% CI]", "CER", "S_rate", "D_rate", "I_rate", "n_perfect"]
    return "\n".join(["## 2. ASR robustness to white noise (synthetic speech)", "",
                      f"{d['n_sentences']} sentences x {d['n_voices']} voices; seed {d['seed']}. *{d['caveat']}.*", "",
                      to_markdown_table(rows, cols), "",
                      fig("asr_noise_wer.png", "WER vs SNR"), "",
                      fig("asr_noise_error_types_small.png", "error types, small"), ""])


def section_fit() -> str:
    d = load("fit_strategies.json")
    if not d:
        return "## 3. Duration-fitting strategies\n\n*not run*\n"
    out = ["## 3. Duration-fitting strategies (controlled slots)", "",
           f"{d['n_sentences']} sentences, LLM `{d['model']}`, voice `{d['voice']}`, tolerance {d['tolerance_ms']} ms; "
           f"LLM calls {d['llm_calls']} ({d['llm_cache_hits']} from cache). *{d['caveat']}.*", ""]
    cols = ["strategy", "n", "overrun_mean_ms", "overrun_median_ms", "fit_within_200ms", "mean_rate_applied",
            "share_beyond_natural_rate", "adapted_rate", "chrf_mean", "numbers_kept_rate"]
    for ratio in d["ratios"]:
        out += [f"### Slot = {int(ratio * 100)}% of the natural duration", "",
                to_markdown_table([r for r in d["summary"] if r["ratio"] == ratio], cols), ""]
    out += [fig("fit_overrun_boxplots.png", "overrun by strategy"), "", fig("fit_share_within_200ms.png", "share within 200 ms"), "",
            fig("fit_text_similarity.png", "text similarity"), "", fig("fit_mean_speedup.png", "mean speed-up"), ""]
    return "\n".join(out)


def section_repeat() -> str:
    """Compare the final fit_strategies run with the first one (made before the number-guard fix)."""
    first, last = load("fit_strategies_run1.json"), load("fit_strategies.json")
    if not (first and last):
        return ""
    key = lambda t: (t["sentence"], t["ratio"], t["strategy"])
    a, b = {key(t): t for t in first["trials"]}, {key(t): t for t in last["trials"]}
    changed = [k for k in a if a[k]["final_text"] != b[k]["final_text"]]
    cases = sorted({(k[0], k[1]) for k in changed})
    out = ["### Repeat run", "",
           f"The study was run twice, with the number guard fixed in between. {len(changed)} of {len(a)} trials ended with a "
           f"different text; they come from {len(cases)} sentence and slot combination(s): {cases} (sentence index, slot ratio)."]
    if cases == [(3, 0.6)]:
        out.append("For that one, the rewritten candidate sits exactly at the 25 % speed-up limit, and five identical syntheses "
                   "of it lasted 2780, 2780, 2780, 2780 and 2790 ms. With a 2220 ms slot, 2780 ms needs +25 % (accepted) and "
                   "2790 ms needs +26 % (rejected). A 10 ms jitter flipped the decision, so a hard limit makes single "
                   "outcomes fragile.")
    out += ["", "Tables above are from the final run.", ""]
    rows = []
    for ratio in last["ratios"]:
        for strategy in last["strategies"]:
            ra = next(r for r in first["summary"] if r["ratio"] == ratio and r["strategy"] == strategy)
            rb = next(r for r in last["summary"] if r["ratio"] == ratio and r["strategy"] == strategy)
            if ra["overrun_mean_ms"] != rb["overrun_mean_ms"] or ra["fit_within_200ms"] != rb["fit_within_200ms"]:
                rows.append({"ratio": ratio, "strategy": strategy,
                             "overrun_mean_ms first": ra["overrun_mean_ms"], "overrun_mean_ms final": rb["overrun_mean_ms"],
                             "fit<=200ms first": ra["fit_within_200ms"], "fit<=200ms final": rb["fit_within_200ms"]})
    if rows:
        out += [to_markdown_table(rows, list(rows[0])), ""]
    return "\n".join(out)


def section_e2e() -> str:
    p = R / "e2e" / "out" / "audit.json"
    if not p.exists():
        return "## 4. End-to-end smoke test\n\n*not run*\n"
    a = json.loads(p.read_text(encoding="utf-8"))
    cols = ["id", "target_ms", "final_duration_ms", "overflow_ms", "rate_applied", "adapted"]
    return "\n".join(["## 4. End-to-end smoke test (real Whisper, Ollama and edge-tts)", "",
                      f"strategy `{a.get('strategy')}`; {len(a['segments'])} segments from a generated Japanese clip.", "",
                      to_markdown_table(a["segments"], cols), ""])


def main() -> None:
    body = ["# Results", "",
            "Generated by `experiments/make_results_md.py` from the JSON files in this folder. All studies are **pilots**: "
            "small samples, synthetic data, one language pair. Read each caveat before drawing a conclusion.", "",
            section_timing(), section_asr(), section_fit(), section_repeat(), section_e2e()]
    (R / "RESULTS.md").write_text("\n".join(body), encoding="utf-8")
    print("written", R / "RESULTS.md")


if __name__ == "__main__":
    main()

"""Whisper WER on clean synthetic French speech with added white Gaussian noise.

8 sentences x 2 voices (edge-tts), SNR clean/20/10/5/0/-5 dB, faster-whisper base and small on CPU.
Bootstrap 95 % intervals over the 16 clips. Limits are stored in the results file.

    python experiments/exp_asr_noise.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing import media  # noqa: E402
from dubbing.backends.asr_whisper import FasterWhisperASR  # noqa: E402
from dubbing.backends.tts_edge import EdgeTTS  # noqa: E402
from dubbing.eval import asr, noise, plots, stats  # noqa: E402

SEED = 20261
SNRS: list[float | None] = [None, 20, 10, 5, 0, -5]   # None means clean
MODELS = ["base", "small"]
VOICES = ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"]
SENTENCES = [
    "Bonjour, je voudrais réserver une table pour deux personnes ce soir.",
    "Le train pour Lyon part dans dix minutes depuis le quai numéro quatre.",
    "Nous avons décidé de repousser la réunion à jeudi prochain.",
    "Pouvez-vous m'indiquer le chemin le plus court pour aller à la bibliothèque ?",
    "Il pleut depuis ce matin et les rues sont presque désertes.",
    "Mon frère travaille dans une petite entreprise qui fabrique des vélos.",
    "N'oubliez pas d'apporter vos papiers d'identité avant la fin de la semaine.",
    "Cette recette demande trois œufs, un peu de farine et beaucoup de patience.",
]
CACHE = ROOT / "results" / "cache" / "asr_noise"
OUT = ROOT / "results"


def label(snr) -> str:
    return "clean" if snr is None else f"{snr:g}"


def clean_clips() -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    tts, clips = EdgeTTS(), []
    for v_idx, voice in enumerate(VOICES):
        for s_idx, text in enumerate(SENTENCES):
            wav = CACHE / f"v{v_idx}_s{s_idx}.wav"
            if not wav.exists():
                raw = tts.synthesize(text, CACHE / f"raw_v{v_idx}_s{s_idx}.wav", voice=voice)
                media.to_wav(raw, wav, sample_rate=16000)
                raw.unlink()
            clips.append({"clip": f"v{v_idx}_s{s_idx}", "voice": voice, "sentence": s_idx, "text": text, "wav": wav})
    return clips


def transcribe(model: FasterWhisperASR, samples: np.ndarray, tmp: pathlib.Path) -> str:
    media.write_wav(tmp, 16000, samples)
    return " ".join(t for _, _, t in model.transcribe(tmp, "fr"))


def run() -> dict:
    clips = clean_clips()
    trials: list[dict] = []
    tmp = CACHE / "_noisy.wav"
    for model_name in MODELS:
        model = FasterWhisperASR(model_name)
        t0 = time.time()
        for ci, clip in enumerate(clips):
            _, clean = media.read_wav(clip["wav"])
            for snr in SNRS:
                if snr is None:
                    sig = clean
                else:
                    rng = np.random.default_rng([SEED, ci, int(snr) + 100])
                    sig = noise.add_noise_at_snr(clean, snr, rng)
                hyp = transcribe(model, sig, tmp)
                w = asr.error_counts(clip["text"], hyp, "word")
                c = asr.error_counts(clip["text"], hyp, "char")
                trials.append({"model": model_name, "snr": label(snr), "clip": clip["clip"], "voice": clip["voice"],
                               "sentence": clip["sentence"], "hyp": hyp,
                               "w_ref": w["n_ref"], "w_S": w["S"], "w_D": w["D"], "w_I": w["I"],
                               "c_ref": c["n_ref"], "c_err": c["S"] + c["D"] + c["I"]})
            print(f"  {model_name}: clip {ci + 1}/{len(clips)}  ({time.time() - t0:.0f}s)", flush=True)
    tmp.unlink(missing_ok=True)
    return summarise(trials)


def summarise(trials: list[dict]) -> dict:
    summary = []
    for model in MODELS:
        for snr in SNRS:
            rows = [t for t in trials if t["model"] == model and t["snr"] == label(snr)]
            errs = [t["w_S"] + t["w_D"] + t["w_I"] for t in rows]
            refs = [t["w_ref"] for t in rows]
            wer, lo, hi = stats.bootstrap_ratio_ci(errs, refs, seed=SEED)
            cer, clo, chi = stats.bootstrap_ratio_ci([t["c_err"] for t in rows], [t["c_ref"] for t in rows], seed=SEED)
            tot = sum(refs)
            summary.append({"model": model, "snr": label(snr), "n_clips": len(rows),
                            "wer": wer, "wer_lo": lo, "wer_hi": hi, "cer": cer, "cer_lo": clo, "cer_hi": chi,
                            "S_rate": sum(t["w_S"] for t in rows) / tot, "D_rate": sum(t["w_D"] for t in rows) / tot,
                            "I_rate": sum(t["w_I"] for t in rows) / tot,
                            "n_perfect": sum(1 for e in errs if e == 0)})
    return {"experiment": "asr_noise", "seed": SEED, "snrs": [label(s) for s in SNRS], "models": MODELS,
            "n_sentences": len(SENTENCES), "n_voices": len(VOICES),
            "caveat": ("synthetic speech + white Gaussian noise, 16 clips, French only; not representative of real dubbing "
                       "audio (music, effects); S/D/I split is one of several equal-cost alignments"),
            "summary": summary, "trials": trials}


def figures(result: dict) -> None:
    fig = OUT / "figures"
    curves = {}
    for model in MODELS:
        pts = [r for r in result["summary"] if r["model"] == model]
        curves[f"whisper-{model}"] = [{"label": r["snr"], "x": i, "value": r["wer"], "lo": r["wer_lo"], "hi": r["wer_hi"]}
                                      for i, r in enumerate(pts)]
    plots.wer_vs_snr(curves, fig / "asr_noise_wer.png", "Whisper WER on noisy synthetic speech (n=16 clips, 95% CI)")
    for model in MODELS:
        rows = [{"label": r["snr"], "S": r["S_rate"], "D": r["D_rate"], "I": r["I_rate"]}
                for r in result["summary"] if r["model"] == model]
        plots.error_types_stack(rows, fig / f"asr_noise_error_types_{model}.png",
                                f"Whisper-{model}: error types versus SNR")


def save(result: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "asr_noise.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    with (OUT / "asr_noise_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result["summary"][0]))
        w.writeheader()
        w.writerows(result["summary"])


if __name__ == "__main__":
    res = run()
    save(res)
    figures(res)
    for r in res["summary"]:
        print(f'{r["model"]:6} snr={r["snr"]:>5}  WER {r["wer"]:.3f} [{r["wer_lo"]:.3f}, {r["wer_hi"]:.3f}]  '
              f'S/D/I {r["S_rate"]:.3f}/{r["D_rate"]:.3f}/{r["I_rate"]:.3f}')

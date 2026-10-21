# Japanese to French dubbing with duration fitting

A dubbing pipeline (speech recognition, translation, duration fitting, synthesis, mixing) and the tools to measure
it. It sits next to the full prototype in `src/` (`docs/PROTOTYPE_README.md`); this package is the tested part, with small experiments.

The code was refactored and evaluated with AI-assisted coding, starting from the author's original pipeline.

## What it does
```
audio or video -> Whisper (Japanese) -> speakers -> LLM translation (Ollama)
               -> fit each line to its slot -> French voice (edge-tts) -> mix, subtitles, audit
```
Fitting follows the original rule: if a line needs more than 25 % speed-up, an LLM proposes shorter versions that
must keep the numbers; otherwise the voice is sped up. Five strategies can be compared (`none`, `speedup`,
`stretch_uncapped`, `adapt`, `adapt+speedup`).

## Run it
```bash
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt   # needs ffmpeg and Ollama
python -m unittest discover -s tests
python -m dubbing run clip.wav -o out --strategy adapt+speedup
python experiments/exp_fit_strategies.py       # figures and tables go to results/
python experiments/make_results_md.py
```

## Results (small pilots, see `results/RESULTS.md`)
- **Duration fit of the original run:** 13 of 16 lines overran their slot by 220 to 540 ms. For one engine the dubbed
  duration follows about 0.86 x slot + 0.77 s, a fixed offset rather than a speaking-rate problem.
- **Whisper under white noise** (16 synthetic clips): `small` stays near 3 % WER down to 10 dB and reaches 16 % at
  0 dB and 52 % at -5 dB. `base` goes above 100 % at -5 dB because it invents words.
- **Five fitting strategies** (20 sentences, slots at 90, 75 and 60 % of natural duration): speeding up alone fits
  small gaps but not big ones; unlimited speed-up fits everything but needs +33 % to +67 %; rewriting plus a small
  speed-up fits 90 to 100 % within 200 ms with under 10 % speed-up, at the cost of changing about half of the text
  (chrF about 0.5). Whether the rewrites keep the meaning was **not measured**.

## Limits
- Small samples, synthetic slots and synthetic speech. No human listening or meaning ratings.
- Multi-speaker memory (pyannote), background separation and voice cloning are in the prototype in `src/`, not in the
  tested package; they could not be run on the test PC.
- Only short clips were run end to end, not a full episode.
- A hard 25 % limit is fragile: a 10 ms change in one synthesis flipped a decision (see `RESULTS.md`).

## Layout
| Path | Content |
|---|---|
| `dubbing/` | package: fitting rule, adaptation, speaker memory, audio, engines, pipeline, CLI |
| `dubbing/eval/` | metrics, noise, statistics, plots |
| `experiments/` | the three pilots and the report generator |
| `results/` | JSON, CSV, figures, `RESULTS.md` |
| `tests/` | unit tests; live tests need `DUBBING_LIVE=1` |
| `docs/` | research plan, problems found while testing, ethics |
| `src/`, `translate_video.py` | the full prototype (not covered by these tests) |
| `legacy/` | older scripts, kept locally and not published (see `.gitignore`) |

No commercial media is included; see `docs/ETHICS_AND_DATA.md`. Choose a licence before making this public.

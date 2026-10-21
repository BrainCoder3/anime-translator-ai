# Research plan

Working title: duration-constrained Japanese to French dubbing with persistent speaker identity.

The pipeline works. What it lacked was measurement: its thresholds (`MAX_ATEMPO`, `MAX_NATURAL_RATE`,
`MATCH_THRESHOLD`, `NEW_CHARACTER_THRESHOLD`, `SEMANTIC_SCORE_THRESHOLD`) were set by ear. The plan asks whether
they can be measured and justified.

## Questions

**RQ1, fitting a line to its slot.** To fit a French line into the source duration, is it better to speed the voice
up, to rewrite the text shorter with an LLM, or to do both?
Metrics: residual overrun, speed-up applied, share fitting within a tolerance (`dubbing.eval.timing`); meaning
preserved (human ratings, still to be collected).

**RQ2, speaker memory.** Can the match, margin and new-character thresholds be chosen on labelled data?
Metrics: correct identification, misidentification, false reject and false accept rates over a threshold sweep
(`dubbing.eval.speaker`).

**RQ3, ASR and background sound.** How does Japanese ASR degrade with music and effects, and does it fail by
dropping speech or by inventing text? Metrics: WER/CER with substitutions, deletions and insertions against SNR.

**RQ4, voice cloning.** How close is the synthetic voice to the reference, and does the engine matter?
Metrics: embedding cosine to the reference, A/B listening test.

## Status

| Study | State |
|---|---|
| Timing of the original pipeline's last run (RQ1) | done, descriptive, 16 segments |
| Five fitting strategies on shortened slots (RQ1) | done with synthetic slots; no human meaning ratings |
| Whisper under white noise (RQ3) | done on synthetic French speech, 16 clips; not real dubbing audio |
| Speaker thresholds (RQ2) | not run: needs real embeddings (pyannote) and labelled episodes |
| Voice similarity and A/B test (RQ4) | not run: needs the cloning engines and listeners |

Results and their limits: `results/RESULTS.md`.

## Design rules
Fixed seeds, model versions and thresholds written to the results files, no tuning on the evaluation data, LLM answers
cached so a run can be repeated. Commercial media is never redistributed (`ETHICS_AND_DATA.md`).

## Threats to validity
- small samples and one language pair;
- LLM adaptation and the LLM's own meaning score are not validated against human judgement;
- synthetic slots and synthetic speech stand in for real source audio;
- the speaker embedding model is a source of error besides the thresholds.

## Next
1. Collect human adequacy and naturalness ratings for the fitting strategies (the missing half of RQ1).
2. Repeat the noise study with music and effects mixed in.
3. Run RQ2 once pyannote and labelled episodes are available.

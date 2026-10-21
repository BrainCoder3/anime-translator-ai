# Problems found while testing

Each one is covered by a test.

| # | Problem | Fix |
|---|---|---|
| 1 | Whisper segment timestamps were far too coarse. On a 15 s clip with three sentences they came out as 0-5000, 5000-10000, 10000-15000 ms, while the speech was at about 0.9-3.8 s, 5.6-8.8 s and 10.6-13.5 s. The slot duration is the constraint the whole project works on. Found by the end-to-end run. | Use first and last *word* timestamps (error under about 0.4 s, about twice the transcription time). The regression test fails on the old behaviour. |
| 2 | Whisper can report an end time after the end of the file (5.42 s on a 4.54 s clip). | Ends are clamped to the real duration. |
| 3 | The number guard counted "un" and "une" as the number 1, so dropping the article in "une table" was rejected. | They are excluded by default (`include_ambiguous=True` restores the old behaviour). |
| 4 | The number guard did not see hyphenated numbers ("dix-huit", "vingt-cinq") or digits glued to a letter ("18h"), so it protected nothing in those sentences. Found when the experiment's own "numbers kept" column looked too good. | Compound numbers are parsed to their value and digits are matched even next to letters. |
| 5 | The semantic guard and the LLM prompt contained rules tied to one test scene (joker, recto/verso, light/shadow). | Concepts are a parameter, empty by default. The old list is in `dubbing/lexicons.py`. The prompt is generic. |
| 6 | `pydub` needs `audioop`, which recent Python versions removed. | Audio goes through `wave`, numpy and ffmpeg. |
| 7 | ffmpeg `atempo` is about 1 % off for extreme chained factors (1.2 % at x0.3). | Not an issue for the factors used here (up to 1.25 and 2.0). |
| 8 | When alignments tie, the substitution/deletion/insertion split is ambiguous (the total is exact). | Documented; one test pins each case. |
| 9 | The first version of the tests missed three real mistakes. A mutation test (deliberately breaking the code 14 times) showed it: the 25 % boundary, rescaling versus clipping in the mixer, and the S/D/I counts. | Three tests added. All 14 breakages are now caught. |

## Not verified here
- `PyannoteDiarizer`, and so cross-episode memory fed by real embeddings: pyannote, torch and a token were not available. `CharacterMemory` is tested with synthetic vectors only.
- Voice cloning (Chatterbox) stays in the prototype in `src/`; it needs a bigger GPU than the test PC has.
- edge-tts and Ollama are not deterministic. The experiments cache the LLM answers.

import json
import pathlib
import re
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing import media  # noqa: E402
from dubbing.config import Config  # noqa: E402
from dubbing.eval import timing  # noqa: E402
from dubbing.pipeline import STRATEGIES, DubbingPipeline  # noqa: E402

SR = 24000
MS_PER_CHAR = 60

UTTERANCES = [(0, 2000, "a"), (2500, 4000, "b"), (4500, 5300, "c")]
# a fits its slot, b and c are too long (c contains the number 3)
TRANSLATIONS = {"a": "Bonjour tout le monde ici",
                "b": "Voici une très longue phrase à raccourcir",
                "c": "Il faut gagner 3 fois"}
CANDIDATES = {
    TRANSLATIONS["b"]: [{"text": "Phrase à raccourcir", "valid": True, "semantic_score": 96, "reason": "ok"}],
    TRANSLATIONS["c"]: [{"text": "Gagner vite", "valid": True, "semantic_score": 99, "reason": "loses the 3"}],
}


class FakeASR:
    def transcribe(self, wav_path, language):
        return list(UTTERANCES)


class FakeTranslator:
    def __init__(self):
        self.asked = []

    def translate(self, text, source_lang, target_lang, *, previous="", next_=""):
        return TRANSLATIONS[text]

    def ask(self, prompt):
        self.asked.append(prompt)
        original = re.search(r"PHRASE À ADAPTER : (.*)", prompt).group(1).strip()
        return json.dumps({"candidates": CANDIDATES.get(original, [])})


class FakeTTS:
    """60 ms per character, divided by (1 + rate/100), with 100 ms of silence on both sides."""
    engine_name = "fake-tts"

    def __init__(self):
        self.calls = []

    def synthesize(self, text, out_wav, *, rate_percent=0, voice=None):
        self.calls.append((text, rate_percent, voice))
        n = int(SR * len(text) * MS_PER_CHAR / 1000 / (1 + rate_percent / 100))
        t = np.arange(n) / SR
        tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        pad = np.zeros(int(SR * 0.1), dtype=np.float32)
        return media.write_wav(out_wav, SR, np.concatenate([pad, tone, pad]))


def make_input(directory: pathlib.Path) -> pathlib.Path:
    t = np.arange(int(16000 * 6)) / 16000
    return media.write_wav(directory / "in.wav", 16000, (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32))


def build(strategy):
    tr, tts = FakeTranslator(), FakeTTS()
    return DubbingPipeline(Config(), FakeASR(), tr, tts, strategy=strategy, log=lambda m: None), tr, tts


@unittest.skipUnless(media.FFMPEG and __import__("shutil").which("ffmpeg"), "ffmpeg not found")
class PipelineTests(unittest.TestCase):
    def run_strategy(self, strategy):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = pathlib.Path(self.tmp.name)
        pipe, tr, tts = build(strategy)
        res = pipe.run(make_input(d), d / "out")
        return res, tr, tts, d / "out"

    def segs(self, res):
        return {s["id"]: s for s in res.audit["segments"]}

    def test_default_strategy_behaves_as_the_original_rule(self):
        res, tr, tts, out = self.run_strategy("adapt+speedup")
        s = self.segs(res)
        # segment 1 fits
        self.assertEqual((s[1]["rate_applied"], s[1]["adapted"], s[1]["overflow_ms"]), (0, False, 0))
        # segment 2 is rewritten shorter and then fits
        self.assertTrue(s[2]["adapted"])
        self.assertEqual(res.segments[1].text_fr, "Phrase à raccourcir")
        self.assertEqual(s[2]["overflow_ms"], 0)
        # segment 3: the only candidate loses the number 3, so the text stays and the speed-up is capped
        self.assertFalse(s[3]["adapted"])
        self.assertEqual(res.segments[2].text_fr, TRANSLATIONS["c"])
        self.assertEqual(s[3]["rate_applied"], 25)
        self.assertGreater(s[3]["overflow_ms"], 0)
        self.assertEqual(s[3]["attempts"][0]["rejected"][0]["rejected_by"], "guard")
        # the LLM is only asked for segments 2 and 3
        self.assertEqual(len(tr.asked), 2)

    def test_outputs_are_written_and_consistent(self):
        res, _, _, out = self.run_strategy("adapt+speedup")
        for key in ("voice", "srt", "segments", "audit"):
            self.assertTrue(res.paths[key].exists(), key)
        self.assertGreaterEqual(media.wav_duration_ms(res.paths["voice"]), 5800 - 50)
        self.assertEqual(res.paths["srt"].read_text(encoding="utf-8").count("-->"), 3)
        audit = json.loads(res.paths["audit"].read_text(encoding="utf-8"))
        blob = json.dumps(audit, ensure_ascii=False)
        for text in list(TRANSLATIONS.values()) + ["Phrase à raccourcir", "Gagner vite"]:
            self.assertNotIn(text, blob)
        stats = timing.overflow_stats(audit["segments"])
        self.assertEqual(stats["n_segments"], 3)
        self.assertEqual(stats["n_overrun"], 1)
        for rec in audit["segments"]:
            self.assertEqual(rec["overflow_ms"], max(0, rec["final_duration_ms"] - rec["target_ms"]))

    def test_none_strategy_leaves_the_raw_overrun(self):
        res, tr, tts, _ = self.run_strategy("none")
        s = self.segs(res)
        self.assertEqual([s[i]["rate_applied"] for i in (1, 2, 3)], [0, 0, 0])
        self.assertTrue(all(not s[i]["adapted"] for i in (1, 2, 3)))
        self.assertGreater(s[2]["overflow_ms"], 500)
        self.assertEqual(tr.asked, [])

    def test_speedup_only_caps_at_25_percent_and_never_rewrites(self):
        res, tr, _, _ = self.run_strategy("speedup")
        s = self.segs(res)
        self.assertEqual(s[2]["rate_applied"], 25)
        self.assertFalse(s[2]["adapted"])
        self.assertEqual(tr.asked, [])

    def test_uncapped_stretch_fits_but_exceeds_the_natural_limit(self):
        res, _, _, _ = self.run_strategy("stretch_uncapped")
        s = self.segs(res)
        self.assertGreater(s[2]["rate_applied"], 25)
        self.assertGreater(s[3]["rate_applied"], 25)
        self.assertLessEqual(s[3]["overflow_ms"], 60)

    def test_adapt_only_does_not_speed_up(self):
        res, _, _, _ = self.run_strategy("adapt")
        s = self.segs(res)
        self.assertEqual([s[i]["rate_applied"] for i in (1, 2, 3)], [0, 0, 0])
        self.assertTrue(s[2]["adapted"])

    def test_adaptation_failure_keeps_the_original_text(self):
        d = pathlib.Path(tempfile.mkdtemp())
        pipe, tr, _ = build("adapt+speedup")
        tr.ask = lambda prompt: (_ for _ in ()).throw(TimeoutError("ollama down"))
        res = pipe.run(make_input(d), d / "out")
        self.assertEqual(res.segments[1].text_fr, TRANSLATIONS["b"])
        self.assertIn("TimeoutError", json.dumps(res.audit["segments"][1]["attempts"]))

    def test_synthesis_is_cached(self):
        _, _, tts, _ = self.run_strategy("adapt+speedup")
        keys = [(t, r) for t, r, _ in tts.calls]
        self.assertEqual(len(keys), len(set(keys)))

    def test_unknown_strategy_and_empty_input(self):
        with self.assertRaises(ValueError):
            build("magic")
        self.assertEqual(len(STRATEGIES), 5)
        d = pathlib.Path(tempfile.mkdtemp())
        pipe, _, _ = build("none")
        pipe.asr = type("Silent", (), {"transcribe": lambda self, p, l: []})()
        with self.assertRaises(RuntimeError):
            pipe.run(make_input(d), d / "out")


if __name__ == "__main__":
    unittest.main()

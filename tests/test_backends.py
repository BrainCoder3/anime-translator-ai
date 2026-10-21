import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing.backends.translate_ollama import OllamaClient, OllamaTranslator, clean_translation  # noqa: E402
from dubbing.backends.tts_edge import rate_string  # noqa: E402
from dubbing.diarization import assign_speakers, overlap_ms  # noqa: E402
from dubbing.schema import Segment  # noqa: E402

LIVE = os.environ.get("DUBBING_LIVE") == "1"


def seg(i, a, b):
    return Segment(i, a, b, "x")


class DiarizationTests(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(overlap_ms(0, 10, 5, 20), 5)
        self.assertEqual(overlap_ms(0, 5, 5, 9), 0)

    def test_largest_overlap_wins(self):
        turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 2.0, "end": 6.0, "speaker": "B"}]
        got = assign_speakers([seg(1, 500, 1500), seg(2, 1500, 5000)], turns)
        self.assertEqual([g["speaker"] for g in got], ["A", "B"])
        self.assertEqual(got[1]["overlap_ms"], 3000)

    def test_micro_utterance_between_same_speaker_goes_to_the_other(self):
        turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 8.0, "end": 9.0, "speaker": "B"},
                 {"start": 4.0, "end": 6.0, "speaker": "A"}]
        # segment 2 overlaps nobody and sits between two lines of A
        got = assign_speakers([seg(1, 0, 2000), seg(2, 2500, 3500), seg(3, 4000, 6000)], turns)
        self.assertEqual([g["speaker"] for g in got], ["A", "B", "A"])
        self.assertEqual(got[1]["fallback"], "dialogue_alternation")

    def test_nearest_turn_fallback(self):
        turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}, {"start": 1.0, "end": 2.0, "speaker": "B"},
                 {"start": 9.0, "end": 10.0, "speaker": "C"}]
        got = assign_speakers([seg(1, 8000, 8400)], turns)
        self.assertEqual(got[0]["speaker"], "C")
        self.assertEqual(got[0]["fallback"], "nearest_turn")

    def test_no_turns(self):
        self.assertEqual(assign_speakers([seg(1, 0, 100)], [])[0]["speaker"], None)


class TranslationCleaningTests(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(clean_translation('  « Bonjour tout le monde »  '), "Bonjour tout le monde")
        self.assertEqual(clean_translation("<think>je réfléchis\nlongtemps</think>\nSalut !"), "Salut !")
        self.assertEqual(clean_translation("Ligne un\n\nLigne deux"), "Ligne un Ligne deux")
        self.assertEqual(clean_translation('"Citation"'), "Citation")

    def test_rate_string(self):
        self.assertEqual(rate_string(25), "+25%")
        self.assertEqual(rate_string(0), "+0%")
        self.assertEqual(rate_string(-10), "-10%")


class _Handler(BaseHTTPRequestHandler):
    """Fake /api/chat: records requests and can fail the first N calls."""
    requests: list = []
    fail_first = 0
    answer = "Traduction fausse"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append(body)
        if len(_Handler.requests) <= _Handler.fail_first:
            self.send_response(500)
            self.end_headers()
            return
        payload = json.dumps({"message": {"content": _Handler.answer}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class OllamaClientTests(unittest.TestCase):
    def setUp(self):
        _Handler.requests, _Handler.fail_first, _Handler.answer = [], 0, "Bonjour"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_request_shape(self):
        out = OllamaClient(self.url, "m", retries=0).chat("salut", json_mode=True, num_predict=77)
        self.assertEqual(out, "Bonjour")
        req = _Handler.requests[0]
        self.assertEqual(req["model"], "m")
        self.assertEqual(req["format"], "json")
        self.assertFalse(req["stream"])
        self.assertEqual(req["options"]["num_predict"], 77)
        self.assertNotIn("think", req)

    def test_think_flag_is_forwarded_when_set(self):
        OllamaClient(self.url, "m", retries=0, think=False).chat("x")
        self.assertIs(_Handler.requests[0]["think"], False)

    def test_retry_after_server_error(self):
        _Handler.fail_first = 1
        out = OllamaClient(self.url, "m", retries=1).chat("x")
        self.assertEqual(out, "Bonjour")
        self.assertEqual(len(_Handler.requests), 2)

    def test_gives_up_after_retries(self):
        _Handler.fail_first = 99
        with self.assertRaises(RuntimeError):
            OllamaClient(self.url, "m", retries=1).chat("x")
        self.assertEqual(len(_Handler.requests), 2)

    def test_translator_builds_prompt_and_cleans_output(self):
        _Handler.answer = ' "Salut le monde" '
        tr = OllamaTranslator(OllamaClient(self.url, "m", retries=0))
        out = tr.translate("こんにちは世界", "ja", "fr", previous="avant", next_="après")
        self.assertEqual(out, "Salut le monde")
        prompt = _Handler.requests[0]["messages"][0]["content"]
        for needle in ("japonais", "français", "こんにちは世界", "avant", "après"):
            self.assertIn(needle, prompt)

    def test_empty_translation_is_an_error(self):
        _Handler.answer = "<think>rien</think>"
        with self.assertRaises(RuntimeError):
            OllamaTranslator(OllamaClient(self.url, "m", retries=0)).translate("x", "ja", "fr")

    def test_unreachable_server(self):
        with self.assertRaises(RuntimeError):
            OllamaClient("http://127.0.0.1:9", "m", retries=0, timeout_s=2).chat("x")


@unittest.skipUnless(LIVE, "set DUBBING_LIVE=1 to run tests that need Whisper, Ollama and internet")
class LiveTests(unittest.TestCase):
    def test_tts_asr_roundtrip(self):
        from dubbing import media
        from dubbing.backends.asr_whisper import FasterWhisperASR
        from dubbing.backends.tts_edge import EdgeTTS
        from dubbing.eval import asr
        text = "Bonjour, je voudrais réserver une table pour deux personnes ce soir."
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            tts = EdgeTTS()
            normal = tts.synthesize(text, d / "n.wav")
            fast = tts.synthesize(text, d / "f.wav", rate_percent=25)
            self.assertLess(media.wav_duration_ms(fast), media.wav_duration_ms(normal))
            wav16 = media.to_wav(normal, d / "n16.wav")
            got = FasterWhisperASR("small").transcribe(wav16, "fr")
            hyp = " ".join(t for _, _, t in got)
            self.assertLess(asr.wer(text, hyp), 0.2, hyp)

    def test_utterance_spans_match_the_real_speech(self):
        """Regression: segment timestamps came out in blocks of 5 s on a clip like this."""
        import asyncio
        import numpy as np
        import edge_tts
        from dubbing import media
        from dubbing.backends.asr_whisper import FasterWhisperASR
        sentences = ["こんにちは、今日はとてもいい天気ですね。", "明日の朝八時に、駅の前で会いましょう。"]
        sr, gap = 16000, np.zeros(int(16000 * 0.7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            parts = [gap]
            for i, text in enumerate(sentences):
                asyncio.run(edge_tts.Communicate(text, "ja-JP-NanamiNeural").save(str(d / f"{i}.mp3")))
                parts += [media.read_wav(media.to_wav(d / f"{i}.mp3", d / f"{i}.wav"))[1], gap]
            wav = media.write_wav(d / "clip.wav", sr, np.concatenate(parts))
            x = np.concatenate(parts)
            idx = np.flatnonzero(np.abs(x) > 0.01)
            cuts = np.flatnonzero(np.diff(idx) > int(0.4 * sr))
            truth = list(zip([idx[0]] + [idx[c + 1] for c in cuts], [idx[c] for c in cuts] + [idx[-1]]))
            truth = [(a * 1000 // sr, b * 1000 // sr) for a, b in truth]
            got = FasterWhisperASR("small").transcribe(wav, "ja")
        self.assertEqual(len(got), len(truth), (got, truth))
        for (gs, ge, _), (ts, te) in zip(got, truth):
            inter = max(0, min(ge, te) - max(gs, ts))
            union = max(ge, te) - min(gs, ts)
            self.assertGreater(inter / union, 0.7, f"span {gs}-{ge} vs real {ts}-{te}")

    def test_ollama_translation(self):
        out = OllamaTranslator(OllamaClient()).translate("ありがとうございます。", "ja", "fr")
        self.assertTrue(any(w in out.lower() for w in ("merci", "remercie")), out)


if __name__ == "__main__":
    unittest.main()

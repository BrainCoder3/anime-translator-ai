import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing import media  # noqa: E402

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
SR = 16000


def tone(seconds: float, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class WavTests(unittest.TestCase):
    def test_roundtrip(self):
        x = tone(0.5)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "a.wav"
            media.write_wav(p, SR, x)
            sr, y = media.read_wav(p)
            self.assertEqual(sr, SR)
            self.assertEqual(y.size, x.size)
            self.assertLess(float(np.abs(x - y).max()), 1e-3)
            self.assertEqual(media.wav_duration_ms(p), 500)

    def test_clipping_on_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "a.wav"
            media.write_wav(p, SR, np.array([2.0, -2.0], dtype=np.float32))
            _, y = media.read_wav(p)
            self.assertLessEqual(float(np.abs(y).max()), 1.0)


class TrimTests(unittest.TestCase):
    def test_trim_removes_silence_but_keeps_padding(self):
        sil = np.zeros(SR, dtype=np.float32)
        x = np.concatenate([sil, tone(0.5), sil])
        y = media.trim_silence(x, SR, pad_ms=50)
        self.assertLess(y.size, x.size)
        self.assertAlmostEqual(y.size / SR, 0.5 + 0.1, delta=0.03)

    def test_trim_handles_degenerate_input(self):
        self.assertEqual(media.trim_silence(np.zeros(0, dtype=np.float32), SR).size, 0)
        z = np.zeros(1000, dtype=np.float32)
        self.assertEqual(media.trim_silence(z, SR).size, 1000)


class MixTests(unittest.TestCase):
    def test_clips_land_at_their_start_time(self):
        a, b = tone(0.2, amp=0.3), tone(0.2, freq=880, amp=0.3)
        out = media.mix_timeline([(0, a), (500, b)], SR, total_ms=1000)
        self.assertEqual(out.size, SR)
        self.assertGreater(float(np.abs(out[: int(0.2 * SR)]).max()), 0.2)
        self.assertEqual(float(np.abs(out[int(0.25 * SR): int(0.45 * SR)]).max()), 0.0)
        self.assertGreater(float(np.abs(out[int(0.5 * SR): int(0.7 * SR)]).max()), 0.2)

    def test_canvas_grows_and_overlap_is_rescaled_not_clipped(self):
        a = tone(0.2, amp=0.9)
        out = media.mix_timeline([(0, a), (0, a)], SR, total_ms=100)
        self.assertEqual(out.size, int(0.2 * SR))
        self.assertAlmostEqual(float(np.abs(out).max()), 1.0, places=5)


class MixRescaleTests(unittest.TestCase):
    def test_rescale_preserves_relative_levels(self):
        a = np.array([0.9, 0.45], dtype=np.float32)
        b = np.array([0.9, 0.0], dtype=np.float32)
        out = media.mix_timeline([(0, a), (0, b)], 1000, total_ms=2)
        self.assertAlmostEqual(float(out[0]), 1.0, places=5)
        self.assertAlmostEqual(float(out[1]), 0.25, places=5)


class SrtTests(unittest.TestCase):
    def test_time_format(self):
        self.assertEqual(media.srt_time(3_723_456), "01:02:03,456")
        self.assertEqual(media.srt_time(0), "00:00:00,000")

    def test_write_srt(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "a.srt"
            media.write_srt([(0, 1500, "Bonjour"), (2000, 3000, "Salut")], p)
            txt = p.read_text(encoding="utf-8")
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nBonjour", txt)
        self.assertIn("2\n00:00:02,000 --> 00:00:03,000\nSalut", txt)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not found")
class FfmpegTests(unittest.TestCase):
    def test_tempo_changes_duration_by_the_factor(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src = media.write_wav(d / "a.wav", SR, tone(2.0))
            for factor in (1.25, 0.8, 3.0, 0.3):
                out = media.tempo(src, d / f"o_{factor}.wav", factor)
                got = media.wav_duration_ms(out)
                expected = 2000 / factor
                # atempo is about 1 % off for extreme chained factors
                self.assertAlmostEqual(got, expected, delta=max(40, 0.02 * expected), msg=f"factor {factor}")
        with self.assertRaises(ValueError):
            media.tempo("x", "y", 0)

    def test_duration_ms_and_to_wav(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src = media.write_wav(d / "a.wav", SR, tone(1.5))
            self.assertAlmostEqual(media.duration_ms(src), 1500, delta=5)
            out = media.to_wav(src, d / "b.wav", sample_rate=8000)
            self.assertEqual(media.read_wav(out)[0], 8000)

    def test_ffmpeg_errors_are_reported(self):
        with self.assertRaises(RuntimeError):
            media.to_wav("does_not_exist.mp3", "out.wav")


if __name__ == "__main__":
    unittest.main()

import json
import math
import pathlib
import statistics
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing.eval import plots, stats, textsim  # noqa: E402


class ChrfTests(unittest.TestCase):
    def test_identity_and_case_space_insensitivity(self):
        self.assertEqual(textsim.chrf("Bonjour le monde", "bonjour  le MONDE"), 1.0)

    def test_disjoint_is_zero(self):
        self.assertEqual(textsim.chrf("aaaa", "bbbb"), 0.0)

    def test_empty_cases(self):
        self.assertEqual(textsim.chrf("", ""), 1.0)
        self.assertEqual(textsim.chrf("abc", ""), 0.0)

    def test_dropping_text_lowers_score_and_is_ordered(self):
        full = "il faut gagner trois victoires pour finir"
        mild = "il faut gagner trois victoires"
        heavy = "gagner"
        a, b = textsim.chrf(full, mild), textsim.chrf(full, heavy)
        self.assertTrue(0 < b < a < 1.0, (a, b))

    def test_recall_weighted(self):
        # beta=2 weights recall more than precision
        ref = "abcdefgh"
        self.assertGreater(textsim.chrf(ref, ref + "xy"), textsim.chrf(ref, "abcd"))


class StatsTests(unittest.TestCase):
    def test_bootstrap_ci_brackets_the_mean_and_is_reproducible(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        p, lo, hi = stats.bootstrap_ci(vals, seed=1)
        self.assertEqual(p, statistics.mean(vals))
        self.assertTrue(lo < p < hi)
        self.assertEqual(stats.bootstrap_ci(vals, seed=1), (p, lo, hi))
        _, lo2, hi2 = stats.bootstrap_ci(vals, seed=2)
        self.assertLess(abs(lo2 - lo), 0.5)
        self.assertLess(abs(hi2 - hi), 0.5)

    def test_constant_values_have_a_degenerate_interval(self):
        self.assertEqual(stats.bootstrap_ci([3.0] * 5), (3.0, 3.0, 3.0))

    def test_ratio_ci(self):
        point, lo, hi = stats.bootstrap_ratio_ci([1, 0, 2, 1], [4, 4, 4, 4], seed=0)
        self.assertAlmostEqual(point, 4 / 16)
        self.assertTrue(lo <= point <= hi)

    def test_linear_fit_exact_and_noisy(self):
        f = stats.linear_fit([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(f["slope"], 2.0)
        self.assertAlmostEqual(f["intercept"], 1.0)
        self.assertAlmostEqual(f["r2"], 1.0)
        g = stats.linear_fit([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])
        self.assertTrue(0 < g["r2"] < 1)
        self.assertEqual(g["n"], 5)

    def test_linear_fit_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            stats.linear_fit([1, 2], [1, 2])
        with self.assertRaises(ValueError):
            stats.linear_fit([2, 2, 2], [1, 2, 3])

    def test_pilot_fit_matches_a_hand_computation(self):
        data = json.loads((ROOT / "data" / "audit_v15_5_sanitized.json").read_text(encoding="utf-8"))
        pts = [s for s in data["segments"] if s["engine"].startswith("Chatterbox")]
        f = stats.linear_fit([s["target_ms"] for s in pts], [s["final_duration_ms"] for s in pts])
        self.assertEqual(f["n"], 14)
        self.assertAlmostEqual(f["slope"], 0.863, places=3)
        self.assertAlmostEqual(f["intercept"], 772, delta=1)
        self.assertAlmostEqual(f["r2"], 0.954, places=3)

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            stats.bootstrap_ci([])
        with self.assertRaises(ValueError):
            stats.bootstrap_ratio_ci([1], [1, 2])


class NoiseTests(unittest.TestCase):
    def test_target_snr_is_reached(self):
        import numpy as np
        from dubbing.eval import noise
        t = np.arange(16000) / 16000
        x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        for snr in (20, 10, 0, -5):
            y = noise.add_noise_at_snr(x, snr, np.random.default_rng(1))
            self.assertAlmostEqual(noise.measured_snr_db(x, y), snr, delta=0.05, msg=f"{snr} dB")

    def test_seeded_and_silent_input(self):
        import numpy as np
        from dubbing.eval import noise
        x = np.ones(1000, dtype=np.float32)
        a = noise.add_noise_at_snr(x, 5, np.random.default_rng(7))
        b = noise.add_noise_at_snr(x, 5, np.random.default_rng(7))
        self.assertTrue(np.array_equal(a, b))
        with self.assertRaises(ValueError):
            noise.add_noise_at_snr(np.zeros(10, dtype=np.float32), 5, np.random.default_rng(0))


def is_png(path: pathlib.Path) -> bool:
    return path.exists() and path.stat().st_size > 2000 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


class PlotTests(unittest.TestCase):
    def test_all_figures_render(self):
        data = json.loads((ROOT / "data" / "audit_v15_5_sanitized.json").read_text(encoding="utf-8"))
        segs = data["segments"]
        curves = {"small": [{"label": l, "x": i, "value": v, "lo": v - .02, "hi": v + .02}
                            for i, (l, v) in enumerate([("clean", .05), ("10", .1), ("0", .4)])]}
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            self.assertTrue(is_png(plots.timing_scatter(segs, d / "a.png", tol_ms=200)))
            self.assertTrue(is_png(plots.overrun_bars(segs, d / "b.png", tol_ms=50)))
            self.assertTrue(is_png(plots.wer_vs_snr(curves, d / "c.png")))
            self.assertTrue(is_png(plots.error_types_stack(
                [{"label": "clean", "S": .03, "D": .01, "I": .01}, {"label": "0", "S": .2, "D": .1, "I": .1}], d / "d.png")))
            self.assertTrue(is_png(plots.strategy_boxplots(
                {"ratio 0.75": {"none": [500, 600, 700], "adapt": [0, 10, 40]}}, d / "e.png")))
            self.assertTrue(is_png(plots.grouped_bars(
                {"g1": {"a": 1.0, "b": 2.0}, "g2": {"a": 2.0, "b": 1.5}}, d / "f.png",
                errors={"g1": {"a": (.8, 1.2), "b": (1.5, 2.5)}, "g2": {"a": (1.5, 2.5), "b": (1.2, 1.8)}})))


if __name__ == "__main__":
    unittest.main()

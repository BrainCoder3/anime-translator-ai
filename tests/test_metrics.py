import json
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing.eval import asr, speaker, timing  # noqa: E402


class TimingTests(unittest.TestCase):
    def test_real_audit_matches_known_values(self):
        data = json.loads((ROOT / "data" / "audit_v15_5_sanitized.json").read_text(encoding="utf-8"))
        segs = data["segments"]
        s = timing.overflow_stats(segs)
        self.assertEqual(s["n_segments"], 16)
        self.assertEqual(s["n_overrun"], 13)
        self.assertAlmostEqual(s["overrun_rate"], 13 / 16)
        self.assertEqual(s["max_overrun_ms"], 540)
        # recomputed overruns match the ids flagged in the file
        flagged = sorted(x["id"] for x in segs if x["final_duration_ms"] > x["target_ms"])
        self.assertEqual(flagged, sorted(data["overflow_segments"]))

    def test_group_by_engine_separates_engines(self):
        data = json.loads((ROOT / "data" / "audit_v15_5_sanitized.json").read_text(encoding="utf-8"))
        g = timing.by_group(data["segments"], "engine")
        self.assertEqual(g["CosyVoice 3"]["n_overrun"], 0)
        self.assertEqual(g["Chatterbox V15.3 reused"]["n_overrun"], 13)

    def test_tolerance_and_two_sided_fit(self):
        segs = [{"target_ms": 1000, "final_duration_ms": 1100},
                {"target_ms": 1000, "final_duration_ms": 900},
                {"target_ms": 1000, "final_duration_ms": 1500}]
        s = timing.overflow_stats(segs, tolerance_ms=150)
        self.assertEqual(s["n_overrun"], 1)
        self.assertAlmostEqual(s["fit_rate"], 2 / 3)
        self.assertAlmostEqual(s["mean_abs_error_ms"], (100 + 100 + 500) / 3)

    def test_percentile_interpolation(self):
        self.assertEqual(timing._percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(timing._percentile([10], 95), 10)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            timing.overflow_stats([])
        with self.assertRaises(ValueError):
            timing.overflow_stats([{"target_ms": 0, "final_duration_ms": 10}])

    def test_stretch_stats(self):
        st = timing.stretch_stats([{"atempo": 1.0}, {"atempo": 1.2}, {}])
        self.assertEqual(st["n_with_atempo"], 2)
        self.assertEqual(st["n_stretched"], 1)
        self.assertEqual(st["max_factor"], 1.2)


class AsrTests(unittest.TestCase):
    def test_known_wer(self):
        # two substitutions: sat->sit and the->very
        c = asr.error_counts("the cat sat on the mat", "the cat sit on very mat")
        self.assertEqual((c["S"], c["D"], c["I"]), (2, 0, 0))
        self.assertAlmostEqual(c["rate"], 2 / 6)

    def test_deletion_and_insertion(self):
        c = asr.error_counts("a b c d", "a c d e f")
        self.assertEqual(c["D"], 1)
        self.assertEqual(c["I"], 2)
        self.assertEqual(c["S"], 0)
        self.assertAlmostEqual(c["rate"], 3 / 4)

    def test_insertion_followed_by_deletion_keeps_separate_counts(self):
        # unique optimal alignment: insert "x", delete "d"; any other costs 4
        c = asr.error_counts("a b c d e f", "x a b c e f")
        self.assertEqual((c["S"], c["D"], c["I"]), (0, 1, 1))
        self.assertAlmostEqual(c["rate"], 2 / 6)

    def test_total_rate_is_unaffected_by_tied_alignments(self):
        # two substitutions or one insertion plus one deletion: same cost, same rate
        c = asr.error_counts("a b c d", "x a c d")
        self.assertAlmostEqual(c["rate"], 2 / 4)
        self.assertEqual(c["S"] + c["D"] + c["I"], 2)

    def test_perfect_and_normalisation(self):
        self.assertEqual(asr.wer("Bonjour, le monde !", "bonjour le monde"), 0.0)
        self.assertEqual(asr.wer("Ça va?", "ça va"), 0.0)

    def test_wer_can_exceed_one(self):
        self.assertAlmostEqual(asr.wer("a", "b c d"), 3.0)

    def test_cer(self):
        self.assertAlmostEqual(asr.cer("abcd", "abxd"), 0.25)

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            asr.wer("", "anything")

    def test_corpus_is_micro_averaged(self):
        r = asr.corpus_error_rate([("a b", "a b"), ("c d e f", "c d e x")])
        self.assertAlmostEqual(r["rate"], 1 / 6)


class SpeakerTests(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(speaker.cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(speaker.cosine([1, 1], [2, 2]), 1.0)
        with self.assertRaises(ValueError):
            speaker.cosine([0, 0], [1, 0])

    def test_prototypes_and_identify(self):
        protos = speaker.prototypes([[1, 0], [0.9, 0.1], [0, 1]], ["A", "A", "B"])
        self.assertEqual(set(protos), {"A", "B"})
        lab, best, _ = speaker.identify([0.95, 0.05], protos, threshold=0.8)
        self.assertEqual(lab, "A")
        self.assertGreater(best, 0.99)

    def test_unknown_rejected_by_threshold(self):
        protos = {"A": [1, 0], "B": [0, 1]}
        lab, _, _ = speaker.identify([1, 1], protos, threshold=0.9)
        self.assertIsNone(lab)

    def test_margin_rejects_ambiguous(self):
        protos = {"A": [1, 0], "B": [0.9, 0.436]}
        probe = [0.95, 0.2]
        self.assertIsNotNone(speaker.identify(probe, protos, threshold=0.5, margin=0.0)[0])
        self.assertIsNone(speaker.identify(probe, protos, threshold=0.5, margin=0.2)[0])

    def test_open_set_sweep(self):
        protos = {"A": [1, 0], "B": [0, 1]}
        items = [([1, 0.05], "A"), ([0.05, 1], "B"), ([0.7, 0.7], "A"),
                 ([-1, -1], "C")]
        low, high = speaker.open_set_eval(items, protos, thresholds=[-1.0, 0.95])
        # permissive: nothing is rejected, but the exact A/B tie is misidentified
        self.assertAlmostEqual(low["correct_id_rate"], 2 / 3)
        self.assertAlmostEqual(low["misid_rate"], 1 / 3)
        self.assertAlmostEqual(low["false_reject_rate"], 0.0)
        self.assertAlmostEqual(low["false_accept_rate"], 1.0)
        self.assertAlmostEqual(high["false_accept_rate"], 0.0)
        self.assertAlmostEqual(high["false_reject_rate"], 1 / 3)
        self.assertEqual((high["n_known"], high["n_unknown"]), (3, 1))

    def test_margin_turns_a_tie_into_a_rejection(self):
        protos = {"A": [1, 0], "B": [0, 1]}
        tie = [0.7, 0.7]
        self.assertIsNotNone(speaker.identify(tie, protos, threshold=0.0, margin=0.0)[0])
        self.assertIsNone(speaker.identify(tie, protos, threshold=0.0, margin=0.05)[0])

    def test_similarity_to_reference(self):
        self.assertAlmostEqual(speaker.similarity_to_reference([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertTrue(math.isfinite(speaker.similarity_to_reference([1, 0], [1, 1])))


if __name__ == "__main__":
    unittest.main()

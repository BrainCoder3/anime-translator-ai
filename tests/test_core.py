import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing import adaptation, audit, fit, lexicons, textutil  # noqa: E402
from dubbing.config import FitConfig, SpeakerThresholds  # noqa: E402
from dubbing.schema import Segment  # noqa: E402
from dubbing.speaker_memory import CharacterMemory, Status, cosine, decide, normalize  # noqa: E402

CFG = FitConfig()


class SchemaTests(unittest.TestCase):
    def test_target_and_roundtrip(self):
        s = Segment(1, 1000, 3500, "テスト", text_fr="Test")
        self.assertEqual(s.target_ms, 2500)
        self.assertEqual(Segment.from_dict(s.to_dict()), s)

    def test_invalid_span(self):
        with self.assertRaises(ValueError):
            Segment(1, 2000, 2000, "x")


class TextUtilTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(textutil.canonical_number_tokens("Il en faut trois ou 12."), {"3", "12"})
        self.assertEqual(textutil.canonical_number_tokens("une table"), set())
        self.assertEqual(textutil.canonical_number_tokens("une table", include_ambiguous=True), {"1"})

    def test_hyphenated_and_compound_numbers(self):
        tok = textutil.canonical_number_tokens
        self.assertEqual(tok("à dix-huit heures"), {"18"})
        self.assertEqual(tok("moins de vingt-cinq ans"), {"25"})
        self.assertEqual(tok("quatre-vingt-dix euros"), {"90"})
        self.assertEqual(tok("soixante-dix-sept"), {"77"})
        self.assertEqual(tok("vingt-et-un et vingt et un"), {"21"})
        self.assertEqual(tok("il a peut-être raison"), set())

    def test_digits_glued_to_letters(self):
        tok = textutil.canonical_number_tokens
        self.assertEqual(tok("rendez-vous à 18h"), {"18"})
        self.assertEqual(tok("25ans"), {"25"})
        self.assertEqual(tok("environ 3,5 km"), {"3,5"})
        self.assertEqual(tok("1234"), {"1234"})

    def test_spelled_and_digit_forms_match_in_the_guard(self):
        orig = "La boutique ferme à dix-huit heures"
        self.assertTrue(textutil.basic_semantic_guard(orig, "Ferme à 18h")[0])
        ok, why = textutil.basic_semantic_guard(orig, "Elle ferme tôt")
        self.assertFalse(ok)
        self.assertIn("'18'", why)

    def test_guard_number_preserved_or_lost(self):
        self.assertEqual(textutil.basic_semantic_guard("Il faut trois victoires", "Trois victoires")[0], True)
        ok, why = textutil.basic_semantic_guard("Il faut trois victoires", "Il faut gagner")
        self.assertFalse(ok)
        self.assertIn("number lost", why)

    def test_guard_empty(self):
        self.assertFalse(textutil.basic_semantic_guard("a", "  ")[0])

    def test_concept_guard_is_off_by_default_and_works_when_given(self):
        orig, cand = "Choisis ton camp.", "Lumière ou ombre ?"
        self.assertTrue(textutil.basic_semantic_guard(orig, cand)[0])
        ok, why = textutil.basic_semantic_guard(orig, cand, lexicons.LEGACY_TEST_SCENE)
        self.assertFalse(ok)
        self.assertIn("concept added", why)
        # dropping a concept is allowed, adding one is not
        self.assertTrue(textutil.basic_semantic_guard("La lumière ou l'ombre ?", "Lumière ?",
                                                      lexicons.LEGACY_TEST_SCENE)[0])

    def test_extract_json(self):
        self.assertEqual(textutil.extract_json_payload('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(textutil.extract_json_payload('Voici : {"a": 2} merci'), {"a": 2})
        with self.assertRaises(ValueError):
            textutil.extract_json_payload("pas de json")

    def test_word_budget(self):
        self.assertEqual(textutil.word_budget(1.0), 3)
        self.assertEqual(textutil.word_budget(5.0), 17)


class FitTests(unittest.TestCase):
    def test_required_rate(self):
        self.assertEqual(fit.required_rate(900, 1000), 0)
        self.assertEqual(fit.required_rate(1250, 1000), 25)
        self.assertEqual(fit.required_rate(2000, 1000), 100)
        with self.assertRaises(ValueError):
            fit.required_rate(1, 0)

    def test_decide(self):
        self.assertFalse(fit.decide(1100, 1000, CFG).needs_adaptation)
        d = fit.decide(1500, 1000, CFG)
        self.assertTrue(d.needs_adaptation)
        self.assertEqual(d.speedup_rate, 25)
        self.assertTrue(fit.decide(1040, 1000, CFG).acceptable_now)
        self.assertFalse(fit.decide(1100, 1000, CFG).acceptable_now)

    def test_adaptation_boundary_is_strict(self):
        """25 % is still natural, 26 % is not."""
        self.assertEqual(fit.required_rate(1250, 1000), 25)
        self.assertFalse(fit.decide(1250, 1000, CFG).needs_adaptation)
        self.assertEqual(fit.required_rate(1260, 1000), 26)
        self.assertTrue(fit.decide(1260, 1000, CFG).needs_adaptation)
        self.assertEqual(fit.decide(1260, 1000, CFG).speedup_rate, 25)

    def test_residual_overrun_after_speedup(self):
        self.assertAlmostEqual(fit.residual_overrun_ms(1250, 1000, 25), 0.0)
        self.assertAlmostEqual(fit.residual_overrun_ms(1500, 1000, 25), 200.0)


class AdaptationTests(unittest.TestCase):
    def test_parse_candidates_tolerates_noise(self):
        payload = {"candidates": [{"text": "A", "valid": True, "semantic_score": "95"},
                                  {"text": "", "valid": True}, "junk",
                                  {"text": "B", "semantic_score": "n/a"}, {"text": "C"}]}
        got = adaptation.parse_candidates(payload, limit=2)
        self.assertEqual([c.text for c in got], ["A", "B"])
        self.assertEqual((got[0].semantic_score, got[1].semantic_score), (95, 0))

    def test_vet_rejects_for_each_reason(self):
        cands = [adaptation.Candidate("Trois", True, 99),
                 adaptation.Candidate("Gagne", True, 99),
                 adaptation.Candidate("Trois ok", True, 80),
                 adaptation.Candidate("Trois go", False, 99)]
        kept, rejected = adaptation.vet_candidates("Il faut trois victoires", cands, CFG)
        self.assertEqual([c.text for c in kept], ["Trois"])
        self.assertEqual([r["rejected_by"] for r in rejected], ["guard", "llm_score", "llm_score"])

    def test_choose_prefers_score_then_rate(self):
        a = adaptation.Candidate("A", True, 95)
        b = adaptation.Candidate("B", True, 99)
        c = adaptation.Candidate("C", True, 99)
        # target 1000 ms: B and C score 99, C needs less speed-up
        best = adaptation.choose_candidate([(a, 1100), (b, 1200), (c, 1100)], 1000, CFG)
        self.assertEqual(best.text, "C")
        self.assertIsNone(adaptation.choose_candidate([(a, 2000)], 1000, CFG))

    def test_adapt_end_to_end_with_fakes(self):
        answer = json.dumps({"candidates": [{"text": "Trois victoires", "valid": True, "semantic_score": 97},
                                            {"text": "Gagne", "valid": True, "semantic_score": 99}]})
        measured = {"Trois victoires": 1100}
        res = adaptation.adapt("Il faut gagner trois victoires pour finir", 1000, ask=lambda p: answer,
                               measure_ms=lambda t: measured[t], cfg=CFG)
        self.assertTrue(res.adapted)
        self.assertEqual(res.text, "Trois victoires")
        blob = json.dumps(res.attempts)
        self.assertNotIn("Trois victoires", blob)

    def test_adapt_falls_back_to_original_on_llm_failure(self):
        def boom(prompt):
            raise TimeoutError("ollama down")
        res = adaptation.adapt("Texte original", 1000, ask=boom, measure_ms=lambda t: 0, cfg=CFG)
        self.assertFalse(res.adapted)
        self.assertEqual(res.text, "Texte original")
        self.assertIn("TimeoutError", res.attempts[0]["error"])

    def test_adapt_keeps_original_when_nothing_fits(self):
        answer = json.dumps({"candidates": [{"text": "Court", "valid": True, "semantic_score": 99}]})
        res = adaptation.adapt("Texte", 1000, ask=lambda p: answer, measure_ms=lambda t: 3000, cfg=CFG)
        self.assertFalse(res.adapted)
        self.assertEqual(res.text, "Texte")

    def test_prompt_mentions_budget_and_has_no_scene_specific_rules(self):
        p = adaptation.build_prompt("Bonjour", 2.0, CFG)
        self.assertIn("7 mots", p)
        for word in ("Joker", "recto", "verso", "lumière"):
            self.assertNotIn(word, p)


class SpeakerMemoryTests(unittest.TestCase):
    TH = SpeakerThresholds(match=0.84, margin=0.08, new_character=0.72)

    def test_decide_regions(self):
        self.assertIs(decide(0.90, 0.20, self.TH), Status.MATCHED)
        self.assertIs(decide(0.90, 0.05, self.TH), Status.AMBIGUOUS)
        self.assertIs(decide(0.80, 0.30, self.TH), Status.AMBIGUOUS)
        self.assertIs(decide(0.50, 0.30, self.TH), Status.NEW)
        self.assertIs(decide(0.84, 0.08, self.TH), Status.MATCHED)

    def test_normalize(self):
        self.assertAlmostEqual(sum(x * x for x in normalize([3, 4])), 1.0)
        self.assertIsNone(normalize([0, 0]))
        self.assertIsNone(normalize([float("nan"), 1]))
        self.assertIsNone(normalize([]))

    def test_first_episode_creates_distinct_characters_with_distinct_voices(self):
        mem = CharacterMemory(self.TH)
        out = mem.recognize({"S0": [1, 0], "S1": [0, 1]}, "ep1")
        self.assertEqual({a.status for a in out.values()}, {Status.NEW})
        self.assertEqual(len({a.character for a in out.values()}), 2)
        self.assertEqual(len({a.voice for a in out.values()}), 2)

    def test_second_episode_matches_and_updates_centroid_once(self):
        mem = CharacterMemory(self.TH)
        mem.recognize({"S0": [1, 0], "S1": [0, 1]}, "ep1")
        out = mem.recognize({"A": [0.99, 0.05], "B": [0.05, 0.99]}, "ep2")
        self.assertEqual([a.status for a in out.values()], [Status.MATCHED, Status.MATCHED])
        self.assertEqual({a.character for a in out.values()}, {"CHARACTER_001", "CHARACTER_002"})
        self.assertEqual(mem.meta["CHARACTER_001"]["embedding_count"], 2)
        before = list(mem.embeddings["CHARACTER_001"])
        mem.recognize({"A": [0.99, 0.05], "B": [0.05, 0.99]}, "ep2")
        self.assertEqual(mem.meta["CHARACTER_001"]["embedding_count"], 2)
        self.assertEqual(mem.embeddings["CHARACTER_001"], before)

    def test_unknown_speaker_becomes_new(self):
        mem = CharacterMemory(self.TH)
        mem.recognize({"S0": [1, 0, 0]}, "ep1")
        out = mem.recognize({"X": [0, 0, 1]}, "ep2")
        self.assertIs(out["X"].status, Status.NEW)
        self.assertEqual(out["X"].character, "CHARACTER_002")

    def test_one_to_one_assignment_flags_duplicates(self):
        mem = CharacterMemory(self.TH)
        mem.recognize({"S0": [1, 0]}, "ep1")
        out = mem.recognize({"A": [1, 0.01], "B": [1, 0.02]}, "ep2")
        self.assertIs(out["A"].status, Status.MATCHED)
        self.assertIs(out["B"].status, Status.AMBIGUOUS_DUPLICATE)
        self.assertIsNone(out["B"].character)
        self.assertEqual(len(mem.meta), 1)

    def test_ambiguous_does_not_touch_memory(self):
        mem = CharacterMemory(self.TH)
        mem.recognize({"S0": [1, 0]}, "ep1")
        out = mem.recognize({"A": [0.78, 0.62]}, "ep2")
        self.assertIs(out["A"].status, Status.AMBIGUOUS)
        self.assertEqual(len(mem.meta), 1)
        self.assertEqual(mem.meta["CHARACTER_001"]["embedding_count"], 1)

    def test_invalid_embedding_is_rejected(self):
        with self.assertRaises(ValueError):
            CharacterMemory().recognize({"S0": [0, 0]}, "ep1")

    def test_save_and_load(self):
        mem = CharacterMemory(self.TH)
        mem.recognize({"S0": [1, 0], "S1": [0, 1]}, "ep1")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "mem.json"
            mem.save(p)
            again = CharacterMemory.load(p)
        self.assertEqual(again.meta, mem.meta)
        self.assertEqual(again.th, self.TH)
        self.assertAlmostEqual(cosine(again.embeddings["CHARACTER_001"], mem.embeddings["CHARACTER_001"]), 1.0)


class AuditTests(unittest.TestCase):
    def test_build_and_roundtrip(self):
        recs = [audit.SegmentAudit(1, "C1", "edge-tts", 1000, 1200, 200),
                audit.SegmentAudit(2, "C1", "edge-tts", 1000, 900, 0)]
        a = audit.build_audit(recs, {"x": 1})
        self.assertEqual(a["overflow_segments"], [1])
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "audit.json"
            audit.save_audit(a, p)
            self.assertEqual(audit.load_audit(p), a)

    def test_compatible_with_timing_metrics(self):
        from dubbing.eval import timing
        recs = [audit.SegmentAudit(1, "C1", "edge-tts", 1000, 1200, 200)]
        stats = timing.overflow_stats(audit.build_audit(recs, {})["segments"])
        self.assertEqual(stats["n_overrun"], 1)


if __name__ == "__main__":
    unittest.main()

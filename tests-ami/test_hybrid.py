import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ami-nano"))

from hybrid import evaluate_candidate, match_knowledge, respond
from chat_es import load_dialogs


class HybridQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogs = load_dialogs()

    def test_known_question_uses_local_knowledge_with_explanation_and_example(self):
        result = respond("¿Quién eres?", self.dialogs, candidate="texto neuronal irrelevante")
        self.assertEqual(result["source"], "local_knowledge")
        self.assertTrue(result["quality"]["accepted"])
        self.assertIn("Explicación:", result["text"])
        self.assertIn("Ejemplo relacionado:", result["text"])
        self.assertIn("AMILCAR AI", result["text"])

    def test_unknown_question_does_not_pass_gibberish(self):
        result = respond(
            "explica la fotosintesis de las plantas",
            self.dialogs,
            candidate="zzzzzzzzzzzzzzzz qqqqqqqqqqqqqqq",
        )
        self.assertEqual(result["source"], "safe_fallback")
        self.assertFalse(result["quality"]["accepted"])
        self.assertIn("no inventar", result["text"].lower())

    def test_quality_rules_reject_empty_or_repetitive_text(self):
        self.assertFalse(evaluate_candidate("explica plantas", "")["accepted"])
        repetitive = evaluate_candidate(
            "explica plantas",
            "aaaaaa aaaaaa aaaaaa aaaaaa aaaaaa",
        )
        self.assertFalse(repetitive["accepted"])
        self.assertFalse(repetitive["checks"]["no_repeated_run"])

    def test_candidate_requires_lexical_relation_to_question(self):
        candidate = (
            "La música produce sonidos variados y melodías agradables "
            "para escuchar durante el día."
        )
        result = evaluate_candidate("explica fotosintesis plantas", candidate)
        self.assertFalse(result["checks"]["question_relevance"])
        self.assertFalse(result["accepted"])

    def test_candidate_quality_checks_can_accept_a_well_formed_related_text(self):
        candidate = (
            "La fotosintesis permite a las plantas transformar la luz solar "
            "en energia quimica para crecer y producir alimento."
        )
        result = evaluate_candidate("explica fotosintesis plantas", candidate)
        self.assertTrue(result["accepted"], result["checks"])
        self.assertGreaterEqual(len(result["relevant_terms"]), 2)

    def test_local_match_reports_score(self):
        match = match_knowledge("hola", self.dialogs)
        self.assertIsNotNone(match)
        self.assertEqual(match["item"]["intent"], "saludo")
        self.assertEqual(match["score"], 1.0)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ami-nano"))
import chat_es

class AmiNano21KnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dialogs = chat_es.load_dialogs()

    def test_knowledge_base_is_loaded(self):
        self.assertGreaterEqual(len(self.dialogs), 40)

    def test_common_questions_have_answers(self):
        cases = {
            "hola": "hola",
            "que es javascript": "javascript",
            "que es python": "python",
            "que es github": "github",
            "necesitas internet": "internet",
            "quien te creo": "amilcar",
            "que es una red neuronal": "red neuronal",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                answer = chat_es.respond(question, self.dialogs).lower()
                self.assertIn(expected, answer)

    def test_unknown_question_uses_safe_fallback(self):
        answer = chat_es.respond("cuanto mide un dragon de marte", self.dialogs)
        self.assertIn("No encuentro", answer)

    def test_empty_message_is_handled(self):
        self.assertIn("pregunta", chat_es.respond("   ", self.dialogs))

    def test_data_has_valid_schema(self):
        for item in self.dialogs:
            self.assertTrue(item["patterns"])
            self.assertTrue(item["responses"])
            self.assertTrue(all(isinstance(p, str) for p in item["patterns"]))
            self.assertTrue(all(isinstance(a, str) for a in item["responses"]))

if __name__ == "__main__":
    unittest.main()

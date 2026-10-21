import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "ami-nano" / "chat_es.py"

class SpanishChatTests(unittest.TestCase):
    def test_common_questions_return_spanish_answers(self):
        env = os.environ.copy()
        cases = {
            "hola": "¡Hola!",
            "¿Quién eres?": "Soy AMI Nano",
            "¿Cómo estás?": "funcionando localmente",
            "¿Necesitas internet?": "localmente",
            "¿Qué opinas de aprender cosas nuevas?": "ampliar nuestros conocimientos",
            "¿Dónde sos tú?": "dispositivo donde se ejecuta",
            "¿Cómo puedo ayudarte a mejorar?": "data/dialogos-es.json",
            "¿Qué opinas de la inteligencia artificial?": "puede ser útil",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                result = subprocess.run(
                    [sys.executable, str(CHAT), question],
                    cwd=ROOT, env=env, text=True, capture_output=True, timeout=5,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected.lower(), result.stdout.lower())

    def test_unknown_question_uses_honest_fallback(self):
        result = subprocess.run(
            [sys.executable, str(CHAT), "resuelve una ecuacion avanzada de astrofisica"],
            cwd=ROOT, text=True, capture_output=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Todavía no sé responder", result.stdout)

if __name__ == "__main__":
    unittest.main()

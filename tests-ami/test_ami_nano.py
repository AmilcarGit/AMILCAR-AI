import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "ami-nano" / "train.py"
GENERATE = ROOT / "ami-nano" / "generate.py"

class AmiNanoTests(unittest.TestCase):
    def test_training_and_generation_from_temporary_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.txt"
            model = Path(tmp) / "model.json"
            corpus.write_text(
                ("hola ami. como estas? soy ami nano. " * 8),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["AMI_CORPUS"] = str(corpus)
            env["AMI_MODEL_PATH"] = str(model)
            env["AMI_EPOCHS"] = "3"

            trained = subprocess.run(
                [sys.executable, str(TRAIN)],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertTrue(model.is_file(), "No se creó el checkpoint")

            data = json.loads(model.read_text(encoding="utf-8"))
            self.assertEqual(data.get("format"), "ami-nano-neural-bigram-v2")
            self.assertTrue(data.get("trained_from_scratch"))
            self.assertEqual(data.get("architecture"), "one-hot-character-to-softmax")
            self.assertEqual(len(data.get("weights", [])), len(data.get("vocab", [])))
            self.assertTrue(data.get("weights"))

            generated = subprocess.run(
                [sys.executable, str(GENERATE), "hola"],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertTrue(generated.stdout.strip(), "La generación salió vacía")
            self.assertIn("hola", generated.stdout.lower())

    def test_generation_fails_cleanly_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["AMI_MODEL_PATH"] = str(Path(tmp) / "missing.json")
            result = subprocess.run(
                [sys.executable, str(GENERATE), "hola"],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("npm run ami:train", result.stderr)

if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "ai" / "local_worker.py"

class AmiNano21WorkerTests(unittest.TestCase):
    def test_sdk_worker_answers_from_local_knowledge_without_checkpoint(self):
        request = {"id": 7, "action": "chat", "message": "que es javascript"}
        result = subprocess.run(
            [sys.executable, str(WORKER)],
            input=json.dumps(request, ensure_ascii=False) + "\n",
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 2, result.stdout)
        ready = json.loads(lines[0])
        response = json.loads(lines[1])
        self.assertTrue(ready.get("ready"), ready)
        self.assertEqual(response.get("id"), 7)
        self.assertIn("JavaScript", response.get("reply", ""))
        self.assertEqual(response.get("model"), "AMI Nano 2.1")
        self.assertEqual(response.get("source"), "knowledge")

if __name__ == "__main__":
    unittest.main()

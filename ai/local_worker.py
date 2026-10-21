from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ai.model.inference import ModelRunner

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("AMILCAR_AI_MODEL_PATH", str(ROOT / "artifacts" / "ami-base.pt")))


def write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    try:
        runner = ModelRunner(MODEL_PATH)
    except Exception as exc:
        write({"ready": False, "error": f"No se pudo cargar el modelo: {exc}"})
        return

    write({
        "ready": True,
        "available": runner.available,
        "model": "AMI-Base",
        "model_path": str(MODEL_PATH),
    })

    for line in sys.stdin:
        request: object = {}
        try:
            request = json.loads(line)
            request_id = request.get("id")
            action = request.get("action", "chat")

            if action == "health":
                response = {
                    "id": request_id,
                    "status": "ok",
                    "model": "AMI-Base",
                    "trained": runner.available,
                }
            elif action == "chat":
                message = request.get("message")
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("Escribe un mensaje no vacío.")
                if not runner.available:
                    response = {
                        "id": request_id,
                        "error": "No se encontró un modelo entrenado. Ejecuta: python -m ai.training.train",
                        "code": "MODEL_NOT_FOUND",
                    }
                else:
                    reply = runner.generate(message.strip())
                    response = {
                        "id": request_id,
                        "reply": reply,
                        "model": "AMI-Base",
                        "trained": True,
                    }
            else:
                response = {"id": request_id, "error": "Acción no admitida.", "code": "INVALID_ACTION"}
            write(response)
        except Exception as exc:
            write({
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": str(exc),
                "code": "LOCAL_INFERENCE_ERROR",
            })


if __name__ == "__main__":
    main()

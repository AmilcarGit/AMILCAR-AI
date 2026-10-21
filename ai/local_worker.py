from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("AMILCAR_AI_MODEL_PATH", str(ROOT / "artifacts" / "ami-base.pt")))
CHAT_PATH = ROOT / "ami-nano" / "chat_es.py"
CHAT_MODULE = None
CHAT_LOAD_ERROR = None
try:
    spec = importlib.util.spec_from_file_location("ami_nano_chat_es", CHAT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("No se pudo cargar el chat local en español.")
    CHAT_MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CHAT_MODULE)
except Exception as exc:
    CHAT_LOAD_ERROR = str(exc)

try:
    from ai.model.inference import ModelRunner
except Exception as exc:
    ModelRunner = None
    MODEL_IMPORT_ERROR = str(exc)
else:
    MODEL_IMPORT_ERROR = None


def write(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    runner = None
    model_error = MODEL_IMPORT_ERROR
    if ModelRunner is not None:
        try:
            runner = ModelRunner(MODEL_PATH)
        except Exception as exc:
            model_error = str(exc)

    knowledge_ready = CHAT_MODULE is not None
    if not knowledge_ready and runner is None:
        write({
            "ready": False,
            "error": "No se pudo iniciar AMI Nano 2.1. " + str(CHAT_LOAD_ERROR or model_error or "Motor local no disponible."),
        })
        return

    write({
        "ready": True,
        "available": bool(runner and runner.available),
        "knowledge_ready": knowledge_ready,
        "model": "AMI Nano 2.1",
        "model_path": str(MODEL_PATH),
    })

    for line in sys.stdin:
        request: object = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Cada solicitud debe ser un objeto JSON.")
            request_id = request.get("id")
            action = request.get("action", "chat")

            if action == "health":
                write({
                    "id": request_id,
                    "status": "ok",
                    "model": "AMI Nano 2.1",
                    "trained": bool(runner and runner.available),
                    "knowledge_ready": knowledge_ready,
                })
                continue

            if action != "chat":
                write({"id": request_id, "error": "Acción no admitida.", "code": "INVALID_ACTION"})
                continue

            message = request.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("Escribe un mensaje no vacío.")

            reply = None
            source = None
            if knowledge_ready:
                known_reply = CHAT_MODULE.respond(message.strip())
                if not known_reply.startswith("No encuentro una respuesta suficientemente fiable"):
                    reply, source = known_reply, "knowledge"
            if reply is None and runner is not None and runner.available:
                reply, source = runner.generate(message.strip()), "neural"
            if reply is None:
                reply = (
                    known_reply if knowledge_ready else
                    "AMI Nano 2.1 no tiene un modelo neuronal entrenado y la base de conocimientos no está disponible."
                )
                source = "knowledge-fallback" if knowledge_ready else "unavailable"

            write({
                "id": request_id,
                "reply": reply,
                "model": "AMI Nano 2.1",
                "trained": bool(runner and runner.available),
                "source": source,
            })
        except Exception as exc:
            write({
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": str(exc),
                "code": "LOCAL_INFERENCE_ERROR",
            })


if __name__ == "__main__":
    main()

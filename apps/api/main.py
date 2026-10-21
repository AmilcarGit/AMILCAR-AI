from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai.model.inference import ModelRunner

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("AMILCAR_AI_MODEL_PATH", str(ROOT / "artifacts" / "ami-base.pt")))
MAX_MESSAGE_CHARS = int(os.getenv("AMILCAR_AI_MAX_MESSAGE_CHARS", "2000"))
MAX_NEW_TOKENS = int(os.getenv("AMILCAR_AI_MAX_NEW_TOKENS", "180"))

app = FastAPI(title="AMILCAR AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
runner = ModelRunner(MODEL_PATH)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    model: str
    trained: bool


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "app": "AMILCAR AI", "model": "AMI-Base", "trained": runner.available}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Escribe un mensaje.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="El mensaje supera el límite permitido.")
    if not runner.available:
        raise HTTPException(status_code=503, detail="AMI-Base todavía no está entrenado. Ejecuta: python -m ai.training.train")
    try:
        reply = runner.generate(message, max_new_tokens=MAX_NEW_TOKENS)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="No se pudo generar la respuesta.") from exc
    return ChatResponse(reply=reply, model="AMI-Base", trained=True)

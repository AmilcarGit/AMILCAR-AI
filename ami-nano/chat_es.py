#!/usr/bin/env python3
"""Chat local en español para AMI Nano: respuestas por intención, sin APIs ni dependencias."""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AMI_DIALOGS_PATH", ROOT / "data" / "dialogos-es.json"))

def normalize(text):
    text = unicodedata.normalize("NFD", text.lower().strip())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    return " ".join(text.split())

def load_dialogs():
    with DATA.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("El archivo de diálogos debe contener una lista.")
    return data

def respond(message, dialogs=None):
    if not isinstance(message, str) or not message.strip():
        return "Escríbeme algo y trataré de ayudarte."
    dialogs = dialogs if dialogs is not None else load_dialogs()
    query = normalize(message)
    query_words = set(query.split())
    best = None
    best_score = 0.0
    for item in dialogs:
        patterns = item.get("patterns", [])
        for pattern in patterns:
            candidate = normalize(pattern)
            if not candidate:
                continue
            if query == candidate:
                score = 1.0
            else:
                words = set(candidate.split())
                overlap = len(query_words & words) / max(1, len(query_words | words))
                score = overlap
                if candidate in query or query in candidate:
                    score = max(score, 0.82)
            if score > best_score:
                best, best_score = item, score
    if best is not None and best_score >= 0.45:
        responses = best.get("responses", [])
        if responses:
            return responses[0]
    return (
        "Todavía no sé responder eso con seguridad. Soy AMI Nano, una IA local experimental. "
        "Puedes enseñarme agregando ejemplos a data/dialogos-es.json."
    )

def main():
    try:
        dialogs = load_dialogs()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("No se pudieron cargar los diálogos: " + str(exc), file=sys.stderr)
        return 2
    message = " ".join(sys.argv[1:]).strip()
    if message:
        print(respond(message, dialogs))
        return 0
    print("AMI Nano — chat local en español. Escribe 'salir' para terminar.")
    try:
        while True:
            message = input("Tú: ").strip()
            if normalize(message) in {"salir", "adios", "hasta luego"}:
                print("AMI: ¡Hasta luego! Seguiré mejorando.")
                break
            print("AMI: " + respond(message, dialogs))
    except (EOFError, KeyboardInterrupt):
        print("\nAMI: ¡Hasta luego!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

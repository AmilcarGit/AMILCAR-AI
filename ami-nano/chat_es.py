#!/usr/bin/env python3
"""Chat local en español para AMI Nano 2.1: conocimientos y coincidencia de intención."""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIALOGS = Path(os.environ.get("AMI_DIALOGS_PATH", ROOT / "data" / "dialogos-es.json"))
KNOWLEDGE = Path(os.environ.get("AMI_KNOWLEDGE_PATH", ROOT / "data" / "conocimientos-es.json"))
FALLBACK = (
    "No encuentro una respuesta suficientemente fiable en mi base local. "
    "Puedes agregar una pregunta y una respuesta revisada a data/conocimientos-es.json."
)

def normalize(text):
    text = unicodedata.normalize("NFD", str(text).lower().strip())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    return " ".join(text.split())

def load_file(path):
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError(f"{path.name} debe contener una lista JSON.")
    clean = []
    for item in data:
        if not isinstance(item, dict):
            continue
        patterns = item.get("patterns", [])
        responses = item.get("responses", [])
        if isinstance(patterns, list) and patterns and isinstance(responses, list) and responses:
            clean.append({"intent": str(item.get("intent", "general")), "patterns": patterns, "responses": responses})
    return clean

def load_dialogs():
    result = []
    for path in (DIALOGS, KNOWLEDGE):
        if path.exists():
            result.extend(load_file(path))
    if not result:
        raise ValueError("No hay diálogos ni conocimientos locales cargados.")
    return result

def similarity(query, candidate):
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    qwords, cwords = set(query.split()), set(candidate.split())
    overlap = len(qwords & cwords) / max(1, len(qwords | cwords))
    if len(query) >= 4 and (candidate in query or query in candidate):
        overlap = max(overlap, 0.84)
    return overlap

def respond(message, dialogs=None):
    if not isinstance(message, str) or not message.strip():
        return "Escríbeme una pregunta y trataré de ayudarte."
    try:
        dialogs = dialogs if dialogs is not None else load_dialogs()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "No pude cargar la base local de conocimientos: " + str(exc)
    query = normalize(message)
    if not query:
        return "Escríbeme una pregunta con palabras para poder buscar una respuesta."
    best, best_score, second_score = None, 0.0, 0.0
    for item in dialogs:
        score = max((similarity(query, normalize(p)) for p in item["patterns"]), default=0.0)
        if score > best_score:
            second_score, best, best_score = best_score, item, score
        elif score > second_score:
            second_score = score
    if best is not None and best_score >= 0.42 and (best_score - second_score >= 0.035 or best_score >= 0.9):
        return str(best["responses"][0])
    return FALLBACK

def main():
    try:
        dialogs = load_dialogs()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("No se pudieron cargar los datos locales: " + str(exc), file=sys.stderr)
        return 2
    message = " ".join(sys.argv[1:]).strip()
    if message:
        print(respond(message, dialogs))
        return 0
    print("AMI Nano 2.1 — chat local en español. Escribe 'salir' para terminar.")
    try:
        while True:
            message = input("Tú: ").strip()
            if normalize(message) in {"salir", "adios", "hasta luego"}:
                print("AMI: ¡Hasta luego!")
                break
            print("AMI: " + respond(message, dialogs))
    except (EOFError, KeyboardInterrupt):
        print("\nAMI: ¡Hasta luego!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

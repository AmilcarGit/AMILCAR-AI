#!/usr/bin/env python3
"""Entrena desde cero un modelo estadístico de caracteres; solo biblioteca estándar."""
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.environ.get("AMI_CORPUS", os.path.join(ROOT, "data", "corpus.txt"))
MODEL = os.environ.get("AMI_MODEL_PATH", os.path.join(ROOT, "artifacts", "ami-nano.json"))
DEMO = """hola soy ami nano un modelo experimental creado por amilcar ai.
estoy aprendiendo a generar texto a partir de ejemplos locales.
amilcar ai es un proyecto abierto para experimentar con modelos propios.
hola como estas puedo generar texto corto pero todavía tengo capacidades limitadas.
la inteligencia artificial se desarrolla con datos pruebas y mejoras progresivas.
"""

def main():
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    if not os.path.exists(CORPUS):
        with open(CORPUS, "w", encoding="utf-8") as f:
            f.write(DEMO)
        print("Creado corpus de demostración:", CORPUS)
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read().lower()
    if len(text.strip()) < 20:
        raise SystemExit("El corpus necesita al menos 20 caracteres de texto.")
    counts = defaultdict(Counter)
    for a, b in zip(text, text[1:]):
        counts[a][b] += 1
    transitions = {a: dict(c) for a, c in counts.items()}
    payload = {
        "format": "ami-nano-char-bigram-v1",
        "trained_from_scratch": True,
        "corpus_chars": len(text),
        "vocab": sorted(set(text)),
        "transitions": transitions,
    }
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    tmp = MODEL + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, MODEL)
    print("Modelo entrenado desde cero.")
    print("Caracteres:", len(text), "| vocabulario:", len(payload["vocab"]))
    print("Checkpoint:", MODEL)

if __name__ == "__main__":
    main()

"""Convierte data/dialogos-es.json y data/knowledge-es.json en un corpus de conversación.

Cada ejemplo usa el mismo formato con el que el modelo responde después:

    Usuario: hola
    AMILCAR AI: ¡Hola! ...

Las preguntas se guardan normalizadas (minúsculas, sin tildes ni signos), igual que las recibe el modelo.

Uso:
    python -m ai.training.build_dialog_corpus --output data/dialogos_train.txt
    python -m ai.training.train --data data/dialogos_train.txt --steps 3000
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIALOGS = ROOT / "data" / "dialogos-es.json"
DEFAULT_KNOWLEDGE = ROOT / "data" / "knowledge-es.json"
DEFAULT_OUTPUT = ROOT / "data" / "dialogos_train.txt"


_NON_TEXT = re.compile(r"[^a-z0-9ñ\s]")


def normalize_question(text: str) -> str:
    """Misma normalización que `normalize()` de ai/native-engine.js.

    Minúsculas, sin tildes (ni la tilde de la ñ), sin signos y con espacios simples. El bot aplica
    esto al mensaje antes de pasárselo al modelo, así que se entrena con exactamente esa forma.
    """
    decomposed = unicodedata.normalize("NFD", str(text))
    stripped = "".join(ch for ch in decomposed if not "\u0300" <= ch <= "\u036f")
    return " ".join(_NON_TEXT.sub(" ", stripped.lower()).split())


def _variants(question: str) -> list[str]:
    normalized = normalize_question(question)
    return [normalized] if normalized else []


def _pairs(dialogs: list[dict], knowledge: list[dict]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in dialogs:
        patterns = item.get("patterns", [])
        responses = item.get("responses", [])
        if not isinstance(patterns, list) or not isinstance(responses, list):
            raise ValueError(f"Intención mal formada: {item.get('intent')!r}")
        for pattern in patterns:
            for question in _variants(str(pattern)):
                for response in responses:
                    pairs.append((question, " ".join(str(response).split())))
    for entry in knowledge:
        answer = " ".join(str(entry.get("answer", "")).split())
        for term in entry.get("terms", []):
            for question in _variants(str(term)):
                if answer:
                    pairs.append((question, answer))
    return pairs


def build_corpus(dialogs: list[dict], knowledge: list[dict] | None = None) -> str:
    """Devuelve el texto de entrenamiento. Sin duplicados y en orden estable."""
    seen: set[tuple[str, str]] = set()
    blocks: list[str] = []
    for question, answer in _pairs(dialogs, knowledge or []):
        if (question, answer) in seen:
            continue
        seen.add((question, answer))
        blocks.append(f"Usuario: {question}\nAMILCAR AI: {answer}")
    if not blocks:
        raise ValueError("No se encontró ningún par pregunta/respuesta.")
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un corpus de conversación para entrenar AMI-Base.")
    parser.add_argument("--dialogs", type=Path, default=DEFAULT_DIALOGS)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))
    knowledge = json.loads(args.knowledge.read_text(encoding="utf-8")) if args.knowledge.is_file() else []
    corpus = build_corpus(dialogs, knowledge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    print(f"{corpus.count('Usuario: ')} ejemplos, {len(corpus)} caracteres -> {args.output}")


if __name__ == "__main__":
    main()

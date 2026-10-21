"""Convierte data/dialogos-es.json y data/knowledge-es.json en un corpus de conversación.

Cada ejemplo usa el mismo formato con el que el modelo responde después:

    Usuario: hola
    AMILCAR AI: ¡Hola! ...

Uso:
    python -m ai.training.build_dialog_corpus --output data/dialogos_train.txt
    python -m ai.training.train --data data/dialogos_train.txt --steps 3000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIALOGS = ROOT / "data" / "dialogos-es.json"
DEFAULT_KNOWLEDGE = ROOT / "data" / "knowledge-es.json"
DEFAULT_OUTPUT = ROOT / "data" / "dialogos_train.txt"


def _variants(question: str) -> list[str]:
    """Forma original y con mayúscula inicial: así el modelo ve cómo escribe la gente."""
    question = " ".join(question.split())
    if not question:
        return []
    capitalized = question[0].upper() + question[1:]
    return [question] if capitalized == question else [question, capitalized]


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

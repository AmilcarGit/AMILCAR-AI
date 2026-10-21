import json
from pathlib import Path

import pytest

from ai.training.build_dialog_corpus import DEFAULT_DIALOGS, DEFAULT_KNOWLEDGE, build_corpus

DIALOGS = [{"intent": "saludo", "patterns": ["hola", "buenas"], "responses": ["¡Hola! Soy AMILCAR AI."]}]
KNOWLEDGE = [{"id": "gravedad", "terms": ["que es la gravedad"], "answer": "Es la fuerza que atrae  los cuerpos."}]


def test_corpus_uses_the_same_format_the_model_answers_with() -> None:
    corpus = build_corpus(DIALOGS)
    assert "Usuario: hola\nAMILCAR AI: ¡Hola! Soy AMILCAR AI." in corpus
    assert "Usuario: Hola\nAMILCAR AI:" in corpus  # variante con mayúscula
    assert corpus.endswith("\n")


def test_corpus_includes_knowledge_and_normalizes_spaces() -> None:
    corpus = build_corpus(DIALOGS, KNOWLEDGE)
    assert "AMILCAR AI: Es la fuerza que atrae los cuerpos." in corpus


def test_corpus_has_no_duplicate_examples() -> None:
    corpus = build_corpus(DIALOGS + DIALOGS)
    blocks = corpus.strip().split("\n\n")
    assert len(blocks) == len(set(blocks))


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_corpus([])


def test_real_project_data_builds_a_usable_corpus() -> None:
    dialogs = json.loads(Path(DEFAULT_DIALOGS).read_text(encoding="utf-8"))
    knowledge = json.loads(Path(DEFAULT_KNOWLEDGE).read_text(encoding="utf-8"))
    corpus = build_corpus(dialogs, knowledge)
    assert corpus.count("Usuario: ") > 100
    assert len(corpus) > 100  # el entrenamiento exige al menos 100 caracteres

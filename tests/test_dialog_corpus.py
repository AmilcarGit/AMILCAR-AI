import json
from pathlib import Path

import pytest

from ai.training.build_dialog_corpus import DEFAULT_DIALOGS, DEFAULT_KNOWLEDGE, DEFAULT_OFFTOPIC, build_corpus, normalize_question

DIALOGS = [{"intent": "saludo", "patterns": ["hola", "buenas"], "responses": ["¡Hola! Soy AMILCAR AI."]}]
KNOWLEDGE = [{"id": "gravedad", "terms": ["que es la gravedad"], "answer": "Es la fuerza que atrae  los cuerpos."}]


def test_corpus_uses_the_same_format_the_model_answers_with() -> None:
    corpus = build_corpus(DIALOGS)
    assert "Usuario: hola\nAMILCAR AI: ¡Hola! Soy AMILCAR AI." in corpus
    assert "Usuario: Hola" not in corpus  # solo se guarda la forma normalizada
    assert corpus.endswith("\n")


def test_questions_are_normalized_like_the_javascript_engine() -> None:
    assert normalize_question("¿Quién te creó?") == "quien te creo"
    assert normalize_question("  ¡Hola,   AMI!! ") == "hola ami"
    assert normalize_question("¿Cómo te llamas, ñandú?") == "como te llamas nandu"
    assert normalize_question("???") == ""


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
    assert "¿" not in corpus.split("AMILCAR AI:")[0]
    assert len(corpus) > 100  # el entrenamiento exige al menos 100 caracteres


def test_offtopic_questions_get_the_refusal_answer() -> None:
    offtopic = {"respuesta": "No sé  eso.", "preguntas": ["¿Cuál es la capital de Perú?", "cual es la capital de peru"]}
    corpus = build_corpus(DIALOGS, KNOWLEDGE, offtopic)
    assert corpus.count("Usuario: cual es la capital de peru\nAMILCAR AI: No sé eso.") == 1  # sin duplicados


def test_real_offtopic_data_is_clean_and_does_not_overlap_in_domain_questions() -> None:
    offtopic = json.loads(Path(DEFAULT_OFFTOPIC).read_text(encoding="utf-8"))
    dialogs = json.loads(Path(DEFAULT_DIALOGS).read_text(encoding="utf-8"))
    knowledge = json.loads(Path(DEFAULT_KNOWLEDGE).read_text(encoding="utf-8"))
    questions = offtopic["preguntas"]
    assert len(questions) >= 100 and len(set(questions)) == len(questions)
    assert all(q == normalize_question(q) for q in questions)
    known = {normalize_question(x) for item in dialogs for x in item["patterns"]}
    known |= {normalize_question(x) for entry in knowledge for x in entry["terms"]}
    assert not known & set(questions)
    assert offtopic["respuesta"].strip()

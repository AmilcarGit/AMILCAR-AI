from pathlib import Path

import pytest

from ai.tokenizer import EOS_TOKEN, BPETokenizer

CORPUS = (
    "La inteligencia artificial puede ayudar a escribir, aprender y programar. "
    "El entrenamiento de un modelo de lenguaje consiste en predecir el siguiente token. "
    "La evaluación ayuda a descubrir errores y medir las mejoras de una versión.\n"
) * 20


def test_roundtrip_with_accents_emoji_and_empty_text() -> None:
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=300)
    for text in ["", "Hola, ¿cómo estás?", "Ñandú 🚀 niño", "  espacios   raros\n\ty saltos "]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_compresses_text_and_respects_vocab_size() -> None:
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=320)
    assert tokenizer.vocab_size <= 320
    assert len(tokenizer.encode(CORPUS)) < len(CORPUS.encode("utf-8")) * 0.6


def test_training_is_deterministic() -> None:
    first = BPETokenizer.train(CORPUS, vocab_size=300)
    second = BPETokenizer.train(CORPUS, vocab_size=300)
    assert first.merges == second.merges
    assert first.encode("inteligencia artificial") == second.encode("inteligencia artificial")


def test_save_and_load_keep_identical_behavior(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=300)
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    sample = "El modelo aprende patrones de lenguaje."
    assert loaded.encode(sample) == tokenizer.encode(sample)
    assert loaded.vocab_size == tokenizer.vocab_size


def test_special_tokens_cannot_be_injected_from_plain_text() -> None:
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=300)
    assert tokenizer.eos_id not in tokenizer.encode("hola " + EOS_TOKEN)
    assert tokenizer.eos_id in tokenizer.encode("hola " + EOS_TOKEN, allow_special=True)
    ids = tokenizer.encode("hola", add_eos=True)
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hola"
    assert tokenizer.decode(ids, skip_special=False).endswith(EOS_TOKEN)


def test_invalid_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError):
        BPETokenizer.train(CORPUS, vocab_size=100)
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=300)
    with pytest.raises(ValueError):
        tokenizer.decode([tokenizer.vocab_size + 5])
    with pytest.raises(TypeError):
        tokenizer.encode(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BPETokenizer.from_dict({"type": "otro"})
    with pytest.raises(ValueError):
        BPETokenizer(merges=[(0, 999)])

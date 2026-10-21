import json
from pathlib import Path

import pytest

from ai.training.prepare_data import clean_text, prepare


def _write_corpus(folder: Path) -> None:
    folder.mkdir()
    paragraphs = [f"Este es el párrafo número {i} del corpus de prueba, con texto suficiente." for i in range(60)]
    (folder / "a.txt").write_text("\n\n".join(paragraphs), encoding="utf-8")
    # Repite los 10 primeros (duplicados) y añade uno corto que debe descartarse.
    (folder / "b.md").write_text("\n\n".join(paragraphs[:10] + ["corto"]), encoding="utf-8")


def test_clean_text_normalizes_unicode_controls_and_spaces() -> None:
    assert clean_text("café   \x00mundo\r\nlinea  dos ") == "café mundo\nlinea dos"


def test_prepare_deduplicates_splits_and_writes_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus(corpus)
    out = tmp_path / "out"
    manifest = prepare([corpus], out, val_fraction=0.2, seed=1)

    train = (out / "train.txt").read_text(encoding="utf-8").strip().split("\n\n")
    val = (out / "val.txt").read_text(encoding="utf-8").strip().split("\n\n")
    assert manifest["duplicates_removed"] == 10
    assert len(train) + len(val) == 60
    assert val and train
    assert not set(train) & set(val)
    assert "corto" not in train + val
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved["train_sha256"] == manifest["train_sha256"]
    assert [s["file"] for s in saved["sources"]] == ["a.txt", "b.md"]


def test_prepare_is_reproducible(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus(corpus)
    first = prepare([corpus], tmp_path / "o1", val_fraction=0.2, seed=5)
    second = prepare([corpus], tmp_path / "o2", val_fraction=0.2, seed=5)
    assert first["train_sha256"] == second["train_sha256"]
    assert first["val_sha256"] == second["val_sha256"]


def test_prepare_always_keeps_a_validation_set_on_tiny_corpora(tmp_path: Path) -> None:
    file = tmp_path / "chico.txt"
    file.write_text("Primer párrafo con texto suficiente.\n\nSegundo párrafo con texto suficiente.", encoding="utf-8")
    manifest = prepare([file], tmp_path / "out", val_fraction=0.01)
    assert manifest["val_paragraphs"] == 1 and manifest["train_paragraphs"] == 1


def test_prepare_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare([tmp_path / "no-existe"], tmp_path / "out")
    with pytest.raises(ValueError):
        prepare([tmp_path], tmp_path / "out", val_fraction=0.9)
    empty = tmp_path / "vacia"
    empty.mkdir()
    with pytest.raises(ValueError):
        prepare([empty], tmp_path / "out")

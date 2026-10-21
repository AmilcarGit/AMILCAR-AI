import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ami-nano"))
import hybrid


def test_generate_llm_candidate_uses_local_llama_cli(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"test-model")
    monkeypatch.setenv("AMILCAR_AI_LLM_MODEL", str(model))
    monkeypatch.setenv("AMILCAR_AI_LLM_CLI", "llama-cli")
    monkeypatch.setattr(hybrid.shutil, "which", lambda name: "/usr/bin/llama-cli", raising=False)
    answer = "La gravedad describe la atracción entre objetos con masa. Por ejemplo, mantiene la Luna en órbita alrededor de la Tierra."
    with patch.object(hybrid.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=answer)) as run:
        result = hybrid.generate_llm_candidate("Explícame la gravedad")

    assert result == answer
    command = run.call_args.args[0]
    assert command[0] == "/usr/bin/llama-cli"
    assert str(model) in command
    assert "-cnv" in command
    assert run.call_args.kwargs["timeout"] >= 10


def test_respond_prefers_local_llm_for_unmatched_question(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"test-model")
    monkeypatch.setenv("AMILCAR_AI_LLM_MODEL", str(model))
    monkeypatch.setattr(hybrid.shutil, "which", lambda name: "/usr/bin/llama-cli", raising=False)
    answer = "La gravedad describe la atracción entre objetos con masa. Por ejemplo, mantiene la Luna en órbita alrededor de la Tierra."
    with patch.object(hybrid.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=answer)):
        result = hybrid.respond("Explícame la gravedad cuántica", dialogs=[], candidate="")

    assert result["source"] == "local_llm"
    assert answer in result["text"]
    assert result["quality"]["accepted"] is True


def test_llm_candidate_rejects_empty_or_garbled_text():
    assert hybrid.evaluate_llm_candidate("").get("accepted") is False
    assert hybrid.evaluate_llm_candidate("aaaaa aaaaa aaaaa aaaaa").get("accepted") is False


def test_llm_is_optional_without_model_path(monkeypatch):
    monkeypatch.delenv("AMILCAR_AI_LLM_MODEL", raising=False)
    with patch.object(hybrid.subprocess, "run") as run:
        assert hybrid.generate_llm_candidate("Hola") is None
        run.assert_not_called()

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ami-nano"))
import hybrid


def test_generate_llm_candidate_calls_local_server(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"test-model")
    monkeypatch.setenv("AMILCAR_AI_LLM_MODEL", str(model))
    answer = "La gravedad describe la atracción entre objetos con masa. Por ejemplo, mantiene la Luna en órbita alrededor de la Tierra."
    body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode("utf-8")

    with patch.object(hybrid.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(body)) as request:
        result = hybrid.generate_llm_candidate("Explícame la gravedad")

    assert result == answer
    sent_request = request.call_args.args[0]
    payload = json.loads(sent_request.data.decode("utf-8"))
    assert payload["messages"][1]["content"] == "Explícame la gravedad"
    assert sent_request.full_url == "http://127.0.0.1:8081/v1/chat/completions"
    assert request.call_args.kwargs["timeout"] >= 3


def test_respond_prefers_local_llm_for_unmatched_question(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"test-model")
    monkeypatch.setenv("AMILCAR_AI_LLM_MODEL", str(model))
    answer = "La gravedad describe la atracción entre objetos con masa. Por ejemplo, mantiene la Luna en órbita alrededor de la Tierra."
    body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode("utf-8")
    with patch.object(hybrid.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(body)):
        result = hybrid.respond("Explícame la gravedad cuántica", dialogs=[], candidate="")

    assert result["source"] == "local_llm"
    assert answer in result["text"]
    assert result["quality"]["accepted"] is True


def test_llm_candidate_rejects_empty_or_garbled_text():
    assert hybrid.evaluate_llm_candidate("").get("accepted") is False
    assert hybrid.evaluate_llm_candidate("aaaaa aaaaa aaaaa aaaaa").get("accepted") is False


def test_llm_is_optional_without_model_path(monkeypatch):
    monkeypatch.delenv("AMILCAR_AI_LLM_MODEL", raising=False)
    with patch.object(hybrid.urllib.request, "urlopen") as request:
        assert hybrid.generate_llm_candidate("Hola") is None
        request.assert_not_called()

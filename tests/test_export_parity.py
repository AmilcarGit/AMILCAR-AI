"""El modelo exportado y ejecutado en JavaScript debe dar los mismos logits que PyTorch."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from ai.model.inference import CHECKPOINT_VERSION
from ai.model.transformer import AMITransformer
from ai.tokenizer import BPETokenizer
from ai.training.export_model import export_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CORPUS = "Usuario: hola\nAMILCAR AI: ¡Hola! Soy AMILCAR AI.\n\nUsuario: quien eres\nAMILCAR AI: Una IA experimental.\n" * 12


def _checkpoint(tmp_path: Path, **config) -> tuple[Path, AMITransformer, BPETokenizer]:
    torch.manual_seed(0)
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=290)
    model = AMITransformer(vocab_size=tokenizer.vocab_size, dropout=0.0, **config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # Pesos grandes (los iniciales, 0.02, darían logits casi planos y una prueba débil).
            if name.endswith("norm.weight"):
                parameter.copy_(1.0 + 0.2 * torch.randn_like(parameter))
            else:
                parameter.normal_(0.0, 0.3)
    model.eval()
    path = tmp_path / "ckpt.pt"
    torch.save({
        "format_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": {**config, "dropout": 0.0},
        "metadata": {"total_steps": 3, "seed": 1},
    }, path)
    return path, model, tokenizer


def test_javascript_engine_matches_pytorch_logits(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        pytest.skip("Node.js no está instalado")
    config = {"d_model": 32, "n_heads": 4, "n_layers": 2, "max_seq_len": 24}
    checkpoint, model, tokenizer = _checkpoint(tmp_path, **config)
    output = tmp_path / "model.amw"
    result = export_checkpoint(checkpoint, output)
    assert result["config"]["hidden"] == 88  # ((int(32 * 8 / 3) + 7) // 8) * 8

    ids = tokenizer.encode("Usuario: hola\nAMILCAR AI: ¡Hola! Soy")[:20]
    with torch.no_grad():
        expected = model(torch.tensor([ids]))[0]
    completed = subprocess.run(
        ["node", str(ROOT / "scripts" / "neural-logits.js"), str(output), json.dumps(ids)],
        capture_output=True, text=True, check=True,
    )
    got = torch.tensor(json.loads(completed.stdout))
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=2e-3, rtol=1e-3), float((got - expected).abs().max())


def test_export_rejects_old_checkpoint_format(tmp_path: Path) -> None:
    old = tmp_path / "old.pt"
    torch.save({"vocab": {"a": 0}, "model_state": {}}, old)
    with pytest.raises(ValueError):
        export_checkpoint(old, tmp_path / "out.amw")

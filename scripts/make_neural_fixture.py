"""Genera los archivos de prueba del motor neuronal de JavaScript (tests-node/fixtures/).

No necesita PyTorch: usa una implementación de referencia en numpy, escrita aparte del motor de JS y
sin caché (secuencia completa con máscara causal). El motor de JS debe reproducir sus logits.

    python scripts/make_neural_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.tokenizer import BPETokenizer  # noqa: E402
from ai.training.build_dialog_corpus import build_corpus  # noqa: E402
from ai.training.export_model import read_amw, tensor_names, write_amw  # noqa: E402

FIXTURES = ROOT / "tests-node" / "fixtures"
EPS = 1e-6


def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * weight


def reference_logits(tensors: dict[str, np.ndarray], config: dict[str, int], ids: list[int]) -> np.ndarray:
    d, heads, layers = config["d_model"], config["n_heads"], config["n_layers"]
    hd = d // heads
    length = len(ids)
    x = tensors["token_embedding.weight"][ids]
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, hd, 2) / hd))
    angles = np.outer(np.arange(length), inv_freq)  # (T, hd/2)
    cos, sin = np.cos(angles), np.sin(angles)
    causal = np.triu(np.ones((length, length), dtype=bool), k=1)

    def rope(t: np.ndarray) -> np.ndarray:  # t: (H, T, hd)
        a, b = t[..., : hd // 2], t[..., hd // 2:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)

    for layer in range(layers):
        p = f"blocks.{layer}."
        h = rms_norm(x, tensors[p + "attn_norm.weight"])
        qkv = h @ tensors[p + "attn.qkv.weight"].T
        q, k, v = np.split(qkv, 3, axis=-1)
        q, k, v = (t.reshape(length, heads, hd).transpose(1, 0, 2) for t in (q, k, v))
        q, k = rope(q), rope(k)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
        scores = np.where(causal, -np.inf, scores)
        scores = scores - scores.max(-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(-1, keepdims=True)
        attended = (weights @ v).transpose(1, 0, 2).reshape(length, d)
        x = x + attended @ tensors[p + "attn.proj.weight"].T
        h = rms_norm(x, tensors[p + "mlp_norm.weight"])
        gate = h @ tensors[p + "gate.weight"].T
        up = h @ tensors[p + "up.weight"].T
        x = x + ((gate / (1.0 + np.exp(-gate))) * up) @ tensors[p + "down.weight"].T
    return rms_norm(x, tensors["norm.weight"]) @ tensors["token_embedding.weight"].T


TOKENIZER_SAMPLES = [
    "hola", "Hola, ¿cómo estás?", "quien te creo", "¿Qué es la fotosíntesis?", "ñandú y niño pequeño",
    "snake_case y __init__", "x² + y³ = 25", "tab\tsalto\nlinea\r\nfin", "  espacios   múltiples  ",
    "emoji 🚀🔥 y más 😀", "Usuario: hola\nAMILCAR AI: ¡Hola!", "AMILCAR-AI, versión 2.0.1", "",
    "a", "123456789", "¡¡¡Muy bien!!!", "correo@ejemplo.com http://x.org/a?b=1&c=2", "ÁÉÍÓÚ áéíóú ÜÑ",
    "\n\n\n", "palabra" * 12, "mezcla de ABC y abc 12 34", "日本語のテキスト", "العربية", "́suelto",
]


def main() -> None:
    dialogs = json.loads((ROOT / "data" / "dialogos-es.json").read_text(encoding="utf-8"))
    knowledge = json.loads((ROOT / "data" / "knowledge-es.json").read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.train(build_corpus(dialogs, knowledge), vocab_size=300)

    config = {"vocab_size": tokenizer.vocab_size, "d_model": 16, "n_heads": 2, "n_layers": 2, "max_seq_len": 16, "hidden": 48}
    rng = np.random.default_rng(1234)
    d, h, v = config["d_model"], config["hidden"], config["vocab_size"]
    shapes = {"token_embedding.weight": (v, d), "norm.weight": (d,)}
    for layer in range(config["n_layers"]):
        p = f"blocks.{layer}."
        shapes.update({
            p + "attn_norm.weight": (d,), p + "attn.qkv.weight": (3 * d, d), p + "attn.proj.weight": (d, d),
            p + "mlp_norm.weight": (d,), p + "gate.weight": (h, d), p + "up.weight": (h, d), p + "down.weight": (d, h),
        })
    raw = {}
    for name in tensor_names(config["n_layers"]):
        shape = shapes[name]
        if name.endswith("norm.weight"):
            raw[name] = 1.0 + 0.2 * rng.standard_normal(shape)
        else:
            raw[name] = 0.5 * rng.standard_normal(shape)
    tensors = {name: (list(value.shape), value.astype("<f4").flatten().tolist()) for name, value in raw.items()}

    FIXTURES.mkdir(parents=True, exist_ok=True)
    model_path = FIXTURES / "tiny.amw"
    write_amw(model_path, config, tokenizer.to_dict(), tensors, {"fixture": "random", "total_steps": 0})

    # La referencia lee los pesos desde el propio .amw, así prueba exactamente lo que verá JS.
    _, loaded = read_amw(model_path)
    reference_tensors = {name: np.array(values, dtype=np.float64).reshape(shapes[name]) for name, values in loaded.items()}
    cases = []
    for _ in range(4):
        length = int(rng.integers(1, config["max_seq_len"] + 1))
        ids = [int(i) for i in rng.integers(0, config["vocab_size"] - 2, size=length)]
        logits = reference_logits(reference_tensors, config, ids)
        cases.append({"ids": ids, "logits": np.round(logits, 6).tolist()})
    cases.append({"ids": [5], "logits": np.round(reference_logits(reference_tensors, config, [5]), 6).tolist()})
    (FIXTURES / "tiny.expected.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")

    parity = [{"text": text, "ids": tokenizer.encode(text)} for text in TOKENIZER_SAMPLES]
    for item in parity:
        assert tokenizer.decode(item["ids"]) == item["text"], item["text"]
    (FIXTURES / "tokenizer-parity.json").write_text(
        json.dumps({"tokenizer": tokenizer.to_dict(), "cases": parity}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Fixtures en {FIXTURES}: modelo {model_path.stat().st_size / 1024:.0f} KB, {len(cases)} casos, {len(parity)} textos")


if __name__ == "__main__":
    main()

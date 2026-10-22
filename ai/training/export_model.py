"""Exporta un checkpoint de AMI-Base al formato .amw, que lee el motor de JavaScript.

El motor de JS (ai/neural-engine.js) ejecuta el modelo sin Python ni PyTorch, así que sirve en
Termux. Entrena en una PC o en GitHub Actions y copia solo el archivo .amw.

Uso:
    python -m ai.training.export_model --checkpoint artifacts/ami-base.pt --output models/ami-base.amw

Formato (todo little-endian):
    4 bytes   magia "AMIW"
    4 bytes   versión del formato (uint32)
    4 bytes   tamaño N del encabezado JSON (uint32)
    N bytes   encabezado JSON en UTF-8, rellenado con espacios hasta múltiplo de 4
    ...       tensores float32 consecutivos; en el encabezado cada uno indica su offset (en floats)
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from array import array
from pathlib import Path
from typing import Iterable, Mapping, Sequence

MAGIC = b"AMIW"
FORMAT_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "ami-base.pt"
DEFAULT_OUTPUT = ROOT / "models" / "ami-base.amw"


def tensor_names(n_layers: int) -> list[str]:
    """Orden fijo de los tensores. La proyección de salida comparte pesos con el embedding."""
    names = ["token_embedding.weight"]
    for layer in range(n_layers):
        prefix = f"blocks.{layer}."
        names += [
            prefix + "attn_norm.weight",
            prefix + "attn.qkv.weight",
            prefix + "attn.proj.weight",
            prefix + "mlp_norm.weight",
            prefix + "gate.weight",
            prefix + "up.weight",
            prefix + "down.weight",
        ]
    names.append("norm.weight")
    return names


def _floats_to_bytes(values: Iterable[float]) -> bytes:
    data = array("f", values)
    if sys.byteorder == "big":
        data.byteswap()
    return data.tobytes()


def write_amw(
    path: str | Path,
    config: Mapping[str, int],
    tokenizer: Mapping[str, object],
    tensors: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    metadata: Mapping[str, object] | None = None,
) -> int:
    """Escribe el archivo y devuelve su tamaño en bytes. `tensors`: nombre -> (forma, valores planos)."""
    expected = tensor_names(int(config["n_layers"]))
    missing = [name for name in expected if name not in tensors]
    if missing:
        raise ValueError(f"Faltan tensores: {missing[:3]}")

    index = []
    chunks = []
    offset = 0
    for name in expected:
        shape, values = tensors[name]
        count = 1
        for dim in shape:
            count *= int(dim)
        if len(values) != count:
            raise ValueError(f"El tensor {name} tiene {len(values)} valores y su forma pide {count}.")
        index.append({"name": name, "shape": [int(d) for d in shape], "offset": offset})
        chunks.append(_floats_to_bytes(values))
        offset += count

    header = json.dumps(
        {
            "format_version": FORMAT_VERSION,
            "config": {key: int(value) for key, value in config.items()},
            "tokenizer": tokenizer,
            "tensors": index,
            "metadata": dict(metadata or {}),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    header += b" " * (-len(header) % 4)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<II", FORMAT_VERSION, len(header)))
        handle.write(header)
        for chunk in chunks:
            handle.write(chunk)
    return target.stat().st_size


def read_amw(path: str | Path) -> tuple[dict[str, object], dict[str, "array[float]"]]:
    """Lee un .amw. Devuelve (encabezado, {nombre: array('f') plano}). Útil para pruebas y diagnóstico."""
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError("No es un archivo .amw de AMILCAR AI.")
    version, header_size = struct.unpack("<II", raw[4:12])
    if version != FORMAT_VERSION:
        raise ValueError(f"Versión de formato no compatible: {version}")
    header = json.loads(raw[12:12 + header_size].decode("utf-8"))
    data_start = 12 + header_size
    tensors: dict[str, "array[float]"] = {}
    for item in header["tensors"]:
        count = 1
        for dim in item["shape"]:
            count *= dim
        start = data_start + item["offset"] * 4
        values = array("f")
        values.frombytes(raw[start:start + count * 4])
        if sys.byteorder == "big":
            values.byteswap()
        tensors[item["name"]] = values
    return header, tensors


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """Convierte un checkpoint de entrenamiento (formato 2) en un archivo .amw."""
    import torch  # solo hace falta al exportar, no para usar el modelo en JS

    from ai.model.inference import CHECKPOINT_VERSION
    from ai.tokenizer import BPETokenizer

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("El checkpoint tiene un formato anterior; reentrena con ai.training.train.")
    state = checkpoint["model_state"]
    tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
    config = dict(checkpoint["config"])
    n_layers = int(config["n_layers"])

    tensors = {}
    for name in tensor_names(n_layers):
        tensor = state[name].detach().cpu().float().contiguous()
        tensors[name] = (list(tensor.shape), tensor.flatten().tolist())

    embedding_rows = tensors["token_embedding.weight"][0][0]
    if embedding_rows != tokenizer.vocab_size:
        raise ValueError("El vocabulario del tokenizador no coincide con el del modelo.")
    hidden = tensors["blocks.0.gate.weight"][0][0]
    export_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": int(config["d_model"]),
        "n_heads": int(config["n_heads"]),
        "n_layers": n_layers,
        "max_seq_len": int(config["max_seq_len"]),
        "hidden": int(hidden),
    }
    metadata = {key: value for key, value in checkpoint.get("metadata", {}).items() if isinstance(value, (int, float, str, bool))}
    metadata["parameters"] = sum(len(values) for _, values in tensors.values())
    size = write_amw(output_path, export_config, tokenizer.to_dict(), tensors, metadata)
    return {"bytes": size, "config": export_config, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta un checkpoint de AMI-Base al formato .amw (JavaScript).")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = export_checkpoint(args.checkpoint, args.output)
    print(f"Exportado a {args.output} ({result['bytes'] / 1024:.0f} KB)")
    print(f"Configuración: {result['config']}")


if __name__ == "__main__":
    main()

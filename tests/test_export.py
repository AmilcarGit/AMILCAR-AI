import json
import struct
from pathlib import Path

import pytest

from ai.training.export_model import MAGIC, read_amw, tensor_names, write_amw

CONFIG = {"vocab_size": 6, "d_model": 4, "n_heads": 2, "n_layers": 1, "max_seq_len": 8, "hidden": 8}
SHAPES = {
    "token_embedding.weight": (6, 4), "blocks.0.attn_norm.weight": (4,), "blocks.0.attn.qkv.weight": (12, 4),
    "blocks.0.attn.proj.weight": (4, 4), "blocks.0.mlp_norm.weight": (4,), "blocks.0.gate.weight": (8, 4),
    "blocks.0.up.weight": (8, 4), "blocks.0.down.weight": (4, 8), "norm.weight": (4,),
}


def _tensors(scale: float = 1.0) -> dict:
    result = {}
    for index, name in enumerate(tensor_names(1)):
        shape = SHAPES[name]
        count = 1
        for dim in shape:
            count *= dim
        result[name] = (list(shape), [scale * (index + 1) + i * 0.25 for i in range(count)])
    return result


def test_write_and_read_roundtrip_keeps_every_value(tmp_path: Path) -> None:
    path = tmp_path / "m.amw"
    tokenizer = {"type": "byte-level-bpe", "format_version": 1, "merges": [], "special_tokens": []}
    size = write_amw(path, CONFIG, tokenizer, _tensors(), {"total_steps": 5})
    assert path.stat().st_size == size
    assert path.read_bytes()[:4] == MAGIC
    header, loaded = read_amw(path)
    assert header["config"]["d_model"] == 4
    assert header["metadata"] == {"total_steps": 5}
    assert header["tokenizer"] == tokenizer
    for name, (shape, values) in _tensors().items():
        assert len(loaded[name]) == len(values)
        assert all(abs(a - b) < 1e-6 for a, b in zip(loaded[name], values)), name


def test_data_section_is_aligned_to_four_bytes(tmp_path: Path) -> None:
    path = tmp_path / "m.amw"
    write_amw(path, CONFIG, {"type": "x"}, _tensors(), {"nota": "añadir acentos cambia el tamaño"})
    raw = path.read_bytes()
    header_size = struct.unpack("<I", raw[8:12])[0]
    assert (12 + header_size) % 4 == 0
    json.loads(raw[12:12 + header_size].decode("utf-8"))


def test_write_rejects_missing_or_misshaped_tensors(tmp_path: Path) -> None:
    missing = _tensors()
    del missing["norm.weight"]
    with pytest.raises(ValueError):
        write_amw(tmp_path / "a.amw", CONFIG, {}, missing)
    wrong = _tensors()
    wrong["norm.weight"] = ([4], [1.0, 2.0])
    with pytest.raises(ValueError):
        write_amw(tmp_path / "b.amw", CONFIG, {}, wrong)


def test_read_rejects_files_that_are_not_amw(tmp_path: Path) -> None:
    path = tmp_path / "x.amw"
    path.write_bytes(b"no es un modelo")
    with pytest.raises(ValueError):
        read_amw(path)

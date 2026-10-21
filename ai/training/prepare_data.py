"""Prepara un corpus de texto para entrenar: limpia, deduplica y separa entrenamiento/validación.

Uso:
    python -m ai.training.prepare_data MIS_TEXTOS/ otro.txt --output data/processed

Solo usa texto del que tengas derecho a entrenar. Revisa procedencia y licencia de cada fuente:
el manifiesto que se genera registra los archivos usados y sus hashes para que sea reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "processed"
TEXT_SUFFIXES = {".txt", ".md"}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """Normaliza Unicode (NFC), quita caracteres de control y espacios sobrantes."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines)


def split_paragraphs(text: str, min_chars: int) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", clean_text(text)):
        block = block.strip()
        if len(block) >= min_chars and "�" not in block:
            paragraphs.append(block)
    return paragraphs


def collect_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(sorted(p for p in item.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES))
        elif item.is_file():
            files.append(item)
        else:
            raise FileNotFoundError(f"No existe la ruta: {item}")
    if not files:
        raise ValueError("No se encontraron archivos .txt o .md para procesar.")
    return files


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_validation(paragraph: str, seed: int, val_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{paragraph}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < val_fraction


def prepare(
    inputs: list[Path],
    output_dir: Path,
    val_fraction: float = 0.02,
    min_chars: int = 20,
    seed: int = 42,
) -> dict[str, object]:
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction debe estar entre 0 y 0.5.")
    if min_chars < 1:
        raise ValueError("min_chars debe ser al menos 1.")

    sources = []
    seen: set[str] = set()
    train: list[str] = []
    validation: list[str] = []
    duplicates = 0
    for path in collect_files(inputs):
        raw = path.read_bytes()
        paragraphs = split_paragraphs(raw.decode("utf-8", errors="replace"), min_chars)
        kept = 0
        for paragraph in paragraphs:
            key = hashlib.sha1(paragraph.lower().encode("utf-8")).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            kept += 1
            (validation if _is_validation(paragraph, seed, val_fraction) else train).append(paragraph)
        sources.append({"file": path.name, "sha256": _sha256_bytes(raw), "paragraphs_kept": kept})

    if len(train) + len(validation) < 2:
        raise ValueError("Hay muy poco texto útil: se necesitan al menos 2 párrafos distintos.")
    if not validation:  # corpus pequeño: garantiza un conjunto de validación no vacío
        validation.append(train.pop())
    if not train:
        raise ValueError("Todo el texto cayó en validación; añade más datos o baja val_fraction.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_bytes = ("\n\n".join(train) + "\n").encode("utf-8")
    val_bytes = ("\n\n".join(validation) + "\n").encode("utf-8")
    (output_dir / "train.txt").write_bytes(train_bytes)
    (output_dir / "val.txt").write_bytes(val_bytes)
    manifest: dict[str, object] = {
        "seed": seed,
        "val_fraction": val_fraction,
        "min_chars": min_chars,
        "train_paragraphs": len(train),
        "val_paragraphs": len(validation),
        "train_chars": len(train_bytes.decode("utf-8")),
        "val_chars": len(val_bytes.decode("utf-8")),
        "duplicates_removed": duplicates,
        "train_sha256": _sha256_bytes(train_bytes),
        "val_sha256": _sha256_bytes(val_bytes),
        "sources": sources,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpia, deduplica y divide un corpus de texto.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Archivos o carpetas con .txt/.md")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.inputs, args.output, args.val_fraction, args.min_chars, args.seed)
    print(
        f"Entrenamiento: {manifest['train_paragraphs']} párrafos ({manifest['train_chars']} caracteres)\n"
        f"Validación: {manifest['val_paragraphs']} párrafos ({manifest['val_chars']} caracteres)\n"
        f"Duplicados eliminados: {manifest['duplicates_removed']}\n"
        f"Guardado en: {args.output}"
    )


if __name__ == "__main__":
    main()

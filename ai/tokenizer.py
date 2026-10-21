"""Tokenizador BPE a nivel de bytes, escrito desde cero y sin dependencias externas.

Funciona con cualquier texto (acentos, ñ, emojis) porque parte de los 256 bytes posibles:
no existe el token "desconocido". Las fusiones (merges) se aprenden del corpus propio.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# Separa el texto en palabras (con su espacio inicial), números, signos y espacios.
PRETOKEN_PATTERN = re.compile(r"\s?[^\W\d_]+|\s?\d+|\s?[^\s\w]+|\s+", re.UNICODE)
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"
DEFAULT_SPECIAL_TOKENS = (PAD_TOKEN, EOS_TOKEN)
FORMAT_VERSION = 1


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    def __init__(
        self,
        merges: Iterable[tuple[int, int]] = (),
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        self.merges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in merges]
        self.special_tokens: list[str] = list(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Los tokens especiales no pueden repetirse.")

        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._ranks: dict[tuple[int, int], int] = {}
        for rank, (a, b) in enumerate(self.merges):
            new_id = 256 + rank
            if a not in self._vocab or b not in self._vocab or a >= new_id or b >= new_id:
                raise ValueError(f"Fusión {rank} inválida: referencia ids que aún no existen.")
            self._vocab[new_id] = self._vocab[a] + self._vocab[b]
            self._ranks[(a, b)] = rank
        first_special = 256 + len(self.merges)
        self.special_ids: dict[str, int] = {
            token: first_special + i for i, token in enumerate(self.special_tokens)
        }
        self._special_by_id = {token_id: token for token, token_id in self.special_ids.items()}
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._word_cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------ propiedades
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def pad_id(self) -> int | None:
        return self.special_ids.get(PAD_TOKEN)

    @property
    def eos_id(self) -> int | None:
        return self.special_ids.get(EOS_TOKEN)

    # ------------------------------------------------------------------ entrenamiento
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Aprende fusiones del corpus. Es determinista: el mismo texto da el mismo tokenizador.

        Recalcula los pares en cada iteración, lo cual va bien hasta decenas de MB.
        Para corpus mucho mayores conviene entrenar el tokenizador con una muestra.
        """
        specials = list(special_tokens)
        base = 256 + len(specials)
        if vocab_size < base:
            raise ValueError(f"vocab_size debe ser al menos {base}.")
        if min_frequency < 1:
            raise ValueError("min_frequency debe ser al menos 1.")

        word_counts = Counter(PRETOKEN_PATTERN.findall(text))
        sequences = {word: list(word.encode("utf-8")) for word in word_counts}
        merges: list[tuple[int, int]] = []
        for _ in range(vocab_size - base):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for word, ids in sequences.items():
                frequency = word_counts[word]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += frequency
            if not pair_counts:
                break
            # Mayor frecuencia; en empates gana el par más pequeño (determinista).
            pair, count = max(pair_counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
            if count < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            for word, ids in sequences.items():
                if len(ids) > 1:
                    sequences[word] = _merge_pair(ids, pair, new_id)
        return cls(merges, specials)

    # ------------------------------------------------------------------ codificación
    def _encode_word(self, word: str) -> tuple[int, ...]:
        cached = self._word_cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(ids, ids[1:]):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None or best_rank is None:
                break
            ids = _merge_pair(ids, best_pair, 256 + best_rank)
        result = tuple(ids)
        if len(self._word_cache) < 100_000:
            self._word_cache[word] = result
        return result

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in PRETOKEN_PATTERN.findall(text):
            ids.extend(self._encode_word(word))
        return ids

    def encode(self, text: str, allow_special: bool = False, add_eos: bool = False) -> list[int]:
        """Convierte texto en ids.

        Con allow_special=False (por defecto) un texto como "<|endoftext|>" escrito por el
        usuario se codifica como texto normal y no puede colarse como token de control.
        """
        if not isinstance(text, str):
            raise TypeError("El texto debe ser str.")
        ids: list[int] = []
        if allow_special and self._special_pattern is not None:
            for part in self._special_pattern.split(text):
                if part in self.special_ids:
                    ids.append(self.special_ids[part])
                elif part:
                    ids.extend(self._encode_plain(part))
        else:
            ids = self._encode_plain(text)
        if add_eos:
            if self.eos_id is None:
                raise ValueError("Este tokenizador no tiene token de fin de texto.")
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chunks: list[bytes] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self._vocab:
                chunks.append(self._vocab[token_id])
            elif token_id in self._special_by_id:
                if not skip_special:
                    chunks.append(self._special_by_id[token_id].encode("utf-8"))
            else:
                raise ValueError(f"Id de token fuera de rango: {token_id}")
        return b"".join(chunks).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistencia
    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "type": "byte-level-bpe",
            "merges": [[a, b] for a, b in self.merges],
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BPETokenizer":
        if data.get("type") != "byte-level-bpe":
            raise ValueError("El archivo no describe un tokenizador BPE de AMILCAR AI.")
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError("Versión de formato de tokenizador no compatible.")
        merges = data.get("merges")
        specials = data.get("special_tokens")
        if not isinstance(merges, list) or not isinstance(specials, list):
            raise ValueError("Tokenizador mal formado.")
        return cls([(int(a), int(b)) for a, b in merges], [str(t) for t in specials])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

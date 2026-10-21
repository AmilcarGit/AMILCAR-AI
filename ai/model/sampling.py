"""Muestreo del siguiente token con temperatura, top-k y top-p (núcleo).

Está escrito en Python puro sobre listas de logits para que sea fácil de probar y de leer.
"""
from __future__ import annotations

import math
import random
from typing import Protocol, Sequence


class _Rng(Protocol):
    def choices(self, population: Sequence[int], weights: Sequence[float], k: int) -> list[int]: ...


def sample_from_logits(
    logits: Sequence[float],
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: _Rng | None = None,
) -> int:
    """Devuelve el id elegido.

    - temperature <= 0: elige siempre el más probable (determinista).
    - top_k > 0: solo considera los k logits más altos.
    - top_p < 1: se queda con el grupo más pequeño cuya probabilidad acumulada llega a top_p.
    Los logits con valor -inf nunca se eligen.
    """
    if not logits:
        raise ValueError("Se necesitan logits para muestrear.")
    if top_k < 0:
        raise ValueError("top_k no puede ser negativo.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p debe estar en el intervalo (0, 1].")

    count = len(logits)
    if temperature <= 0:
        return max(range(count), key=logits.__getitem__)

    scaled = [value / temperature for value in logits]
    order = sorted(range(count), key=scaled.__getitem__, reverse=True)
    if top_k:
        order = order[:top_k]
    peak = scaled[order[0]]
    if peak == -math.inf:
        raise ValueError("Todos los logits son -inf; no hay token posible.")
    weights = [math.exp(scaled[index] - peak) for index in order]

    if top_p < 1.0:
        total = sum(weights)
        cumulative = 0.0
        cut = len(order)
        for position, weight in enumerate(weights):
            cumulative += weight / total
            if cumulative >= top_p:
                cut = position + 1
                break
        order, weights = order[:cut], weights[:cut]

    generator: _Rng = rng if rng is not None else random  # type: ignore[assignment]
    return generator.choices(order, weights=weights, k=1)[0]

"""Programación de la tasa de aprendizaje: calentamiento lineal y descenso con coseno."""
from __future__ import annotations

import math


def lr_at(
    step: int,
    total_steps: int,
    max_lr: float,
    warmup_steps: int = 0,
    min_ratio: float = 0.1,
) -> float:
    """Tasa de aprendizaje para el paso `step` (empieza en 0).

    Sube linealmente durante `warmup_steps`, luego baja con coseno hasta
    `max_lr * min_ratio` al final del entrenamiento.
    """
    if total_steps < 1:
        raise ValueError("total_steps debe ser mayor que cero.")
    if max_lr <= 0:
        raise ValueError("max_lr debe ser mayor que cero.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps no puede ser negativo.")
    if not 0 <= min_ratio <= 1:
        raise ValueError("min_ratio debe estar entre 0 y 1.")
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return max_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_ratio + (1.0 - min_ratio) * cosine)

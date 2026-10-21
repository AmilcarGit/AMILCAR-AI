import random

import pytest

from ai.model.sampling import sample_from_logits
from ai.training.schedule import lr_at

LOGITS = [0.0, 3.0, 1.0, 2.5, -1.0]


def test_zero_temperature_is_greedy() -> None:
    assert sample_from_logits(LOGITS, temperature=0) == 1


def test_top_k_limits_candidates() -> None:
    rng = random.Random(0)
    picks = {sample_from_logits(LOGITS, temperature=5.0, top_k=2, rng=rng) for _ in range(300)}
    assert picks <= {1, 3}
    assert picks == {1, 3}


def test_top_p_keeps_only_the_nucleus() -> None:
    rng = random.Random(1)
    skewed = [10.0, 0.0, 0.0, 0.0]
    picks = {sample_from_logits(skewed, temperature=1.0, top_p=0.9, rng=rng) for _ in range(200)}
    assert picks == {0}


def test_negative_infinity_is_never_chosen() -> None:
    rng = random.Random(2)
    blocked = [float("-inf"), 1.0, 1.0]
    picks = {sample_from_logits(blocked, temperature=1.0, rng=rng) for _ in range(200)}
    assert 0 not in picks


def test_sampling_is_reproducible_with_a_seeded_rng() -> None:
    first = [sample_from_logits(LOGITS, 1.0, rng=random.Random(7)) for _ in range(5)]
    second = [sample_from_logits(LOGITS, 1.0, rng=random.Random(7)) for _ in range(5)]
    assert first == second


def test_sampling_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        sample_from_logits([])
    with pytest.raises(ValueError):
        sample_from_logits(LOGITS, top_k=-1)
    with pytest.raises(ValueError):
        sample_from_logits(LOGITS, top_p=0)
    with pytest.raises(ValueError):
        sample_from_logits([float("-inf")] * 3)


def test_learning_rate_warms_up_then_decays() -> None:
    values = [lr_at(step, 100, 1e-3, warmup_steps=10) for step in range(100)]
    assert values[0] < values[5] < values[9]
    assert abs(values[10] - 1e-3) < 1e-9
    assert all(a >= b for a, b in zip(values[10:], values[11:]))
    assert values[-1] >= 1e-3 * 0.1 - 1e-12


def test_learning_rate_validates_arguments() -> None:
    with pytest.raises(ValueError):
        lr_at(0, 0, 1e-3)
    with pytest.raises(ValueError):
        lr_at(0, 10, 0)
    with pytest.raises(ValueError):
        lr_at(0, 10, 1e-3, warmup_steps=-1)
    assert lr_at(3, 5, 1e-3, warmup_steps=10) == pytest.approx(1e-3 * 4 / 10)

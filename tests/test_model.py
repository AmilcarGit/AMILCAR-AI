import pytest
import torch

from ai.model.transformer import AMITransformer


def _small_model(**overrides) -> AMITransformer:
    config = dict(vocab_size=24, d_model=32, n_heads=4, n_layers=2, max_seq_len=16)
    config.update(overrides)
    torch.manual_seed(0)
    model = AMITransformer(**config)
    model.eval()
    return model


def test_model_returns_next_token_logits() -> None:
    model = AMITransformer(vocab_size=24, d_model=32, n_heads=4, n_layers=1, max_seq_len=16)
    logits = model(torch.randint(0, 24, (2, 8)))
    assert logits.shape == (2, 8, 24)


def test_output_projection_shares_weights_with_embedding() -> None:
    model = _small_model()
    assert model.output.weight is model.token_embedding.weight


def test_model_is_causal() -> None:
    model = _small_model()
    ids = torch.randint(0, 24, (1, 10))
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 24
    with torch.no_grad():
        first = model(ids)
        second = model(changed)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)


def test_kv_cache_matches_full_forward() -> None:
    model = _small_model()
    ids = torch.randint(0, 24, (2, 12))
    with torch.no_grad():
        full = model(ids)
        head, past = model(ids[:, :7], use_cache=True)
        tail_parts = []
        for position in range(7, 12):
            step, past = model(ids[:, position:position + 1], past_key_values=past, use_cache=True)
            tail_parts.append(step)
        cached = torch.cat([head, *tail_parts], dim=1)
    assert torch.allclose(full, cached, atol=1e-4)


def test_cache_longer_than_context_is_rejected() -> None:
    model = _small_model(max_seq_len=8)
    with torch.no_grad():
        _, past = model(torch.randint(0, 24, (1, 8)), use_cache=True)
        with pytest.raises(ValueError):
            model(torch.randint(0, 24, (1, 1)), past_key_values=past, use_cache=True)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        AMITransformer(vocab_size=24, d_model=30, n_heads=4)
    with pytest.raises(ValueError):
        AMITransformer(vocab_size=24, d_model=12, n_heads=4)  # head_dim impar: RoPE lo exige par

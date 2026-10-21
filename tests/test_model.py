import torch
from ai.model.transformer import AMITransformer


def test_model_returns_next_token_logits() -> None:
    model = AMITransformer(vocab_size=24, d_model=32, n_heads=4, n_layers=1, max_seq_len=16)
    logits = model(torch.randint(0, 24, (2, 8)))
    assert logits.shape == (2, 8, 24)

from pathlib import Path

import torch

from ai.model.inference import ModelRunner
from ai.training.train import DEFAULT_DATA, train

SMALL = dict(vocab_size=300, d_model=32, n_heads=4, n_layers=1, max_seq_len=16, block_size=8, batch_size=2)


def test_training_saves_loadable_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ami-test.pt"
    train(DEFAULT_DATA, checkpoint, steps=1, seed=7)
    runner = ModelRunner(checkpoint)
    assert runner.available
    assert isinstance(runner.generate("Hola", max_new_tokens=2), str)


def test_training_can_resume_existing_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ami-resume.pt"
    train(DEFAULT_DATA, checkpoint, steps=1, seed=11)
    metrics = train(DEFAULT_DATA, checkpoint, steps=1, seed=12, resume=True)
    runner = ModelRunner(checkpoint)
    assert runner.available
    assert metrics["final_loss"] > 0
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["metadata"]["total_steps"] == 2


def test_training_reports_validation_loss_and_perplexity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ami-val.pt"
    metrics = train(DEFAULT_DATA, checkpoint, steps=2, seed=3, **SMALL)
    assert metrics["val_loss"] > 0
    assert metrics["perplexity"] > 1
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["metadata"]["val_loss"] == metrics["val_loss"]
    assert saved["format_version"] == 2


def test_generation_survives_going_past_the_context_window(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ami-long.pt"
    train(DEFAULT_DATA, checkpoint, steps=1, seed=5, **SMALL)
    runner = ModelRunner(checkpoint)
    text = runner.generate("Hola", max_new_tokens=60, seed=1)  # supera max_seq_len=16
    assert isinstance(text, str) and text


def test_generation_is_reproducible_with_a_seed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ami-seed.pt"
    train(DEFAULT_DATA, checkpoint, steps=1, seed=9, **SMALL)
    runner = ModelRunner(checkpoint)
    assert runner.generate("Hola", max_new_tokens=8, seed=4) == runner.generate("Hola", max_new_tokens=8, seed=4)


def test_legacy_checkpoint_is_reported_instead_of_crashing(tmp_path: Path) -> None:
    legacy = tmp_path / "viejo.pt"
    torch.save({"vocab": {"a": 0}, "reverse_vocab": ["a"], "config": {}, "model_state": {}}, legacy)
    runner = ModelRunner(legacy)
    assert not runner.available
    assert runner.load_error is not None and "formato anterior" in runner.load_error

from pathlib import Path

from ai.model.inference import ModelRunner
from ai.training.train import DEFAULT_DATA, train


def test_training_saves_loadable_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ami-test.pt"
    train(DEFAULT_DATA, checkpoint, steps=1, seed=7)
    runner = ModelRunner(checkpoint)
    assert runner.available
    assert isinstance(runner.generate("Hola", max_new_tokens=2), str)

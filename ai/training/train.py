from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ai.model.transformer import AMITransformer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "sample_corpus.txt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "ami-base.pt"


def train(data_path: Path, output_path: Path, steps: int, seed: int = 42) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    text = data_path.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("El corpus debe tener al menos 100 caracteres.")
    chars = sorted(set(text))
    vocab = {char: index for index, char in enumerate(chars)}
    encoded = torch.tensor([vocab[char] for char in text], dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {"d_model": 64, "n_heads": 4, "n_layers": 2, "max_seq_len": 128, "dropout": 0.1}
    model = AMITransformer(vocab_size=len(vocab), **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    seq_len = min(64, len(encoded) - 1)
    losses = []
    model.train()
    for step in range(max(1, steps)):
        starts = torch.randint(0, len(encoded) - seq_len - 1, (16,))
        x = torch.stack([encoded[i:i + seq_len] for i in starts]).to(device)
        y = torch.stack([encoded[i + 1:i + seq_len + 1] for i in starts]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"paso {step + 1}/{steps} - pérdida={losses[-1]:.4f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.cpu().state_dict(), "vocab": vocab, "reverse_vocab": chars,
                "config": config, "metadata": {"seed": seed, "steps": steps,
                "corpus_chars": len(text), "final_loss": losses[-1]}}, output_path)
    return {"initial_loss": losses[0], "final_loss": losses[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena AMI-Base desde pesos aleatorios.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train(args.data, args.output, args.steps, args.seed)
    print(f"Checkpoint guardado en: {args.output}")
    print(f"Pérdida inicial: {metrics['initial_loss']:.4f}; final: {metrics['final_loss']:.4f}")


if __name__ == "__main__":
    main()

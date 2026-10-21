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


def train(
    data_path: Path,
    output_path: Path,
    steps: int,
    seed: int = 42,
    resume: bool = False,
) -> dict[str, float]:
    if steps < 1:
        raise ValueError("steps debe ser mayor que cero.")
    random.seed(seed)
    torch.manual_seed(seed)
    text = data_path.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("El corpus debe tener al menos 100 caracteres.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"No existe el checkpoint para continuar: {output_path}")
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
        vocab = checkpoint["vocab"]
        chars = checkpoint["reverse_vocab"]
        unknown = sorted(set(text) - set(vocab))
        if unknown:
            sample = "".join(unknown[:20])
            raise ValueError(
                "El corpus contiene caracteres que no están en el vocabulario guardado "
                f"({sample!r}). Entrena una nueva versión sin --resume para crear vocabulario nuevo."
            )
        config = checkpoint["config"]
        model = AMITransformer(vocab_size=len(vocab), **config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        previous_steps = int(checkpoint.get("metadata", {}).get("total_steps", 0))
    else:
        chars = sorted(set(text + "Usuario: \nAMILCAR AI:"))
        vocab = {char: index for index, char in enumerate(chars)}
        config = {"d_model": 64, "n_heads": 4, "n_layers": 2, "max_seq_len": 128, "dropout": 0.1}
        model = AMITransformer(vocab_size=len(vocab), **config).to(device)
        previous_steps = 0

    encoded = torch.tensor([vocab[char] for char in text], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    seq_len = min(64, len(encoded) - 1)
    losses: list[float] = []
    model.train()
    for step in range(steps):
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
    torch.save({
        "model_state": model.cpu().state_dict(),
        "vocab": vocab,
        "reverse_vocab": chars,
        "config": config,
        "metadata": {
            "seed": seed,
            "last_run_steps": steps,
            "total_steps": previous_steps + steps,
            "corpus_chars": len(text),
            "final_loss": losses[-1],
            "resumed": resume,
        },
    }, output_path)
    return {"initial_loss": losses[0], "final_loss": losses[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena AMI-Base desde cero o continúa un checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continúa el checkpoint existente; conserva el vocabulario.")
    args = parser.parse_args()
    metrics = train(args.data, args.output, args.steps, args.seed, args.resume)
    print(f"Checkpoint guardado en: {args.output}")
    print(f"Pérdida inicial: {metrics['initial_loss']:.4f}; final: {metrics['final_loss']:.4f}")


if __name__ == "__main__":
    main()

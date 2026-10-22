from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ai.model.inference import CHECKPOINT_VERSION
from ai.model.transformer import AMITransformer
from ai.tokenizer import BPETokenizer
from ai.training.schedule import lr_at

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "sample_corpus.txt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "ami-base.pt"


def get_batch(
    data: torch.Tensor, seq_len: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ventanas aleatorias: x son los tokens e y los mismos tokens desplazados una posición."""
    starts = torch.randint(0, len(data) - seq_len, (batch_size,)).tolist()
    x = torch.stack([data[i:i + seq_len] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts]).to(device)
    return x, y


@torch.no_grad()
def evaluate(
    model: AMITransformer, data: torch.Tensor, seq_len: int, batch_size: int, batches: int, device: torch.device
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, seq_len, batch_size, device)
        logits = model(x)
        losses.append(float(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))))
    model.train()
    return sum(losses) / len(losses)


def train(
    data_path: Path,
    output_path: Path,
    steps: int,
    seed: int = 42,
    resume: bool = False,
    val_path: Path | None = None,
    vocab_size: int = 512,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    max_seq_len: int = 256,
    block_size: int = 128,
    batch_size: int = 16,
    lr: float = 3e-4,
    warmup_steps: int | None = None,
    eval_batches: int = 8,
    val_fraction: float = 0.05,
) -> dict[str, float]:
    if steps < 1:
        raise ValueError("steps debe ser mayor que cero.")
    if batch_size < 1 or eval_batches < 1 or block_size < 2:
        raise ValueError("batch_size y eval_batches deben ser >= 1 y block_size >= 2.")
    if lr <= 0:
        raise ValueError("lr debe ser mayor que cero.")
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
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("El checkpoint tiene un formato anterior; entrena uno nuevo sin --resume.")
        tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
        config = checkpoint["config"]
        model = AMITransformer(vocab_size=tokenizer.vocab_size, **config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        previous_steps = int(checkpoint.get("metadata", {}).get("total_steps", 0))
    else:
        tokenizer = BPETokenizer.train(text, vocab_size)
        config = {
            "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
            "max_seq_len": max_seq_len, "dropout": 0.1,
        }
        model = AMITransformer(vocab_size=tokenizer.vocab_size, **config).to(device)
        previous_steps = 0

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if val_path is not None:
        train_ids = ids
        val_ids = torch.tensor(tokenizer.encode(val_path.read_text(encoding="utf-8")), dtype=torch.long)
    elif val_fraction > 0 and len(ids) >= 200:
        held_out = max(32, int(len(ids) * val_fraction))  # el final del corpus se reserva para validar
        train_ids, val_ids = ids[:-held_out], ids[-held_out:]
    else:
        train_ids = val_ids = ids  # val_fraction=0 o corpus diminuto: se valida sobre el mismo texto
    if len(train_ids) < 10:
        raise ValueError("El corpus tiene muy pocos tokens para entrenar.")

    seq_len = min(block_size, config["max_seq_len"], len(train_ids) - 2)
    val_len = min(seq_len, len(val_ids) - 2)
    if val_len < 1:
        val_ids, val_len = train_ids, seq_len

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=lr
    )
    warmup = warmup_steps if warmup_steps is not None else min(100, max(1, steps // 10))
    report_every = max(1, steps // 10)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        current_lr = lr_at(step, steps, lr, warmup)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        x, y = get_batch(train_ids, seq_len, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % report_every == 0:
            val_now = evaluate(model, val_ids, val_len, batch_size, eval_batches, device)
            print(
                f"paso {step + 1}/{steps} - pérdida={losses[-1]:.4f} - "
                f"validación={val_now:.4f} - lr={current_lr:.2e}"
            )

    val_loss = evaluate(model, val_ids, val_len, batch_size, eval_batches, device)
    perplexity = math.exp(min(val_loss, 20.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": CHECKPOINT_VERSION,
        "model_state": model.cpu().state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": config,
        "metadata": {
            "seed": seed,
            "last_run_steps": steps,
            "total_steps": previous_steps + steps,
            "corpus_chars": len(text),
            "corpus_tokens": int(len(ids)),
            "vocab_size": tokenizer.vocab_size,
            "final_loss": losses[-1],
            "val_loss": val_loss,
            "perplexity": perplexity,
            "resumed": resume,
        },
    }, output_path)
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "val_loss": val_loss,
        "perplexity": perplexity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena AMI-Base desde cero o continúa un checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--val", type=Path, default=None, help="Archivo de validación (p. ej. data/processed/val.txt).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continúa el checkpoint existente; conserva tokenizador y arquitectura.")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Parte final del corpus reservada para validar; 0 = entrenar con todo (validación sobre lo mismo).")
    args = parser.parse_args()
    metrics = train(
        args.data, args.output, args.steps, args.seed, args.resume, args.val,
        args.vocab_size, args.d_model, args.n_heads, args.n_layers,
        args.max_seq_len, args.block_size, args.batch_size, args.lr, val_fraction=args.val_fraction,
    )
    print(f"Checkpoint guardado en: {args.output}")
    print(
        f"Pérdida inicial: {metrics['initial_loss']:.4f}; final: {metrics['final_loss']:.4f}; "
        f"validación: {metrics['val_loss']:.4f}; perplexity: {metrics['perplexity']:.2f}"
    )


if __name__ == "__main__":
    main()

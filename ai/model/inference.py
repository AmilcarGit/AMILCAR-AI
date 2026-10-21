from __future__ import annotations

import random
from pathlib import Path

import torch

from ai.model.sampling import sample_from_logits
from ai.model.transformer import AMITransformer
from ai.tokenizer import BPETokenizer

CHECKPOINT_VERSION = 2
FALLBACK_REPLY = "Todavía estoy aprendiendo. Prueba otra pregunta."


class LegacyCheckpointError(RuntimeError):
    """El checkpoint se guardó con una arquitectura o formato anterior y hay que reentrenar."""


class ModelRunner:
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model: AMITransformer | None = None
        self.tokenizer: BPETokenizer | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.available = False
        self.load_error: str | None = None
        if self.checkpoint_path.is_file():
            try:
                self.load()
            except LegacyCheckpointError as exc:
                self.load_error = str(exc)

    def load(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if checkpoint.get("format_version") != CHECKPOINT_VERSION:
            raise LegacyCheckpointError(
                "El checkpoint usa un formato anterior (otra arquitectura y tokenizador). "
                "Reentrena con: python -m ai.training.train"
            )
        self.tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
        self.model = AMITransformer(vocab_size=self.tokenizer.vocab_size, **checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.available = True

    @torch.inference_mode()
    def generate(
        self,
        message: str,
        max_new_tokens: int = 180,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> str:
        if not self.available or self.model is None or self.tokenizer is None:
            raise RuntimeError("El modelo aún no está entrenado")
        model, tokenizer = self.model, self.tokenizer
        rng = random.Random(seed) if seed is not None else random.Random()
        limit = model.max_seq_len

        context = tokenizer.encode("Usuario: " + message + "\nAMILCAR AI:")
        context = context[-(limit - 1):] or [tokenizer.eos_id if tokenizer.eos_id is not None else 0]
        all_ids = list(context)
        generated: list[int] = []

        logits, past = model(torch.tensor([context], dtype=torch.long, device=self.device), use_cache=True)
        for _ in range(max(1, min(max_new_tokens, 512))):
            scores = logits[0, -1, :].tolist()
            if tokenizer.pad_id is not None:
                scores[tokenizer.pad_id] = float("-inf")
            token = sample_from_logits(scores, temperature, top_k, top_p, rng)
            if token == tokenizer.eos_id:
                break
            generated.append(token)
            all_ids.append(token)
            if "\nUsuario:" in tokenizer.decode(generated):
                break
            if past[0][0].shape[2] >= limit:
                # Caché llena: se reconstruye con la mitad más reciente del contexto.
                window = all_ids[-max(1, limit // 2):]
                logits, past = model(torch.tensor([window], dtype=torch.long, device=self.device), use_cache=True)
            else:
                logits, past = model(
                    torch.tensor([[token]], dtype=torch.long, device=self.device),
                    past_key_values=past,
                    use_cache=True,
                )
        answer = tokenizer.decode(generated).split("\nUsuario:", 1)[0].strip()
        return answer or FALLBACK_REPLY

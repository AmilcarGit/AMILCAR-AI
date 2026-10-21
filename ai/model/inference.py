from __future__ import annotations

from pathlib import Path
import torch

from ai.model.transformer import AMITransformer


class ModelRunner:
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.available = False
        if self.checkpoint_path.is_file():
            self.load()

    def load(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.vocab = checkpoint["vocab"]
        self.reverse_vocab = checkpoint["reverse_vocab"]
        self.model = AMITransformer(vocab_size=len(self.vocab), **checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.available = True

    @torch.inference_mode()
    def generate(self, message: str, max_new_tokens: int = 180, temperature: float = 0.8) -> str:
        if not self.available or self.model is None:
            raise RuntimeError("El modelo aún no está entrenado")
        prompt = "Usuario: " + message + "\nAMILCAR AI:"
        ids = [self.vocab.get(char, self.vocab.get(" ", 0)) for char in prompt]
        tokens = torch.tensor([ids[-self.model.max_seq_len:] or [0]], dtype=torch.long, device=self.device)
        for _ in range(max(1, min(max_new_tokens, 512))):
            logits = self.model(tokens[:, -self.model.max_seq_len:])[:, -1, :]
            probabilities = torch.softmax(logits / max(temperature, 0.1), dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        generated = "".join(self.reverse_vocab[index] for index in tokens[0].tolist())
        answer = generated.split("AMILCAR AI:", 1)[-1].split("\nUsuario:", 1)[0].strip()
        return answer or "Todavía estoy aprendiendo. Prueba otra pregunta."

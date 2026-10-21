#!/usr/bin/env python3
"""Inferencia local de AMI Nano 2.0; biblioteca estándar de Python."""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("AMI_MODEL_PATH", os.path.join(ROOT, "artifacts", "ami-nano.json"))

def softmax(logits):
    peak = max(logits)
    exps = [math.exp(max(-40.0, min(40.0, x - peak))) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]

def main():
    if not os.path.isfile(MODEL):
        print("No existe el modelo AMI Nano 2.0. Ejecuta npm run ami:train.", file=sys.stderr)
        return 2
    try:
        with open(MODEL, encoding="utf-8") as f:
            model = json.load(f)
        if model.get("format") != "ami-nano-neural-bigram-v2":
            print("Checkpoint incompatible. Ejecuta npm run ami:train para crear AMI Nano 2.0.", file=sys.stderr)
            return 2
        vocab = model["vocab"]
        weights = model["weights"]
        bias = model["bias"]
        if not vocab or len(weights) != len(vocab) or len(bias) != len(vocab):
            raise ValueError("estructura de pesos invalida")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print("No se pudo cargar el modelo:", str(exc), file=sys.stderr)
        return 2
    prompt = " ".join(sys.argv[1:]).strip().lower() or "hola"
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    rng = random.Random()
    known = [ch for ch in prompt if ch in char_to_id]
    current = char_to_id[known[-1]] if known else char_to_id.get(" ", 0)
    output = prompt
    max_new = int(os.environ.get("AMI_MAX_NEW_TOKENS", "180"))
    temperature = max(0.2, min(float(os.environ.get("AMI_TEMPERATURE", "0.85")), 2.0))
    for _ in range(max(1, min(max_new, 500))):
        logits = [(weights[current][j] + bias[j]) / temperature for j in range(len(vocab))]
        probs = softmax(logits)
        current = rng.choices(range(len(vocab)), weights=probs, k=1)[0]
        output += vocab[current]
        if vocab[current] in ".!?\n" and len(output) >= len(prompt) + 35:
            break
    print(output.strip())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

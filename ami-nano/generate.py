#!/usr/bin/env python3
"""Generación local con un modelo de caracteres entrenado desde cero."""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("AMI_MODEL_PATH", os.path.join(ROOT, "artifacts", "ami-nano.json"))

def main():
    if not os.path.isfile(MODEL):
        print("No existe el modelo AMI Nano. Ejecuta npm run ami:train.", file=sys.stderr)
        return 2
    with open(MODEL, encoding="utf-8") as f:
        model = json.load(f)
    prompt = " ".join(sys.argv[1:]).strip().lower()
    if not prompt:
        prompt = "hola"
    transitions = model["transitions"]
    rng = random.Random()
    current = prompt[-1] if prompt[-1] in transitions else (model["vocab"][0] if model["vocab"] else " ")
    output = prompt + " "
    for _ in range(240):
        options = transitions.get(current) or transitions.get(" ")
        if not options:
            break
        chars, weights = zip(*options.items())
        current = rng.choices(chars, weights=weights, k=1)[0]
        output += current
        if current in ".!?\n" and len(output) > len(prompt) + 50:
            break
    print(output.strip())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

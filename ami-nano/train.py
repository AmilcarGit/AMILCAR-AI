#!/usr/bin/env python3
"""Entrena AMI Nano 2.0: red neuronal de caracteres desde pesos aleatorios, sin librerías externas."""
import json
import math
import os
import random
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.environ.get("AMI_CORPUS", os.path.join(ROOT, "data", "corpus.txt"))
MODEL = os.environ.get("AMI_MODEL_PATH", os.path.join(ROOT, "artifacts", "ami-nano.json"))
SEED = int(os.environ.get("AMI_SEED", "7"))
EPOCHS = max(1, min(int(os.environ.get("AMI_EPOCHS", "80")), 2000))
LR = float(os.environ.get("AMI_LR", "0.08"))
DEMO = """hola soy ami nano, una inteligencia artificial experimental creada por amilcar.
hola, como estas? estoy aprendiendo poco a poco.
me llamo ami y funciono localmente en tu telefono.
puedo responder saludos y continuar texto aprendido.
amilcar ai es un proyecto abierto para crear una inteligencia artificial propia.
esta version es pequena y todavia comete errores.
hola amigo, que tal? espero que estes bien.
que puedes hacer? puedo generar texto sencillo usando ejemplos de entrenamiento.
no necesito internet, una api comercial ni modelos preentrenados.
aprendo patrones del texto que me proporcionas.
para mejorar mis respuestas necesito mas ejemplos variados y bien escritos.
la inteligencia artificial se desarrolla con datos, entrenamiento, pruebas y mejoras.
buenos dias. buenas tardes. buenas noches.
gracias por probar ami nano. seguire mejorando con nuevas versiones.
"""

def softmax(logits):
    peak = max(logits)
    exps = [math.exp(max(-40.0, min(40.0, x - peak))) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]

def main():
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    if not os.path.exists(CORPUS):
        with open(CORPUS, "w", encoding="utf-8") as f:
            f.write(DEMO)
        print("Creado corpus de demostracion:", CORPUS)
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read().lower()
    text = text.replace("\r\n", "\n")
    if len(text.strip()) < 50:
        raise SystemExit("El corpus necesita al menos 50 caracteres.")
    vocab = sorted(set(text))
    if len(vocab) > 256:
        raise SystemExit("AMI Nano 2.0 admite hasta 256 caracteres distintos.")
    ids = {ch: i for i, ch in enumerate(vocab)}
    sequence = [ids[ch] for ch in text]
    n = len(vocab)
    rng = random.Random(SEED)
    weights = [[rng.uniform(-0.01, 0.01) for _ in range(n)] for _ in range(n)]
    bias = [0.0] * n
    learning_rate = LR
    pairs = list(zip(sequence[:-1], sequence[1:]))
    started = time.time()
    for epoch in range(EPOCHS):
        rng.shuffle(pairs)
        loss = 0.0
        for prev, target in pairs:
            probs = softmax([weights[prev][j] + bias[j] for j in range(n)])
            loss -= math.log(max(probs[target], 1e-12))
            for j in range(n):
                grad = probs[j] - (1.0 if j == target else 0.0)
                weights[prev][j] -= learning_rate * grad
                bias[j] -= learning_rate * grad
        if epoch in (EPOCHS // 2, EPOCHS - 1):
            print("Epoca", epoch + 1, "/", EPOCHS, "| perdida media:", round(loss / len(pairs), 4))
    payload = {
        "format": "ami-nano-neural-bigram-v2",
        "version": "2.0.0",
        "trained_from_scratch": True,
        "architecture": "one-hot-character-to-softmax",
        "seed": SEED,
        "epochs": EPOCHS,
        "corpus_chars": len(text),
        "vocab": vocab,
        "weights": weights,
        "bias": bias,
    }
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    tmp = MODEL + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, MODEL)
    print("AMI Nano 2.0 entrenado desde cero.")
    print("Caracteres:", len(text), "| vocabulario:", n, "| ejemplos:", len(pairs))
    print("Tiempo:", round(time.time() - started, 2), "s")
    print("Checkpoint:", MODEL)

if __name__ == "__main__":
    main()

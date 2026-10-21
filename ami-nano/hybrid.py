#!/usr/bin/env python3
"""Chat híbrido experimental: conocimiento local, candidato neuronal y reglas verificables."""
import os
import re
import subprocess
import sys
from pathlib import Path

from chat_es import load_dialogs, normalize

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "ami-nano" / "generate.py"
STOPWORDS = {
    "a", "al", "algo", "como", "con", "de", "del", "desde", "el", "ella",
    "en", "es", "esta", "este", "la", "las", "lo", "los", "me", "mi",
    "para", "por", "que", "se", "si", "sin", "soy", "su", "te", "tengo",
    "tu", "un", "una", "unas", "uno", "unos", "y", "yo",
}
VOWELS = set("aeiouáéíóúü")


def match_knowledge(message, dialogs):
    query = normalize(message)
    query_words = set(query.split())
    best = None
    best_pattern = ""
    best_score = 0.0
    for item in dialogs:
        for pattern in item.get("patterns", []):
            candidate = normalize(pattern)
            if not candidate:
                continue
            if query == candidate:
                score = 1.0
            else:
                words = set(candidate.split())
                score = len(query_words & words) / max(1, len(query_words | words))
                if candidate in query or query in candidate:
                    score = max(score, 0.82)
            if score > best_score:
                best, best_pattern, best_score = item, pattern, score
    if best is not None and best_score >= 0.45:
        return {"item": best, "pattern": best_pattern, "score": best_score}
    return None


def evaluate_candidate(question, candidate):
    text = candidate.strip()
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    meaningful = set(normalize(question).split()) - STOPWORDS
    candidate_words = set(words)
    overlap = meaningful & candidate_words
    vowel_count = sum(char.lower() in VOWELS for char in text)
    letters = sum(char.isalpha() for char in text)
    repeated_run = re.search(r"(.)\1{4,}", text.lower()) is not None
    checks = {
        "nonempty": bool(text),
        "length": 25 <= len(text) <= 600,
        "enough_words": len(words) >= 5,
        "word_diversity": len(candidate_words) / max(1, len(words)) >= 0.35,
        "vowel_ratio": letters > 0 and vowel_count / letters >= 0.25,
        "no_repeated_run": not repeated_run,
        "question_relevance": len(overlap) >= 2,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "word_count": len(words),
        "relevant_terms": sorted(overlap),
    }


def generate_candidate(message, timeout=8):
    env = os.environ.copy()
    env["AMI_MAX_NEW_TOKENS"] = "180"
    prompt = " ".join(message.strip().lower().split())
    try:
        result = subprocess.run(
            [sys.executable, str(GENERATOR), prompt],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if output.lower().startswith(prompt):
        output = output[len(prompt):].strip()
    return output or None


def respond(message, dialogs=None, candidate=None):
    if not isinstance(message, str) or not message.strip():
        return {
            "text": "Escríbeme una pregunta concreta y trataré de ayudarte.",
            "source": "rule",
            "quality": {"accepted": False, "reason": "empty_input"},
        }

    dialogs = dialogs if dialogs is not None else load_dialogs()
    match = match_knowledge(message, dialogs)
    if match:
        item = match["item"]
        answers = item.get("responses", [])
        if answers:
            example = match["pattern"]
            answer = answers[0]
            return {
                "text": (
                    f"Respuesta: {answer}\n\n"
                    f"Explicación: encontré una coincidencia en el conocimiento local "
                    f"para el tema «{item.get('intent', 'general')}». La respuesta procede "
                    f"de data/dialogos-es.json; esta coincidencia no significa que la red "
                    f"neuronal haya aprendido ese contenido.\n\n"
                    f"Ejemplo relacionado: «{example}»."
                ),
                "source": "local_knowledge",
                "quality": {
                    "accepted": True,
                    "knowledge_score": round(match["score"], 3),
                    "intent": item.get("intent", "general"),
                },
            }

    if candidate is None:
        candidate = generate_candidate(message)
    if candidate:
        quality = evaluate_candidate(message, candidate)
        if quality["accepted"]:
            return {
                "text": (
                    "Propuesta experimental del modelo neuronal (no verificada):\n"
                    f"{candidate}\n\n"
                    "Explicación: el texto superó las reglas básicas de longitud, variedad, "
                    "vocales, repetición y relación léxica con la pregunta. Estas reglas no "
                    "garantizan que sea verdadero ni que tenga sentido."
                ),
                "source": "neural_candidate",
                "quality": quality,
            }
    else:
        quality = {"accepted": False, "reason": "model_missing_or_unavailable"}

    if candidate:
        quality = evaluate_candidate(message, candidate)
        reason = "El candidato neuronal no superó todas las reglas de calidad."
    else:
        reason = (
            "No pude generar un candidato neuronal; comprueba que exista el modelo "
            "entrenado con «npm run ami:train»."
        )
    return {
        "text": (
            "Todavía no tengo una respuesta fiable para esa pregunta.\n\n"
            f"Explicación: {reason} La red actual aprende transiciones entre caracteres, "
            "no el significado de las preguntas, así que prefiero no inventar una respuesta.\n\n"
            "Ejemplo para ampliar el conocimiento: añade a data/dialogos-es.json una "
            "intención con varias formas de preguntar y una respuesta revisada. Después "
            "ejecuta «npm run ami:test» para comprobar los casos."
        ),
        "source": "safe_fallback",
        "quality": quality,
    }


def main():
    try:
        dialogs = load_dialogs()
    except (OSError, ValueError) as exc:
        print("No se pudieron cargar los conocimientos locales: " + str(exc), file=sys.stderr)
        return 2
    message = " ".join(sys.argv[1:]).strip()
    if message:
        print(respond(message, dialogs)["text"])
        return 0
    print("AMILCAR AI — chat híbrido local. Escribe 'salir' para terminar.")
    try:
        while True:
            message = input("Tú: ").strip()
            if normalize(message) in {"salir", "adios", "hasta luego"}:
                print("AMI: ¡Hasta luego!")
                break
            result = respond(message, dialogs)
            print("AMI: " + result["text"])
            print("[control de calidad: " + result["source"] + "]")
    except (EOFError, KeyboardInterrupt):
        print("\nAMI: ¡Hasta luego!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

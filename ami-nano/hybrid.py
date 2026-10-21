#!/usr/bin/env python3
"""AMI Nano Evolutiva 2.1: recuperación local y composición controlada en español."""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
import sys
import shutil
import time
from pathlib import Path

from chat_es import load_dialogs, normalize

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "ami-nano" / "generate.py"
STOPWORDS = {
    "a", "al", "algo", "como", "con", "de", "del", "desde", "el", "ella",
    "en", "es", "esta", "este", "la", "las", "lo", "los", "me", "mi",
    "para", "por", "que", "se", "si", "sin", "soy", "su", "te", "tengo",
    "tu", "un", "una", "unas", "uno", "unos", "y", "yo", "qué", "cuál",
    "puedes", "explica", "explicame", "dime", "habla", "sobre", "favor",
}
KNOWLEDGE = {
    "fotosintesis": {
        "terms": {"fotosintesis", "plantas", "luz", "solar", "agua", "dioxido", "carbono", "glucosa", "oxigeno"},
        "answer": (
            "La fotosíntesis es el proceso por el que las plantas, las algas y algunas bacterias "
            "transforman la energía de la luz en energía química. En las plantas ocurre principalmente "
            "en los cloroplastos: usan agua y dióxido de carbono para producir azúcares y liberan oxígeno. "
            "Por ejemplo, una planta recibe luz por sus hojas y utiliza esa energía para fabricar parte "
            "del alimento que necesita para crecer."
        ),
        "intent": "ciencia_fotosintesis",
    },
    "aprendizaje": {
        "terms": {"aprender", "aprendizaje", "conocimiento", "practicar", "preguntas", "errores", "experiencia"},
        "answer": (
            "Aprender cosas nuevas ayuda a ampliar conocimientos, descubrir otras perspectivas y resolver "
            "problemas. Una forma práctica de aprender es estudiar una idea, explicarla con tus propias "
            "palabras, probarla con un ejemplo y comprobar qué entendiste. Equivocarse también sirve si "
            "revisas el error y corriges lo que haga falta."
        ),
        "intent": "aprendizaje",
    },
    "inteligencia_artificial": {
        "terms": {"inteligencia", "artificial", "ia", "modelo", "datos", "automatizar", "privacidad", "errores"},
        "answer": (
            "La inteligencia artificial es un conjunto de técnicas que permite a un sistema realizar tareas "
            "como reconocer patrones, procesar lenguaje o generar contenido. Puede ayudar a estudiar y "
            "automatizar trabajos, pero sus resultados deben verificarse porque también puede equivocarse. "
            "Conviene proteger los datos personales y no confundir una respuesta convincente con una respuesta verdadera."
        ),
        "intent": "inteligencia_artificial",
    },
}

def tokens(text):
    return set(re.findall(r"[a-záéíóúüñ0-9]+", normalize(text))) - STOPWORDS


def match_knowledge(message, dialogs):
    query = normalize(message)
    query_words = tokens(message)
    best = None
    best_pattern = ""
    best_score = 0.0
    for item in dialogs:
        for pattern in item.get("patterns", []):
            candidate = normalize(pattern)
            if not candidate:
                continue
            pattern_words = tokens(pattern)
            if query == candidate:
                score = 1.0
            elif not pattern_words or not query_words:
                score = 0.0
            else:
                precision = len(query_words & pattern_words) / len(query_words)
                recall = len(query_words & pattern_words) / len(pattern_words)
                score = 2 * precision * recall / max(precision + recall, 1e-9)
                if candidate in query or query in candidate:
                    score = max(score, 0.88)
            if score > best_score:
                best, best_pattern, best_score = item, pattern, score
    if best is not None and best_score >= 0.68:
        return {"item": best, "pattern": best_pattern, "score": best_score}
    return None


def retrieve_local_knowledge(message):
    query = tokens(message)
    if not query:
        return None
    best = None
    best_score = 0.0
    for key, item in KNOWLEDGE.items():
        terms = item["terms"]
        overlap = query & terms
        score = len(overlap) / max(1, len(query))
        if key in normalize(message):
            score = max(score, 1.0)
        if score > best_score:
            best, best_score = item, score
    if best is not None and best_score >= 0.25:
        return {"item": best, "score": best_score}
    return None


def evaluate_candidate(question, candidate):
    text = candidate.strip()
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    meaningful = tokens(question)
    candidate_words = set(words)
    overlap = meaningful & candidate_words
    vowel_count = sum(char.lower() in "aeiouáéíóúü" for char in text)
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
            cwd=ROOT, env=env, text=True, capture_output=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if output.lower().startswith(prompt):
        output = output[len(prompt):].strip()
    return output or None


def _ensure_local_llm_server(model_path, endpoint):
    if not endpoint.startswith("http://127.0.0.1:8081/"):
        return False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=1):
            return True
    except (OSError, TimeoutError, urllib.error.URLError):
        pass

    server = shutil.which("llama-server")
    if not server:
        return False
    try:
        subprocess.Popen(
            [server, "-m", str(model_path), "--host", "127.0.0.1", "--port", "8081", "-c", "2048"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        return False

    for _ in range(50):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=1):
                return True
        except (OSError, TimeoutError, urllib.error.URLError):
            continue
    return False


def generate_llm_candidate(message, timeout=None):
    default_model = Path.home() / ".cache" / "amilcar-ai" / "models" / "qwen2.5-0.5b-instruct-q2_k.gguf"
    model_path = os.environ.get("AMILCAR_AI_LLM_MODEL", "").strip() or str(default_model)
    if not Path(model_path).is_file():
        return None

    endpoint = os.environ.get(
        "AMILCAR_AI_LLM_URL",
        "http://127.0.0.1:8081/v1/chat/completions",
    ).strip()
    if not endpoint.startswith(("http://127.0.0.1:", "http://localhost:", "http://[::1]:")):
        return None
    if endpoint == "http://127.0.0.1:8081/v1/chat/completions":
        if not _ensure_local_llm_server(model_path, endpoint):
            return None

    try:
        timeout = timeout or max(3, int(os.environ.get("AMILCAR_AI_LLM_TIMEOUT", "90")))
        max_tokens = max(32, min(400, int(os.environ.get("AMILCAR_AI_LLM_TOKENS", "180"))))
    except ValueError:
        return None

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres AMILCAR AI. Responde en español natural y útil. Explica la idea principal "
                    "con claridad y añade un ejemplo breve si ayuda. Si no sabes algo, dilo con "
                    "honestidad. No inventes fuentes ni repitas la pregunta."
                ),
            },
            {"role": "user", "content": " ".join(message.strip().split())},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, ValueError, urllib.error.URLError):
        return None

    try:
        output = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    for marker in ("<|im_end|>", "<|endoftext|>", "</s>"):
        output = output.split(marker, 1)[0].strip()
    return output or None


def evaluate_llm_candidate(candidate):
    text = candidate.strip()
    words = re.findall(r"[a-záéíóúüñ]+", text.lower())
    letters = sum(char.isalpha() for char in text)
    vowel_count = sum(char.lower() in "aeiouáéíóúü" for char in text)
    repeated_run = re.search(r"(.)\1{4,}", text.lower()) is not None
    accepted = (
        25 <= len(text) <= 1800
        and len(words) >= 5
        and len(set(words)) / max(1, len(words)) >= 0.25
        and letters > 0
        and vowel_count / max(1, letters) >= 0.20
        and not repeated_run
    )
    return {"accepted": accepted, "word_count": len(words), "source": "local_llm"}


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
            answer = answers[0]
            return {
                "text": (
                    f"Respuesta: {answer}\n\n"
                    f"Explicación: recuperé una respuesta revisada de la base local "
                    f"para el tema «{item.get('intent', 'general')}». Esto es recuperación "
                    "de conocimiento, no una prueba de que la red neuronal haya aprendido ese contenido.\n\n"
                    f"Ejemplo relacionado: «{match['pattern']}»."
                ),
                "source": "local_knowledge",
                "quality": {
                    "accepted": True,
                    "knowledge_score": round(match["score"], 3),
                    "intent": item.get("intent", "general"),
                },
            }

    retrieved = retrieve_local_knowledge(message)
    if retrieved:
        item = retrieved["item"]
        return {
            "text": (
                f"Respuesta: {item['answer']}\n\n"
                f"Explicación: construí esta respuesta usando conocimiento local revisado "
                f"del tema «{item['intent']}». La coincidencia léxica fue "
                f"{round(retrieved['score'] * 100)}%; no equivale a una verificación automática de todos los hechos.\n\n"
                "Ejemplo relacionado: puedo explicar el concepto paso a paso o responder una pregunta más concreta."
            ),
            "source": "local_composition",
            "quality": {"accepted": True, "knowledge_score": round(retrieved["score"], 3), "intent": item["intent"]},
        }

    llm_candidate = generate_llm_candidate(message)
    if llm_candidate:
        llm_quality = evaluate_llm_candidate(llm_candidate)
        if llm_quality["accepted"]:
            return {
                "text": (
                    f"{llm_candidate}\n\n"
                    "— Respuesta generada localmente. Puede contener errores; verifica los datos importantes."
                ),
                "source": "local_llm",
                "quality": llm_quality,
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
                    "Explicación: el texto superó reglas básicas de forma y coincidencia léxica. "
                    "Estas comprobaciones no garantizan verdad, coherencia ni comprensión semántica."
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
        reason = "No pude generar un candidato neuronal; comprueba el entrenamiento con «npm run ami:train»."
    return {
        "text": (
            "Todavía no tengo una respuesta fiable para esa pregunta.\n\n"
            f"Explicación: {reason} La red neuronal actual aprende transiciones entre caracteres; "
            "no comprende por sí sola el significado del lenguaje, así que prefiero no inventar.\n\n"
            "Ejemplo para ampliar el conocimiento: añade a data/dialogos-es.json una intención con varias "
            "formas de preguntar y una respuesta revisada. Después ejecuta «npm run ami:test»."
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
    print("AMI Nano Evolutiva 2.1 — chat híbrido local. Escribe 'salir' para terminar.")
    try:
        while True:
            message = input("Tú: ").strip()
            if normalize(message) in {"salir", "adios", "hasta luego"}:
                print("AMI: ¡Hasta luego!")
                break
            result = respond(message, dialogs)
            print("AMI: " + result["text"])
            print("[origen: " + result["source"] + "]")
    except (EOFError, KeyboardInterrupt):
        print("\nAMI: ¡Hasta luego!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

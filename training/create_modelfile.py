#!/usr/bin/env python3
"""Genera un Modelfile de Ollama apuntando al GGUF exportado por Unsloth."""
from pathlib import Path

GGUF_DIR = Path("outputs/ami-qwen35/gguf")
PROMPT_FILE = Path("training/SYSTEM_PROMPT.txt")
OUTPUT = Path("Modelfile")

def main():
    candidates = sorted(GGUF_DIR.glob("*.gguf"))
    if len(candidates) != 1:
        raise SystemExit(
            f"Se esperaba exactamente un GGUF en {GGUF_DIR}, se encontraron {len(candidates)}. "
            "Comprueba la exportación y conserva solo el archivo Q4_K_M que quieras importar."
        )
    system = PROMPT_FILE.read_text(encoding="utf-8").strip()
    escaped = system.replace('"""', '""')
    content = f'''FROM ./{candidates[0].as_posix()}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """
{escaped}
"""
'''
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Creado {OUTPUT}; GGUF: {candidates[0]}")
    print("Desde la raíz del repositorio ejecuta: ollama create ami -f Modelfile")

if __name__ == "__main__":
    main()

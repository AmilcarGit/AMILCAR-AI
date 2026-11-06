# AMI — fine-tuning local con Qwen3.5 + LoRA

Este directorio contiene un flujo **opcional y separado** para ajustar un modelo open-source y ejecutarlo localmente. No cambia el motor JavaScript actual de AMILCAR-AI, no modifica BaseBot y no incorpora APIs externas.

## Modelo y método

- Base: `unsloth/Qwen3.5-4B`
- Método: SFT + LoRA BF16 (no QLoRA de 4 bits).
- Motivo: la guía oficial de Unsloth recomienda BF16 LoRA para Qwen3.5 y desaconseja QLoRA 4-bit por diferencias de cuantización.
- Memoria orientativa oficial: aproximadamente 10 GB de VRAM para Qwen3.5-4B BF16 LoRA. La memoria real depende de secuencia, lote, versiones y entorno.
- Para GPU de 16 GB: empezar con secuencia 1024–2048 y batch 1. Colab gratuito puede cambiar de GPU y disponibilidad.

Documentación oficial: https://unsloth.ai/docs/models/qwen3.5/fine-tune  
Notebook oficial 4B: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_%284B%29_Vision.ipynb

## Archivos

- `data/train.example.jsonl`: formato de dataset y ejemplos iniciales.
- `train.py`: entrenamiento, validación y guardado de adaptador LoRA.
- `evaluate.py`: prueba conversacional del adaptador.
- `SYSTEM_PROMPT.txt`: identidad y reglas de AMI.
- `Modelfile.example`: plantilla para Ollama.
- `DATASET_PROMPT.txt`: prompt para crear lotes de ejemplos con otra IA.

## Preparación

En Linux, WSL o Google Colab con GPU NVIDIA:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade unsloth unsloth_zoo datasets trl
```

Qwen3.5 requiere Transformers v5. La instalación reciente de Unsloth normalmente gestiona esta dependencia; si el entorno conserva una versión anterior, actualízala según la guía oficial de Unsloth.

En Colab, sube la carpeta `training/qwen35-lora` o clona el repositorio y cambia al directorio:

```bash
git clone https://github.com/AmilcarGit/AMILCAR-AI.git
cd AMILCAR-AI/training/qwen35-lora
```

## Dataset

Copia el ejemplo y crea el archivo real:

```bash
mkdir -p data outputs
cp data/train.example.jsonl data/train.jsonl
```

Cada línea es un objeto JSON con `messages`, compatible con el chat template del tokenizer:

```json
{"messages":[{"role":"user","content":"Explícame la gravedad con un ejemplo."},{"role":"assistant","content":"La gravedad es la interacción por la que los objetos con masa se atraen. Por ejemplo, la Tierra atrae una pelota hacia el suelo cuando la sueltas."}]}
```

Añade cientos de conversaciones revisadas. Evita duplicados, datos personales, hechos inventados y respuestas que simulen certeza. Mantén un archivo de evaluación separado o usa la división automática reproducible del script.

## Entrenar

```bash
python train.py
```

El script separa el 10% para evaluación, entrena tres épocas inicialmente y guarda el adaptador en `outputs/ami-qwen35-lora`. No subas pesos ni checkpoints al repositorio.

## Evaluar

```bash
python evaluate.py
```

Compara las respuestas del modelo base y del modelo ajustado. Evalúa exactitud, utilidad, naturalidad, español neutro y reconocimiento de incertidumbre. La pérdida de entrenamiento por sí sola no demuestra una mejora.

## Exportar a GGUF

Cuando el entrenamiento termine, puedes descomentar el bloque de exportación de `train.py` o ejecutar:

```python
# Ejecutar después de cargar el adaptador entrenado en train.py:
model.save_pretrained_gguf(
    "outputs/ami-qwen35-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

Comprueba el nombre real del archivo GGUF producido. Conserva el mismo chat template usado durante el entrenamiento; una plantilla incorrecta puede degradar las respuestas en Ollama.

## Ollama

Copia `Modelfile.example` como `Modelfile`, cambia la ruta de `FROM` por el archivo GGUF real y ejecuta:

```bash
ollama create ami -f Modelfile
ollama run ami
```

## Limitaciones reales

Un ajuste LoRA enseña patrones de respuesta, tono y comportamiento. No crea conocimiento general desde cero ni actualiza hechos automáticamente. Un modelo 4B será mucho menos capaz que los modelos grandes de ChatGPT/Grok en razonamiento complejo, programación extensa, precisión y contexto. Para conocimiento privado o verificable, añade una recuperación documental local (RAG) y pruebas automatizadas.

## Licencia y redistribución

Revisa la licencia y condiciones de uso del modelo base antes de redistribuir pesos derivados. El código de este directorio no incluye pesos del modelo.

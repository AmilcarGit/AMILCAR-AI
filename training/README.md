# Fine-tuning local de AMI (LoRA + Unsloth)

Esta carpeta contiene un flujo opcional de fine-tuning supervisado (SFT) para personalizar el estilo de AMI. No reemplaza el motor JavaScript ni cambia la instalación del SDK. El entrenamiento requiere una GPU NVIDIA CUDA; el modelo final se puede exportar a GGUF y ejecutar con Ollama.

## Modelo recomendado

- Base: `unsloth/Qwen3.5-4B`.
- Método: LoRA BF16/FP16 con Unsloth.
- Exportación: GGUF `Q4_K_M`.
- Motivo: Unsloth publica soporte específico para Qwen3.5-4B y estima aproximadamente 10 GB de VRAM para BF16 LoRA. Qwen3.5 no es recomendable para QLoRA 4-bit según la guía oficial; por eso esta receta usa LoRA de 16 bits, no QLoRA.

Fuentes oficiales:
- https://unsloth.ai/docs/models/qwen3.5/fine-tune
- https://ollama.com/download
- https://github.com/ollama/ollama/blob/main/docs/import.mdx

## 1. Entorno

La opción más sencilla es Google Colab con GPU NVIDIA. Sube o clona el repositorio y ejecuta desde su raíz:

```bash
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo datasets trl
```

En Colab, si la instalación actualiza paquetes ya cargados, reinicia el entorno antes de importar Unsloth. Qwen3.5 necesita Transformers v5; las versiones actuales de Unsloth gestionan esta dependencia según su guía.

Comprueba la GPU:

```python
import torch
print(torch.cuda.get_device_name(0))
print(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
```

## 2. Dataset

Copia los ejemplos:

```bash
cp training/data/train.example.jsonl training/data/train.jsonl
cp training/data/eval.example.jsonl training/data/eval.jsonl
```

Cada línea es un objeto JSON con una lista `messages` y roles `user`/`assistant`. Los archivos `.example.jsonl` son solo muestras pequeñas, no un dataset suficiente para un modelo final.

Objetivo inicial: 300–1.000 conversaciones revisadas. Mantén evaluación separada: reserva preguntas distintas de las de entrenamiento (10–15% como referencia). Incluye explicación, conversación, programación, razonamiento, límites e incertidumbre. No generes ejemplos falsos ni repitas plantillas cambiando solo unas palabras.

Puedes usar otra IA para proponer lotes de 30–50 ejemplos, pero revisa cada respuesta, elimina hechos dudosos y valida que cada línea sea JSON válido. No incluyas datos personales o secretos.

## 3. Entrenar

Desde la raíz del repositorio:

```bash
python training/train.py
```

Variables opcionales:

```bash
AMI_MAX_SEQ_LENGTH=1024 AMI_EPOCHS=2 python training/train.py
```

El script guarda el adaptador en `outputs/ami-qwen35/lora-adapter` y solicita una exportación GGUF Q4_K_M en `outputs/ami-qwen35/gguf`.

Si aparece falta de memoria, reduce primero `AMI_MAX_SEQ_LENGTH` a 1024. La memoria real depende del hardware, longitud, versión de CUDA y tamaño del dataset.

## 4. Evaluar

```bash
python training/evaluate.py
```

Evalúa manualmente factualidad, utilidad, naturalidad, seguimiento de instrucciones y reconocimiento de incertidumbre. Compara el modelo base y el adaptado con las mismas preguntas que no aparezcan en el entrenamiento. Una pérdida menor no demuestra por sí sola que las respuestas sean mejores.

## 5. Crear el Modelfile y ejecutar con Ollama

Después de terminar la exportación:

```bash
python training/create_modelfile.py
ollama create ami -f Modelfile
ollama run ami
```

El generador comprueba que exista exactamente un GGUF en el directorio de salida y crea el `Modelfile` en la raíz. Si la exportación genera varios archivos GGUF, selecciona manualmente el archivo Q4_K_M y ajusta el script.

## Limitaciones realistas

- Un modelo de 4B parámetros no iguala la capacidad general, razonamiento, herramientas ni conocimientos de modelos grandes.
- 300–1.000 ejemplos pueden ajustar tono y patrones básicos; no bastan para enseñar todo el conocimiento general.
- El fine-tuning no elimina las alucinaciones ni actualiza automáticamente hechos.
- Para conocimiento propio o cambiante, añade recuperación documental local (RAG) y fuentes verificables.
- Ollama y el GGUF requieren memoria RAM/VRAM suficiente para inferencia; entrenar en Colab no significa que el modelo vaya a correr rápido en un teléfono.

Este entrenamiento es opcional y separado de la arquitectura local actual de AMILCAR-AI. No usa APIs externas para generar respuestas en tiempo de ejecución.

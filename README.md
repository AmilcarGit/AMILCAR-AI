# AMILCAR AI

IA abierta, local y entrenable desde cero. MVP experimental creado por AmilcarGit.

## Incluye
- Interfaz cyberpunk adaptable a móvil y escritorio con React, Vite y TypeScript.
- API local FastAPI.
- Transformer causal pequeño de PyTorch, inicializado desde pesos aleatorios.
- Entrenamiento reproducible, checkpoints y pruebas básicas.
- Sin claves comerciales obligatorias.

## Requisitos
Python 3.11 o 3.12, Node.js 20+ y npm. PyTorch debe ser compatible con tu plataforma.

## Instalación
Clona el repositorio:

    git clone https://github.com/AmilcarGit/AMILCAR-AI.git
    cd AMILCAR-AI

Crea un entorno virtual:

    python -m venv .venv

Linux/macOS/Termux: `source .venv/bin/activate`  
Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Instala dependencias:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

Entrena el modelo desde cero:

    python -m ai.training.train --steps 300

Inicia la API:

    uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

En otra terminal, inicia la interfaz:

    cd apps/web
    npm install
    npm run dev

La API queda en http://127.0.0.1:8000, su documentación en http://127.0.0.1:8000/docs y Vite mostrará la dirección de la interfaz.

## API
- `GET /health`: estado y disponibilidad del modelo.
- `POST /api/chat`: generación con AMI-Base.
- `GET /docs`: documentación interactiva.

Ejemplo de cuerpo JSON para el chat: `{"message":"Hola AMILCAR AI"}`.

## Modelo
AMI-Base es un Transformer pequeño y experimental. Usa el corpus de demostración de `data/sample_corpus.txt`. El modelo se inicializa con pesos aleatorios; no es un modelo comercial preentrenado. El corpus de ejemplo solo valida el flujo técnico y no produce inteligencia avanzada. Verifica procedencia y licencias antes de entrenar con datos externos.

Los pesos se guardan en `artifacts/ami-base.pt`, excluido de Git. Si el modelo aún no existe, la API responde con un mensaje explícito para entrenarlo.

## Pruebas
Ejecuta `pytest`. Para compilar la interfaz, entra a `apps/web`, ejecuta `npm install` y `npm run build`.

## Seguridad
La API escucha en `127.0.0.1` por defecto. No la expongas a Internet sin autenticación, HTTPS, límites de frecuencia y revisión de seguridad. No guardes claves ni datos privados en el código, corpus o checkpoints. El modelo no ejecuta comandos del sistema.

## Estado
MVP inicial: chat web, API, modelo entrenable y CI. Faltan evaluación amplia, streaming, persistencia de conversaciones y entrenamiento de alta calidad. Las respuestas del modelo experimental pueden ser incoherentes.

## Licencia
El código usa MIT. La licencia del código no concede derechos sobre datos externos; revisa por separado las licencias de los datos que incorpores.

Contribuciones bienvenidas. Mantén los cambios pequeños, añade pruebas y evita romper funciones existentes.

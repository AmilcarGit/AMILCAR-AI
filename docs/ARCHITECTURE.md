# Arquitectura

- apps/web: interfaz React que se comunica con la API HTTP.
- apps/api: validación, salud del servicio y llamadas al motor local.
- ai/model: arquitectura Transformer y generación de texto.
- ai/training: entrenamiento, optimización y checkpoints.
- ai/tokenizer.py: tokenizador BPE a nivel de bytes, propio y sin dependencias.
- ai/training/prepare_data.py: limpieza, deduplicación y división train/val con manifiesto reproducible.
- data: corpus de demostración. Los datos adicionales requieren revisión de procedencia y licencia.

Flujo: el usuario envía texto a /api/chat; FastAPI valida la solicitud; el motor tokeniza y genera texto; la API devuelve la respuesta. No se ejecutan comandos de sistema ni herramientas arbitrarias.

Decisiones del MVP: monolito modular, ejecución local, modelo pequeño experimental y sin proveedor comercial obligatorio.

Próximos pasos: evaluación independiente, checkpoints versionados con hashes, historial local optativo, streaming y autenticación antes de cualquier despliegue remoto.

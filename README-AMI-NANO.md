# AMILCAR AI — AMI Nano 2.1

AMI Nano 2.1 combina dos componentes locales y ligeros: una base de conocimientos y diálogos en español, y el experimento neuronal AMI Nano entrenado desde pesos aleatorios. No utiliza APIs comerciales, servicios web, PyTorch, NumPy ni pesos preentrenados. Es un MVP educativo, no un modelo de lenguaje grande.

## Termux / Android ARM64

```bash
pkg install nodejs python git -y
git clone -b feat/ami-nano-2-1-local-es https://github.com/AmilcarGit/AMILCAR-AI.git
cd AMILCAR-AI
npm install
npm run ami:chat-es -- "¿Qué es JavaScript?"
npm run ami:chat-interactivo
npm run ami:test
npm run ami:train
npm run ami:chat -- "hola"
```

La base de conocimientos y el chat funcionan sin entrenar el modelo neuronal y no descargan modelos. El entrenamiento neuronal es un componente separado: `ami:train` genera un checkpoint local pequeño en `artifacts/ami-nano.json`.

## Cómo funciona el chat español

- `data/dialogos-es.json`: saludos y conversación inicial.
- `data/conocimientos-es.json`: preguntas y respuestas locales sobre programación, Termux, GitHub, seguridad, IA y conceptos básicos.
- `ami-nano/chat_es.py`: normalización de texto, coincidencia local de patrones y respuesta conservadora cuando no encuentra una coincidencia fiable.
- `ami-nano/train.py` y `ami-nano/generate.py`: experimento neuronal independiente de caracteres.

Para agregar conocimiento, añade un objeto con `intent`, `patterns` y `responses` a `data/conocimientos-es.json`. Usa respuestas revisadas, ejemplos originales o contenido cuya licencia permita su distribución. No agregues mensajes privados, contraseñas ni claves.

## Pruebas y límites

Ejecuta `npm run ami:test`. Las pruebas verifican la carga de la base, preguntas representativas, respuesta de reserva para consultas desconocidas y el esquema de datos. Estas pruebas no demuestran comprensión general ni que la calidad sea equivalente a un LLM.

La base de conocimiento responde con textos revisados; no es el modelo neuronal quien redacta esas respuestas. La red de caracteres se entrena desde cero, pero por su tamaño solo aprende patrones muy simples y puede generar texto incoherente. AMI no se autoactualiza ni modifica el bot anfitrión.

## Integración con bots

El SDK de AMILCAR AI debe ejecutarse como dependencia local del bot. Para Termux, instala Python por separado con `pkg install python`. Integra la capa de chat solo después de comprobar los comandos existentes del bot; limita la longitud del mensaje, captura errores y no envíes mensajes privados a entrenamiento automáticamente.

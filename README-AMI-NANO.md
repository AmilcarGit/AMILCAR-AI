# AMILCAR AI — AMI Nano (prototipo local)

AMILCAR AI es un proyecto open source para desarrollar modelos propios desde cero. Esta rama prepara un MVP que funciona en Termux/Android ARM64 sin API web, sin APIs comerciales y sin pesos preentrenados.

## Alcance honesto de AMI Nano v0

La primera versión es un **modelo estadístico de caracteres (bigramas) entrenado desde cero**, no un Transformer ni un LLM moderno. Aprende transiciones de caracteres a partir de un corpus de texto local y genera texto corto. No usa pesos ajenos. Sirve para validar entrenamiento, guardado/carga, inferencia local e integración con bots antes de desarrollar una red neuronal pequeña.

## Requisitos

- Node.js 18+.
- Python 3 (en Termux: `pkg install python`).
- No necesita PyTorch, NumPy, C/C++, GPU, servidor HTTP ni claves API.
- El prototipo no descarga datos ni modelos durante la instalación.

## Termux

```bash
pkg update
pkg install nodejs python git -y
git clone https://github.com/AmilcarGit/AMILCAR-AI.git
cd AMILCAR-AI
git checkout feat/ami-nano-termux
npm install
npm test
npm run ami:train
npm run ami:chat -- "hola"
```

La primera ejecución de entrenamiento crea un corpus de demostración local y aprende un modelo nuevo. Sustituye `data/corpus.txt` por textos propios o con licencia compatible. No uses mensajes privados de WhatsApp sin permiso.

## Integración en BaseBot

```js
const { AmilcarAI } = require("amilcar-ai");
const ia = new AmilcarAI({ mode: "local" });
const result = await ia.chat("Hola");
console.log(result.reply);
```

La integración del SDK con AMI Nano se está preparando en esta rama; no instales esta rama en el bot estable hasta completar las pruebas de integración.

## Limitaciones

- La calidad inicial será muy baja: un modelo de caracteres no entiende el mundo ni sigue instrucciones como un LLM.
- Los checkpoints son archivos JSON pequeños y deben regenerarse después de cambiar el corpus.
- La instalación no descarga pesos de terceros.
- El entrenamiento evolutivo significa por ahora repetir entrenamiento con corpus versionado y comparar métricas; no se permite que el bot reescriba automáticamente su propio código ni sus pesos en producción.
- Los archivos de modelo grandes no deben guardarse en Git. Para este MVP, los checkpoints se generan localmente y se ignoran por Git.

## Privacidad y seguridad

Todo se ejecuta localmente. No se envían mensajes a servicios externos. No entrenes con conversaciones privadas ni credenciales. Limita los tamaños de entrada y salida en producción.

## Próximas fases

1. Validar el MVP en Termux ARM64.
2. Conectar el SDK y BaseBot sin modificar otros comandos.
3. Añadir evaluación reproducible y versionado de corpus.
4. Desarrollar una red neuronal pequeña con inicialización aleatoria propia.
5. Diseñar un ejecutor C/C++ propio solo cuando las pruebas demuestren que es necesario.

# AMILCAR AI

IA abierta y experimental de AmilcarGit, con una interfaz cyberpunk, una API local y AMI-Base, un modelo pequeño entrenable desde pesos aleatorios. Incluye un SDK oficial para integrar el motor con bots de WhatsApp hechos en Node.js.

## Integrar en un bot de WhatsApp

Instala el SDK directamente desde este repositorio:

    npm install github:AmilcarGit/AMILCAR-AI

El SDK funciona con Node.js 18 o superior y no necesita dependencias npm adicionales. Es compatible con CommonJS y puede importarse desde proyectos Node.js que usen `require`.

```js
const { AmilcarAI } = require("amilcar-ai");

const ia = new AmilcarAI({
  mode: "local",
  timeout: 120000
});

async function consultarIA(texto) {
  const resultado = await ia.chat(texto);
  return resultado.reply;
}
```

Métodos disponibles:
- `await ia.chat("Hola")`: devuelve `{ reply, model, trained }`.
- `await ia.generate("Hola")`: devuelve solo el texto de la respuesta.
- `await ia.health()`: comprueba el estado del motor.
- Opciones: `mode`, `python`, `timeout`, `maxMessageLength`, `baseURL`, `chatPath`, `healthPath`.

Los errores incluyen `code` y, cuando corresponde, `status` para poder gestionarlos sin detener el bot.

### Motor híbrido local (experimental)

El SDK incluye el modo `hybrid`, que usa respuestas revisadas y conocimiento local en español. No requiere API HTTP, PyTorch ni un checkpoint AMI-Base; sí requiere Python 3 y que se mantengan los archivos `ami-nano/` y `data/` incluidos en el paquete.

```js
const { AmilcarAI } = require("amilcar-ai");

const ia = new AmilcarAI({ mode: "hybrid", timeout: 45000 });
const resultado = await ia.chat("Explícame la fotosíntesis");
console.log(resultado.reply);
```

Instala el SDK desde la rama oficial `main` con:

    npm install github:AmilcarGit/AMILCAR-AI

El modo híbrido es experimental: busca coincidencias en diálogos y temas locales, y aplica reglas para rechazar candidatos neuronales de baja calidad. No garantiza responder sobre cualquier tema y debe indicar cuando no encuentra información fiable. El generador neuronal es opcional y necesita un archivo de modelo compatible; sin ese archivo, el motor conserva las respuestas locales y usa el fallback honesto. Cada consulta ejecuta Python, así que la latencia puede ser mayor que la de un worker persistente.

### Ejemplo para Baileys

Consulta `examples/baileys-handler.cjs`. En tu bot instala el SDK y copia el ejemplo a tu estructura; luego registra el manejador con tu socket:

```js
const { installAmilcarAI } = require("./examples/baileys-handler.cjs");

installAmilcarAI(sock, { prefix: "!ia" });
```

El ejemplo responde solo cuando el mensaje empieza con `!ia`, evita responder a mensajes enviados por el propio bot y permite limitar los chats mediante `allowedJids`. Ajusta el import si copias el archivo a otra carpeta. Baileys se instala por separado según la versión que utilice tu bot.

### Importante: requisitos reales del modo local

El SDK prioriza el modo local, sin URL ni servidor HTTP. Sin embargo, `npm install github:AmilcarGit/AMILCAR-AI` instala el código JavaScript; no instala Python/PyTorch ni incluye actualmente un checkpoint AMI entrenado. Por eso el modo local todavía requiere un entorno Python compatible, las dependencias del motor y un archivo de pesos indicado por `AMILCAR_AI_MODEL_PATH`. Ejecuta `npm run check:local` para comprobar estos requisitos. No se debe ejecutar entrenamiento pesado durante `npm install`. El modo API sigue disponible explícitamente con `mode: "api"` y `AMILCAR_AI_URL`, pero es opcional.

## Ejecutar el motor AMILCAR AI

### Requisitos
Python 3.11 o 3.12, Node.js 18 o superior y npm. PyTorch debe ser compatible con tu sistema.

### Instalación

    git clone https://github.com/AmilcarGit/AMILCAR-AI.git
    cd AMILCAR-AI
    python -m venv .venv

Activa el entorno:
- Linux/macOS/Termux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Instala dependencias Python y entrena el modelo de demostración:

    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python -m ai.training.train --steps 300

Inicia la API en una terminal:

    uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

La API queda en `http://127.0.0.1:8000`; la documentación interactiva está en `http://127.0.0.1:8000/docs`. Si el bot se ejecuta en otro contenedor o servidor, `127.0.0.1` apuntará a ese entorno, no al servidor de IA: configura una dirección de red privada protegida y accesible desde el bot.

### Interfaz web opcional

En otra terminal, dentro del repositorio:

    cd apps/web
    npm install
    npm run dev

## Usar AMILCAR AI sin web ni API

El SDK puede ejecutar el motor de Python como un proceso local persistente. Esto no necesita navegador, página web, servidor HTTP ni una API comercial. Sí necesita Python, PyTorch, las dependencias del proyecto y un checkpoint entrenado. El paquete GitHub aún no distribuye automáticamente ese checkpoint; por tanto, la instalación completamente automática descrita como objetivo todavía no está terminada.

### Preparar el motor local

Clona el repositorio y entra en él:

    git clone https://github.com/AmilcarGit/AMILCAR-AI.git
    cd AMILCAR-AI
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m ai.training.train --steps 300

Para entrenar más pasos sobre el checkpoint existente, usando el mismo vocabulario:

    python -m ai.training.train --steps 1000 --resume

El entrenamiento progresivo solo admite caracteres que ya estén en el vocabulario del checkpoint. Si el corpus necesita caracteres nuevos, crea una nueva versión del modelo sin `--resume` y evalúa antes de reemplazar la versión anterior.

### Usar el motor desde Node.js

Instala el SDK en el proyecto del bot:

    npm install github:AmilcarGit/AMILCAR-AI

Configura la ruta del checkpoint creado en el clon del repositorio. Ajusta la ruta a tu sistema operativo:

    export AMILCAR_AI_MODE=local
    export AMILCAR_AI_MODEL_PATH=/ruta/absoluta/AMILCAR-AI/artifacts/ami-base.pt
    export AMILCAR_AI_PYTHON=/ruta/absoluta/AMILCAR-AI/.venv/bin/python

Ejemplo JavaScript:

```js
const { AmilcarAI } = require("amilcar-ai");

const ia = new AmilcarAI({
  mode: process.env.AMILCAR_AI_MODE || "local",
  timeout: 120000
});

const respuesta = await ia.generate("Hola, ¿quién eres?");
console.log(respuesta);

// Cuando el bot se apague, cierra el proceso local:
await ia.close();
```

El proceso local carga el checkpoint una vez y atiende varias solicitudes. En producción, cierra el cliente al apagar el bot. Si prefieres usar la API existente, omite `mode: "local"` o establece `AMILCAR_AI_MODE=api`.

**Importante:** el paquete instalado desde GitHub contiene el código, pero no incluye automáticamente los pesos generados por tu entrenamiento porque los archivos `.pt` están excluidos de Git. `AMILCAR_AI_MODEL_PATH` apunta al checkpoint que generaste en tu clon local. No subas checkpoints grandes o de baja calidad sin evaluarlos; Git LFS o GitHub Releases son mejores opciones cuando crezcan.

## API HTTP
- `GET /health`: estado del servicio y disponibilidad del modelo.
- `POST /api/chat`: genera una respuesta a partir de `{"message":"Hola AMILCAR AI"}`.
- `GET /docs`: documentación de FastAPI.

## Modelo y limitaciones

AMI-Base es un Transformer pequeño de PyTorch que empieza con pesos aleatorios. El corpus incluido es solo una demostración técnica; no es un modelo avanzado ni un reemplazo de una IA comercial. Sus respuestas pueden ser incoherentes. Antes de entrenar con datos externos, revisa su procedencia, permisos y licencia. Los checkpoints se guardan localmente en `artifacts/ami-base.pt` y no se incluyen en Git por defecto. Para distribuir un checkpoint validado y pequeño se puede añadir expresamente; para archivos grandes, usa Git LFS o GitHub Releases. No confundas un checkpoint experimental con un modelo conversacional competente.

## Pruebas

SDK Node.js:

    npm test

Backend Python:

    pytest

Interfaz:

    cd apps/web
    npm install
    npm run build

GitHub Actions comprueba automáticamente SDK, backend e interfaz.

## Seguridad

La API escucha en `127.0.0.1` por defecto. No la expongas públicamente sin HTTPS, autenticación, límites de frecuencia y revisión de configuración. No guardes claves, mensajes privados ni datos sensibles en el repositorio o corpus. El modelo no ejecuta comandos del sistema.

## Licencia y contribuciones

El código está bajo MIT. Esta licencia no otorga derechos sobre datos externos que incorpores. Consulta `CONTRIBUTING.md` y `SECURITY.md` antes de contribuir o reportar una vulnerabilidad.

<div align="center">

<img src="https://files.catbox.moe/qbg4v1.png" alt="AMILCAR AI — logo oficial" width="220">

# AMILCAR AI

### Inteligencia artificial experimental, abierta y local

**Un motor de IA diseñado para aprender, experimentar e integrarse con tus proyectos.**

<p>
  <a href="https://github.com/AmilcarGit/AMILCAR-AI"><img src="https://img.shields.io/badge/GitHub-AMILCAR--AI-181717?style=for-the-badge&logo=github" alt="Repositorio GitHub"></a>
  <a href="https://github.com/AmilcarGit/AMILCAR-AI/blob/main/LICENSE"><img src="https://img.shields.io/badge/Licencia-MIT-blue?style=for-the-badge" alt="Licencia MIT"></a>
  <a href="https://github.com/AmilcarGit/AMILCAR-AI/actions"><img src="https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" alt="GitHub Actions"></a>
  <a href="https://github.com/AmilcarGit/AMILCAR-AI/issues"><img src="https://img.shields.io/badge/Soporte-Abrir_issue-8957e5?style=for-the-badge&logo=github" alt="Reportar un problema"></a>
</p>

<p>
  <a href="#-introducción">Introducción</a> ·
  <a href="#-características">Características</a> ·
  <a href="#-instalación-rápida">Instalación</a> ·
  <a href="#-integración-con-bots-de-whatsapp">WhatsApp</a> ·
  <a href="#-desarrollo-y-pruebas">Pruebas</a>
</p>

</div>

---

## 👋 Introducción

**AMILCAR AI** es un proyecto de inteligencia artificial de código abierto desarrollado por [AmilcarGit](https://github.com/AmilcarGit). Reúne un SDK para Node.js, un motor experimental en Python, una API HTTP opcional y herramientas para investigar generación de texto y modelos locales.

La meta es que un desarrollador pueda incorporar AMILCAR AI a su propio proyecto sin tener que construir toda la capa de integración desde cero. Uno de los casos de uso principales es la integración con bots de WhatsApp: **cada bot conserva su propia sesión, configuración y datos**, mientras que la instalación de AMILCAR AI genera el texto que el bot puede enviar.

> **Estado del proyecto:** experimental y en desarrollo activo. Las capacidades dependen del modo seleccionado y de los componentes instalados. El modelo AMI-Base entrenado desde cero es experimental; lee las limitaciones antes de usarlo en producción.

## ✨ Características

- **SDK para Node.js:** integra el motor desde proyectos JavaScript compatibles con CommonJS.
- **Modos de ejecución:** `local`, `hybrid` y `api`, según el flujo que necesites.
- **Integración con WhatsApp:** ejemplo de manejador para Baileys que permite responder solo al prefijo configurado.
- **Generación local experimental:** el modo híbrido puede usar un modelo GGUF mediante `llama.cpp`, sin enviar las consultas a una API comercial.
- **Conocimiento local:** recuperación de contenido y reglas de respuesta en español.
- **API HTTP opcional:** backend FastAPI con endpoints de salud y chat.
- **Interfaz web:** aplicación web incluida en el repositorio.
- **Pruebas automatizadas:** pruebas de Node.js, Python y compilación web mediante GitHub Actions.

## 🚀 Instalación rápida

### Requisitos básicos

- Node.js 18 o superior para el SDK.
- Git y conexión a Internet para instalar desde GitHub.
- Para funciones específicas: Python compatible, dependencias del backend o `llama.cpp`, según el modo elegido.

Instala el SDK en la carpeta de tu proyecto:

```bash
npm install github:AmilcarGit/AMILCAR-AI
```

Para importar el SDK:

```js
const { AmilcarAI } = require("amilcar-ai");

const ia = new AmilcarAI({
  mode: "hybrid",
  timeout: 120000
});

async function responder(texto) {
  const resultado = await ia.chat(texto);
  return resultado.reply;
}
```

Este ejemplo muestra cómo llamar al SDK; el resultado real depende del motor disponible y de su configuración.

## 🤖 Integración con bots de WhatsApp

AMILCAR AI es el componente que genera la respuesta; **no necesita iniciar una sesión de WhatsApp propia**. El bot anfitrión mantiene su autenticación y decide qué mensajes procesar y dónde enviar la respuesta.

```text
Mensaje entrante
      ↓
Bot de WhatsApp (su propia sesión)
      ↓
AMILCAR AI (genera el texto)
      ↓
El bot envía la respuesta
```

Hay un ejemplo de integración con Baileys en [`examples/baileys-handler.cjs`](examples/baileys-handler.cjs).

```js
const { installAmilcarAI } = require("./examples/baileys-handler.cjs");

installAmilcarAI(sock, { prefix: "!ia" });
```

El ejemplo requiere un socket de Baileys ya conectado. Está pensado como punto de partida: adapta el import a la ubicación del archivo y revisa la versión de Baileys que utiliza tu bot. No reemplaza el sistema de sesión, el almacenamiento ni la lógica de conexión del bot.

### Métodos del SDK

| Método | Descripción |
| --- | --- |
| `ia.chat("Hola")` | Devuelve un objeto con la respuesta y metadatos. |
| `ia.generate("Hola")` | Devuelve únicamente el texto generado. |
| `ia.health()` | Comprueba el estado del motor disponible. |
| `ia.close()` | Cierra los recursos del cliente cuando corresponda. |

Las llamadas a `chat`, `generate` y `health` son asíncronas. Maneja los errores con `try/catch` para evitar que una consulta fallida detenga el bot.

## 🧠 Modos de funcionamiento

### `hybrid` — conocimiento local y generación opcional

```js
const { AmilcarAI } = require("amilcar-ai");

const ia = new AmilcarAI({ mode: "hybrid", timeout: 120000 });
const resultado = await ia.chat("Explícame cómo funciona la gravedad");
console.log(resultado.reply);
```

El modo híbrido busca primero información y respuestas locales. La generación mediante un modelo GGUF es opcional; si el modelo o el servidor no están disponibles, el sistema puede recurrir al comportamiento alternativo configurado. El modo es experimental y no garantiza una respuesta correcta para todas las preguntas.

### `local` — motor Python experimental

El modo local del motor AMI-Base requiere Python, las dependencias indicadas en `requirements.txt` y un checkpoint compatible. No des por hecho que esos pesos están incluidos al instalar el SDK desde GitHub.

Consulta [Preparar el motor local](#-ejecutar-el-motor-completo) y comprueba los requisitos antes de seleccionar este modo.

### `api` — backend HTTP opcional

El modo API utiliza una instancia del backend configurada por el usuario. No es obligatorio para el SDK ni para el modo híbrido local.

## 📲 Modelo GGUF local en Android / Termux

El paquete incluye un instalador de preparación que intenta descargar un modelo GGUF durante `npm install` y reutilizarlo desde:

```text
~/.cache/amilcar-ai/models/
```

En Termux, el instalador también intenta instalar `llama.cpp` mediante `pkg`. El SDK puede intentar iniciar `llama-server` en `127.0.0.1:8081` al necesitar la generación, si el modelo y el ejecutable están disponibles.

**Ten en cuenta:**

- La descarga requiere conexión a Internet y varios cientos de megabytes libres.
- Si la descarga falla, la instalación del paquete intenta continuar; eso no significa que la generación GGUF esté lista.
- La instalación automática del motor puede variar según la plataforma. En sistemas que no sean Termux, instala `llama.cpp` de forma compatible y asegúrate de que `llama-server` esté disponible en el `PATH`.
- Para omitir la descarga durante la instalación, define `AMILCAR_AI_AUTO_INSTALL_MODEL=0`.
- Mantén el servidor enlazado a `127.0.0.1`; no lo expongas directamente a Internet.
- El modelo pequeño sirve para experimentar y puede producir respuestas incorrectas, repetitivas o poco profundas.

## 🛠️ Ejecutar el motor completo

Para trabajar en el repositorio, clónalo:

```bash
git clone https://github.com/AmilcarGit/AMILCAR-AI.git
cd AMILCAR-AI
```

### Backend Python

Se recomienda Python 3.11 o 3.12, con una versión de PyTorch compatible con el dispositivo:

```bash
python -m venv .venv
```

Activa el entorno:

- Linux, macOS o Termux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Instala dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para entrenar el modelo de demostración:

```bash
python -m ai.training.train --steps 300
```

Para iniciar la API local:

```bash
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

- API local: `http://127.0.0.1:8000`
- Documentación interactiva: `http://127.0.0.1:8000/docs`

El entrenamiento es una tarea separada y no debe confundirse con instalar el SDK en un bot.

### Interfaz web

```bash
cd apps/web
npm install
npm run dev
```

## ⚙️ Configuración

| Variable | Propósito |
| --- | --- |
| `AMILCAR_AI_MODE` | Selecciona el modo de ejecución utilizado por la integración. |
| `AMILCAR_AI_MODEL_PATH` | Ruta al checkpoint de AMI-Base para el motor Python. |
| `AMILCAR_AI_PYTHON` | Ejecutable de Python que utilizará el motor local. |
| `AMILCAR_AI_LLM_MODEL` | Ruta opcional a un modelo GGUF. |
| `AMILCAR_AI_LLM_URL` | Endpoint local compatible con el formato de chat; por defecto, `http://127.0.0.1:8081/v1/chat/completions`. |
| `AMILCAR_AI_LLM_TIMEOUT` | Tiempo máximo de espera para la generación local, en segundos. |
| `AMILCAR_AI_LLM_TOKENS` | Límite de tokens de respuesta, dentro del rango admitido por el motor. |
| `AMILCAR_AI_AUTO_INSTALL_MODEL` | Usa `0` para desactivar la descarga automática durante la instalación. |

La disponibilidad y el efecto de cada variable dependen del modo y de la versión instalada. No publiques claves ni información privada en el repositorio.

## 🧪 Desarrollo y pruebas

SDK Node.js:

```bash
npm test
```

Backend Python:

```bash
pytest
```

Compilación de la interfaz:

```bash
cd apps/web
npm install
npm run build
```

Los flujos de GitHub Actions ejecutan comprobaciones automatizadas. Consulta la pestaña [Actions](https://github.com/AmilcarGit/AMILCAR-AI/actions) para ver el resultado de cada ejecución.

## 🗺️ Hoja de ruta

- [x] SDK de Node.js para integrar AMILCAR AI.
- [x] Modo híbrido experimental con conocimiento local.
- [x] Ejemplo de integración con Baileys.
- [x] Preparación inicial para descargar un modelo GGUF local.
- [ ] Validar la instalación automática en diferentes versiones de Termux y Linux.
- [ ] Mejorar la calidad de generación y ampliar las evaluaciones en español.
- [ ] Documentar con pruebas los escenarios de instalación, recuperación de errores y uso prolongado.

Las casillas reflejan el estado documental del proyecto y deben actualizarse a medida que se validen las funciones.

## 🔐 Seguridad y privacidad

- Cada bot debe gestionar su propia sesión de WhatsApp y sus permisos.
- No compartas archivos de sesión, credenciales, tokens ni mensajes privados.
- Mantén las APIs y los servidores de inferencia en interfaces privadas; para acceso remoto, añade autenticación y transporte seguro.
- Comprueba el origen, los permisos y la licencia de los datos usados para entrenar.
- Evalúa las respuestas del modelo antes de utilizarlas en flujos importantes.

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Puedes abrir un issue para reportar errores, proponer una mejora o enviar un pull request con cambios concretos y pruebas cuando sea posible.

- [Ver issues abiertos](https://github.com/AmilcarGit/AMILCAR-AI/issues)
- [Proponer una mejora](https://github.com/AmilcarGit/AMILCAR-AI/issues/new)
- [Explorar el código](https://github.com/AmilcarGit/AMILCAR-AI/tree/main)

## 📄 Licencia

El código de este proyecto se distribuye bajo la licencia [MIT](LICENSE). La licencia del código no sustituye los términos de licencias de modelos o datos externos que decidas incorporar.

---

<div align="center">

**Creado por [AmilcarGit](https://github.com/AmilcarGit)**

*Construyendo AMILCAR AI paso a paso: abierta, experimental y enfocada en la integración local.*

<a href="https://github.com/AmilcarGit/AMILCAR-AI">⭐ Explorar el repositorio</a>

</div>

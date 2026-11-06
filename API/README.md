# API de Mistral para AMILCAR-AI

Esta carpeta contiene un cliente opcional para generar respuestas con Mistral AI mediante su API HTTP.

## Seguridad

**No publiques tu API Key en GitHub.** Este repositorio es público. La clave se lee desde la variable de entorno `MISTRAL_API_KEY`, nunca desde un archivo versionado.

## Configuración

En Termux/Linux:

```bash
export MISTRAL_API_KEY="TU_CLAVE_PRIVADA"
export MISTRAL_MODEL="mistral-small-latest"
```

En Vercel, agrega `MISTRAL_API_KEY` en Project Settings → Environment Variables y vuelve a desplegar.

## Uso desde Node.js 18+

Desde la raíz del repositorio:

```js
const { MistralClient } = require("./API/mistral");

const mistral = new MistralClient();

async function responder(texto) {
  const resultado = await mistral.chat(texto);
  return resultado.reply;
}

responder("Hola, ¿quién eres?")
  .then(console.log)
  .catch(console.error);
```

También puedes enviar historial y un mensaje de sistema:

```js
const resultado = await mistral.chat("Explícame la fotosíntesis", {
  messages: [
    { role: "system", content: "Eres AMI, asistente en español neutro, claro y directo." },
    { role: "user", content: "Explícame la fotosíntesis" }
  ],
  temperature: 0.7,
  maxTokens: 1024
});
```

## Notas

- Endpoint: `https://api.mistral.ai/v1/chat/completions`.
- Modelo predeterminado: `mistral-small-latest`; puedes cambiarlo con `MISTRAL_MODEL`.
- Necesita conexión a Internet y una clave válida con cuota disponible.
- El plan gratuito, límites y modelos disponibles dependen de tu cuenta de Mistral.
- Este cliente no entrena ni modifica los pesos del modelo. La memoria y el aprendizaje local son funciones separadas.

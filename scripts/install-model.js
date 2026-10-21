"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");
const { Readable } = require("stream");
const { pipeline } = require("stream/promises");

const MODEL_NAME = "qwen2.5-0.5b-instruct-q2_k.gguf";
const MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf?download=true";
const MIN_BYTES = 300 * 1024 * 1024;
const CACHE_DIR = path.join(os.homedir(), ".cache", "amilcar-ai", "models");
const MODEL_PATH = path.join(CACHE_DIR, MODEL_NAME);

async function download(url, destination, redirects = 0) {
  if (redirects > 8) throw new Error("Demasiadas redirecciones al descargar el modelo.");
  const response = await fetch(url, { redirect: "manual", signal: AbortSignal.timeout(120000) });
  if ([301, 302, 303, 307, 308].includes(response.status)) {
    const location = response.headers.get("location");
    if (!location) throw new Error("La descarga devolvió una redirección sin destino.");
    return download(new URL(location, url).toString(), destination, redirects + 1);
  }
  if (!response.ok || !response.body) throw new Error("Descarga HTTP " + response.status);
  const temp = destination + ".part";
  try {
    await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(temp));
    const size = fs.statSync(temp).size;
    if (size < MIN_BYTES) throw new Error("El archivo descargado es demasiado pequeño (" + size + " bytes); se rechaza por seguridad.");
    fs.renameSync(temp, destination);
    console.log("Modelo descargado: " + destination + " (" + Math.round(size / 1024 / 1024) + " MB).");
  } catch (error) {
    try { fs.rmSync(temp, { force: true }); } catch {}
    throw error;
  }
}

async function main() {
  if (process.env.AMILCAR_AI_AUTO_INSTALL_MODEL === "0") {
    console.log("[AMILCAR AI] Descarga automática desactivada por AMILCAR_AI_AUTO_INSTALL_MODEL=0.");
    return;
  }

  fs.mkdirSync(CACHE_DIR, { recursive: true });
  if (fs.existsSync(MODEL_PATH) && fs.statSync(MODEL_PATH).size >= MIN_BYTES) {
    console.log("[AMILCAR AI] Modelo local ya disponible; no se vuelve a descargar.");
  } else {
    console.log("[AMILCAR AI] Preparando modelo local Qwen2.5-0.5B (aprox. 415 MB).");
    console.log("[AMILCAR AI] El modelo se guarda en la caché del usuario y se reutiliza entre instalaciones.");
    await download(MODEL_URL, MODEL_PATH);
  }

  const isTermux = process.platform === "android" || (process.env.PREFIX || "").includes("com.termux");
  if (isTermux) {
    const check = spawnSync("sh", ["-lc", "command -v llama-server"], { encoding: "utf8" });
    if (check.status !== 0) {
      console.log("[AMILCAR AI] Instalando el motor llama.cpp en Termux...");
      const install = spawnSync("pkg", ["install", "-y", "llama-cpp"], { stdio: "inherit" });
      if (install.error || install.status !== 0) {
        console.warn("[AMILCAR AI] No se pudo instalar llama-cpp automáticamente. Ejecuta: pkg install llama-cpp");
      }
    } else {
      console.log("[AMILCAR AI] Motor llama-server disponible.");
    }
  } else {
    console.log("[AMILCAR AI] Modelo listo. Para generación automática se necesita llama.cpp (llama-server) instalado en este sistema.");
  }
}

main().catch((error) => {
  console.warn("[AMILCAR AI] No se pudo preparar el modelo durante npm install: " + error.message);
  console.warn("[AMILCAR AI] La instalación del bot continuará; puedes volver a ejecutar: node node_modules/amilcar-ai/scripts/install-model.js");
});

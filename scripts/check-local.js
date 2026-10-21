"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const root = path.resolve(__dirname, "..");
const modelPath = process.env.AMILCAR_AI_MODEL_PATH || path.join(root, "artifacts", "ami-base.pt");
const python = process.env.AMILCAR_AI_PYTHON || (process.platform === "win32" ? "python" : "python3");

console.log("AMILCAR AI — diagnóstico del modo local");
console.log("Python:", python);
console.log("Checkpoint:", modelPath);

const py = spawnSync(python, ["-c", "import torch; print('PyTorch', torch.__version__)"], { encoding: "utf8" });
if (py.error || py.status !== 0) {
  console.error("Falta Python o PyTorch. Instala Python compatible y las dependencias de AMILCAR AI.");
  if (py.stderr) console.error(py.stderr.trim());
  process.exitCode = 1;
} else {
  console.log(py.stdout.trim());
}

if (!fs.existsSync(modelPath)) {
  console.error("No se encontró un checkpoint entrenado. El paquete de GitHub todavía no incluye pesos AMI distribuidos.");
  console.error("Entrena un modelo en el repositorio AMILCAR-AI o configura AMILCAR_AI_MODEL_PATH.");
  process.exitCode = 1;
} else {
  console.log("Checkpoint encontrado.");
}

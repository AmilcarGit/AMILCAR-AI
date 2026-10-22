#!/usr/bin/env node
"use strict";

// Uso: node scripts/neural-logits.js modelo.amw "[1,2,3]"
// Imprime en JSON los logits de cada posición. Sirve para comparar el motor de JS con PyTorch.

const { NeuralModel } = require("../ai/neural-engine.js");

const [modelPath, idsJson] = process.argv.slice(2);
if (!modelPath || !idsJson) {
  console.error('Uso: node scripts/neural-logits.js modelo.amw "[1,2,3]"');
  process.exit(2);
}
const ids = JSON.parse(idsJson);
if (!Array.isArray(ids) || !ids.every(Number.isInteger)) {
  console.error("El segundo argumento debe ser un arreglo JSON de enteros.");
  process.exit(2);
}
process.stdout.write(JSON.stringify(NeuralModel.load(modelPath).forwardAll(ids)));

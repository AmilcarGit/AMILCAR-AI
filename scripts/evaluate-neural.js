#!/usr/bin/env node
"use strict";

/**
 * Evalúa un modelo .amw y escribe un informe en Markdown.
 *
 *   node scripts/evaluate-neural.js [--model models/ami-base.amw] [--out models/ami-base.report.md]
 *
 * Mide (1) cuántas respuestas de dialogos-es.json y knowledge-es.json reproduce exactamente,
 * (2) la confianza del modelo ante preguntas conocidas, reformuladas y ajenas al tema, y
 * (3) propone un umbral de confianza para el bot.
 */

const fs = require("node:fs");
const path = require("node:path");
const { NeuralAmilcarAI } = require("../neural.js");
const { normalize } = require("../ai/native-engine.js");

const ROOT = path.join(__dirname, "..");

const PARAPHRASES = [
  "oye quien eres tu",
  "hola amigo como estas",
  "dime quien te hizo",
  "cuentame que es la gravedad",
  "para que sirves tu",
  "que me puedes explicar de la fotosintesis",
  "eres una ia",
  "buenas noches como te va"
];

const UNRELATED = [
  "cual es la capital de francia",
  "quien gano el mundial de futbol",
  "cuanto cuesta un iphone",
  "recomiendame una pelicula",
  "cuentame un chiste",
  "como hago pan en casa",
  "dame una receta de ceviche",
  "cual es tu color favorito",
  "explicame la teoria de la relatividad",
  "cuantos huesos tiene el cuerpo humano",
  "traduce hello world al frances",
  "escribe un poema sobre el mar"
];

function parseArgs(argv) {
  const args = { model: path.join(ROOT, "models", "ami-base.amw"), out: path.join(ROOT, "models", "ami-base.report.md") };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--model") args.model = argv[++i];
    else if (argv[i] === "--out") args.out = argv[++i];
  }
  return args;
}

/** Pares (pregunta normalizada, respuesta esperada) de los archivos de datos del proyecto. */
function trainingPairs() {
  const dialogs = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "dialogos-es.json"), "utf8"));
  const knowledge = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "knowledge-es.json"), "utf8"));
  const pairs = new Map();
  for (const item of dialogs) {
    for (const pattern of item.patterns || []) {
      const question = normalize(pattern);
      for (const response of item.responses || []) if (question) pairs.set(question + "\u0000" + response, [question, response]);
    }
  }
  for (const entry of knowledge) {
    for (const term of entry.terms || []) {
      const question = normalize(term);
      if (question && entry.answer) pairs.set(question + "\u0000" + entry.answer, [question, entry.answer]);
    }
  }
  return [...pairs.values()];
}

const squash = (text) => String(text).replace(/\s+/g, " ").trim();
const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);
const cut = (text, n = 90) => (text.length > n ? text.slice(0, n - 1) + "…" : text).replace(/\|/g, "\\|").replace(/\n/g, " ");

async function evaluate(modelPath) {
  const ia = new NeuralAmilcarAI({ modelPath, maxNewTokens: 200 });
  const health = await ia.health();
  const pairs = trainingPairs();

  const known = [];
  for (const [question, expected] of pairs) {
    const result = await ia.chat(question);
    known.push({ question, expected: squash(expected), reply: squash(result.reply), confidence: result.confidence, intent: result.intent, exact: squash(result.reply) === squash(expected) });
  }
  const run = async (questions) => {
    const rows = [];
    for (const question of questions) {
      const result = await ia.chat(question);
      rows.push({ question, reply: squash(result.reply), confidence: result.confidence, intent: result.intent, refused: result.intent === "unknown" });
    }
    return rows;
  };
  const paraphrases = await run(PARAPHRASES);
  const unrelated = await run(UNRELATED);

  // El umbral solo importa para respuestas que el modelo NO rechazó: separa reformulaciones válidas de
  // preguntas ajenas que contestó igualmente.
  const acceptedParaphrases = paraphrases.filter((row) => !row.refused).map((row) => row.confidence);
  const acceptedUnrelated = unrelated.filter((row) => !row.refused).map((row) => row.confidence);
  const lowestParaphrase = acceptedParaphrases.length ? Math.min(...acceptedParaphrases) : 1;
  const highestUnrelated = acceptedUnrelated.length ? Math.max(...acceptedUnrelated) : 0;
  const separable = lowestParaphrase > highestUnrelated;
  const suggested = separable ? Number(((lowestParaphrase + highestUnrelated) / 2).toFixed(2)) : 0.5;
  const lowestGood = Math.min(1, ...known.filter((row) => row.exact).map((row) => row.confidence));
  return { health, known, paraphrases, unrelated, lowestGood, highestUnrelated, lowestParaphrase, separable, suggested };
}

function report(result, modelPath) {
  const exact = result.known.filter((row) => row.exact).length;
  const lines = [];
  lines.push("# Informe de evaluación de AMI-Base (motor JavaScript)", "");
  const refusedKnown = result.known.filter((row) => row.intent === "unknown").length;
  const refusedUnrelated = result.unrelated.filter((row) => row.refused).length;
  const refusedParaphrases = result.paraphrases.filter((row) => row.refused).length;
  lines.push(`- Modelo: \`${path.basename(modelPath)}\``);
  lines.push(`- Preguntas conocidas con respuesta exacta: **${exact} de ${result.known.length}** (${((100 * exact) / Math.max(1, result.known.length)).toFixed(1)} %); rechazadas por error: ${refusedKnown}`);
  lines.push(`- Confianza mínima entre las respuestas exactas: ${result.lowestGood.toFixed(3)}`);
  lines.push(`- Reformuladas: ${result.paraphrases.length - refusedParaphrases} contestadas y ${refusedParaphrases} rechazadas (confianza mínima de las contestadas: ${result.lowestParaphrase.toFixed(3)})`);
  lines.push(`- Preguntas ajenas al tema: **${refusedUnrelated} de ${result.unrelated.length} rechazadas correctamente** (de las contestadas, confianza máxima ${result.highestUnrelated.toFixed(3)})`);
  lines.push(`- **Umbral de confianza sugerido: ${result.suggested}**${result.separable ? "" : " (las contestadas no se separan por confianza; se usa el valor por defecto)"}`, "");

  const table = (title, rows) => {
    lines.push(`## ${title}`, "", "| Pregunta | Confianza | Respuesta |", "|---|---|---|");
    for (const row of rows) lines.push(`| ${cut(row.question, 50)} | ${row.confidence.toFixed(3)} | ${row.refused ? "🚫 " : ""}${cut(row.reply)} |`);
    lines.push("");
  };
  table("Preguntas reformuladas", result.paraphrases);
  table("Preguntas ajenas al tema (lo ideal: 🚫 rechazadas)", result.unrelated);
  const failures = result.known.filter((row) => !row.exact).slice(0, 15);
  if (failures.length) {
    lines.push("## Preguntas conocidas que no reprodujo exactamente (primeras 15)", "", "| Pregunta | Confianza | Respuesta del modelo |", "|---|---|---|");
    for (const row of failures) lines.push(`| ${cut(row.question, 50)} | ${row.confidence.toFixed(3)} | ${cut(row.reply)} |`);
    lines.push("");
  }
  return lines.join("\n");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const result = await evaluate(args.model);
  const text = report(result, args.model);
  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  fs.writeFileSync(args.out, text + "\n", "utf8");
  console.log(text);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message);
    process.exit(1);
  });
}

module.exports = { evaluate, report, trainingPairs, PARAPHRASES, UNRELATED };

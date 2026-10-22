"use strict";

const fs = require("node:fs");
const path = require("node:path");
const engine = require("./ai/neural-engine.js");
const { normalize } = require("./ai/native-engine.js");

const DEFAULT_MODEL_PATH = path.join(__dirname, "models", "ami-base.amw");

function fail(message, code) {
  const error = new Error(message);
  error.name = "AmilcarAIError";
  error.code = code;
  return error;
}

function numberOption(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Modelo AMI-Base ejecutado en JavaScript puro: genera cada respuesta token por token con la red
 * neuronal entrenada, sin Python, sin PyTorch y sin APIs externas (funciona en Termux).
 *
 * Los pesos se entrenan aparte (ver README) y se guardan en models/ami-base.amw.
 */
class NeuralAmilcarAI {
  constructor(options = {}) {
    this.mode = "neural";
    this.modelPath = options.modelPath || process.env.AMILCAR_AI_NEURAL_MODEL || DEFAULT_MODEL_PATH;
    this.temperature = numberOption(options.temperature, 0);
    this.topK = numberOption(options.topK, 0);
    this.topP = numberOption(options.topP, 1);
    this.maxNewTokens = numberOption(options.maxNewTokens, 160);
    this.maxPromptChars = numberOption(options.maxPromptChars, 300);
    this.maxMessageLength = Number.isSafeInteger(options.maxMessageLength) && options.maxMessageLength > 0 ? options.maxMessageLength : 2000;
    this.model = null;
  }

  /** ¿Existe el archivo del modelo? No lo carga. */
  isAvailable() {
    try {
      return fs.statSync(this.modelPath).isFile();
    } catch {
      return false;
    }
  }

  load() {
    if (this.model) return this.model;
    if (!this.isAvailable()) {
      throw fail("No se encontró el modelo neuronal en " + this.modelPath + ". Entrénalo y expórtalo a .amw (ver README).", "MODEL_NOT_FOUND");
    }
    try {
      this.model = engine.NeuralModel.load(this.modelPath);
    } catch (cause) {
      const error = fail("No se pudo cargar el modelo neuronal: " + cause.message, "MODEL_INVALID");
      error.cause = cause;
      throw error;
    }
    return this.model;
  }

  async chat(message, options = {}) {
    const text = typeof message === "string" ? message : message?.message;
    if (typeof text !== "string" || !text.trim()) {
      const error = new TypeError("El mensaje debe ser un texto no vacío.");
      error.code = "INVALID_MESSAGE";
      throw error;
    }
    if (text.length > this.maxMessageLength) throw fail("El mensaje supera el máximo de " + this.maxMessageLength + " caracteres.", "MESSAGE_TOO_LONG");

    const model = this.load();
    // Igual que en el entrenamiento: minúsculas, sin tildes ni signos.
    const question = normalize(text).slice(0, this.maxPromptChars).trim();
    if (!question) return { reply: "", model: "AMI-Base-Neural", trained: true, mode: "neural", intent: "neural", confidence: 0, tokens: 0 };

    const result = engine.generate(model, "Usuario: " + question + "\nAMILCAR AI:", {
      temperature: numberOption(options.temperature, this.temperature),
      topK: numberOption(options.topK, this.topK),
      topP: numberOption(options.topP, this.topP),
      maxNewTokens: numberOption(options.maxNewTokens, this.maxNewTokens),
      seed: options.seed
    });
    return {
      reply: result.text,
      model: "AMI-Base-Neural",
      trained: true,
      mode: "neural",
      intent: "neural",
      confidence: Number(result.confidence.toFixed(4)),
      tokens: result.tokens
    };
  }

  async generate(message, options) {
    return (await this.chat(message, options)).reply;
  }

  async health() {
    const available = this.isAvailable();
    const info = { ready: available, model: "AMI-Base-Neural", mode: "neural", local: true, externalApi: false, modelPath: this.modelPath };
    if (available) {
      const model = this.load();
      info.parameters = model.metadata.parameters || null;
      info.trainedSteps = model.metadata.total_steps || null;
    }
    return info;
  }

  async close() {
    this.model = null;
  }
}

module.exports = { NeuralAmilcarAI, DEFAULT_MODEL_PATH, NeuralEngine: engine };

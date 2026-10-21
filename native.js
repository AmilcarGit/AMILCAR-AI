"use strict";

const engine = require("./ai/native-engine.js");

class NativeAmilcarAI {
  constructor(options = {}) {
    this.mode = "native";
    this.maxMessageLength = Number.isSafeInteger(options.maxMessageLength) && options.maxMessageLength > 0
      ? options.maxMessageLength
      : 2000;
  }

  async chat(message) {
    const text = typeof message === "string" ? message : message?.message;
    if (typeof text !== "string" || !text.trim()) {
      const error = new TypeError("El mensaje debe ser un texto no vacío.");
      error.code = "INVALID_MESSAGE";
      throw error;
    }
    if (text.length > this.maxMessageLength) {
      const error = new Error("El mensaje supera el máximo de " + this.maxMessageLength + " caracteres.");
      error.name = "AmilcarAIError";
      error.code = "MESSAGE_TOO_LONG";
      throw error;
    }
    const result = engine.answer(text);
    return {
      reply: result.reply,
      model: "AMILCAR-Native-1",
      trained: false,
      mode: "native",
      intent: result.intent,
      confidence: result.confidence
    };
  }

  async generate(message) {
    return (await this.chat(message)).reply;
  }

  async health() {
    return { ready: true, model: "AMILCAR-Native-1", mode: "native", local: true, externalApi: false };
  }

  async close() {}
}

module.exports = { NativeAmilcarAI, AmilcarNativeEngine: engine };

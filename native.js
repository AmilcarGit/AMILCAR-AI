"use strict";

const engine = require("./ai/native-engine.js");

class NativeAmilcarAI {
  constructor(options = {}) {
    this.mode = "native";
    this.history = [];
    this.knowledge = engine.KNOWLEDGE.map((entry) => ({ ...entry, terms: [...entry.terms] }));
    this.maxHistory = Number.isSafeInteger(options.maxHistory) && options.maxHistory >= 0 ? Math.min(options.maxHistory, 20) : 6;
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
    const previous = [...this.history].reverse().find((item) => item.intent && item.intent !== "unknown");
    const result = engine.answer(text, { lastIntent: previous?.intent || "", knowledge: this.knowledge });
    if (this.maxHistory > 0) {
      this.history.push({ role: "user", text: text.trim() });
      this.history.push({ role: "assistant", text: result.reply, intent: result.intent });
      if (this.history.length > this.maxHistory * 2) this.history.splice(0, this.history.length - this.maxHistory * 2);
    }
    return {
      reply: result.reply,
      model: "AMILCAR-Native-1",
      trained: false,
      mode: "native",
      intent: result.intent,
      confidence: result.confidence
    };
  }

  getHistory() {
    return this.history.map((item) => ({ ...item }));
  }

  clearHistory() {
    this.history = [];
  }

  addKnowledge(entry) {
    return engine.addKnowledge(entry, this.knowledge);
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

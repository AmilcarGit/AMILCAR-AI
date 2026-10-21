"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { NativeAmilcarAI, AmilcarNativeEngine } = require("../native.js");

test("native engine answers greetings without network access", async () => {
  const ia = new NativeAmilcarAI();
  const result = await ia.chat("¡Hola!");
  assert.equal(result.mode, "native");
  assert.equal(result.model, "AMILCAR-Native-1");
  assert.match(result.reply, /hola/i);
  assert.equal((await ia.health()).externalApi, false);
});

test("native engine matches local knowledge in Spanish", async () => {
  const ia = new NativeAmilcarAI();
  const result = await ia.chat("¿Qué es la fotosíntesis?");
  assert.equal(result.intent, "fotosintesis");
  assert.match(result.reply, /plantas/i);
});

test("native engine honestly falls back on unknown questions", async () => {
  const result = await new NativeAmilcarAI().chat("Explícame un tema totalmente desconocido para este motor");
  assert.equal(result.intent, "unknown");
  assert.match(result.reply, /no tengo conocimiento suficiente|no quiero inventar|aún no sé responder/i);
});

test("native engine validates empty and oversized messages", async () => {
  const ia = new NativeAmilcarAI({ maxMessageLength: 4 });
  await assert.rejects(ia.chat(""), { code: "INVALID_MESSAGE" });
  await assert.rejects(ia.chat("demasiado largo"), { code: "MESSAGE_TOO_LONG" });
});

test("native engine normalizes Spanish accents", () => {
  assert.equal(AmilcarNativeEngine.normalize("FOTOSÍNTESIS"), "fotosintesis");
});

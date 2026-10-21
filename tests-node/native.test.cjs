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

test("native engine uses recent topic for simple follow-up questions", async () => {
  const ia = new NativeAmilcarAI();
  const first = await ia.chat("¿Qué es la gravedad?");
  const second = await ia.chat("Dame un ejemplo");
  assert.equal(first.intent, "gravedad");
  assert.equal(second.intent, "gravedad");
  assert.match(second.reply, /Ejemplo:/);
});

test("native engine stores bounded per-instance history and can clear it", async () => {
  const ia = new NativeAmilcarAI({ maxHistory: 1 });
  await ia.chat("Hola");
  await ia.chat("¿Qué es la fotosíntesis?");
  assert.equal(ia.getHistory().length, 2);
  ia.clearHistory();
  assert.equal(ia.getHistory().length, 0);
});

test("native engine accepts new local knowledge without network access", async () => {
  const ia = new NativeAmilcarAI();
  ia.addKnowledge({
    id: "mi_tema_local",
    terms: ["que es mi tema", "mi tema local"],
    answer: "Este es un conocimiento agregado en la instalación local."
  });
  const result = await ia.chat("¿Qué es mi tema?");
  assert.equal(result.intent, "mi_tema_local");
  assert.match(result.reply, /instalación local/);
});

test("native engine rejects invalid knowledge entries", () => {
  assert.throws(() => AmilcarNativeEngine.addKnowledge({ id: "invalido", terms: [], answer: "x" }), /terms/);
});

test("native engine retrieves topics from the editable Spanish knowledge file", async () => {
  const ia = new NativeAmilcarAI();
  const solar = await ia.chat("¿Qué planetas forman el sistema solar?");
  const water = await ia.chat("Explícame el ciclo del agua");
  const internet = await ia.chat("¿Qué es internet?");
  assert.equal(solar.intent, "sistema_solar");
  assert.match(solar.reply, /Mercurio.*Neptuno/);
  assert.equal(water.intent, "ciclo_del_agua");
  assert.match(water.reply, /evaporación/i);
  assert.equal(internet.intent, "internet");
  assert.match(internet.reply, /TCP\/IP/);
});

test("native engine normalizes spaces in custom knowledge identifiers", async () => {
  const ia = new NativeAmilcarAI();
  ia.addKnowledge({
    id: "tema personalizado",
    terms: ["mi tema personalizado"],
    answer: "Respuesta local personalizada."
  });
  const result = await ia.chat("mi tema personalizado");
  assert.equal(result.intent, "tema_personalizado");
  assert.equal(result.reply, "Respuesta local personalizada.");
});

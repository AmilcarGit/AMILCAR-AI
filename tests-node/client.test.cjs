"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { AmilcarAI, AmilcarAIError } = require("../index.js");

test("chat sends the message and returns normalized response", async () => {
  let request;
  const ai = new AmilcarAI({ mode: "api",
    baseURL: "http://127.0.0.1:8000/",
    fetch: async (url, options) => {
      request = { url, options };
      return new Response(JSON.stringify({ reply: "¡Hola!", model: "AMI-Base", trained: true }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
  });
  const result = await ai.chat("  Hola  ");
  assert.equal(request.url, "http://127.0.0.1:8000/api/chat");
  assert.equal(JSON.parse(request.options.body).message, "Hola");
  assert.deepEqual(result, { reply: "¡Hola!", model: "AMI-Base", trained: true });
});

test("generate returns only the reply text", async () => {
  const ai = new AmilcarAI({ mode: "api", fetch: async () => new Response(JSON.stringify({ reply: "Respuesta" }), { status: 200 }) });
  assert.equal(await ai.generate("Hola"), "Respuesta");
});

test("health checks the configured endpoint", async () => {
  let requestedUrl = "";
  const ai = new AmilcarAI({ mode: "api", fetch: async (url) => { requestedUrl = url; return new Response(JSON.stringify({ status: "ok" }), { status: 200 }); } });
  assert.deepEqual(await ai.health(), { status: "ok" });
  assert.equal(requestedUrl, "http://127.0.0.1:8000/health");
});

test("rejects empty and oversized messages before sending", async () => {
  const ai = new AmilcarAI({ mode: "api", maxMessageLength: 4, fetch: async () => { throw new Error("should not call"); } });
  await assert.rejects(ai.chat("  "), TypeError);
  await assert.rejects(ai.chat("demasiado"), (error) => error instanceof AmilcarAIError && error.code === "MESSAGE_TOO_LONG");
});

test("turns API errors into actionable errors", async () => {
  const ai = new AmilcarAI({ mode: "api", fetch: async () => new Response(JSON.stringify({ detail: "Modelo no entrenado" }), { status: 503 }) });
  await assert.rejects(ai.chat("Hola"), (error) => error instanceof AmilcarAIError && error.status === 503 && error.message === "Modelo no entrenado");
});

test("validates base URL and timeout options", () => {
  assert.throws(() => new AmilcarAI({ mode: "api", baseURL: "file:///tmp" }), TypeError);
  assert.throws(() => new AmilcarAI({ mode: "api", timeout: 0 }), TypeError);
});

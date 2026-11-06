"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { AmilcarAI } = require("../index.js");

test("modo mistral envía la consulta al endpoint y normaliza la respuesta", async () => {
  let request;
  const ai = new AmilcarAI({
    mode: "mistral",
    apiKey: "test-key",
    fetch: async (url, options) => {
      request = { url, options };
      return new Response(JSON.stringify({
        model: "mistral-small-latest",
        choices: [{ message: { content: "¡Hola!" } }],
        usage: { total_tokens: 4 }
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
  });

  const result = await ai.chat(" Hola ");
  assert.equal(request.url, "https://api.mistral.ai/v1/chat/completions");
  assert.equal(request.options.headers.Authorization, "Bearer test-key");
  assert.equal(JSON.parse(request.options.body).messages[0].content, "Hola");
  assert.equal(result.reply, "¡Hola!");
  assert.equal(result.mode, "mistral");
  assert.equal(result.provider, "mistral");
  assert.equal((await ai.health()).ready, true);
});

test("modo mistral requiere MISTRAL_API_KEY", async () => {
  const previous = process.env.MISTRAL_API_KEY;
  delete process.env.MISTRAL_API_KEY;
  try {
    const ai = new AmilcarAI({ mode: "mistral", fetch: async () => { throw new Error("No debe llamar a fetch"); } });
    await assert.rejects(ai.chat("Hola"), (error) => error.code === "MISSING_API_KEY");
  } finally {
    if (previous === undefined) delete process.env.MISTRAL_API_KEY;
    else process.env.MISTRAL_API_KEY = previous;
  }
});

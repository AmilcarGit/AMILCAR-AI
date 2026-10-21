"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { AmilcarAI } = require("../index.js");

test("cloud mode sends a chat completion request and returns the answer", async () => {
  let request;
  const ia = new AmilcarAI({
    mode: "cloud",
    apiKey: "test-key",
    model: "openrouter/free",
    fetch: async (url, init) => {
      request = { url, init };
      return {
        ok: true,
        status: 200,
        json: async () => ({
          model: "example/free-model",
          choices: [{ message: { content: "Hola, soy AMILCAR AI." } }]
        })
      };
    }
  });

  const result = await ia.chat("Hola");

  assert.equal(request.url, "https://openrouter.ai/api/v1/chat/completions");
  assert.equal(request.init.headers.Authorization, "Bearer test-key");
  assert.equal(JSON.parse(request.init.body).model, "openrouter/free");
  assert.equal(JSON.parse(request.init.body).messages[0].content, "Hola");
  assert.equal(result.reply, "Hola, soy AMILCAR AI.");
  assert.equal(result.mode, "cloud");
  assert.equal(result.model, "example/free-model");
});

test("cloud mode reports a missing API key without making a request", async () => {
  const previousOpenRouterKey = process.env.OPENROUTER_API_KEY;
  const previousAmilcarKey = process.env.AMILCAR_AI_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  delete process.env.AMILCAR_AI_API_KEY;

  try {
    const ia = new AmilcarAI({
      mode: "cloud",
      apiKey: "",
      fetch: async () => {
        throw new Error("fetch should not be called without an API key");
      }
    });
    await assert.rejects(ia.chat("Hola"), { code: "MISSING_API_KEY" });
  } finally {
    if (previousOpenRouterKey === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = previousOpenRouterKey;
    if (previousAmilcarKey === undefined) delete process.env.AMILCAR_AI_API_KEY;
    else process.env.AMILCAR_AI_API_KEY = previousAmilcarKey;
  }
});

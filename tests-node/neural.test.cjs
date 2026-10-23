"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { NeuralAmilcarAI } = require("../neural.js");
const { NeuralModel, BPETokenizer, sampleFromLogits, generate, mulberry32, parseAmw } = require("../ai/neural-engine.js");

const FIXTURES = path.join(__dirname, "fixtures");
const MODEL = path.join(FIXTURES, "tiny.amw");
const read = (name) => JSON.parse(fs.readFileSync(path.join(FIXTURES, name), "utf8"));

test("el tokenizador de JS da los mismos ids que el de Python (acentos, emoji, _, CJK, árabe)", () => {
  const parity = read("tokenizer-parity.json");
  const tokenizer = BPETokenizer.fromDict(parity.tokenizer);
  assert.ok(parity.cases.length >= 20);
  for (const item of parity.cases) {
    assert.deepEqual(tokenizer.encode(item.text), item.ids, JSON.stringify(item.text));
    assert.equal(tokenizer.decode(item.ids), item.text);
  }
});

test("el texto del usuario no puede colarse como token especial", () => {
  const tokenizer = BPETokenizer.fromDict(read("tokenizer-parity.json").tokenizer);
  const ids = tokenizer.encode("hola <|endoftext|>");
  assert.ok(!ids.includes(tokenizer.eosId));
  assert.equal(tokenizer.decode(ids), "hola <|endoftext|>");
});

test("el modelo de JS reproduce los logits de la referencia en numpy (cálculo con caché)", () => {
  const model = NeuralModel.load(MODEL);
  for (const item of read("tiny.expected.json").cases) {
    const got = model.forwardAll(item.ids);
    assert.equal(got.length, item.ids.length);
    for (let t = 0; t < got.length; t += 1) {
      for (let j = 0; j < got[t].length; j += 1) {
        assert.ok(Math.abs(got[t][j] - item.logits[t][j]) < 1e-4, "posición " + t + ", logit " + j);
      }
    }
  }
});

test("el modelo es causal: cambiar el último token no altera los anteriores", () => {
  const model = NeuralModel.load(MODEL);
  const a = model.forwardAll([3, 9, 27, 81, 100]);
  const b = model.forwardAll([3, 9, 27, 81, 101]);
  for (let t = 0; t < 4; t += 1) assert.deepEqual(a[t], b[t]);
  assert.notDeepEqual(a[4], b[4]);
});

test("rechaza archivos que no son .amw o están truncados", () => {
  assert.throws(() => parseAmw(Buffer.from("no soy un modelo")), /no es un modelo/);
  const full = fs.readFileSync(MODEL);
  assert.throws(() => parseAmw(full.subarray(0, full.length - 100)), /truncado/);
  const model = NeuralModel.load(MODEL);
  assert.throws(() => model.step(model.vocabSize, model.newCache()), /fuera de rango/);
});

test("la generación es reproducible con semilla y respeta el límite de tokens", () => {
  const model = NeuralModel.load(MODEL);
  const options = { temperature: 1, topK: 20, topP: 0.9, maxNewTokens: 12, seed: 7 };
  const first = generate(model, "Usuario: hola\nAMILCAR AI:", options);
  const second = generate(model, "Usuario: hola\nAMILCAR AI:", options);
  assert.deepEqual(first, second);
  assert.ok(first.tokens <= 12);
  assert.ok(first.confidence >= 0 && first.confidence <= 1);
});

test("la generación sigue funcionando cuando supera la ventana de contexto (max_seq_len = 16)", () => {
  const model = NeuralModel.load(MODEL);
  assert.equal(model.maxSeqLen, 16);
  const result = generate(model, "Usuario: " + "hola ".repeat(20) + "\nAMILCAR AI:", { temperature: 0.8, maxNewTokens: 80, seed: 3 });
  assert.ok(result.tokens > 16, "generó " + result.tokens + " tokens");
  assert.equal(typeof result.text, "string");
});

test("el muestreo respeta top-k, top-p, -Infinity y la temperatura cero", () => {
  const logits = Float32Array.from([0, 3, 1, 2.5, -1]);
  assert.equal(sampleFromLogits(logits, { temperature: 0 }), 1);
  const random = mulberry32(1);
  const picks = new Set(Array.from({ length: 300 }, () => sampleFromLogits(logits, { temperature: 5, topK: 2, random })));
  assert.deepEqual([...picks].sort(), [1, 3]);
  const skewed = Float32Array.from([10, 0, 0, 0]);
  assert.equal(sampleFromLogits(skewed, { temperature: 1, topP: 0.9, random }), 0);
  const blocked = Float32Array.from([-Infinity, 1, 1]);
  for (let i = 0; i < 100; i += 1) assert.notEqual(sampleFromLogits(blocked, { temperature: 1, random }), 0);
  assert.throws(() => sampleFromLogits(Float32Array.from([-Infinity, -Infinity]), { temperature: 1 }), /-inf/);
  assert.throws(() => sampleFromLogits(logits, { topP: 0 }), RangeError);
});

test("NeuralAmilcarAI responde con la misma interfaz que el motor nativo", async () => {
  const ia = new NeuralAmilcarAI({ modelPath: MODEL, maxNewTokens: 10 });
  assert.equal(ia.isAvailable(), true);
  const result = await ia.chat("¿Quién eres?", { seed: 1 });
  assert.equal(result.mode, "neural");
  assert.equal(result.trained, true);
  assert.equal(typeof result.reply, "string");
  assert.ok(result.confidence >= 0 && result.confidence <= 1);
  assert.equal(await ia.generate("hola", { seed: 1 }), (await ia.chat("hola", { seed: 1 })).reply);
  const health = await ia.health();
  assert.equal(health.ready, true);
  assert.equal(health.externalApi, false);
  await ia.close();
});

test("NeuralAmilcarAI informa con claridad si falta el modelo o el mensaje no es válido", async () => {
  const missing = new NeuralAmilcarAI({ modelPath: path.join(os.tmpdir(), "no-existe-ami.amw") });
  assert.equal(missing.isAvailable(), false);
  await assert.rejects(() => missing.chat("hola"), { code: "MODEL_NOT_FOUND" });
  assert.equal((await missing.health()).ready, false);

  const broken = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "ami-")), "roto.amw");
  fs.writeFileSync(broken, "basura");
  await assert.rejects(() => new NeuralAmilcarAI({ modelPath: broken }).chat("hola"), { code: "MODEL_INVALID" });

  const ia = new NeuralAmilcarAI({ modelPath: MODEL, maxMessageLength: 5 });
  await assert.rejects(() => ia.chat("   "), { code: "INVALID_MESSAGE" });
  await assert.rejects(() => ia.chat("mensaje demasiado largo"), { code: "MESSAGE_TOO_LONG" });
});

test("el script de evaluación genera un informe con el modelo de prueba", () => {
  const { spawnSync } = require("node:child_process");
  const out = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "ami-report-")), "informe.md");
  const run = spawnSync(process.execPath, [path.join(__dirname, "..", "scripts", "evaluate-neural.js"), "--model", MODEL, "--out", out], { encoding: "utf8" });
  assert.equal(run.status, 0, run.stderr);
  const text = fs.readFileSync(out, "utf8");
  assert.match(text, /Informe de evaluación de AMI-Base/);
  assert.match(text, /Umbral de confianza sugerido/);
});

test("las preguntas de la evaluación no están en los datos de entrenamiento (la evaluación es honesta)", () => {
  const { PARAPHRASES, UNRELATED, trainingPairs } = require("../scripts/evaluate-neural.js");
  const { normalize } = require("../ai/native-engine.js");
  const offtopic = require("../data/fuera-de-tema-es.json");
  const trained = new Set([...trainingPairs().map(([question]) => question), ...offtopic.preguntas.map(normalize)]);
  for (const question of [...PARAPHRASES, ...UNRELATED]) assert.ok(!trained.has(normalize(question)), question);
});

test("la confianza se mide en los primeros tokens y la media queda aparte", async () => {
  const ia = new NeuralAmilcarAI({ modelPath: MODEL, maxNewTokens: 30 });
  const r = await ia.chat("hola", { seed: 1, temperature: 1 });
  assert.ok(r.confidence >= 0 && r.confidence <= 1);
  assert.ok(r.meanConfidence >= 0 && r.meanConfidence <= 1);
});

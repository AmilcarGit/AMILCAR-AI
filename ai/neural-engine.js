"use strict";

/**
 * Motor neuronal de AMILCAR AI en JavaScript puro: ejecuta el Transformer entrenado (ai/model/transformer.py)
 * sin Python, sin PyTorch y sin dependencias. Pensado para Termux.
 *
 * Contiene: tokenizador BPE a nivel de bytes (equivale a ai/tokenizer.py), lector del formato .amw,
 * el modelo con caché de claves/valores y el muestreo (temperatura, top-k, top-p).
 */

const fs = require("node:fs");

const MAGIC = "AMIW";
const FORMAT_VERSION = 1;

// ---------------------------------------------------------------------------------------------
// Tokenizador BPE (mismo comportamiento que ai/tokenizer.py)
// ---------------------------------------------------------------------------------------------

// Letras (y números "raros"), dígitos, símbolos (incluido "_") y espacios. La última alternativa
// recoge cualquier carácter restante: codificar nunca pierde texto.
const PRETOKEN = /\s?[\p{L}\p{No}\p{Nl}]+|\s?\p{Nd}+|\s?[^\s\p{L}\p{N}]+|\s+|[\s\S]/gu;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8");

function mergePair(ids, a, b, newId) {
  const merged = [];
  for (let i = 0; i < ids.length; i += 1) {
    if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
      merged.push(newId);
      i += 1;
    } else {
      merged.push(ids[i]);
    }
  }
  return merged;
}

class BPETokenizer {
  constructor(merges = [], specialTokens = ["<|pad|>", "<|endoftext|>"]) {
    this.merges = merges.map(([a, b]) => [Number(a), Number(b)]);
    this.specialTokens = [...specialTokens];
    this.vocab = [];
    for (let i = 0; i < 256; i += 1) this.vocab.push(Uint8Array.of(i));
    this.ranks = new Map();
    this.merges.forEach(([a, b], rank) => {
      const newId = 256 + rank;
      if (!(a < newId && b < newId && a >= 0 && b >= 0)) throw new Error("Fusión inválida en el tokenizador: " + rank);
      const left = this.vocab[a];
      const right = this.vocab[b];
      const joined = new Uint8Array(left.length + right.length);
      joined.set(left, 0);
      joined.set(right, left.length);
      this.vocab.push(joined);
      this.ranks.set(a * 1048576 + b, rank);
    });
    const firstSpecial = 256 + this.merges.length;
    this.specialIds = new Map(this.specialTokens.map((token, i) => [token, firstSpecial + i]));
    this.specialById = new Map(this.specialTokens.map((token, i) => [firstSpecial + i, token]));
    this.wordCache = new Map();
  }

  static fromDict(data) {
    if (!data || data.type !== "byte-level-bpe" || data.format_version !== 1) {
      throw new Error("Tokenizador no compatible.");
    }
    return new BPETokenizer(data.merges, data.special_tokens);
  }

  get vocabSize() {
    return 256 + this.merges.length + this.specialTokens.length;
  }

  get padId() {
    return this.specialIds.has("<|pad|>") ? this.specialIds.get("<|pad|>") : null;
  }

  get eosId() {
    return this.specialIds.has("<|endoftext|>") ? this.specialIds.get("<|endoftext|>") : null;
  }

  encodeWord(word) {
    const cached = this.wordCache.get(word);
    if (cached) return cached;
    let ids = Array.from(textEncoder.encode(word));
    while (ids.length > 1) {
      let bestRank = -1;
      let bestA = 0;
      let bestB = 0;
      for (let i = 0; i < ids.length - 1; i += 1) {
        const rank = this.ranks.get(ids[i] * 1048576 + ids[i + 1]);
        if (rank !== undefined && (bestRank === -1 || rank < bestRank)) {
          bestRank = rank;
          bestA = ids[i];
          bestB = ids[i + 1];
        }
      }
      if (bestRank === -1) break;
      ids = mergePair(ids, bestA, bestB, 256 + bestRank);
    }
    if (this.wordCache.size < 100000) this.wordCache.set(word, ids);
    return ids;
  }

  /** Texto -> ids. El texto del usuario nunca se interpreta como token especial. */
  encode(text) {
    const ids = [];
    for (const match of String(text).matchAll(PRETOKEN)) {
      for (const id of this.encodeWord(match[0])) ids.push(id);
    }
    return ids;
  }

  decode(ids) {
    const chunks = [];
    let total = 0;
    for (const id of ids) {
      if (id >= 0 && id < this.vocab.length) {
        chunks.push(this.vocab[id]);
        total += this.vocab[id].length;
      } else if (!this.specialById.has(id)) {
        throw new Error("Id de token fuera de rango: " + id);
      }
    }
    const bytes = new Uint8Array(total);
    let position = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, position);
      position += chunk.length;
    }
    return textDecoder.decode(bytes);
  }
}

// ---------------------------------------------------------------------------------------------
// Formato .amw
// ---------------------------------------------------------------------------------------------

function parseAmw(buffer) {
  if (buffer.length < 12 || buffer.toString("latin1", 0, 4) !== MAGIC) {
    throw new Error("El archivo no es un modelo .amw de AMILCAR AI.");
  }
  const version = buffer.readUInt32LE(4);
  if (version !== FORMAT_VERSION) throw new Error("Versión de formato .amw no compatible: " + version);
  const headerSize = buffer.readUInt32LE(8);
  if (12 + headerSize > buffer.length) throw new Error("Archivo .amw truncado (encabezado).");
  const header = JSON.parse(buffer.toString("utf8", 12, 12 + headerSize));
  const dataStart = 12 + headerSize;
  const tensors = new Map();
  for (const item of header.tensors) {
    const count = item.shape.reduce((a, b) => a * b, 1);
    const start = dataStart + item.offset * 4;
    if (start + count * 4 > buffer.length) throw new Error("Archivo .amw truncado (tensor " + item.name + ").");
    // slice copia los bytes: así el Float32Array queda alineado sin importar el offset del Buffer.
    const copy = buffer.buffer.slice(buffer.byteOffset + start, buffer.byteOffset + start + count * 4);
    tensors.set(item.name, { shape: item.shape, data: new Float32Array(copy) });
  }
  return { header, tensors };
}

// ---------------------------------------------------------------------------------------------
// Modelo
// ---------------------------------------------------------------------------------------------

class NeuralModel {
  constructor(header, tensors) {
    const c = header.config;
    this.dModel = c.d_model;
    this.nHeads = c.n_heads;
    this.nLayers = c.n_layers;
    this.maxSeqLen = c.max_seq_len;
    this.vocabSize = c.vocab_size;
    this.headDim = this.dModel / this.nHeads;
    if (!Number.isInteger(this.headDim) || this.headDim % 2) throw new Error("Configuración de modelo inválida.");
    this.tokenizer = BPETokenizer.fromDict(header.tokenizer);
    if (this.tokenizer.vocabSize !== this.vocabSize) throw new Error("El tokenizador no coincide con el vocabulario del modelo.");
    this.metadata = header.metadata || {};

    const get = (name, ...shape) => {
      const tensor = tensors.get(name);
      if (!tensor) throw new Error("Falta el tensor " + name);
      if (tensor.shape.length !== shape.length || tensor.shape.some((dim, i) => dim !== shape[i])) {
        throw new Error("Forma inesperada en " + name + ": " + JSON.stringify(tensor.shape));
      }
      return tensor.data;
    };
    const D = this.dModel;
    const H = c.hidden;
    this.hidden = H;
    this.embedding = get("token_embedding.weight", this.vocabSize, D);
    this.layers = [];
    for (let i = 0; i < this.nLayers; i += 1) {
      const p = "blocks." + i + ".";
      this.layers.push({
        attnNorm: get(p + "attn_norm.weight", D),
        qkv: get(p + "attn.qkv.weight", 3 * D, D),
        proj: get(p + "attn.proj.weight", D, D),
        mlpNorm: get(p + "mlp_norm.weight", D),
        gate: get(p + "gate.weight", H, D),
        up: get(p + "up.weight", H, D),
        down: get(p + "down.weight", D, H)
      });
    }
    this.finalNorm = get("norm.weight", D);

    // Tablas de RoPE: coseno y seno por posición y par de dimensiones.
    const half = this.headDim / 2;
    this.cos = new Float32Array(this.maxSeqLen * half);
    this.sin = new Float32Array(this.maxSeqLen * half);
    for (let pos = 0; pos < this.maxSeqLen; pos += 1) {
      for (let j = 0; j < half; j += 1) {
        const angle = pos * (1 / Math.pow(10000, (2 * j) / this.headDim));
        this.cos[pos * half + j] = Math.cos(angle);
        this.sin[pos * half + j] = Math.sin(angle);
      }
    }

    // Memoria de trabajo reutilizable.
    this.x = new Float32Array(D);
    this.normed = new Float32Array(D);
    this.qkvOut = new Float32Array(3 * D);
    this.attnOut = new Float32Array(D);
    this.projOut = new Float32Array(D);
    this.gateOut = new Float32Array(H);
    this.upOut = new Float32Array(H);
    this.downOut = new Float32Array(D);
    this.scores = new Float32Array(this.maxSeqLen);
    this.logits = new Float32Array(this.vocabSize);
  }

  static fromBuffer(buffer) {
    const { header, tensors } = parseAmw(buffer);
    return new NeuralModel(header, tensors);
  }

  static load(path) {
    return NeuralModel.fromBuffer(fs.readFileSync(path));
  }

  /** Caché vacía: claves y valores por capa, con la posición actual en `length`. */
  newCache() {
    const layers = [];
    for (let i = 0; i < this.nLayers; i += 1) {
      layers.push({ k: new Float32Array(this.maxSeqLen * this.dModel), v: new Float32Array(this.maxSeqLen * this.dModel) });
    }
    return { layers, length: 0 };
  }

  /**
   * Procesa un token en la posición `cache.length` y la avanza. Devuelve los logits del siguiente
   * token (un Float32Array reutilizado: cópialo si lo necesitas después) o null si wantLogits es false.
   */
  step(tokenId, cache, wantLogits = true) {
    const pos = cache.length;
    if (pos >= this.maxSeqLen) throw new Error("La secuencia supera max_seq_len; reconstruye la caché con una ventana menor.");
    if (!(tokenId >= 0 && tokenId < this.vocabSize)) throw new Error("Id de token fuera de rango: " + tokenId);
    const D = this.dModel;
    const hd = this.headDim;
    const half = hd / 2;
    const x = this.x;
    x.set(this.embedding.subarray(tokenId * D, (tokenId + 1) * D));

    for (let l = 0; l < this.nLayers; l += 1) {
      const layer = this.layers[l];
      const kv = cache.layers[l];
      rmsNorm(x, layer.attnNorm, this.normed);
      matVec(layer.qkv, 3 * D, D, this.normed, this.qkvOut);

      // RoPE sobre q y k (cada cabeza por separado) y guardado de k y v en la caché.
      const q = this.qkvOut;
      for (let h = 0; h < this.nHeads; h += 1) {
        const base = h * hd;
        for (let j = 0; j < half; j += 1) {
          const c = this.cos[pos * half + j];
          const s = this.sin[pos * half + j];
          const q1 = q[base + j];
          const q2 = q[base + half + j];
          q[base + j] = q1 * c - q2 * s;
          q[base + half + j] = q1 * s + q2 * c;
          const k1 = q[D + base + j];
          const k2 = q[D + base + half + j];
          kv.k[pos * D + base + j] = k1 * c - k2 * s;
          kv.k[pos * D + base + half + j] = k1 * s + k2 * c;
        }
      }
      for (let i = 0; i < D; i += 1) kv.v[pos * D + i] = q[2 * D + i];

      // Atención causal: la consulta nueva ve todas las posiciones 0..pos.
      const scale = 1 / Math.sqrt(hd);
      const out = this.attnOut;
      for (let h = 0; h < this.nHeads; h += 1) {
        const base = h * hd;
        let max = -Infinity;
        for (let t = 0; t <= pos; t += 1) {
          let dot = 0;
          const kBase = t * D + base;
          for (let j = 0; j < hd; j += 1) dot += q[base + j] * kv.k[kBase + j];
          dot *= scale;
          this.scores[t] = dot;
          if (dot > max) max = dot;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t += 1) {
          const e = Math.exp(this.scores[t] - max);
          this.scores[t] = e;
          sum += e;
        }
        for (let j = 0; j < hd; j += 1) out[base + j] = 0;
        for (let t = 0; t <= pos; t += 1) {
          const weight = this.scores[t] / sum;
          const vBase = t * D + base;
          for (let j = 0; j < hd; j += 1) out[base + j] += weight * kv.v[vBase + j];
        }
      }
      matVec(layer.proj, D, D, out, this.projOut);
      for (let i = 0; i < D; i += 1) x[i] += this.projOut[i];

      // Capa de alimentación SwiGLU.
      rmsNorm(x, layer.mlpNorm, this.normed);
      matVec(layer.gate, this.hidden, D, this.normed, this.gateOut);
      matVec(layer.up, this.hidden, D, this.normed, this.upOut);
      for (let i = 0; i < this.hidden; i += 1) {
        const g = this.gateOut[i];
        this.gateOut[i] = (g / (1 + Math.exp(-g))) * this.upOut[i];
      }
      matVec(layer.down, D, this.hidden, this.gateOut, this.downOut);
      for (let i = 0; i < D; i += 1) x[i] += this.downOut[i];
    }
    cache.length = pos + 1;
    if (!wantLogits) return null;
    rmsNorm(x, this.finalNorm, this.normed);
    matVec(this.embedding, this.vocabSize, D, this.normed, this.logits);
    return this.logits;
  }

  /** Logits de cada posición de `ids` (para pruebas y diagnóstico). */
  forwardAll(ids) {
    const cache = this.newCache();
    return ids.map((id) => Array.from(this.step(id, cache, true)));
  }
}

function rmsNorm(x, weight, out) {
  let sum = 0;
  for (let i = 0; i < x.length; i += 1) sum += x[i] * x[i];
  const scale = 1 / Math.sqrt(sum / x.length + 1e-6);
  for (let i = 0; i < x.length; i += 1) out[i] = x[i] * scale * weight[i];
}

/** out[r] = suma_c W[r * cols + c] * x[c], con W en orden fila (salida, entrada) como en PyTorch. */
function matVec(W, rows, cols, x, out) {
  for (let r = 0; r < rows; r += 1) {
    let sum = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c += 1) sum += W[base + c] * x[c];
    out[r] = sum;
  }
}

// ---------------------------------------------------------------------------------------------
// Muestreo y generación
// ---------------------------------------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Equivale a ai/model/sampling.py. Los logits en -Infinity nunca se eligen. */
function sampleFromLogits(logits, { temperature = 0.8, topK = 0, topP = 1, random = Math.random } = {}) {
  if (!logits.length) throw new Error("Se necesitan logits para muestrear.");
  if (topK < 0) throw new RangeError("topK no puede ser negativo.");
  if (!(topP > 0 && topP <= 1)) throw new RangeError("topP debe estar en el intervalo (0, 1].");
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i += 1) if (logits[i] > logits[best]) best = i;
    return best;
  }
  let order = Array.from({ length: logits.length }, (_, i) => i).sort((a, b) => logits[b] - logits[a]);
  if (topK) order = order.slice(0, topK);
  const peak = logits[order[0]] / temperature;
  if (peak === -Infinity) throw new Error("Todos los logits son -inf; no hay token posible.");
  let weights = order.map((i) => Math.exp(logits[i] / temperature - peak));
  if (topP < 1) {
    const total = weights.reduce((a, b) => a + b, 0);
    let cumulative = 0;
    let cut = order.length;
    for (let i = 0; i < weights.length; i += 1) {
      cumulative += weights[i] / total;
      if (cumulative >= topP) {
        cut = i + 1;
        break;
      }
    }
    order = order.slice(0, cut);
    weights = weights.slice(0, cut);
  }
  const total = weights.reduce((a, b) => a + b, 0);
  let draw = random() * total;
  for (let i = 0; i < order.length; i += 1) {
    draw -= weights[i];
    if (draw <= 0) return order[i];
  }
  return order[order.length - 1];
}

/** log(softmax(logits))[index], con la estabilidad numérica habitual. */
function logProbability(logits, index) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) sum += Math.exp(logits[i] - max);
  return logits[index] - max - Math.log(sum);
}

const STOP_PATTERN = /\n\s*\n|\nUsuario:/;

/**
 * Genera la respuesta a `prompt` (el texto ya con formato "Usuario: ...\nAMILCAR AI:").
 * Devuelve { text, confidence, tokens }. `confidence` es la probabilidad media geométrica de los
 * tokens elegidos según el propio modelo (0 a 1): baja cuando el modelo "improvisa".
 */
function generate(model, prompt, options = {}) {
  const tokenizer = model.tokenizer;
  const limit = model.maxSeqLen;
  const maxNewTokens = Math.max(1, Math.min(Number.isFinite(options.maxNewTokens) ? options.maxNewTokens : 160, 512));
  const random = options.seed === undefined || options.seed === null ? Math.random : mulberry32(options.seed);
  const sampling = {
    temperature: options.temperature ?? 0,
    topK: options.topK ?? 0,
    topP: options.topP ?? 1,
    random
  };

  let context = tokenizer.encode(prompt).slice(-(limit - 1));
  if (!context.length) context = [tokenizer.eosId !== null ? tokenizer.eosId : 0];
  const allIds = [...context];
  const generated = [];
  const cache = model.newCache();
  const prefill = (ids) => {
    cache.length = 0;
    let logits = null;
    for (let i = 0; i < ids.length; i += 1) logits = model.step(ids[i], cache, i === ids.length - 1);
    return logits;
  };

  let logits = prefill(context);
  let logSum = 0;
  let text = "";
  for (let n = 0; n < maxNewTokens; n += 1) {
    const scores = Float32Array.from(logits);
    if (tokenizer.padId !== null) scores[tokenizer.padId] = -Infinity;
    const token = sampleFromLogits(scores, sampling);
    if (token === tokenizer.eosId) break;
    logSum += logProbability(scores, token);
    generated.push(token);
    allIds.push(token);
    text = tokenizer.decode(generated);
    if (STOP_PATTERN.test(text)) break;
    if (cache.length >= limit) {
      // Caché llena: se reconstruye con la mitad más reciente del contexto.
      logits = prefill(allIds.slice(-Math.max(1, Math.floor(limit / 2))));
    } else {
      logits = model.step(token, cache, true);
    }
  }
  const answer = text.split(STOP_PATTERN)[0].trim();
  const confidence = generated.length ? Math.exp(logSum / generated.length) : 0;
  return { text: answer, confidence, tokens: generated.length };
}

module.exports = {
  BPETokenizer,
  NeuralModel,
  parseAmw,
  sampleFromLogits,
  logProbability,
  generate,
  mulberry32,
  FORMAT_VERSION
};

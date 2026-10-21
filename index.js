"use strict";

const path = require("path");
const { spawn, execFile } = require("child_process");
const { promisify } = require("util");
const readline = require("readline");
const execFileAsync = promisify(execFile);

class AmilcarAIError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "AmilcarAIError";
    this.code = options.code || "AMILCAR_AI_ERROR";
    this.status = options.status ?? null;
    this.details = options.details ?? null;
    if (options.cause) this.cause = options.cause;
  }
}

class AmilcarAI {
  constructor(options = {}) {
    this.mode = options.mode || process.env.AMILCAR_AI_MODE || "local";
    if (!["api", "local", "hybrid", "cloud"].includes(this.mode)) throw new TypeError("mode debe ser 'api', 'local', 'hybrid' o 'cloud'.");
    this.timeout = positiveInteger(options.timeout ?? 30000, "timeout");
    this.maxMessageLength = positiveInteger(options.maxMessageLength ?? 2000, "maxMessageLength");
    this.python = options.python || process.env.AMILCAR_AI_PYTHON || "python";
    this._worker = null;
    this._workerReady = null;
    this._pending = new Map();
    this._nextId = 1;

    if (this.mode === "cloud") {
      const baseURL = options.baseURL || process.env.AMILCAR_AI_URL || "https://openrouter.ai/api/v1";
      let parsed;
      try { parsed = new URL(baseURL); }
      catch { throw new TypeError("baseURL debe ser una URL HTTP o HTTPS válida."); }
      if (!["http:", "https:"].includes(parsed.protocol)) throw new TypeError("baseURL solo admite HTTP o HTTPS.");
      this.baseURL = parsed.toString().replace(/\\/$/, "");
      this.chatPath = normalizePath(options.chatPath || "/chat/completions");
      this.fetch = options.fetch || globalThis.fetch;
      this.apiKey = options.apiKey || process.env.OPENROUTER_API_KEY || process.env.AMILCAR_AI_API_KEY || "";
      this.model = options.model || process.env.AMILCAR_AI_MODEL || "openrouter/free";
      if (typeof this.fetch !== "function") throw new Error("AMILCAR AI necesita Node.js 18 o superior, con fetch disponible.");
    }

    if (this.mode === "api") {
      const baseURL = options.baseURL || process.env.AMILCAR_AI_URL || "http://127.0.0.1:8000";
      let parsed;
      try { parsed = new URL(baseURL); }
      catch { throw new TypeError("baseURL debe ser una URL HTTP o HTTPS válida."); }
      if (!["http:", "https:"].includes(parsed.protocol)) throw new TypeError("baseURL solo admite HTTP o HTTPS.");
      this.baseURL = parsed.toString().replace(/\/$/, "");
      this.chatPath = normalizePath(options.chatPath || "/api/chat");
      this.healthPath = normalizePath(options.healthPath || "/health");
      this.fetch = options.fetch || globalThis.fetch;
      if (typeof this.fetch !== "function") throw new Error("AMILCAR AI necesita Node.js 18 o superior, con fetch disponible.");
    }
  }

  async health(options = {}) {
    if (this.mode === "cloud") return { ready: Boolean(this.apiKey), model: this.model, mode: "cloud", local: false };
    if (this.mode === "hybrid") return { ready: true, model: "AMI-Nano-Hybrid", mode: "hybrid", local: true };
    if (this.mode === "local") {
      await this._ensureLocal();
      return this._sendLocal({ action: "health" }, options.signal);
    }
    return this._request(this.healthPath, { method: "GET" }, options.signal);
  }

  async chat(message, options = {}) {
    const text = typeof message === "string" ? message : message?.message;
    if (typeof text !== "string" || !text.trim()) throw new TypeError("El mensaje debe ser un texto no vacío.");
    if (text.length > this.maxMessageLength) {
      throw new AmilcarAIError("El mensaje supera el máximo de " + this.maxMessageLength + " caracteres.", { code: "MESSAGE_TOO_LONG" });
    }

    if (this.mode === "hybrid") return this._chatHybrid(text.trim(), options.signal);

    if (this.mode === "cloud") {
      if (!this.apiKey) throw new AmilcarAIError("Falta OPENROUTER_API_KEY. Crea una clave en OpenRouter y configúrala como variable de entorno.", { code: "MISSING_API_KEY" });
      const result = await this._request(this.chatPath, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + this.apiKey },
        body: JSON.stringify({ model: this.model, messages: [{ role: "user", content: text.trim() }] })
      }, options.signal);
      const content = result?.choices?.[0]?.message?.content;
      const reply = typeof content === "string" ? content : Array.isArray(content) ? content.filter((part) => part && typeof part.text === "string").map((part) => part.text).join("\\n") : "";
      if (!reply.trim()) throw new AmilcarAIError("El proveedor no devolvió texto en choices[0].message.content.", { code: "INVALID_RESPONSE", details: result });
      return { reply: reply.trim(), model: typeof result.model === "string" ? result.model : this.model, trained: false, mode: "cloud" };
    }

    if (this.mode === "local") {
      await this._ensureLocal();
      const result = await this._sendLocal({ action: "chat", message: text.trim() }, options.signal);
      if (result.error) throw new AmilcarAIError(result.error, { code: result.code || "LOCAL_INFERENCE_ERROR" });
      return { reply: result.reply, model: result.model || "AMI-Base", trained: Boolean(result.trained) };
    }

    const result = await this._request(this.chatPath, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ message: text.trim() })
    }, options.signal);
    if (typeof result.reply !== "string") throw new AmilcarAIError("La API respondió sin un campo 'reply' válido.", { code: "INVALID_RESPONSE" });
    return { reply: result.reply, model: typeof result.model === "string" ? result.model : "AMILCAR AI", trained: Boolean(result.trained) };
  }

  async _chatHybrid(message, signal) {
    if (signal?.aborted) throw new AmilcarAIError("La solicitud fue cancelada.", { code: "ABORTED" });
    const scriptPath = path.join(__dirname, "ami-nano", "hybrid.py");
    const timeout = this.timeout;
    try {
      const { stdout } = await execFileAsync(this.python, [scriptPath, message], {
        cwd: __dirname, env: process.env, timeout, maxBuffer: 1024 * 1024, encoding: "utf8", signal
      });
      const reply = stdout.trim();
      if (!reply) throw new AmilcarAIError("El motor híbrido no devolvió texto.", { code: "EMPTY_RESPONSE" });
      return { reply, model: process.env.AMILCAR_AI_LLM_MODEL ? "AMI-Nano-Hybrid + GGUF local" : "AMI-Nano-Hybrid", trained: false, mode: "hybrid" };
    } catch (cause) {
      if (cause instanceof AmilcarAIError) throw cause;
      if (signal?.aborted || cause?.name === "AbortError") throw new AmilcarAIError("La solicitud fue cancelada.", { code: "ABORTED", cause });
      if (cause?.code === "ETIMEDOUT" || cause?.killed) throw new AmilcarAIError("El motor híbrido superó el tiempo límite de " + timeout + " ms.", { code: "TIMEOUT", cause });
      if (cause?.code === "ENOENT") throw new AmilcarAIError("No se encontró Python. Instala Python 3 y configura AMILCAR_AI_PYTHON si es necesario.", { code: "PYTHON_START_FAILED", cause });
      throw new AmilcarAIError("No se pudo ejecutar ami-nano/hybrid.py. Comprueba que el paquete incluya el motor híbrido y sus archivos de conocimiento.", { code: "HYBRID_START_FAILED", cause });
    }
  }

  async generate(message, options = {}) {
    return (await this.chat(message, options)).reply;
  }

  async close() {
    const child = this._worker;
    this._worker = null;
    this._workerReady = null;
    for (const [id, item] of this._pending) {
      clearTimeout(item.timer);
      item.reject(new AmilcarAIError("El motor local se cerró.", { code: "WORKER_CLOSED" }));
      this._pending.delete(id);
    }
    if (child && !child.killed) child.kill();
  }

  async _ensureLocal() {
    if (this._workerReady) return this._workerReady;
    this._workerReady = new Promise((resolve, reject) => {
      const workerPath = path.join(__dirname, "ai", "local_worker.py");
      const child = spawn(this.python, ["-u", workerPath], {
        cwd: __dirname,
        stdio: ["pipe", "pipe", "pipe"],
        env: process.env
      });
      this._worker = child;
      const lines = readline.createInterface({ input: child.stdout });
      let initialized = false;

      lines.on("line", (line) => {
        let data;
        try { data = JSON.parse(line); }
        catch { return; }
        if (!initialized) {
          initialized = true;
          if (!data.ready) {
            const err = new AmilcarAIError(data.error || "No se pudo iniciar el motor local.", { code: "LOCAL_START_FAILED" });
            reject(err);
            child.kill();
            return;
          }
          resolve(data);
          return;
        }
        const pending = this._pending.get(data.id);
        if (!pending) return;
        clearTimeout(pending.timer);
        this._pending.delete(data.id);
        if (data.error) pending.reject(new AmilcarAIError(data.error, { code: data.code || "LOCAL_INFERENCE_ERROR" }));
        else pending.resolve(data);
      });

      child.stderr.on("data", (chunk) => {
        if (process.env.AMILCAR_AI_DEBUG === "1") process.stderr.write("[AMILCAR AI] " + chunk.toString());
      });

      child.on("error", (cause) => {
        const err = new AmilcarAIError("No se pudo iniciar Python. Instala Python y las dependencias de AMILCAR AI.", { code: "PYTHON_START_FAILED", cause });
        if (!initialized) reject(err);
        this._rejectPending(err);
        this._worker = null;
        this._workerReady = null;
      });

      child.on("exit", (code) => {
        const err = new AmilcarAIError("El motor local terminó (código " + code + ").", { code: "WORKER_EXITED" });
        if (!initialized) reject(err);
        this._rejectPending(err);
        this._worker = null;
        this._workerReady = null;
      });
    });
    return this._workerReady;
  }

  _sendLocal(payload, signal) {
    if (signal?.aborted) return Promise.reject(new AmilcarAIError("La solicitud fue cancelada.", { code: "ABORTED" }));
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this._pending.delete(id);
        reject(new AmilcarAIError("La solicitud local superó " + this.timeout + " ms.", { code: "TIMEOUT" }));
      }, this.timeout);
      const abort = () => {
        clearTimeout(timer);
        this._pending.delete(id);
        reject(new AmilcarAIError("La solicitud fue cancelada.", { code: "ABORTED" }));
      };
      if (signal) signal.addEventListener("abort", abort, { once: true });
      this._pending.set(id, {
        resolve: (value) => { signal?.removeEventListener("abort", abort); resolve(value); },
        reject: (error) => { signal?.removeEventListener("abort", abort); reject(error); },
        timer
      });
      try {
        this._worker.stdin.write(JSON.stringify({ ...payload, id }) + "\n");
      } catch (cause) {
        clearTimeout(timer);
        this._pending.delete(id);
        reject(new AmilcarAIError("No se pudo enviar la solicitud al motor local.", { code: "LOCAL_WRITE_FAILED", cause }));
      }
    });
  }

  _rejectPending(error) {
    for (const [id, item] of this._pending) {
      clearTimeout(item.timer);
      item.reject(error);
      this._pending.delete(id);
    }
  }

  async _request(pathname, init, externalSignal) {
    const controller = new AbortController();
    let timedOut = false;
    const timeoutId = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeout);
    const abortFromCaller = () => controller.abort(externalSignal.reason);
    if (externalSignal) {
      if (externalSignal.aborted) abortFromCaller();
      else externalSignal.addEventListener("abort", abortFromCaller, { once: true });
    }
    try {
      const response = await this.fetch(this.baseURL + pathname, {
        ...init, signal: controller.signal,
        headers: { Accept: "application/json", ...(init.headers || {}) }
      });
      let data;
      try { data = await response.json(); }
      catch (cause) { throw new AmilcarAIError("La API devolvió una respuesta que no es JSON válido.", { code: "INVALID_JSON", status: response.status, cause }); }
      if (!response.ok) {
        const detail = typeof data?.detail === "string" ? data.detail : "HTTP " + response.status;
        throw new AmilcarAIError(detail, { code: "API_ERROR", status: response.status, details: data });
      }
      return data;
    } catch (cause) {
      if (cause instanceof AmilcarAIError) throw cause;
      if (timedOut) throw new AmilcarAIError("La solicitud superó el tiempo límite de " + this.timeout + " ms.", { code: "TIMEOUT", cause });
      if (externalSignal?.aborted) throw new AmilcarAIError("La solicitud fue cancelada.", { code: "ABORTED", cause });
      throw new AmilcarAIError("No se pudo conectar con AMILCAR AI. Comprueba que la API esté encendida y que baseURL sea correcta.", { code: "CONNECTION_ERROR", cause });
    } finally {
      clearTimeout(timeoutId);
      externalSignal?.removeEventListener("abort", abortFromCaller);
    }
  }
}

function normalizePath(path) {
  if (typeof path !== "string" || !path.startsWith("/") || path.startsWith("//")) throw new TypeError("Las rutas de la API deben comenzar con una sola barra (/).");
  return path;
}
function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) throw new TypeError(name + " debe ser un entero positivo.");
  return value;
}

exports.AmilcarAI = AmilcarAI;
exports.AmilcarAIError = AmilcarAIError;

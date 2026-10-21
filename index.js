"use strict";

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
    const baseURL = options.baseURL || process.env.AMILCAR_AI_URL || "http://127.0.0.1:8000";
    let parsed;
    try { parsed = new URL(baseURL); }
    catch { throw new TypeError("baseURL debe ser una URL HTTP o HTTPS válida."); }
    if (!["http:", "https:"].includes(parsed.protocol)) throw new TypeError("baseURL solo admite HTTP o HTTPS.");
    this.baseURL = parsed.toString().replace(/\/$/, "");
    this.chatPath = normalizePath(options.chatPath || "/api/chat");
    this.healthPath = normalizePath(options.healthPath || "/health");
    this.timeout = positiveInteger(options.timeout ?? 30000, "timeout");
    this.maxMessageLength = positiveInteger(options.maxMessageLength ?? 2000, "maxMessageLength");
    this.fetch = options.fetch || globalThis.fetch;
    if (typeof this.fetch !== "function") throw new Error("AMILCAR AI necesita Node.js 18 o superior, con fetch disponible.");
  }

  async health(options = {}) {
    return this._request(this.healthPath, { method: "GET" }, options.signal);
  }

  async chat(message, options = {}) {
    const text = typeof message === "string" ? message : message?.message;
    if (typeof text !== "string" || !text.trim()) throw new TypeError("El mensaje debe ser un texto no vacío.");
    if (text.length > this.maxMessageLength) {
      throw new AmilcarAIError("El mensaje supera el máximo de " + this.maxMessageLength + " caracteres.", { code: "MESSAGE_TOO_LONG" });
    }
    const result = await this._request(this.chatPath, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ message: text.trim() })
    }, options.signal);
    if (typeof result.reply !== "string") throw new AmilcarAIError("La API respondió sin un campo 'reply' válido.", { code: "INVALID_RESPONSE" });
    return {
      reply: result.reply,
      model: typeof result.model === "string" ? result.model : "AMILCAR AI",
      trained: Boolean(result.trained)
    };
  }

  async generate(message, options = {}) {
    return (await this.chat(message, options)).reply;
  }

  async _request(path, init, externalSignal) {
    const controller = new AbortController();
    let timedOut = false;
    const timeoutId = setTimeout(() => { timedOut = true; controller.abort(); }, this.timeout);
    const abortFromCaller = () => controller.abort(externalSignal.reason);
    if (externalSignal) {
      if (externalSignal.aborted) abortFromCaller();
      else externalSignal.addEventListener("abort", abortFromCaller, { once: true });
    }
    try {
      const response = await this.fetch(this.baseURL + path, {
        ...init,
        signal: controller.signal,
        headers: { Accept: "application/json", ...(init.headers || {}) }
      });
      let data;
      try { data = await response.json(); }
      catch (cause) {
        throw new AmilcarAIError("La API devolvió una respuesta que no es JSON válido.", { code: "INVALID_JSON", status: response.status, cause });
      }
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

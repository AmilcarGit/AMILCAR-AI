"use strict";

/**
 * Cliente oficial-compatible para Mistral Chat Completions.
 * La clave se lee únicamente desde MISTRAL_API_KEY.
 * Nunca escribas la clave dentro del repositorio.
 */

class MistralAPIError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "MistralAPIError";
    this.status = options.status ?? null;
    this.code = options.code || "MISTRAL_API_ERROR";
    this.details = options.details ?? null;
  }
}

class MistralClient {
  constructor(options = {}) {
    this.apiKey = options.apiKey || process.env.MISTRAL_API_KEY || "";
    this.model = options.model || process.env.MISTRAL_MODEL || "mistral-small-latest";
    this.baseURL = (options.baseURL || "https://api.mistral.ai/v1").replace(/\/$/, "");
    this.timeout = Number.isInteger(options.timeout) && options.timeout > 0 ? options.timeout : 60000;
    this.fetch = options.fetch || globalThis.fetch;

    if (typeof this.fetch !== "function") {
      throw new Error("MistralClient requiere Node.js 18 o superior (fetch).");
    }
  }

  async chat(message, options = {}) {
    if (!this.apiKey) {
      throw new MistralAPIError(
        "Falta MISTRAL_API_KEY. Configura la clave como variable de entorno; no la guardes en el código.",
        { code: "MISSING_API_KEY" }
      );
    }

    const text = typeof message === "string" ? message : message?.message;
    if (typeof text !== "string" || !text.trim()) {
      throw new TypeError("El mensaje debe ser un texto no vacío.");
    }

    const messages = Array.isArray(options.messages) && options.messages.length
      ? options.messages
      : [{ role: "user", content: text.trim() }];

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);
    const abort = () => controller.abort(options.signal?.reason);
    if (options.signal) {
      if (options.signal.aborted) abort();
      else options.signal.addEventListener("abort", abort, { once: true });
    }

    try {
      const response = await this.fetch(this.baseURL + "/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          Authorization: "Bearer " + this.apiKey
        },
        body: JSON.stringify({
          model: options.model || this.model,
          messages,
          temperature: options.temperature ?? 0.7,
          max_tokens: options.maxTokens ?? 1024
        }),
        signal: controller.signal
      });

      const data = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = data?.message || data?.error?.message || ("HTTP " + response.status);
        throw new MistralAPIError("Mistral rechazó la solicitud: " + detail, {
          status: response.status,
          code: "MISTRAL_HTTP_ERROR",
          details: data
        });
      }

      const reply = data?.choices?.[0]?.message?.content;
      if (typeof reply !== "string" || !reply.trim()) {
        throw new MistralAPIError("Mistral no devolvió texto en la respuesta.", {
          code: "INVALID_RESPONSE",
          details: data
        });
      }

      return {
        reply: reply.trim(),
        model: data.model || options.model || this.model,
        provider: "mistral",
        usage: data.usage || null
      };
    } catch (error) {
      if (error instanceof MistralAPIError) throw error;
      if (controller.signal.aborted) {
        throw new MistralAPIError("La solicitud a Mistral fue cancelada o superó el tiempo límite.", {
          code: "MISTRAL_TIMEOUT",
          cause: error
        });
      }
      throw new MistralAPIError("No se pudo conectar con Mistral: " + error.message, {
        code: "MISTRAL_NETWORK_ERROR",
        cause: error
      });
    } finally {
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", abort);
    }
  }
}

module.exports = { MistralClient, MistralAPIError };

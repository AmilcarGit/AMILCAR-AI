"use strict";

const { AmilcarAI } = require("amilcar-ai");

const ai = new AmilcarAI({
  baseURL: process.env.AMILCAR_AI_URL || "http://127.0.0.1:8000",
  timeout: 45000
});

function getMessageText(message) {
  const content = message?.message;
  return (
    content?.conversation ||
    content?.extendedTextMessage?.text ||
    content?.imageMessage?.caption ||
    content?.videoMessage?.caption ||
    ""
  ).trim();
}

function installAmilcarAI(sock, options = {}) {
  const prefix = options.prefix || "!ia";
  const allowedJids = options.allowedJids ? new Set(options.allowedJids) : null;

  sock.ev.on("messages.upsert", async ({ messages, type }) => {
    if (type !== "notify") return;
    for (const message of messages || []) {
      const jid = message?.key?.remoteJid;
      if (!jid || message.key.fromMe || jid === "status@broadcast") continue;
      if (allowedJids && !allowedJids.has(jid)) continue;

      const text = getMessageText(message);
      if (!text.toLowerCase().startsWith(prefix.toLowerCase())) continue;
      const prompt = text.slice(prefix.length).trim();
      if (!prompt) {
        await sock.sendMessage(jid, { text: "Escribe tu pregunta después de " + prefix + "." }, { quoted: message });
        continue;
      }

      try {
        const result = await ai.chat(prompt);
        await sock.sendMessage(jid, { text: result.reply }, { quoted: message });
      } catch (error) {
        const reply = error?.code === "API_ERROR" && error.status === 503
          ? "AMILCAR AI todavía no está entrenada. Revisa el servicio del modelo."
          : "No pude conectar con AMILCAR AI. Inténtalo de nuevo en unos momentos.";
        await sock.sendMessage(jid, { text: reply }, { quoted: message });
      }
    }
  });
}

module.exports = { installAmilcarAI };

"use strict";

const KNOWLEDGE = require("../data/knowledge-es.json");

const GREETINGS = [
  "¡Hola! Soy AMILCAR AI. ¿Qué te gustaría consultar?",
  "¡Buenas! Estoy aquí para ayudarte. ¿Qué necesitas?",
  "¡Hola! Cuéntame qué tienes en mente."
];

const FALLBACKS = [
  "Todavía no tengo conocimiento suficiente para responder eso con fiabilidad. Puedes reformular la pregunta o añadir el tema a la base de conocimientos local de AMILCAR AI.",
  "No quiero inventar una respuesta. Mi motor nativo funciona sin APIs, pero por ahora solo cubre los temas y patrones incluidos en el proyecto. Podemos ampliar esa base con respuestas revisadas.",
  "Aún no sé responder bien a esa pregunta. Si me das más contexto o una pregunta más concreta, intentaré encontrar una coincidencia en mis conocimientos locales."
];

function normalize(value) {
  return String(value)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9ñ\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function scoreEntry(query, entry) {
  let best = 0;
  for (const phrase of entry.terms) {
    const candidate = normalize(phrase);
    if (!candidate) continue;
    if (query === candidate) best = Math.max(best, 1);
    else if (query.includes(candidate)) best = Math.max(best, Math.min(0.96, candidate.split(" ").length / Math.max(1, query.split(" ").length) + 0.35));
    else {
      const words = candidate.split(" ").filter((word) => word.length > 2);
      if (!words.length) continue;
      const queryWords = new Set(query.split(" "));
      const overlap = words.filter((word) => queryWords.has(word)).length;
      best = Math.max(best, overlap / words.length * 0.72);
    }
  }
  return best;
}

function calculateLocally(message) {
  if (typeof message !== "string" || message.length > 120) return null;
  const normalized = message.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().trim();
  const cleaned = normalized.replace(/^[¿?¡!.,;:]+|[¿?¡!.,;:]+$/g, "").trim();
  const match = cleaned.match(/^(?:(?:cuanto es|calcula|calcular|resuelve|resuelve esto|resultado de)\s+)?([0-9\s.+*/^()%-]+)$/);
  if (!match) return null;
  const expression = match[1].trim();
  if (!/[+*/^()-]/.test(expression) || !/^[0-9\s.+*/^()-]+$/.test(expression)) return null;
  const tokens = expression.match(/\d+(?:\.\d+)?|[()+*/^-]/g) || [];
  if (!tokens.length || tokens.join("") !== expression.replace(/\s+/g, "")) return null;
  let position = 0;
  function parsePrimary() {
    const token = tokens[position++];
    if (token === "(") {
      const value = parseExpression();
      if (tokens[position++] !== ")") throw new Error("Paréntesis incompletos");
      return value;
    }
    if (token === "+") return parsePrimary();
    if (token === "-") return -parsePrimary();
    if (token === undefined || !/^\d+(?:\.\d+)?$/.test(token)) throw new Error("Número no válido");
    return Number(token);
  }
  function parsePower() {
    let value = parsePrimary();
    if (tokens[position] === "^") {
      position++;
      value = value ** parsePower();
    }
    return value;
  }
  function parseTerm() {
    let value = parsePower();
    while (tokens[position] === "*" || tokens[position] === "/") {
      const operator = tokens[position++];
      const right = parsePower();
      if (operator === "/" && right === 0) throw new Error("No se puede dividir entre cero");
      value = operator === "*" ? value * right : value / right;
    }
    return value;
  }
  function parseExpression() {
    let value = parseTerm();
    while (tokens[position] === "+" || tokens[position] === "-") {
      const operator = tokens[position++];
      const right = parseTerm();
      value = operator === "+" ? value + right : value - right;
    }
    return value;
  }
  try {
    const result = parseExpression();
    if (position !== tokens.length || !Number.isFinite(result)) return null;
    const rounded = Number(result.toPrecision(12));
    return { reply: "El resultado es " + rounded + ".", intent: "math_calculation", confidence: 1 };
  } catch {
    return null;
  }
}

function answer(message, context = {}) {
  if (typeof message !== "string" || !message.trim()) {
    return { reply: "Escríbeme una pregunta o un mensaje para empezar.", intent: "empty", confidence: 1 };
  }

  const query = normalize(message);
  const calculation = calculateLocally(message);
  if (calculation) return calculation;
  const previousIntent = context && typeof context.lastIntent === "string" ? context.lastIntent : "";
  const followUp = /^(explica mas|explicame mas|amplia|amplia la respuesta|dame mas detalles|continua|sigue|pon un ejemplo|dame un ejemplo|otro ejemplo|por que|porque|como asi)$/.test(query);
  if (followUp && previousIntent) {
    const previous = KNOWLEDGE.find((item) => item.id === previousIntent);
    if (previous) {
      const examples = {
        fotosintesis: "Por ejemplo, una planta junto a una ventana recibe luz y la utiliza para fabricar azúcares.",
        gravedad: "Por ejemplo, cuando sueltas una pelota, la gravedad la acelera hacia el suelo.",
        programacion: "Por ejemplo, puedes empezar con una función pequeña, probarla con dos entradas y luego manejar errores.",
        inteligencia_artificial: "Por ejemplo, un sistema puede reconocer patrones en texto, pero sus resultados deben verificarse.",
        whatsapp_bots: "Por ejemplo, el bot recibe un mensaje, llama al motor y envía el texto que este devuelve."
      };
      return {
        reply: previous.answer + "\n\nEjemplo: " + (examples[previousIntent] || "Podemos dividir el tema en partes pequeñas y revisar una por una."),
        intent: previousIntent,
        confidence: 0.8,
        followUp: true
      };
    }
  }
  if (/^(hola|holi|buenas|buenos dias|buenas tardes|buenas noches|hey|hi|hello)(\s|$)/.test(query)) {
    return { reply: GREETINGS[hash(query) % GREETINGS.length], intent: "greeting", confidence: 0.99 };
  }
  if (/^(gracias|muchas gracias|te lo agradezco|thank you)(\s|$)/.test(query)) {
    return { reply: "¡De nada! Si necesitas otra cosa, dime y seguimos.", intent: "gratitude", confidence: 0.99 };
  }
  if (/^(adios|hasta luego|nos vemos|chao|bye)(\s|$)/.test(query)) {
    return { reply: "¡Hasta luego! Aquí estaré si quieres volver a consultar algo.", intent: "goodbye", confidence: 0.99 };
  }
  if (query.includes("que puedes hacer") || query === "ayuda" || query === "ayudame") {
    return {
      reply: "Puedo conversar de forma básica, responder saludos y agradecimientos, y buscar coincidencias en conocimientos locales sobre IA, programación, bots de WhatsApp, privacidad, aprendizaje, fotosíntesis y gravedad. No tengo acceso a Internet ni genero conocimiento general ilimitado. Podemos ampliar el motor añadiendo temas revisados.",
      intent: "capabilities",
      confidence: 1
    };
  }
  if (query.includes("que hora es") || query.includes("fecha de hoy") || query === "que dia es hoy") {
    const now = new Date();
    const date = new Intl.DateTimeFormat("es-PE", { dateStyle: "full" }).format(now);
    const time = new Intl.DateTimeFormat("es-PE", { timeStyle: "short" }).format(now);
    return { reply: "Según la fecha y hora local del dispositivo: " + date + ", " + time + ".", intent: "local_time", confidence: 0.95 };
  }
  if (query.includes("como estas") || query.includes("como te encuentras")) {
    return { reply: "¡Gracias por preguntar! Estoy funcionando y listo para ayudarte dentro de mis conocimientos locales. ¿Qué te gustaría hacer?", intent: "wellbeing", confidence: 0.98 };
  }

  let bestEntry = null;
  let bestScore = 0;
  for (const entry of KNOWLEDGE) {
    const score = scoreEntry(query, entry);
    if (score > bestScore) {
      bestEntry = entry;
      bestScore = score;
    }
  }
  if (bestEntry && bestScore >= 0.55) {
    return { reply: bestEntry.answer, intent: bestEntry.id, confidence: Number(bestScore.toFixed(2)) };
  }

  return {
    reply: FALLBACKS[hash(query) % FALLBACKS.length],
    intent: "unknown",
    confidence: 0
  };
}

function hash(value) {
  let result = 0;
  for (let i = 0; i < value.length; i += 1) result = (result * 31 + value.charCodeAt(i)) >>> 0;
  return result;
}

function addKnowledge(entry) {
  if (!entry || typeof entry !== "object" || typeof entry.id !== "string" || !entry.id.trim()) {
    throw new TypeError("El conocimiento necesita un id de texto.");
  }
  if (!Array.isArray(entry.terms) || entry.terms.length === 0 || entry.terms.some((term) => typeof term !== "string" || !term.trim())) {
    throw new TypeError("El conocimiento necesita una lista terms con frases no vacías.");
  }
  if (typeof entry.answer !== "string" || !entry.answer.trim()) {
    throw new TypeError("El conocimiento necesita una respuesta de texto no vacía.");
  }
  const normalizedId = normalize(entry.id).replace(/\s+/g, "_");
  const normalizedTerms = entry.terms.map((term) => term.trim());
  const existing = KNOWLEDGE.findIndex((item) => item.id === normalizedId);
  const item = { id: normalizedId, terms: normalizedTerms, answer: entry.answer.trim() };
  if (existing >= 0) KNOWLEDGE[existing] = item;
  else KNOWLEDGE.push(item);
  return { id: normalizedId, terms: normalizedTerms.length };
}

module.exports = { answer, normalize, KNOWLEDGE, addKnowledge, calculateLocally };

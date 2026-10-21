"use strict";

const KNOWLEDGE = [
  {
    id: "inteligencia_artificial",
    terms: ["inteligencia artificial", "que es la ia", "que es inteligencia artificial", "como funciona la ia"],
    answer: "La inteligencia artificial es un área de la informática que crea sistemas capaces de realizar tareas como reconocer patrones, procesar lenguaje y ayudar a resolver problemas. Algunos sistemas siguen reglas y otros aprenden patrones a partir de datos. Sus resultados deben verificarse porque pueden equivocarse."
  },
  {
    id: "amilcar_ai",
    terms: ["amilcar ai", "amilcar-ai", "quien eres", "que eres", "tu nombre"],
    answer: "Soy AMILCAR AI, un asistente experimental que funciona localmente con reglas y conocimientos incluidos en el proyecto. No uso una API externa en el modo nativo. Mis capacidades son limitadas y no debo fingir que conozco algo cuando no tengo información suficiente."
  },
  {
    id: "programacion",
    terms: ["programacion", "programar", "javascript", "node.js", "nodejs", "codigo"],
    answer: "Para programar, divide el problema en pasos pequeños, implementa una parte, ejecútala y prueba los casos normales y los errores. Si compartes el lenguaje, el código y el error exacto, puedo ayudarte a analizarlo con las capacidades de mi motor local."
  },
  {
    id: "whatsapp_bots",
    terms: ["bot de whatsapp", "whatsapp bot", "baileys", "bot whatsapp"],
    answer: "Un bot de WhatsApp recibe eventos de mensajes, aplica sus reglas y decide qué respuesta enviar. Conviene separar la conexión y la sesión de WhatsApp del motor que genera texto. Así cada instalación conserva sus credenciales y AMILCAR AI se ocupa solo de preparar la respuesta."
  },
  {
    id: "aprendizaje",
    terms: ["como aprender", "aprender", "estudiar", "aprendizaje"],
    answer: "Una técnica útil para aprender es estudiar una idea, explicarla con tus propias palabras, practicar con un ejemplo y comprobar qué entendiste. Revisa los errores y vuelve a intentarlo después de un intervalo; la práctica activa suele ser más útil que solo releer."
  },
  {
    id: "privacidad",
    terms: ["privacidad", "datos personales", "seguridad", "mis datos"],
    answer: "Para proteger tus datos, evita publicar contraseñas, tokens, archivos de sesión o información privada. En el modo nativo de AMILCAR AI, el texto se procesa localmente y no se envía a un proveedor de IA; el bot anfitrión y otras dependencias deben revisarse por separado."
  },
  {
    id: "fotosintesis",
    terms: ["fotosintesis", "plantas y luz", "como se alimentan las plantas"],
    answer: "La fotosíntesis permite que las plantas conviertan energía de la luz en energía química. Usan agua y dióxido de carbono para producir azúcares y liberan oxígeno. En las plantas ocurre principalmente en los cloroplastos de sus células."
  },
  {
    id: "gravedad",
    terms: ["gravedad", "que es la gravedad", "fuerza gravitatoria"],
    answer: "La gravedad es la interacción por la que los cuerpos con masa se atraen. En la Tierra nos mantiene sobre el suelo y hace que los objetos caigan cuando no hay otra fuerza que los sostenga. También mantiene a los planetas en órbita alrededor del Sol."
  }
];

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

function answer(message, context = {}) {
  if (typeof message !== "string" || !message.trim()) {
    return { reply: "Escríbeme una pregunta o un mensaje para empezar.", intent: "empty", confidence: 1 };
  }

  const query = normalize(message);
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
        reply: previous.answer + "\\n\\nEjemplo: " + (examples[previousIntent] || "Podemos dividir el tema en partes pequeñas y revisar una por una."),
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
  const normalizedId = normalize(entry.id).replace(/\\s+/g, "_");
  const normalizedTerms = entry.terms.map((term) => term.trim());
  const existing = KNOWLEDGE.findIndex((item) => item.id === normalizedId);
  const item = { id: normalizedId, terms: normalizedTerms, answer: entry.answer.trim() };
  if (existing >= 0) KNOWLEDGE[existing] = item;
  else KNOWLEDGE.push(item);
  return { id: normalizedId, terms: normalizedTerms.length };
}

module.exports = { answer, normalize, KNOWLEDGE, addKnowledge };

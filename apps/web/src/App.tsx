import { FormEvent, useEffect, useRef, useState } from "react";
import { Activity, ArrowUp, Bot, Cpu, Github, Menu, Plus, ShieldCheck, Sparkles, Trash2, Wifi, WifiOff } from "lucide-react";

type Message = { role: "assistant" | "user"; content: string };
const welcome: Message = { role: "assistant", content: "Bienvenido a AMILCAR AI. Soy una IA experimental abierta, ejecutada desde tu propio equipo. Entrena AMI-Base para empezar a generar respuestas. Mis capacidades todavía son limitadas." };

export default function App() {
  const [messages, setMessages] = useState<Message[]>([welcome]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [online, setOnline] = useState(false);
  const [trained, setTrained] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/health").then((r) => r.json()).then((data) => { setOnline(data.status === "ok"); setTrained(Boolean(data.trained)); }).catch(() => setOnline(false));
  }, []);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages, busy]);

  async function send(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    setMessages((old) => [...old, { role: "user", content: message }]);
    setInput("");
    setError("");
    setBusy(true);
    try {
      const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "No se pudo contactar con el modelo.");
      setMessages((old) => [...old, { role: "assistant", content: data.reply }]);
      setTrained(Boolean(data.trained));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Error inesperado.");
    } finally { setBusy(false); }
  }

  function newChat() { setMessages([welcome]); setError(""); }

  return <div className="app-shell">
    <div className="ambient ambient-one" /><div className="ambient ambient-two" />
    <aside className="sidebar">
      <a className="brand" href="https://github.com/AmilcarGit/AMILCAR-AI" target="_blank" rel="noreferrer">
        <span className="brand-mark"><Cpu size={24}/></span><span><strong>AMILCAR<span> AI</span></strong><small>NEURAL SYSTEM v0.1</small></span>
      </a>
      <button className="new-chat" onClick={newChat}><Plus size={17}/> Nueva conversación</button>
      <div className="side-label">SISTEMA</div>
      <div className="side-item"><Activity size={17}/> Panel de actividad</div>
      <div className="side-item"><Bot size={17}/> Modelo AMI-Base</div>
      <div className="side-item"><ShieldCheck size={17}/> Privacidad local</div>
      <div className="side-bottom">
        <div className="system-status"><span className={online ? "status-dot live" : "status-dot"}/>{online ? "API conectada" : "API desconectada"}</div>
        <a className="github-link" href="https://github.com/AmilcarGit/AMILCAR-AI" target="_blank" rel="noreferrer"><Github size={16}/> Ver proyecto en GitHub</a>
        <div className="credit">CREADO POR AMILCARGIT</div>
      </div>
    </aside>
    <main className="main-panel">
      <header className="topbar">
        <div className="mobile-brand"><Menu size={19}/><span>AMILCAR <b>AI</b></span></div>
        <div className="model-chip"><span className="pulse-dot"/> AMI-BASE <span className="version">0.1.0</span></div>
        <div className="top-actions"><span className={online ? "connection" : "connection offline"}>{online ? <Wifi size={14}/> : <WifiOff size={14}/ >}{online ? "LOCAL" : "OFFLINE"}</span><button className="icon-button" aria-label="Nueva conversación" onClick={newChat}><Trash2 size={16}/></button></div>
      </header>
      <section className="chat-area">
        {messages.length === 1 && <div className="hero">
          <div className="orbital"><div className="orbit orbit-a"/><div className="orbit orbit-b"/><div className="core"><Sparkles size={35}/></div><span className="orbit-node node-a"/><span className="orbit-node node-b"/></div>
          <div className="eyebrow"><span/> SISTEMA NEURONAL INICIALIZADO</div>
          <h1>El futuro empieza<br/>con <span>una idea.</span></h1>
          <p>Inteligencia abierta. Control local. Construida para evolucionar contigo.</p>
          <div className="feature-grid">
            <div className="feature-card"><Cpu size={18}/><strong>Modelo propio</strong><small>Entrenable desde cero</small></div>
            <div className="feature-card"><ShieldCheck size={18}/><strong>Privacidad</strong><small>Tu equipo, tus datos</small></div>
            <div className="feature-card"><Github size={18}/><strong>Open source</strong><small>Abierto a contribuciones</small></div>
          </div>
        </div>}
        <div className="messages">
          {messages.length > 1 && messages.slice(1).map((message, index) => <article className={message.role === "user" ? "message user-message" : "message assistant-message"} key={String(index) + message.role}>
            <div className="avatar">{message.role === "user" ? "TÚ" : <Bot size={17}/>}</div><div className="message-body"><div className="message-role">{message.role === "user" ? "USUARIO" : "AMILCAR AI"}</div><p>{message.content}</p></div>
          </article>)}
          {busy && <div className="thinking"><span/><span/><span/> Procesando señal neuronal…</div>}<div ref={endRef}/>
        </div>
      </section>
      <div className="composer-wrap">
        {error && <div className="error-banner">{error}{!trained && <span> · Entrena el modelo con <code>python -m ai.training.train</code></span>}</div>}
        {!trained && <div className="model-note"><Activity size={14}/> Modelo aún no entrenado. Ejecuta el comando que aparece en el README.</div>}
        <form className="composer" onSubmit={send}>
          <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(event); } }} placeholder="Escribe tu mensaje a AMILCAR AI…" rows={1} maxLength={2000} aria-label="Mensaje para AMILCAR AI"/>
          <button className="send-button" disabled={!input.trim() || busy} aria-label="Enviar mensaje"><ArrowUp size={20}/></button>
        </form>
        <div className="composer-footer"><span><ShieldCheck size={13}/> Procesamiento local · sin claves comerciales</span><span>{input.length}/2000</span></div>
      </div>
    </main>
  </div>;
}

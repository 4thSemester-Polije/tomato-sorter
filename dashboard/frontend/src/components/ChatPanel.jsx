import { useEffect, useRef, useState } from "react";
import { getChatHistory, sendChat } from "../api.js";

const KEY = "tomato-chat-session";
const SUGGESTIONS = ["Berapa tomat per grade?", "Apa error terakhir?", "Berapa rata-rata latensi?", "Tampilkan 5 tomat terbaru"];

function newSessionId() {
  // crypto.randomUUID hanya ada di HTTPS atau localhost; cadangan untuk akses lewat http://IP-LAN
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID().replaceAll("-", "");
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
}

export default function ChatPanel() {
  const [sid, setSid] = useState(() => {
    try {
      return sessionStorage.getItem(KEY);
    } catch {
      return null;
    }
  });
  const [messages, setMessages] = useState([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const end = useRef(null);

  useEffect(() => {
    if (sid) getChatHistory(sid).then((h) => setMessages(h.map(({ role, content }) => ({ role, content })))).catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    end.current?.scrollIntoView?.({ block: "end" });
  }, [messages, busy]);

  const ask = async (q) => {
    const message = q.trim();
    if (!message || busy) return;
    setMessages((m) => [...m, { role: "user", content: message }]);
    setText("");
    setBusy(true);
    setError(null);
    try {
      const id = sid || newSessionId();
      if (!sid) {
        setSid(id);
        try {
          sessionStorage.setItem(KEY, id);
        } catch {}
      }
      const r = await sendChat(id, message);
      setMessages((m) => [...m, { role: "assistant", content: r.reply }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    sessionStorage.removeItem(KEY);
    setSid(null);
    setMessages([]);
    setError(null);
  };

  return (
    <section className="card chat">
      <div className="card-head">
        <h2>Tanya data</h2>
        {messages.length > 0 && <button onClick={reset}>Percakapan baru</button>}
      </div>
      <p className="muted small">Asisten hanya membaca data alat. Ia tidak bisa mengontrol mesin.</p>
      <div className="chat-log" aria-live="polite">
        {messages.length === 0 && (
          <div className="chips">
            {SUGGESTIONS.map((s) => <button key={s} className="chip" onClick={() => ask(s)}>{s}</button>)}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>{m.content}</div>
        ))}
        {busy && <div className="bubble assistant muted">Mencari data...</div>}
        {error && <p className="error">{error}</p>}
        <div ref={end} />
      </div>
      <form onSubmit={(e) => { e.preventDefault(); ask(text); }} className="chat-form">
        <input value={text} onChange={(e) => setText(e.target.value)} maxLength={1000} placeholder="Tanya tentang hasil sortir..." aria-label="Pertanyaan" />
        <button type="submit" disabled={busy || !text.trim()}>Kirim</button>
      </form>
    </section>
  );
}

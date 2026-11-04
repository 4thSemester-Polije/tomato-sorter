import { useEffect, useRef, useState } from "react";

const MAX_TOMATOES = 25;
const MAX_EVENTS = 50;

const initial = { connected: false, state: null, stats: null, tomatoes: [], events: [], newTomatoTick: 0 };

function reduce(prev, msg) {
  switch (msg.type) {
    case "snapshot":
      return { ...prev, state: msg.state, stats: msg.stats, tomatoes: msg.tomatoes, events: msg.events };
    case "state":
      return { ...prev, state: msg.state };
    case "stats":
      return { ...prev, stats: msg.stats };
    case "tomato":
      return {
        ...prev,
        stats: msg.stats,
        tomatoes: [msg.tomato, ...prev.tomatoes].slice(0, MAX_TOMATOES),
        newTomatoTick: prev.newTomatoTick + 1,
      };
    case "tomato_update":
      return { ...prev, tomatoes: prev.tomatoes.map((t) => (t.id === msg.tomato.id ? msg.tomato : t)) };
    case "event":
      return { ...prev, events: [msg.event, ...prev.events].slice(0, MAX_EVENTS) };
    default:
      return prev;
  }
}

/** Data langsung dari WebSocket /ws. Menyambung ulang otomatis; snapshot baru dikirim server setiap tersambung. */
export function useLive() {
  const [live, setLive] = useState(initial);
  const retry = useRef(0);

  useEffect(() => {
    let ws;
    let timer;
    let closed = false;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => {
        retry.current = 0;
        setLive((p) => ({ ...p, connected: true }));
      };
      ws.onmessage = (e) => setLive((p) => reduce(p, JSON.parse(e.data)));
      ws.onclose = () => {
        setLive((p) => ({ ...p, connected: false }));
        if (!closed) {
          timer = setTimeout(connect, Math.min(1000 * 2 ** retry.current++, 10000));
        }
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      ws && ws.close();
    };
  }, []);

  return live;
}

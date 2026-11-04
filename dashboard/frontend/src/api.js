async function json(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(typeof detail === "string" ? detail : "Permintaan gagal");
  }
  return res.json();
}

export const getTomatoes = ({ grade, limit, offset }) => {
  const q = new URLSearchParams({ limit, offset });
  if (grade) q.set("grade", grade);
  return fetch(`/api/tomatoes?${q}`).then(json);
};

export const sendChat = (session_id, message) =>
  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id, message }),
  }).then(json);

export const getChatHistory = (session_id) => fetch(`/api/chat/${session_id}`).then(json);

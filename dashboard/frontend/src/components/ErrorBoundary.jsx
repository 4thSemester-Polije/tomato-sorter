import React from "react";

/** Bila ada komponen yang error, tampilkan pesannya (bukan layar putih) agar mudah dilaporkan. */
export default class ErrorBoundary extends React.Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("Dashboard error:", error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={{ padding: 24, maxWidth: 720, margin: "0 auto", fontFamily: "system-ui, sans-serif" }}>
        <h1>Terjadi kesalahan di tampilan</h1>
        <p>Muat ulang halaman. Bila berulang, kirim pesan di bawah ini ke pengembang.</p>
        <pre style={{ whiteSpace: "pre-wrap", background: "#eee", color: "#222", padding: 12, borderRadius: 8 }}>
          {String(this.state.error?.stack || this.state.error)}
        </pre>
        <button onClick={() => location.reload()}>Muat ulang</button>
      </div>
    );
  }
}

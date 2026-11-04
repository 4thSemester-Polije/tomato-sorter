import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Saat `npm run dev`, semua panggilan ke backend FastAPI (python -m dashboard) diteruskan lewat proxy.
const backend = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": backend,
      "/stream": backend,
      "/ws": { target: backend, ws: true },
    },
  },
});

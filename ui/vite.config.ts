// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const CHAIN_SERVER = process.env.CHAIN_SERVER_URL || "http://localhost:8009";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: Number(process.env.PORT) || 3000,
    strictPort: true,
    proxy: {
      // The UI calls the relative `/api` it uses behind nginx, so one forwarded
      // port serves both the page and the API. `VITE_API_BASE_URL` set to an
      // absolute chain-server URL is resolved by the browser instead, and
      // fails whenever the browser is not on the machine running the services.
      "/api": {
        target: CHAIN_SERVER,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
        configure: (proxy) => {
          // Server-sent events must not be buffered, or the reply arrives in
          // one lump at the end of the turn instead of streaming.
          proxy.on("proxyRes", (proxyRes) => {
            proxyRes.headers["cache-control"] = "no-cache, no-transform";
          });
        },
      },
    },
  },
  build: {
    // The Dockerfile copies /app/build into nginx.
    outDir: "build",
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});

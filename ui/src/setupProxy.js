/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const { createProxyMiddleware } = require("http-proxy-middleware");

/**
 * Keep local browser API traffic on the UI origin without enabling Create
 * React App's broad package.json proxy. The dedicated prefix limits forwarding
 * to an explicit chain-server boundary and remains usable through remote port
 * forwarding, whose public Host header is not known to the local runner.
 */
module.exports = function setupProxy(app) {
  // This server is development-only. Prevent a remote browser or forwarding
  // layer from retaining an older bundle with a stale API base URL.
  app.use((_request, response, next) => {
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("Cloudflare-CDN-Cache-Control", "no-store");
    next();
  });

  app.use(
    "/local-api",
    createProxyMiddleware({
      target: "http://127.0.0.1:8009",
      changeOrigin: true,
      pathRewrite: { "^/local-api": "" },
      ws: false,
      logLevel: "warn",
    })
  );
};

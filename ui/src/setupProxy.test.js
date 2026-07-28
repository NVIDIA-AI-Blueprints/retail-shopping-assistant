// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { createProxyMiddleware } = require("http-proxy-middleware");
const setupProxy = require("./setupProxy");

jest.mock("http-proxy-middleware", () => ({
  createProxyMiddleware: jest.fn(),
}));

describe("local development proxy", () => {
  test("disables response caching and scopes forwarding to local-api", () => {
    const profileProxy = jest.fn();
    createProxyMiddleware.mockReturnValue(profileProxy);
    const app = { use: jest.fn() };
    setupProxy(app);

    expect(app.use).toHaveBeenCalledTimes(2);

    const cacheMiddleware = app.use.mock.calls[0][0];
    for (const url of [
      "/",
      "/static/js/bundle.js",
      "/local-api/shopper-profiles",
    ]) {
      const response = { setHeader: jest.fn() };
      const next = jest.fn();
      cacheMiddleware({ url }, response, next);

      expect(response.setHeader).toHaveBeenCalledWith(
        "Cache-Control",
        "no-store"
      );
      expect(response.setHeader).toHaveBeenCalledWith(
        "Cloudflare-CDN-Cache-Control",
        "no-store"
      );
      expect(next).toHaveBeenCalledTimes(1);
    }

    expect(createProxyMiddleware).toHaveBeenCalledWith({
      target: "http://127.0.0.1:8009",
      changeOrigin: true,
      pathRewrite: { "^/local-api": "" },
      ws: false,
      logLevel: "warn",
    });
    expect(app.use.mock.calls[1]).toEqual(["/local-api", profileProxy]);
  });
});

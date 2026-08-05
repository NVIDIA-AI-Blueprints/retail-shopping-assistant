/**
 * Dev-server proxy so the API rides the same port the UI does.
 *
 * `REACT_APP_API_BASE_URL=http://localhost:8009` is baked into the bundle at
 * build time and resolved by the *browser*, so it only works when the browser
 * and the chain server share a machine. Reaching the app through a forwarded
 * port — the usual remote-dev setup — forwards 3000 and not 8009, so every
 * request fails before it leaves the browser with "Failed to fetch".
 *
 * Proxying `/api` here lets the UI use the same relative base it uses in
 * production behind nginx, so one forwarded port is enough. The prefix is
 * stripped because the chain server serves `/query/stream`, not
 * `/api/query/stream`.
 */

const { createProxyMiddleware } = require('http-proxy-middleware');

const CHAIN_SERVER = process.env.CHAIN_SERVER_URL || 'http://localhost:8009';

module.exports = function setupProxy(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: CHAIN_SERVER,
      changeOrigin: true,
      // Server-sent events must not be buffered, or the reply arrives in one
      // lump at the end of the turn instead of streaming.
      onProxyRes: (proxyRes) => {
        proxyRes.headers['cache-control'] = 'no-cache, no-transform';
      },
      pathRewrite: { '^/api': '' },
    })
  );
};

import type { NextConfig } from "next";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://127.0.0.1:8000";

const config: NextConfig = {
  reactStrictMode: true,
  experimental: {
    typedRoutes: true,
  },
  // Move the Next dev-mode indicator out of the bottom-left sidebar area.
  devIndicators: {
    position: "bottom-right",
  },
  // Proxy API calls to the FastAPI backend so the SPA can use same-origin fetches.
  async rewrites() {
    return [
      { source: "/api/query", destination: `${FASTAPI_URL}/query` },
      { source: "/api/info", destination: `${FASTAPI_URL}/info` },
      { source: "/api/healthz", destination: `${FASTAPI_URL}/healthz` },
      { source: "/api/cache/clear", destination: `${FASTAPI_URL}/cache/clear` },
    ];
  },
};

export default config;

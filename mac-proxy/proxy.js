#!/usr/bin/env node
/**
 * 内网只读代理（示例实现）
 * ============================================================
 * 适用场景：
 *   助手服务跑在能上公网的机器上，但某些内部接口只允许从内网访问。
 *   在内网机器（办公电脑 / 跳板机）上跑这个代理，服务端通过它转发**只读**请求。
 *
 * 四条安全约束（缺一不可）：
 *   ① 路径白名单：只放行 ALLOW 里显式登记的路径，其余一律 403；
 *   ② 只读：只接受 GET / POST，不透传 PUT / PATCH / DELETE；
 *   ③ 密钥鉴权：调用方必须带 X-Proxy-Key（防止同网段其他设备乱调）；
 *   ④ 凭据不外泄：上游 token 只在内存里用，不出现在响应与日志里。
 *
 * 用法：
 *   PROXY_KEY=<密钥> TARGET=https://internal.example.com node proxy.js
 *   curl -H 'X-Proxy-Key: <密钥>' http://127.0.0.1:9100/health
 */

const http = require("http");
const https = require("https");
const { URL } = require("url");

const PORT = Number(process.env.PROXY_PORT || 9100);
const BIND = process.env.PROXY_BIND || "127.0.0.1";     // 需跨机访问时改成内网 / Tailscale 地址
const KEY = process.env.PROXY_KEY || "";                 // 必填
const TARGET = process.env.TARGET || "";                 // 必填，内部系统基地址
const UPSTREAM_TOKEN = process.env.UPSTREAM_TOKEN || ""; // 可选，上游凭据
const TIMEOUT_MS = Number(process.env.PROXY_TIMEOUT_MS || 8000);

// 白名单：按需添加，支持前缀匹配
const ALLOW = (process.env.PROXY_ALLOW || "/api/status,/api/query")
  .split(",").map(s => s.trim()).filter(Boolean);

function allowed(path) {
  return ALLOW.some(p => path === p || path.startsWith(p + "/") || path.startsWith(p + "?"));
}

function send(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

const server = http.createServer((req, res) => {
  if (!KEY || !TARGET) {
    return send(res, 500, { ok: false, error: "PROXY_KEY / TARGET 未配置" });
  }
  if (req.headers["x-proxy-key"] !== KEY) {
    return send(res, 401, { ok: false, error: "unauthorized" });
  }
  const u = new URL(req.url, "http://local");
  if (u.pathname === "/health") {
    return send(res, 200, { ok: true, target: new URL(TARGET).host, allow: ALLOW });
  }
  if (req.method !== "GET" && req.method !== "POST") {
    return send(res, 405, { ok: false, error: "只读代理：仅支持 GET / POST" });
  }
  if (!allowed(u.pathname)) {
    return send(res, 403, { ok: false, error: "path not allowed" });
  }

  const base = new URL(TARGET);
  const mod = base.protocol === "https:" ? https : http;
  const headers = { "Content-Type": req.headers["content-type"] || "application/json" };
  if (UPSTREAM_TOKEN) headers["Authorization"] = `Bearer ${UPSTREAM_TOKEN}`;

  const up = mod.request(
    {
      protocol: base.protocol,
      hostname: base.hostname,
      port: base.port || (base.protocol === "https:" ? 443 : 80),
      path: base.pathname.replace(/\/$/, "") + u.pathname + u.search,
      method: req.method,
      headers,
      timeout: TIMEOUT_MS,
    },
    (upRes) => {
      let data = "";
      upRes.setEncoding("utf8");
      upRes.on("data", (c) => { data += c; });
      upRes.on("end", () => {
        res.writeHead(upRes.statusCode || 502, { "Content-Type": "application/json; charset=utf-8" });
        res.end(data);
      });
    }
  );
  up.on("timeout", () => up.destroy(new Error("upstream timeout")));
  up.on("error", (e) => {
    // ⚠️ 不回传上游地址与凭据，只给一个中性错误
    console.error("[proxy]", e.message);
    send(res, 502, { ok: false, error: "该数据暂时无法取到" });
  });

  if (req.method === "POST") req.pipe(up); else up.end();
});

server.listen(PORT, BIND, () => {
  console.log(`[proxy] listening on ${BIND}:${PORT} → ${TARGET || "(未配置)"}`);
});

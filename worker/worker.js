/**
 * 运维知识库 AI 问答 —— 云端网关（路线A 的"另一处 API"）
 *
 * 职责：
 *   1. 静态页面（public/index.html 报障表单，绑在 ask.example.com 根路径）
 *   2. 用户访问 POST /ask、/summarize、/upload 时校验 API Key，探测本机隧道入口（KB_ORIGIN）
 *   3. 在线 → 转发给本机服务，返回结果
 *   4. 离线（本机关机/断网/tunnel 断）→ 返回 503 "服务不在线，请联系 xxx"
 *   5. GET /healthz 免 Key 转发探活（页面顶部在线状态用）；
 *      GET /img/*   免 Key 转发报障截图（页面 <img src> 带不了鉴权头，靠 id 随机不可猜）
 *
 * ⚠️ 这里是**白名单**转发：本机新增 POST 接口后，除了本机 main.py 加路由，
 *    还必须把路径加进下方 FORWARD_POST_PATHS，否则公网访问会 404。
 *
 * 部署：见 README.md「第二步」。本文件不含任何密钥，密钥放 Worker 的 Secret 里。
 */

const PROBE_TIMEOUT_MS = 2500;   // 探活首次超时：本机离线时用户不用等太久
const PROBE_RETRY_TIMEOUT_MS = 7000; // 首次失败后的放宽超时（见 probe 注释）
const PROBE_CACHE_MS = 5000;     // 探活结果短缓存：避免同一 isolate 内连续重复探活
// 转发超时（2026-10-10 评审 ⑧ 对齐三层；2026-10-12 随轮数 8→12 再放大）：
//   前端 fetch 240s / 后端排队等待上限 100s / deep 单问答 ~80s（p90，12 轮后可能到 120s）。
//   ⇒ deep 上限 12 轮后，最坏 = 排队未完就先等 100s 后快速回绝（不走这里）；
//     真跑的问答单笔 60~140s，故取 210s 给足余量，且**小于前端 240s** ——
//     宁可网关先断（错误更友好），也不要前端先断（用户只看到干等）。
const FORWARD_TIMEOUT_MS = 210000;

// 需要校验 Key 并转发到本机服务的 POST 路径（白名单：新增本机接口只改这里）
// ⚠️ 精确匹配，子路径要单独列（如 /coach/active）。
const FORWARD_POST_PATHS = new Set([
  "/ask", "/ask-stream", "/summarize", "/upload", "/upload-zip",
  "/feedback", "/cases", "/coach", "/coach/active", "/retry",
  // 会话（跨设备续聊）+ 手动清理（2026-09-22）
  "/session/new", "/session/sync", "/gc",
  // 访问认证（2026-09-22）：登录/退出本身必须免验证，否则没法登录
  "/auth/login", "/auth/logout",
  // 管理员面板（2026-09-25）：走 POST 是为了拿到网关注入的 X-Auth-User，
  // 后端据此判管理员；非管理员返回 403，前端就隐藏入口。
  "/admin/status", "/admin/user/disable", "/admin/user/enable",
  "/admin/user/rename",     // 2026-11-02 管理员改显示名（老邀请的匿名名 → 真名）
  "/admin/reload", "/admin/invite/create", "/admin/invite/revoke",
  "/admin/usage",     // 2026-10-12 使用统计（最近 N 天用量与解决情况）
]);

// 免 Key 转发到本机服务的 GET 路径前缀（页面 <img src> 无法带鉴权头；
// 图片 id 是 32 位随机 hex、简图名是内容 sha1 前 16 位，都不可猜）
// ⚠️ /diagram/ 漏了会很难查：请求会落到下面第 4 段的静态资源，返回 index.html，
//    <img> 拿到 HTML 后显示裂图 —— 表现为「本机看得到图、公网看不到」。
// ⚠️ /session/ 也免 Key：会话 ID 是 70bit 随机不可猜凭证（同 /img/ 思路），
//    主人 2026-09-22 定的是「只靠会话 ID 就能继续」，所以不再叠一层 Key。
// ⚠️ "/queue-status" 必须在这里：前端轮询「我前面还有几位」时走 GET，
//    网关没登记就直接 404（这个坑踩过 —— 新接口上线别忘了这一行）。
// ⚠️ 另注：GET 转发用的是 `origin + url.pathname`，**query 会被丢掉**，
//    所以排队查询的 rid 走**路径**而不是 ?rid=xxx。
const FORWARD_GET_PREFIXES = ["/img/", "/diagram/", "/session/", "/auth/", "/queue-status"];

/* ================== 访问认证（2026-09-22 主人定） ==================
   动机：助手跑在总部管理员权限上，内部不需要权限分级，但**要挡住外部人员嗅探数据**。
   原来的网关密钥硬编码在页面 JS 里（F12 可见）= 等于没有访问控制，这一层补上。

   方案：**凭据（TOTP 6 位动态码 **或** 共享长期授权码 + 员工编号）** → 本机签发**签名设备令牌**
        → **在网关离线验签**（不查库、不回源）。
        token = "<exp>.<fp8>.<uid>.<sig>"，sig = HMAC-SHA256(exp.fp8.uid, AUTH_SECRET) 前 32 位
        AUTH_SECRET 同时配在本机和 Worker（wrangler secret），两边算法必须一致。

   ⭐ 2026-10-09 主人放开内测：门禁页**多了一条入口** —— 共享长期授权码 + 用户自报的员工编号。
      新用户不必再绑 Authenticator（那一步最劝退）；已绑的老用户输 6 位动态码照走。
      ⚠️ 网关**不校验授权码**（它离线、不查库）—— 授权码的正确性由**后端** /auth/login 判定；
         网关只负责"验签令牌、放行页面"。所以改授权码**不必重新部署 Worker**。

   ⚠️ 全站拦：页面本身也在 Cloudflare 上，所以门禁必须放在**本 Worker 的最前面**，
      静态资源与转发都在它之后 —— 否则"只有接口被保护、页面照样能打开"。
   ⚠️ 令牌 Cookie 是 HttpOnly（JS 偷不走、XSS 也读不到）；设备标识单独放一个可读 Cookie
      （Worker 要拿它跟令牌里的 fp8 比对）。这两个 Cookie 名要与后端 config 一致。
   ⚠️ 设备标识仍只比对**前 8 位**（与后端 FP_LEN 一致）—— 2026-10-09 改成随机 ID 之后
      它已经不漂移了，但**不能改比对长度**：改了会让所有已签发的令牌立刻失效、全员被踢。

   ⭐ 2026-10-09 改动：`kb_fp` 的值从「浏览器指纹」换成「**随机设备 ID**」。
      起因：用户输完动态码报「缺少浏览器指纹」—— 查下来 `browserFp()` 依赖
      `crypto.subtle` / Canvas，在老内核、国产浏览器"转码/云加速"模式、部分内置 WebView
      里**根本不存在** ⇒ 抛异常；而那段的调用处**没有 try/catch** ⇒ `FP` 永远是空串，
      前端还完全静默（唯一症状是底部一直停在「正在生成设备标识…」）。
      为什么能换成随机 ID：**指纹与随机 ID 的安全性完全同级** —— 两者都是客户端自己说了算、
      都能被伪造、服务端都无法独立验证。真正的凭证是「动态码 + HMAC 签名令牌」，设备标识
      只是"同一台设备免重复输码"的便利标签（旧注释其实已自认：「真正的凭证是 Cookie 本身」）。
      换来的好处：① 不依赖 crypto.subtle/Canvas ⇒ 老设备也能用、**永不为空**；
                  ② 不再漂移 ⇒ 浏览器升级不会被踢。
      ⚠️ Cookie 名 `kb_fp` **保持不变**（改名会把在用的人全部踢出去重验）；
         并且**先读后写**（有值就沿用）⇒ 老用户（Cookie 里存的是旧指纹值）**完全无感**。 */
const AUTH_COOKIE = "kb_device";
const AUTH_FP_COOKIE = "kb_fp";
// 设备令牌有效期（天）。⚠️ 必须与后端 config.AUTH_TTL_DAYS **保持一致** ——
// 后端是真正签发的地方，这里只用来写页面文案（写错不会导致验签失败，但会**骗人**）。
// 2026-10-09 主人放开内测：7 → 15。2026-10-12：15 → 30。
const AUTH_TTL_DAYS = 30;

function parseCookies(request) {
  const h = request.headers.get("Cookie") || "";
  const out = {};
  for (const part of h.split(";")) {
    const i = part.indexOf("=");
    if (i > 0) out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

async function hmacHex(secret, msg) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(msg));
  return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, "0")).join("");
}

/** 离线验签。返回 [是否有效, 原因, uid] —— 与 app/auth.py::verify 的判据保持一致。
 *  ⚠️ 令牌是 **4 段**：`<exp>.<fp8>.<uid>.<sig>`。uid 是「谁」的标识（签发时记进审计），
 *     网关只负责验签，不解释 uid —— 身份追溯靠后端 data/auth_audit.jsonl。 */
async function verifyToken(token, fp, secret) {
  const parts = String(token || "").split(".");
  if (parts.length !== 4) return [false, "令牌格式不对", ""];
  const [expS, fp8, uid, sig] = parts;
  const exp = parseInt(expS, 10);
  if (!exp) return [false, "令牌格式不对", ""];
  if (exp < Math.floor(Date.now() / 1000)) return [false, "验证已过期，请重新验证", ""];
  const expect = (await hmacHex(secret, `${exp}.${fp8}.${uid}`)).slice(0, 32);
  if (expect.length !== sig.length) return [false, "令牌无效", ""];
  let diff = 0;
  for (let i = 0; i < sig.length; i++) diff |= sig.charCodeAt(i) ^ expect.charCodeAt(i);
  if (diff !== 0) return [false, "令牌无效", ""];
  if (fp && fp.slice(0, 8) !== fp8) return [false, "换了设备或浏览器，请重新验证", ""];
  return [true, "ok", uid];
}

/** 验证页（全站门禁）。授权码 / 动态码 + 设备标识都在这页里处理，验过之后整站才可用。
 *
 *  ⭐ 2026-10-09 主人放开内测，入口**两条**：
 *    ① 授权码（管理员发的，**长期有效**）+ 你的员工编号（只作身份记录）—— 新用户走这条；
 *    ② 6 位动态码（TOTP）—— 已绑过 Authenticator 的老用户不受影响，继续用。
 *    判据是「6 位纯数字 → 动态码；其它 → 授权码」，与后端 main.auth_login 的分支一致。
 */
function loginPage(reason) {
  const msg = reason && reason !== "ok" ? reason
    : `新用户：填管理员发的<b>授权码</b> + 你的<b>LDAP 账号</b>。验证一次，这台设备 ${AUTH_TTL_DAYS} 天内免验证。`;
  const html = `<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>身份验证 · POS 智能报障助手</title>
<style>
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
       background:#f6f8fc;font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:#0f172a}
  .card{width:min(400px,92vw);background:#fff;border:1px solid #dbeafe;border-radius:16px;
        padding:26px 24px 22px;box-shadow:0 10px 30px rgba(37,99,235,.10)}
  .hd{display:flex;align-items:center;gap:9px;font-size:17px;font-weight:700}
  .sub{font-size:12.5px;color:#64748b;margin-top:7px;line-height:1.7}
  label{display:block;font-size:13px;color:#334155;margin:18px 0 7px;font-weight:500}
  label .opt{font-weight:400;color:#94a3b8;font-size:11.5px}
  input{width:100%;padding:11px 13px;border:1px solid #cbd5e1;border-radius:10px;font-size:15px;outline:none}
  input:focus{border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.13)}
  button{width:100%;margin-top:16px;padding:12px;border:0;border-radius:10px;cursor:pointer;
         background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;font-size:15px;font-weight:600}
  button:disabled{opacity:.5;cursor:not-allowed}
  .err{margin-top:13px;font-size:13px;color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;
       border-radius:9px;padding:9px 11px;display:none}
  .fp{margin-top:16px;padding-top:13px;border-top:1px dashed #e2e8f0;font-size:11.5px;color:#94a3b8;line-height:1.6}
</style></head><body>
<div class="card">
  <div class="hd">🔐 身份验证</div>
  <div class="sub">${msg}</div>
  <label for="code">授权码（必填）</label>
  <input id="code" type="text" autocomplete="off" autofocus
         placeholder="管理员发的授权码（已绑动态令牌的填 6 位动态码）">
  <label for="staff">你的 LDAP 账号 <span class="opt" id="staffHint">（用授权码时必填 · 用于登记是谁在使用）</span></label>
  <input id="staff" type="text" autocomplete="off" placeholder="例如 zuoyuan1（与域账号一致），仅作身份记录">
  <button id="go">验证并进入</button>
  <div class="err" id="err"></div>
  <div class="fp" id="fpbox">设备标识：读取中…</div>
</div>
<script>
/* 设备标识：**随机生成一次 → 存 Cookie → 之后一直沿用**（2026-10-09 从"浏览器指纹"改过来）。
   它只用于「区分是不是同一台设备/浏览器」，不作为强身份凭证 ——
   真正的凭证是 TOTP 动态码 + 服务端签名的令牌。

   ⚠️ 为什么不用指纹：指纹要"算"，而算它依赖 crypto.subtle / Canvas，这两样在
      老内核、国产浏览器"转码/云加速"模式、部分内置 WebView 里**根本不存在** ⇒ 抛异常。
      叠加"调用处没有 try/catch"，就会静默变成一个空串，用户只会看到后端的
      「缺少浏览器指纹」，而且反复重输动态码也没用。
      随机 ID 不依赖任何现代 API：只要 Cookie 能写就一定有值 ⇒ **永远非空**。
   ⚠️ 安全性没有降低：指纹和随机 ID 都是**客户端自己说了算**、都能被伪造、服务端都无法
      独立验证 —— 两者是同一等级。真正的门是「动态码 + 签名令牌」。 */
const $ = id => document.getElementById(id);

function genId(){
  // 三级降级，任一级成功即可。Math.random 是 ES1 就有的，兜底保证**一定非空**。
  try { const u = crypto.randomUUID(); if (u) return u.replace(/-/g, "").slice(0, 16); } catch (e) {}
  try {
    const a = new Uint8Array(8); crypto.getRandomValues(a);
    return [...a].map(b => b.toString(16).padStart(2, "0")).join("");
  } catch (e) {}
  return (Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2)
          + Date.now().toString(36)).slice(0, 16);
}

function getDeviceId(){
  // ⚠️ 整体包一层：任何意外都**不许抛出去**（抛出去 = 脚本死 = 按钮点不动）。
  //    这次的 bug 教训就是"静默崩溃"，所以这里宁可退化也不崩。
  try {
    // ① 先读：**有值就沿用** —— 关键，无条件覆盖会把老 Cookie 每次换成新值
    //    （那正是"浏览器升级/指纹漂移就被踢"的根源）。老用户 Cookie 里存的旧指纹值照样沿用。
    const m = document.cookie.match(/(?:^|;\s*)${AUTH_FP_COOKIE}=([^;]+)/);
    if (m && m[1] && m[1] !== "undefined" && m[1] !== "null") {
      try { return decodeURIComponent(m[1]); } catch (e) { return m[1]; }
    }
    // ② 没有才生成并写入。有效期 1 年（比令牌的 ${AUTH_TTL_DAYS} 天长：换令牌不必换设备标识）
    const id = genId();
    document.cookie = "${AUTH_FP_COOKIE}=" + id + "; Path=/; Secure; SameSite=Lax; Max-Age=" + (365 * 86400);
    return id;
  } catch (e) {
    return genId();            // 读/写 Cookie 都失败也要给个值（内存态，本次会话内可用）
  }
}

let FP = getDeviceId();   // 同步拿到，**不存在竞态**（原来是 await，提交时可能还没算完）
$("fpbox").textContent = "设备标识：" + FP.slice(0, 8) + "…（本浏览器专用；换浏览器或换设备需重新验证；验证状态保留 ${AUTH_TTL_DAYS} 天）";
function showErr(m){ $("err").textContent = m; $("err").style.display = "block"; }
async function submit(){
  const code = $("code").value.trim();
  const staff  = $("staff").value.trim();
  if (!code) { $("code").focus(); return; }
  /* 分支判据与后端一致：**6 位纯数字 = 动态码**（LDAP 账号可留空），其它 = 授权码（必填 LDAP 账号）。
     ⚠️ 前端这层只是"提前给一句人话"，**不是**校验 —— 最终以后端为准。 */
  const isTotp = /^[0-9]{6}$/.test(code);
  if (!isTotp && !staff) {
    showErr("请把「你的 LDAP 账号」也填上 —— 用授权码进入时需要它来记录是谁在使用");
    $("staff").focus();
    return;
  }
  $("go").disabled = true; $("err").style.display = "none";
  try {
    const r = await fetch("/auth/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      // ⚠️ 双保险：设备标识绝不允许发空值 —— 空的会被后端判「缺少浏览器指纹」，
      //    而用户完全看不懂那句话（2026-10-09 就是这么踩的）。
      body: JSON.stringify({ code, staff_id: staff, fp: FP || getDeviceId() })
    });
    const j = await r.json().catch(()=>({}));
    if (!r.ok) throw new Error(j.detail || ("验证失败（" + r.status + "）"));
    /* ⚠️ 2026-11-02：**先确认令牌真的落到了浏览器，再跳**。
       某些环境（Safari 的 ITP、Firefox 的 ETP、隐私/无痕模式、企业微信等内嵌 WebView）
       会静默丢弃 Set-Cookie。原写法直接 location.replace("/")，结果又被门禁弹回登录页 ——
       用户看到的就是「验证码没错、点了却进不去」，而且**一句提示都没有**。
       这里补一次落地校验：失败就明说原因，别再让用户对着登录页猜。 */
    let landed = true;
    try {
      const w = await fetch("/auth/whoami", { cache: "no-store" });
      const wj = await w.json().catch(()=>({}));
      if (wj && wj.ok === false) landed = false;
    } catch (e) { /* 探测本身失败就别拦人 —— 交给下一次导航去验证 */ }
    if (!landed) {
      $("go").disabled = false;
      // 附一段环境自检，让用户（或我们）一眼看出是 HTTPS/Cookie/隐私模式哪一环
      let diag = [];
      try {
        diag.push(location.protocol === "https:" ? "HTTPS ✓" : "非 HTTPS ✗");
        let cookieOK = false;
        try {
          document.cookie = "kb_probe=1; Path=/; SameSite=Lax"
            + (location.protocol === "https:" ? "; Secure" : "") + "; Max-Age=60";
          cookieOK = document.cookie.indexOf("kb_probe=") >= 0;
          document.cookie = "kb_probe=; Path=/; Max-Age=0";
        } catch (e) {}
        diag.push(cookieOK ? "Cookie 可写 ✓" : "Cookie 被禁用 ✗");
      } catch (e) {}
      showErr("验证已通过，但浏览器没有保存登录状态（Cookie 被拦）。"
            + "请退出无痕/隐私模式、允许本站使用 Cookie，或改用 Chrome / Edge 再试。"
            + "（自检：" + diag.join(" · ") + "）");
      return;
    }
    location.replace("/");                     // 验过 → 进站
  } catch (e) {
    showErr(e.message || "验证失败，请重试");
    $("go").disabled = false;
  }
}
/* 「LDAP 账号」的必填提示**随输入联动**：6 位纯数字 = 动态码路径 → **可留空**；
   其它 = 授权码路径 → **必填**（后端 find_or_create_by_staff 也会强制，这里只是提前给句人话）。
   ⚠️ 整段 try/catch —— 它只管文案，**绝不允许**把脚本带崩（会变成"按钮点不动"那种事故）。 */
function syncStaffHint(){
  try {
    const el = $("staffHint"); if (!el) return;
    const isTotp = /^[0-9]{6}$/.test(($("code").value || "").trim());
    el.textContent = isTotp
      ? "（可留空 · 已绑动态令牌的老用户不用填）"
      : "（用授权码时必填 · 用于登记是谁在使用）";
  } catch (e) {}
}
$("code").addEventListener("input", syncStaffHint);
syncStaffHint();
$("go").addEventListener("click", submit);
$("code").addEventListener("keydown", e => { if (e.key === "Enter") { $("staff").value ? submit() : $("staff").focus(); } });
$("staff").addEventListener("keydown", e => { if (e.key === "Enter") submit(); });
</script></body></html>`;
  return new Response(html, {
    status: 401,
    headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" },
  });
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", ...CORS },
  });
}

function offlineResponse() {
  return json(
    { status: "offline", message: "服务不在线，请联系 xxx（运维值班）" },
    503
  );
}

/** 带超时的 fetch，返回 true=服务在线 / false=离线。
 *
 *  ⚠️⚠️ 2026-09-25 重写 —— 这里曾是「公网偶发 503」的根源，记清楚：
 *  实测（小主机上连测 8 次 `ai.域名/healthz`）：走隧道单次要 **1.2 秒**（本机只要 1.5 毫秒），
 *  而且**偶发 502**（8 次中 1 次）。原因是隧道到 Cloudflare 边缘的往返 + 偶发抖动。
 *  而原先的实现：**一次 fetch、超时 2.5 秒，失败就直接判"服务离线"** →
 *  只要隧道抖一下（>2.5s）或偶发一个 502，整个请求就被判成 503「服务不在线」，
 *  用户看到的是"服务挂了"，实际服务好好的。
 *
 *  现在的策略：
 *    ① 首次用 2.5 秒（保持"真离线时快速失败"的体验）；
 *    ② 失败（超时 / 连接错误 / 非 2xx）后，用 **7 秒**放宽超时**再试一次** —— 抖动是瞬时的；
 *    ③ 加 **5 秒结果缓存**，避免同一 isolate 内连续请求反复付这 1.2 秒的隧道开销。
 *  ⚠️ 缓存是 per-isolate 的（Workers 的全局变量不跨 isolate 共享），只做"降低重复探活"用，
 *     不承担正确性 —— 所以缓存为"在线"时若服务刚挂，仍由转发那一步兜底（会返回离线）。
 */
let _probeCache = { at: 0, ok: false };

async function probe(origin, secretKey) {
  const now = Date.now();
  if (_probeCache.at && now - _probeCache.at < PROBE_CACHE_MS) {
    return _probeCache.ok;
  }
  let ok = false;
  for (const timeoutMs of [PROBE_TIMEOUT_MS, PROBE_RETRY_TIMEOUT_MS]) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const headers = {};
      if (secretKey) headers["X-API-Key"] = secretKey;
      const res = await fetch(`${origin}/healthz`, {
        headers,
        signal: controller.signal,
      });
      if (res.ok) { ok = true; break; }
      // 非 2xx：可能是隧道抖动吐的 502 —— 不立刻判死，走下一轮放宽超时再试
    } catch {
      // 超时 / 连接错误：同上
    } finally {
      clearTimeout(timer);
    }
  }
  _probeCache = { at: Date.now(), ok };
  return ok;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    /* ⭐ 2026-11-02：**强制 HTTPS**（必须放在所有路由最前面）。
       根因：登录 Cookie 带 `Secure` 属性，浏览器在 **http 页面**下一律拒收
       ⇒ 用户看到「验证已通过，但浏览器没有保存登录状态」，自检显示
         「非 HTTPS ✗ · Cookie 可写 ✓」。
       为什么不能只在登录接口里处理：页面本身若在 http 下渲染，
       JS 写带 Secure 的 Cookie 会被静默丢弃，用户怎么点都没用。
       所以在这里一刀切：任何 http 请求先 301 到 https，页面一进来就是安全上下文。 */
    if (url.protocol === "http:") {
      return Response.redirect("https://" + url.host + url.pathname + url.search, 301);
    }

    // CORS 预检
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }

    /* ===== 访问认证门禁（2026-09-22）=====
       ⚠️ 必须放在**所有路由之前**：页面是 Cloudflare 上的静态资源，
          只在接口层拦等于白搭（外面照样能打开界面、看到功能与提示词痕迹）。
       只有 /auth/* 放行 —— 不然连登录都没法登。 */
    const authSecret = env.AUTH_SECRET || "";
    // 管理员 uid 列表（2026-09-24 P1）：默认只有 u1（主人本人），可用环境变量 ADMIN_UIDS 覆盖
    const adminUids = String(env.ADMIN_UIDS || "u1").split(",").map(s => s.trim()).filter(Boolean);
    let authUid = "";                      // 门禁验出的身份，供转发时注入 X-Auth-User
    if (authSecret) {                      // 配了密钥才启用；与后端 KB_ACCESS_SECRET 成对开关
      const p = url.pathname;
      // ⚠️ /auth/bind/ 与 /auth/join/ 都要放行：打开这两个页面的人此刻还没有任何凭据
      //    （他们正是来拿凭据的）—— 一个是指定人绑定，一个是群里自助领取。
      const isAuthApi = (p === "/auth/login" || p === "/auth/whoami" || p === "/auth/logout"
                         || p === "/auth/name" || p === "/auth/staff"
                         || p.startsWith("/auth/bind/") || p.startsWith("/auth/join/"));
      // ⭐ 2026-10-28：站点图标必须放行 —— 否则连**登录页**都拿不到 favicon
      //    （请求被 401 成登录页 HTML，浏览器只能显示默认图标）。
      //    图标是公开资源，不含任何业务信息，放行无风险。
      const isIcon = (p === "/favicon.ico" || p === "/apple-touch-icon.png"
                      || /^\/favicon-\d+\.png$/.test(p));
      if (!isAuthApi && !isIcon) {
        const ck = parseCookies(request);
        // ⚠️ 首次访问（一个 Cookie 都没有）是最常见的情况，别把内部原因「令牌格式不对」
        //    当文案甩给用户 —— 那是我们自己的判据，用户看了只会懵。给默认引导语。
        if (!ck[AUTH_COOKIE]) return loginPage("");
        const [ok, why, uid] = await verifyToken(ck[AUTH_COOKIE], ck[AUTH_FP_COOKIE], authSecret);
        if (!ok) return loginPage(why);
        authUid = uid || "";
        // ⚠️ 管理员页面在**网关层**就拦住（2026-09-24 P1）：普通用户看到 404，
        //    比"能打开页面但数据拿不到"更干净，也少暴露一个功能入口。
        if ((p === "/review" || p === "/review.html") && !adminUids.includes(authUid)) {
          return new Response("Not Found", { status: 404, headers: { "Content-Type": "text/plain" } });
        }
      }
    }

    const origin = env.KB_ORIGIN || "";
    if (!origin) {
      return json({ status: "error", message: "网关未配置 KB_ORIGIN，请在 Worker 设置环境变量" }, 500);
    }
    const originKey = env.ORIGIN_API_KEY || "";     // 本机服务侧密钥（对应本机 KB_GATEWAY_API_KEY）
    // 注：原来这里还读过一个 env.GATEWAY_API_KEY（用户侧密钥、页面 JS 内置同一份），
    //     2026-09-24 P0 起**已废弃** —— 前端不再持有任何密钥，不需要再校验它。

    /* ---- 0) 认证接口：单独走一条**透传**通道 ----
       ⚠️ 不能走下面那套 `json(data, res.ok ? 200 : 502)` 包装，那会带来两个 bug（2026-09-22 实测踩到）：
       ① 它 new 一个全新 Response，**原始响应的 Set-Cookie 被吃掉** ——
          现象是登录接口返回 200 看着成功，浏览器却一个 Cookie 都没收到，然后继续被门禁拦，无限循环；
       ② 状态码被压成 200/502 —— 错口令本该 401，前端拿到 502 只能提示"服务异常"，很误导。
       所以这里原样透传状态码与所有 Set-Cookie。 */
    if (url.pathname === "/auth/login" || url.pathname === "/auth/logout"
        || url.pathname === "/auth/whoami" || url.pathname === "/auth/name"
        || url.pathname === "/auth/staff"
        || url.pathname.startsWith("/auth/bind/") || url.pathname.startsWith("/auth/join/")) {
      const online = await probe(origin, originKey);
      if (!online) return offlineResponse();
      // ⚠️ 请求头不能写死 JSON：绑定页那边回的是 text/html，写死会让浏览器把二维码页当 JSON 渲染
      const h = {};
      if (originKey) h["X-API-Key"] = originKey;
      if (request.method === "POST") h["Content-Type"] = "application/json";
      /* ⭐⭐ 2026-11-02：**必须转发 Cookie**（原来漏了，后果不小）。
         Worker 里的 fetch 是**服务端请求**，浏览器不会自动带上 Cookie —— 不显式转发，
         后端就一个 Cookie 都看不到：
           ① `/auth/whoami` 永远返回「令牌格式不对」⇒ 登录成功后的落地自检必定失败，
              用户看到「验证已通过，但浏览器没有保存登录状态」而其实 Cookie 好好的；
           ② `/auth/join/` 的 `kb_claim` 幂等 Cookie **从来没生效过** ⇒
              同一条邀请链接每打开/刷新一次就新建一个身份（线上 202 个受邀身份、
              大量 use_count=0 的「灰度第一批-01…」就是这么堆出来的）。
         实测对照（同一枚有效令牌）：
           · 直连后端带 Cookie → ok:true, uid:u1
           · 经网关带同样的 Cookie → 修复前 ok:false「令牌格式不对」；修复后 ok:true */
      const ck = request.headers.get("Cookie");
      if (ck) h["Cookie"] = ck;
      const res = await fetch(origin + url.pathname, {
        method: request.method,
        headers: h,
        body: request.method === "POST" ? await request.text() : undefined,
      });
      const out = new Response(await res.text(), {
        status: res.status,             // ← 401/410 就是原样，别压成 502
        headers: {
          "Content-Type": res.headers.get("Content-Type") || "application/json; charset=utf-8",
          "Cache-Control": "no-store",  // 绑定页含密钥，绝不能被缓存
          /* ⚠️ 2026-11-02：这里**故意不加 CORS 头**（原先是 `...CORS`）。
             根因：`Access-Control-Allow-Origin: *` 与 `Set-Cookie` 同时出现时，
             Safari（ITP）和 Firefox（ETP）会把响应当成**跨站追踪**处理，进而拦掉
             Set-Cookie —— 现象是「验证通过、点进去又弹回登录页」，且 **Chrome 不复现**
             （这正是用户反馈「只有非 Chrome 不行」的来源）。
             这些接口全是**同源**调用（页面与接口同一个域名），本就不需要任何 CORS 头。 */
        },
      });
      const sc = res.headers.getSetCookie
        ? res.headers.getSetCookie()
        : (res.headers.get("set-cookie") ? [res.headers.get("set-cookie")] : []);
      for (const c of sc) out.headers.append("set-cookie", c);   // ← 令牌能落地全靠这一行
      return out;
    }

    // ---- 1) POST /ask、/summarize、/cases…：探活 + 转发 ----
    if (FORWARD_POST_PATHS.has(url.pathname) && request.method === "POST") {
      /* 2026-09-24 P0：**不再校验前端 Key**。
         原来校验的 GATEWAY_API_KEY 硬编码在页面 JS 里（F12 可得），任何使用者都能复制走，
         进而调用 /cases（拉全量案例库）、/coach（写记录改助手行为）等管理接口。
         现在前端不持有任何密钥：认证完全由上面的 Cookie 门禁负责，
         转发时由网关自己附加 ORIGIN key —— 那是"只有网关和本机知道"的密钥。
         ⚠️ 这一改动必须与前端**同时部署**：前端已不带 Key，若网关仍要求它，/ask 会 401、整站不可用。 */
      const online = await probe(origin, originKey);
      if (!online) return offlineResponse();

      /* ⭐ 2026-10-12（第四~六轮审计 §3.1）：**请求体大小闸门**。
         为什么需要：后端 `CaseIn` 等模型**没有任何长度约束**（截断发生在 `cases._clip`
         的**写入时**），也就意味着 FastAPI 会**先把整个 body 读进内存**再截断；
         Worker 层原先也没有任何 size 检查 ⇒ 一个超大 POST 能一路灌到本机。
         ⚠️ 危害有限（要过门禁；Cloudflare 免费版另有 100MB 上限兜底；9000 只听 127.0.0.1），
            但**早拒绝永远好过晚拒绝** —— 这里用 `content-length` 头，连 body 都不用读。
         ⚠️ 阈值 2MB 的依据：正常最大的是「带日志的 /ask」，实测 34.6 万字日志约 1.4MB；
            本地其实已把日志截到 ±1 小时窗口，所以 2MB 有充足余量。
            （base64 图片不走这里 —— 它走 /upload 且有自己的 6MB 上限，逐张校验。）
         ⚠️ content-length 可能缺失（分块传输）—— 那种情况放过去，交由后端 Pydantic 兜。 */
      /* ⚠️ `/upload` 与 `/upload-zip` 单独放宽到 8MB：两者都是 base64 上传，
         5MB 的文件走 base64 后约 6.7MB，2MB 根本扛不住。
         其余路径仍维持 2MB（见下方原始说明）。 */
      const BIG_BODY_PATHS = new Set(["/upload", "/upload-zip"]);
      const MAX_BODY = BIG_BODY_PATHS.has(url.pathname) ? 8 * 1024 * 1024 : 2 * 1024 * 1024;
      const _cl = parseInt(request.headers.get("content-length") || "0", 10);
      if (_cl > MAX_BODY) {
        return json({ status: "error",
                      message: `请求体过大（${(_cl / 1048576).toFixed(1)}MB，上限 2MB）。` +
                               `若附带日志过长，请缩小到出问题的那几分钟再发。` }, 413);
      }

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), FORWARD_TIMEOUT_MS);
      try {
        // 2026-09-25：先把 body 解析出来，失败返回 400。
        // ⚠️ 原写法 `body: JSON.stringify(await request.json())` 在调用方漏传 body 时
        //    会抛异常 → 被下面的 catch 吞掉 → 返回 503「服务不在线」，
        //    把"请求有问题"误报成"服务挂了"（手工 curl 测试时踩到过，白排查一轮）。
        let payload;
        try {
          payload = await request.json();
        } catch {
          return json({ status: "error", message: "请求体不是合法 JSON（空 body 也会触发）" }, 400);
        }
        const headers = { "Content-Type": "application/json" };
        if (originKey) headers["X-API-Key"] = originKey;
        // ⚠️ 身份由**网关注入**（值取自验签通过的门禁令牌，客户端伪造无效 ——
        //    这里从不用 request.headers 里的同名头，杜绝伪造）2026-09-24 P1
        if (authUid) headers["X-Auth-User"] = authUid;
        const res = await fetch(origin + url.pathname, {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        // ⭐ /ask-stream 是 SSE，**必须流式透传**。
        //    不能像下面那样 `await res.json()` —— 那会等完整响应体才返回，
        //    等于把「后端已开始响应」这个信息吃掉，边缘照样按 100s 掐断。
        //    直接返回 res.body（ReadableStream），边收边转，长任务可跑数分钟。
        if (url.pathname === "/ask-stream") {
          return new Response(res.body, {
            status: res.status || 200,
            headers: {
              ...CORS,
              "Content-Type": "text/event-stream; charset=utf-8",
              "Cache-Control": "no-cache, no-transform",
              "X-Accel-Buffering": "no",
            },
          });
        }
        const data = await res.json();
        // 2026-09-25：**透传后端真实状态码**。原写法 `res.ok ? 200 : 502` 会把后端的
        // 403（权限不足）、400（参数错）一并压成 502 → 前端只提示"服务异常"，
        // 把"你没权限"误导成"服务挂了"。认证通道早已这么做，这里补上。
        return json(data, res.status || (res.ok ? 200 : 502));
      } catch (e) {
        return offlineResponse();
      } finally {
        clearTimeout(timer);
      }
    }

    // ---- 2) GET /healthz：免 Key 转发探活（页面顶部在线状态徽标用） ----
    if (url.pathname === "/healthz" && request.method === "GET") {
      const online = await probe(origin, originKey);
      if (!online) return offlineResponse();
      try {
        const headers = {};
        if (originKey) headers["X-API-Key"] = originKey;
        const res = await fetch(origin + "/healthz", { headers });
        const data = await res.json();
        return json(data, res.ok ? 200 : 502);
      } catch {
        return offlineResponse();
      }
    }

    // ---- 3) GET /img/*：免 Key 转发报障截图（<img src> 带不了鉴权头，靠随机 id 兜底） ----
    if (request.method === "GET" && FORWARD_GET_PREFIXES.some((p) => url.pathname.startsWith(p))) {
      const online = await probe(origin, originKey);
      if (!online) return offlineResponse();
      try {
        const res = await fetch(origin + url.pathname);
        return new Response(res.body, {
          status: res.status,
          headers: {
            ...CORS,
            "Content-Type": res.headers.get("Content-Type") || "application/octet-stream",
            "Cache-Control": "public, max-age=31536000, immutable",
          },
        });
      } catch {
        return offlineResponse();
      }
    }

    // ---- 4) 其余 GET/HEAD：静态页面（public/） ----
    if (request.method === "GET" || request.method === "HEAD") {
      const res = await env.ASSETS.fetch(request);
      return new Response(res.body, {
        status: res.status,
        headers: { ...CORS, ...Object.fromEntries(res.headers) },
      });
    }

    return json({ status: "error", message: "not found" }, 404);
  },
};

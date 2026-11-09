/* POS 智能报障助手 · 前端回归验证（jsdom 真跑一遍页面 + 断言，不是静态检查）

   为什么要它：2026-09-16 的「点生成并发送没反应 / 发送键永久点不动」，
   根因是 msgNode() 引用了未声明的 diagHtml → 每条消息渲染都抛 ReferenceError →
   doSend 在 busy=true 之后崩掉。语法检查、截图法全都看不出来，只有真跑一遍才抓得到。

   运行（改完 source/worker/public/index.html 之后、wrangler deploy 之前跑一次）：
     # 首次：装依赖到隔离工作区，不污染用户环境
     cd /Users/mac/.workbuddy/binaries/node/workspace && npm install jsdom --no-fund --no-audit
     # 每次：
     NODE_PATH=/Users/mac/.workbuddy/binaries/node/workspace/node_modules \
       /Users/mac/.workbuddy/binaries/node/versions/22.22.2-3/bin/node \
       "/Users/mac/工作/示例企业/pos工具/POS_AI/ask-ops/tests/verify_ui.js"

   验证点：
   1) 点「生成并发送」→ 消息真的进了对话、输入框被腾空、busy 被释放（按钮还能再点）
   2) 回答里的 diagram 能渲染成 <img>（曾经因 diagHtml 未定义整条链路崩掉）
   3) 报障卡里填「⑥ 发生时间」后，附加日志不再要求重填时间
   4) 日志上传后出现「已收下」卡片，且可删除
*/
const fs = require("fs");
const { JSDOM } = require("jsdom");

const HTML_PATH = "/Users/mac/工作/示例企业/pos工具/POS_AI/ask-ops/source/worker/public/index.html";
const HTML = fs.readFileSync(HTML_PATH, "utf8");

let pass = 0, fail = 0;
function ok(name, cond, extra) {
  if (cond) { pass++; console.log("  ✅ " + name); }
  else { fail++; console.log("  ❌ " + name + (extra ? "  → " + extra : "")); }
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

function jsonResp(obj) {
  return { ok: true, status: 200, json: async () => obj, text: async () => JSON.stringify(obj) };
}

// 模拟服务端的会话存储：跨 JSDOM 实例共享，用来验证「换台电脑凭 sid 还能拉到」
const store = { sessions: {} };

function makeDom() {
  const calls = [];
  const dom = new JSDOM(HTML, {
    runScripts: "dangerously",
    pretendToBeVisual: true,
    url: "https://ask.example.com/",
    beforeParse(win) {
      win.fetch = async (u, opt) => {
        calls.push({ url: String(u), body: opt && opt.body });
        const url = String(u);
        if (url.includes("/healthz")) return jsonResp({ status: "ok", docs: 123 });
        if (url.includes("/ask")) return jsonResp({
          answer: "先做 A。\n\n| 编码 | 含义 |\n|---|---|\n| RSP:02 | 失败 |",
          sources: [{ file: "a.md", score: 1 }],
          code_sources: [{ file: "x.java", line: 10, score: 1 }],
          diagram: { kind: "timeline", title: "测试时间线", url: "/diagram/abcdef1234567890.svg", svg: "" }
        });
        if (url.includes("/upload")) return jsonResp({ id: "img1", url: "/img/img1" });
        if (url.includes("/feedback")) return jsonResp({ ok: true });
        // 会话：新建 / 同步 / 拉取。⚠️ 顺序要紧："/session/new" 必须先于通用 "/session/" 判断
        if (url.includes("/session/new")) return jsonResp({ sid: "pos-7k2m9xqp4t3a5b", ttl_days: 7 });
        if (url.includes("/session/sync")) {
          const b = JSON.parse(opt.body || "{}");
          store.sessions[b.sid] = { sid: b.sid, title: b.title, created_at: 1,
                                    last_active_at: 2, messages: b.messages || [] };
          return jsonResp({ ok: true, sid: b.sid, messages: (b.messages || []).length });
        }
        if (url.includes("/session/")) {
          const sid = url.split("/session/")[1];
          if (store.sessions[sid]) return jsonResp(store.sessions[sid]);
          return { ok: false, status: 404,
                   json: async () => ({ detail: "会话不存在或已过期（会话仅保留 7 天）" }) };
        }
        return jsonResp({});
      };
      // jsdom 不实现这些浏览器 API，补空实现，避免测试被无关 API 卡住
      win.Element.prototype.scrollIntoView = function () {};
      if (!win.AbortSignal.timeout) win.AbortSignal.timeout = () => new win.AbortController().signal;
      win.HTMLInputElement.prototype.showPicker = function () {};
      // 「继续会话」用 prompt 收 sid、用 confirm 二次确认 —— jsdom 不实现，直接给固定回答
      win.prompt = () => "pos-7k2m9xqp4t3a5b";
      win.confirm = () => true;
    }
  });
  return { dom, win: dom.window, doc: dom.window.document, calls };
}

const modChips = doc => doc.querySelectorAll("#modChips .chip");
const typeChips = doc => doc.querySelectorAll("#typeChips .chip");
// 2026-09-22 起首屏不再是「默认就摆信息卡」，而是先给「📝 报障 / 💬 其他」两个入口 ——
// 凡是要走报障卡的场景，测试都得先点一下「📝 报障」（幂等：已经展开就跳过）
const enterReport = doc => {
  const b = doc.getElementById("optReport");
  if (b) b.click();
  return !!doc.getElementById("icard");
};

(async () => {
  /* ============ 场景 1：报障卡 → 生成并发送 ============ */
  console.log("\n[场景1] 填卡 → 点「生成并发送」");
  {
    const { win, doc, calls } = makeDom();

    // ① 首屏入口（2026-09-22 主人定）：默认先给「报障 / 其他」两块，不再一上来就摆信息卡
    ok("◆ 首屏出现两个入口：报障 / 其他",
       !!doc.getElementById("optReport") && !!doc.getElementById("optOther"));
    ok("◆ 首屏默认**不**摆信息卡（非报障场景不该被卡片拦一道）", !doc.getElementById("icard"));
    ok("旧的纯文本 #fname 已移除", !doc.getElementById("fname"));

    // 2026-09-18 主人反馈「新会话按钮位置怪」→ 从页头挪进输入区工具栏（紧邻发送按钮）
    const ncb = doc.getElementById("newChatBtn");
    ok("「＋ 新报障」按钮存在", !!ncb);
    ok("◆ 它已离开页头、落在输入区工具栏里",
       !!ncb && !ncb.closest("header") && !!ncb.closest(".toolbar"),
       ncb ? (ncb.closest("header") ? "仍在 header 里" : "不在 .toolbar 内") : "按钮缺失");
    if (ncb) {
      // 「＋ 新报障」的语义也升级了：点它直接进报障卡，不再让用户从入口再选一次
      ncb.click();
      ok("◆ 点「＋ 新报障」直接展开报障卡", !!doc.getElementById("icard"));
    }

    // 填卡
    ok("报障卡已渲染", !!doc.getElementById("icard"));
    ok("卡内出现「📎 附加排查日志」入口", !!doc.getElementById("logAddCard"));
    ok("卡内出现日志卡片槽位 #logSlotCard", !!doc.getElementById("logSlotCard"));
    ok("◆ 从首屏「📝 报障」入口进来同样能展开信息卡", enterReport(doc));
    modChips(doc)[0].click();           // ① 涉及功能
    typeChips(doc)[1].click();          // ② 终端类型 = 小POS
    doc.getElementById("fVer").value = "5.7.55";
    doc.getElementById("fDesc").value = "连着几笔交易失败";
    // ⑥ 发生时间（datetime-local）——这一步以前不会同步到 _reportTime
    const fTime = doc.getElementById("fTime");
    fTime.value = "2026-09-16T11:10";
    fTime.dispatchEvent(new win.Event("input", { bubbles: true }));

    ok("卡里填了时间后，底部时间徽标显示出来（=已同步成全页时间源）",
       doc.getElementById("timeVal").textContent.includes("09-16 11:10"),
       "timeVal=" + JSON.stringify(doc.getElementById("timeVal").textContent));

    // 点生成并发送
    doc.getElementById("icSend").click();
    ok("点击后输入框的内容被立刻取走（不再滞留）", doc.getElementById("input").value === "",
       "input=" + JSON.stringify(doc.getElementById("input").value));
    ok("用户消息气泡已出现", doc.querySelectorAll(".msg.user").length === 1);

    await sleep(60);   // 等 fetch 走完

    ok("AI 回答气泡已出现", doc.querySelectorAll(".msg.ai").length === 1);
    const t = doc.querySelector("#typingMsg");
    ok("「正在分析」气泡已撤掉", !t);

    // 核心回归：diagHtml 未定义曾让整个渲染抛错、busy 卡死
    const diagImg = doc.querySelector(".diagwrap img");
    ok("回答里的简图渲染成 <img>", !!diagImg, diagImg ? "" : "没找到 .diagwrap img");
    if (diagImg) ok("简图 src 指向 /diagram/…", diagImg.getAttribute("src") === "/diagram/abcdef1234567890.svg",
                    diagImg.getAttribute("src"));
    ok("表格被渲染成真 <table>", !!doc.querySelector(".msg.ai .md-tbl"));

    // busy 是否释放：再打字，按钮必须能重新亮起
    const sendBtn = doc.getElementById("sendBtn");
    const inp = doc.getElementById("input");
    inp.value = "再问一次";
    inp.dispatchEvent(new win.Event("input", { bubbles: true }));
    ok("◆ busy 已释放：重新打字后发送键恢复可点（原 bug 的根因）", sendBtn.disabled === false,
       "disabled=" + sendBtn.disabled);

    ok("/ask 请求真的发出去了", calls.some(c => c.url.endsWith("/ask")));
    win.close();
  }

  /* ============ 场景 2：卡里已有时间 → 附日志不该再要时间 ============ */
  console.log("\n[场景2] 卡里填了时间 → 附加日志");
  {
    const { win, doc } = makeDom();
    enterReport(doc);                          // 首屏入口 → 先展开报障卡
    const fTime = doc.getElementById("fTime");
    fTime.value = "2026-09-16T11:10";
    fTime.dispatchEvent(new win.Event("input", { bubbles: true }));

    const log = [
      "2026-09-16 11:05:00 INFO  connect ok",
      "2026-09-16 11:10:00 ERROR queryOrgPayNew1 RSP:02 PAYERROR",
      "2026-09-16 11:10:01 ERROR 订单 0007 支付失败",
      "2026-09-16 13:40:00 INFO  无关的其它日志",
    ].join("\n");
    const file = new win.File([log], "2026-09-16.txt", { type: "text/plain" });
    const li = doc.getElementById("logFile");
    Object.defineProperty(li, "files", { value: [file], configurable: true });
    li.dispatchEvent(new win.Event("change", { bubbles: true }));

    await sleep(80);

    const errline = doc.getElementById("errline");
    ok("◆ 没有弹「请补填发生时间」（这是主人反馈的那次多余打扰）",
       errline.style.display !== "block",
       "errline=" + JSON.stringify(errline.textContent));

    const card = doc.querySelector("#logSlotCard .attach");
    ok("卡内出现「已收下」日志卡片", !!card);
    if (card) {
      const txt = card.textContent;
      ok("卡片显示文件名", txt.includes("2026-09-16.txt"));
      ok("卡片显示截取结果（取 N/M 行）", /取\s*\d+\/\d+\s*行/.test(txt), txt);
      ok("卡片不是「待补时间」态", !card.classList.contains("wait"));
    }
    ok("footer 槽位同步渲染（一份状态两个槽位）", !!doc.querySelector("#logSlotFoot .attach"));

    // 改了「发生时间」→ 日志应按新窗口重截（不必重选文件）
    const tookBefore = (doc.querySelector("#logSlotCard .attach .sub").textContent.match(/取\s*(\d+)\//) || [])[1];
    fTime.value = "2026-09-16T13:40";
    fTime.dispatchEvent(new win.Event("change", { bubbles: true }));
    await sleep(80);
    const tookAfter = (doc.querySelector("#logSlotCard .attach .sub").textContent.match(/取\s*(\d+)\//) || [])[1];
    ok("◆ 改时间后按新窗口自动重截（不用重选文件）", tookBefore !== tookAfter,
       `11:10 取 ${tookBefore} 行 → 13:40 取 ${tookAfter} 行`);

    // 可删除
    const rm = doc.querySelector("#logSlotCard .rm");
    ok("卡片带删除按钮", !!rm);
    if (rm) {
      rm.click();
      ok("删除后卡片消失（卡内 + 底部都清空）",
         !doc.querySelector("#logSlotCard .attach") && !doc.querySelector("#logSlotFoot .attach"));
    }
    win.close();
  }

  /* ============ 场景 3：没填时间就传日志 → 收下 + 提示补时间 + 补完自动接上 ============ */
  console.log("\n[场景3] 没填时间就传日志");
  {
    const { win, doc } = makeDom();
    enterReport(doc);                          // 首屏入口 → 先展开报障卡
    const log = "2026-09-16 11:10:00 ERROR 失败\n";
    const file = new win.File([log], "no-time.txt", { type: "text/plain" });
    const li = doc.getElementById("logFile");
    Object.defineProperty(li, "files", { value: [file], configurable: true });
    li.dispatchEvent(new win.Event("change", { bubbles: true }));
    await sleep(40);

    const wait = doc.querySelector("#logSlotCard .attach.wait");
    ok("出现「已收下，待补时间」卡片", !!wait, wait ? "" : "没找到 .attach.wait");
    ok("提示语指向「⑥ 发生时间」", doc.getElementById("errline").textContent.includes("⑥ 发生时间"),
       doc.getElementById("errline").textContent);

    // 卡里补时间 → 应该自动接着截取
    const fTime = doc.getElementById("fTime");
    fTime.value = "2026-09-16T11:10";
    fTime.dispatchEvent(new win.Event("input", { bubbles: true }));
    await sleep(700);   // input 路径走 500ms 防抖兜底

    const done = doc.querySelector("#logSlotCard .attach");
    ok("◆ 补完时间后自动完成截取（不用重选文件）",
       !!done && !done.classList.contains("wait"),
       done ? done.textContent : "没有卡片");
    win.close();
  }

  /* ============ 场景 4：会话 ID（跨设备续聊） ============ */
  console.log("\n[场景4] 会话 ID：同步 → 换台电脑拉取");
  {
    store.sessions = {};                                   // 干净的模拟服务端
    const { win, doc } = makeDom();
    const sidBtn0 = doc.getElementById("sidBtn");
    ok("「📥 继续会话」按钮存在", !!doc.getElementById("joinBtn"));
    ok("会话 ID 按钮落在「复制给人工」那一行（同属交接场景）",
       !!sidBtn0 && !!sidBtn0.closest("#pickBar"));
    ok("一开始还没有会话 ID（按钮隐藏）", !!sidBtn0 && sidBtn0.hidden === true);

    // 走真实路径：填卡 → 发送 → 触发首次同步
    enterReport(doc);                                   // 先过首屏入口
    doc.querySelectorAll("#modChips .chip")[0].click();
    doc.querySelectorAll("#typeChips .chip")[1].click();
    doc.getElementById("fVer").value = "5.7.55";
    doc.getElementById("fDesc").value = "会话同步测试";
    const ft = doc.getElementById("fTime");
    ft.value = "2026-09-16T11:10";
    ft.dispatchEvent(new win.Event("input", { bubbles: true }));
    doc.getElementById("icSend").click();
    await sleep(150);

    const sidBtn = doc.getElementById("sidBtn");
    ok("◆ 首次发送后拿到会话 ID 并显示出来", !!sidBtn && !sidBtn.hidden,
       sidBtn ? sidBtn.textContent : "按钮不存在");
    const saved = store.sessions["pos-7k2m9xqp4t3a5b"];
    ok("◆ 会话已同步到服务端", !!saved, JSON.stringify(Object.keys(store.sessions)));
    ok("同步上去的消息数与页面一致",
       (saved ? saved.messages.length : -1) === doc.querySelectorAll(".msg").length,
       `服务端 ${saved ? saved.messages.length : "?"} / 页面 ${doc.querySelectorAll(".msg").length}`);
    ok("sid 已记进 localStorage", !!win.localStorage.getItem("pos_ask_sid_v1"));
    win.close();

    // 「另一台电脑」：全新实例、空 localStorage，只有会话 ID
    const { win: w2, doc: d2 } = makeDom();
    ok("新机器上确实是空的", d2.querySelectorAll(".msg").length === 0,
       `实际 ${d2.querySelectorAll(".msg").length} 条`);
    d2.getElementById("joinBtn").click();
    await sleep(150);
    ok("◆ 凭会话 ID 把整段会话拉了回来", d2.querySelectorAll(".msg").length === 2,
       `实际 ${d2.querySelectorAll(".msg").length} 条`);
    ok("拉回来的内容对得上", (d2.getElementById("feed").textContent || "").includes("会话同步测试"));
    ok("新机器上也记住了这个 sid", !!w2.localStorage.getItem("pos_ask_sid_v1"));
    w2.close();

    // 拉一个不存在的会话 → 必须如实提示，不能误导用户去「找回」
    const { win: w3, doc: d3 } = makeDom();
    w3.prompt = () => "pos-00000000000000";
    d3.getElementById("joinBtn").click();
    await sleep(150);
    ok("◆ 会话不存在/已过期时给出明确提示",
       (d3.getElementById("toast").textContent || "").includes("已过期"),
       d3.getElementById("toast").textContent);
    w3.close();
  }

  /* ============ 场景 5：自由输入（不点报障、直接打字） ============ */
  console.log("\n[场景5] 其他：不点报障，直接打字 → mode=free");
  {
    const { win, doc, calls } = makeDom();
    ok("首屏是入口卡片，且没摆信息卡",
       !!doc.getElementById("optReport") && !doc.getElementById("icard"));

    doc.getElementById("optOther").click();
    ok("◆ 点「其他」后入口消失、直接可输入", !doc.getElementById("entry"));
    ok("点「其他」不会误开信息卡", !doc.getElementById("icard"));

    const inp = doc.getElementById("input");
    inp.value = "0100000000260101000001 这个单号的手机号是多少";
    inp.dispatchEvent(new win.Event("input", { bubbles: true }));   // ⚠️ 不触发 input，发送键仍是禁用态
    doc.getElementById("sendBtn").click();
    await sleep(150);

    const sent = calls.filter(c => String(c.url).includes("/ask")).pop();
    const body = sent ? JSON.parse(sent.body) : {};
    ok("◆ 自由输入带上了 mode=free（后端据此形状自适应）",
       body.mode === "free", `实际 mode=${JSON.stringify(body.mode)}`);
    ok("自由输入发的是原文，没被套上「【POS 报障】」卡格式",
       !String(body.question || "").includes("▍涉及功能"),
       String(body.question || "").slice(0, 40));
    win.close();
  }

  console.log(`\n结果：${pass} 通过 / ${fail} 失败`);
  process.exit(fail ? 1 : 0);
})();

"""运维知识库 AI 问答 —— 本机服务（只监听 127.0.0.1，公网入口走 Cloudflare Worker）。

接口：
  GET  /healthz   探活（Worker 靠它判断在线/离线）
  POST /ask       提问 {question, history?, images?} -> {answer, sources, code_sources?}
  POST /upload    报障截图存本机 {dataUrl} -> {id, url}（前端再把 id 放进 /ask 的 images）
  GET  /img/{id}  读回报障截图（页面 <img src> 用；id 为随机 32 位，等同不可猜的凭证）
  POST /feedback  上报/回写一个案例（已解决-未解决、审查状态）-> {id, ...}
  POST /cases     案例审查页拉取：{cases, coach, stat}
  POST /coach     写一条教练记录（正确的问题点 / 处理方式 / 线索），此后同类提问会带上它
  POST /coach/active  停用/启用某条教练记录
  POST /retry     用教练记录重跑一个问题，验证「教」的效果
  POST /reload    重扫知识库（KB 文件有增删改后调用）
  POST /code-rebuild  强制重建源码索引（首次/源码大幅变更后调用）
启动：./start.command  或  cd source && .venv/bin/python -m uvicorn app.main:app --port 9000
"""
import base64
import functools
import html as htmllib
import io
import json
import logging
import re
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import Cookie, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .totp_asset import TOTP_APP_QR, TOTP_APP_URL   # 验证器 App 下载二维码 + 下载页地址（内嵌，见该文件说明）
from pydantic import BaseModel, Field

from . import alerts, auth, cases, codesearch, config, intent, knowledge, llm, queue, sessions, totp

app = FastAPI(title="运维知识库问答(本机)", version="1.0.0")


# ---- access log 降噪：丢掉「成功的 /healthz」------------------------------------
# 实测（2026-10-09）：service.log 共 21893 行，其中 **16739 行是 /healthz（76.5%）** ——
#   云端网关每 30 秒探活一次，把真正有用的 `[agent]` / `[tools]` / 审计轨迹全淹了。
#   当天排查那 8 次 `POST /auth/login 500` 时为此刻意翻了几千行无关记录。
# ⚠️ **只丢 2xx 的成功探活**：非 200 的 healthz 一律保留 ——
#   那是「服务挂了 / 隧道断了」的第一现场，正是最需要看到的行。别一刀切成过滤掉整个路径。
# ⚠️ 这是"日志可读性"层面的改动，不影响任何业务行为；要临时看全部探活就 tail 网关侧日志。
class _HealthzNoiseFilter(logging.Filter):
    """丢掉「成功的 /healthz」探活记录。

    ⚠️ 2026-10-09 第一版写错了、**没生效**（线上实测：打完 healthz 日志仍多一行）：
       原来判的是 `" 200 " in msg`，但 uvicorn 的访问日志是
       `logger.info('%s - "%s %s HTTP/%s" %d', ...)` —— `record` 里**只有数字码**，
       末尾那个 `" OK"` 是 `AccessFormatter` 格式化时才补的 ⇒ `getMessage()` 得到的串
       结尾是 `... HTTP/1.1" 200`（**后面没有空格**），于是永远匹配不上。
       现在优先读 `record.status_code`（uvicorn 通过 extra 挂在 record 上），
       拿不到再用正则兜，两种都能work。
    """

    _CODE_RE = re.compile(r'"\s(\d{3})(?!\d)')

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:                       # noqa: BLE001
            return True                         # 解析不了就别拦
        if "/healthz" not in msg:
            return True
        code = getattr(record, "status_code", None)
        if code is None:
            m = self._CODE_RE.search(msg)
            code = int(m.group(1)) if m else None
        if code is None:
            return True                         # 判不出状态码 → 保留，宁可吵也别漏
        return not (200 <= int(code) < 300)     # ⚠️ 只有 2xx 才丢；5xx/503 是"服务挂了"的第一现场


logging.getLogger("uvicorn.access").addFilter(_HealthzNoiseFilter())
print("[log] 已挂载 access-log 降噪过滤器（只丢 2xx 的 /healthz）", flush=True)


@app.on_event("startup")
async def _warm_intranet_status():
    """启动预热（2026-10-12 第三轮评审 P1 改造）。

    ① 内网通道状态预热（后台线程）—— 让页面首帧就能看到「数据在线/离线」，
       而不是先显示「数据检测中」等 25 秒。
    ② ⚠️ 知识库 + 源码索引的加载**移到这里**（原先写在模块级，见下）。
    ③ ⭐ 线程池上限显式化（见下）。

    为什么要搬：`knowledge.reload()` / `codesearch.load()` 原先是**模块顶层调用**
    （夹在 middleware 与 class 定义之间）——
      · 它们在 **import 阶段同步阻塞**（codesearch.load 加载 38MB pickle，8~9 秒），
        早于 uvicorn 完成启动 ⇒ 表现为"服务启动慢"，但读 startup 函数只看得到一行 warmup，
        **容易误判启动很快**；
      · 模块级抛错 = **整个模块 import 失败 = 服务起不来**；
        放在 startup 里则**可控降级**（索引坏了不该让助手整体不可用）。
    """
    # ③ 线程池上限：/ask 是**同步**函数 ⇒ FastAPI 丢进 AnyIO 线程池（默认 40）。
    #    排队 = 阻塞占一个线程（不是轻量等待）⇒ 真实容量是"3 在跑 + ≤37 在等"，
    #    而 37 个线程被占满会让其他同步接口（/cases、/admin/*、/upload）也拿不到线程。
    #    降到 10：让**排队容量变得明确可见**（超出快速回绝 > 默默占线程等 100 秒）。
    #    ⚠️ 必须 > MAX_CONCURRENT_ASK(3)，否则连"正在跑的"都排不上。
    #    ⚠️ 必须在**事件循环内**改（anyio 的 limiter 是 loop-bound）。
    try:
        from anyio.to_thread import current_default_thread_limiter
        _lim = current_default_thread_limiter()
        _want = int(getattr(config, "THREAD_POOL_SIZE", 10) or 0)
        if _want > 0:
            _lim.total_tokens = _want
            print(f"[boot] 线程池上限设为 {_want}（原 {40}）", flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f"[boot] 线程池上限设置失败（不影响功能）：{type(e).__name__}", flush=True)

    try:
        knowledge.reload()
        print(f"[boot] 知识库已加载：{knowledge.count()} 段", flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f"[boot] 知识库加载失败（降级：检索功能不可用）：{type(e).__name__}: {e}",
              flush=True)
    try:
        codesearch.load()
    except Exception as e:                       # noqa: BLE001
        print(f"[boot] 源码索引加载失败（降级：search_code 不可用）：{type(e).__name__}: {e}",
              flush=True)


@app.middleware("http")
async def _revoke_guard(request: Request, call_next):
    """令牌吊销守卫（2026-09-25）。

    ⚠️ 为什么后端还要查一遍：设备令牌在**网关离线验签**（只验签名、不回源查库），
       所以「用户表里已经没有这个人」并不能让他的令牌失效 —— 签名照样验得过，7 天内照用。
       这里按「显式吊销名单 → 已停用 → 已删除」三条判据把它拦住，落实「停用即失效」。
    ⚠️ 只拦**带身份**的请求（网关注入 X-Auth-User 才会走到这）；/healthz、/img/* 等
       免身份路径不受影响。
    """
    uid = (request.headers.get("X-Auth-User") or "").strip()
    if uid and auth.is_revoked(uid):
        return JSONResponse(
            {"detail": "该账号已停用或已失效，请联系管理员重新绑定"},
            status_code=401,
        )
    return await call_next(request)


# ⚠️ 2026-10-12：原先这里是模块级 `knowledge.reload()` + `codesearch.load()`
#   —— 已移进上面的 `_warm_intranet_status`（startup 事件，带降级）。
#   **别搬回来**：模块级加载会让 import 阻塞 8~9 秒、且加载失败直接起不来服务。


class AskIn(BaseModel):
    question: str
    history: list[dict] | None = None  # [{"role": "user"|"assistant", "content": "..."}]
    # 首屏入口选的模式（2026-09-22 起）：
    #   "report" = 走了报障信息卡；"free" / None = 自由输入（查单号、问配置、随便问）
    # ⚠️ free 模式下回答形状要**自适应**（一句话能答完的别硬套三段式），见 llm.py 的 mode 分支
    mode: str | None = None
    # 报障截图。每项二选一：
    #   · 图片 id（如 "a1b2….jpg"）—— 先经 POST /upload 存到本机，常规路径；
    #   · data:image/...;base64,…    —— 直传兜底（上传失败时前端降级用）。
    images: list[str] | None = None
    # 排队用：前端在**发起请求前**生成的随机 id（配合 GET /queue-status/<rid> 查"我排第几"）。
    # ⚠️ 为什么是前端生成：服务端分配票号要等请求进来才知道，而用户提交后就要看到排队位置，
    #    所以 id 必须在发出请求之前就存在。
    rid: str | None = None


class UploadIn(BaseModel):
    dataUrl: str
    name: str | None = None  # 原始文件名，仅用于日志，不参与存储


class SummarizeIn(BaseModel):
    # 用户勾选要复制的对话消息 [{role, content}]
    messages: list[dict] | None = None


class CaseDelIn(BaseModel):
    """删除案例：只要 id。"""
    id: str


class CaseIn(BaseModel):
    """案例上报/回写。字段可缺省：只带 id + status 也能做增量更新。

    ⚠️ 2026-10-12（第四~六轮审计 §3.1/§3.3）：**加长度与格式约束**。
    此前这些字段**完全无约束**，截断只发生在 `cases._clip` 的**写入时** ——
    意味着 FastAPI 会先把整个 body 读进内存（几百 MB 也能收）。
    现在由 Pydantic 在**解析时**拒绝超长值（返回 422），比"先收后截"更早、更省。
    ⚠️ 上限取得比 `cases._clip` 的实际截断值**略宽**（留 10~20% 余量）：
    正常调用不该被拒，只有明显异常的巨型字段才需要挡。
    """
    # ⭐ id 加**格式约束**（审计 §3.3）：它是 `upsert_case` 的合并键 ⇒ 可覆盖已有案例。
    #   前端生成的是 base36（`Date.now().toString(36) + random`），这里限制为 8~64 位
    #   字母数字 + `-`/`_`，既兼容前端格式，又挡住"构造任意字符串当 id"的滥用。
    id: str | None = Field(default=None, max_length=64,
                           pattern=r"^[A-Za-z0-9_-]{8,64}$")
    ts: int | None = None
    question: str | None = Field(default=None, max_length=8000)
    answer: str | None = Field(default=None, max_length=24000)
    verdict: str | None = Field(default=None, max_length=16)
    via: str | None = Field(default=None, max_length=24)
    status: str | None = Field(default=None, max_length=16)
    note: str | None = Field(default=None, max_length=1500)
    tags: list[str] | None = Field(default=None, max_length=20)
    meta: dict | None = None
    imgs: list | None = Field(default=None, max_length=20)
    # ⚠️⚠️ 2026-10-12（第四~六轮审计时发现的一个真 bug）：
    #   前端 `reportCase()` 一直在发 `user` 字段（网关注入的身份，用于「使用统计」），
    #   但**这个模型原先没有声明它** —— 而 `feedback()` 用的是
    #   `body.model_dump(exclude_none=True)`，**只 dump 模型里声明过的字段**
    #   ⇒ 前端传的 user 被**静默丢弃** ⇒ 线上 77 条案例 `user` 全为空
    #   ⇒ 上午刚上线的「使用人数」统计**永远是 0**（因为一条身份都存不下来）。
    #   修法：显式声明该字段（顺带加长度约束）。
    #   ⚠️ 这个坑的教训：**用 Pydantic 模型收请求体时，前端新增字段必须在模型里同步声明**，
    #      否则会被静默吞掉、且不报错 —— 极难发现（前端看起来"发成功了"）。
    user: str | None = Field(default=None, max_length=64)


class CoachIn(BaseModel):
    case_id: str | None = None
    question: str | None = None
    raw: str | None = None              # ⭐ 单框模式（2026-10-29）：整段教练内容，优先于下面三段
    point: str | None = None            # 正确的问题点（旧三段式，保留兼容）
    howto: str | None = None            # 正确的处理方式
    hint: str | None = None             # 给助手的线索（看什么日志/源码/字段）
    tags: list[str] | None = None
    intents: list[str] | None = None     # ⭐ 适用场景（2026-10-29）：声明后按意图精确匹配，不再靠字面
    verdict: str | None = None          # 与该案例一起回写的结论


class CoachActiveIn(BaseModel):
    id: str
    active: bool = False


class RetryIn(BaseModel):
    question: str
    coach_id: str | None = None         # 指定要强制带上的一条教练记录（验证用）


# ---- 报障图片：存取与解析 ----
_IMG_MIME = {"image/jpeg": ".jpg", "image/png": ".png",
             "image/webp": ".webp", "image/gif": ".gif"}
_IMG_EXT_MIME = {v: k for k, v in _IMG_MIME.items()}
_IMG_NAME_RE = re.compile(r"^[0-9a-f]{32}\.(jpg|png|webp|gif)$")
# 简图文件名：SVG 内容的 sha1 前 16 位（tools._persist_diagram 生成）
_DIAG_NAME_RE = re.compile(r"^[0-9a-f]{16}\.svg$")
MAX_IMAGES = 4   # 单次提问最多带几张图（多图会显著增加 token 与耗时）

# ── 内联图片兜底守卫（2026-10-27）──────────────────────────────
# 背景：前端报障截图会先压到「最长边 1400 / JPEG 0.82」（线上实测平均 52KB），
#   但 canvas 压缩失败时（Safari 大图限制等）前端会回退成 `resolve(out || fr.result)`，
#   即**原图 base64**。而 _resolve_images 原先对 data: 开头的内容不做任何检查就直送模型
#   ⇒ 一张 5MB 截图变成 6.7MB base64，单次就能把 prompt 顶到几十万 token（甚至超窗口）。
# 策略：超限的内联图能压就压（Pillow 为可选依赖），压不了就丢弃并告警 —— 绝不拖垮整次提问。
#   正常路径（/upload 存盘 → 只传 id）完全不受影响。
INLINE_MAX_CHARS = 900 * 1024   # base64 字符数，≈675KB 原图；正常压缩截图约 70KB，远低于此
_IMG_EDGE = 1400                # 与前端 index.html 的 IMG_MAX_EDGE 保持一致


def _shrink_data_url(s: str):
    """把超限的 data:image/...;base64 缩到 _IMG_EDGE / JPEG 0.82。失败返回 None。"""
    try:
        import io
        from PIL import Image
    except Exception:
        return None
    try:
        _, b64 = s.split(",", 1)
        raw = base64.b64decode(re.sub(r"\s+", "", b64), validate=True)
        im = Image.open(io.BytesIO(raw))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        sc = min(1.0, _IMG_EDGE / max(w, h))
        if sc < 1.0:
            im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=82, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()
    except Exception:
        return None


def _resolve_images(items: list | None) -> list[str]:
    """把 /ask 里的 images 统一解析成 data URL 列表，供多模态调用。

    - 以 data: 开头的直接用（前端直传兜底）；
    - 其余当作本机存储的图片 id 去 UPLOAD_DIR 读盘并 base64（常规路径）。
    解析不出来的一律跳过，不让「图丢了」把整次提问打挂。
    """
    out: list[str] = []
    for it in (items or [])[:MAX_IMAGES]:
        s = str(it or "").strip()
        if not s:
            continue
        if s.startswith("data:image/"):
            # 超限的内联图先压缩（前端 canvas 失败时会回退传原图，见 INLINE_MAX_CHARS 注释）
            if len(s) <= INLINE_MAX_CHARS:
                out.append(s)
                continue
            shrunk = _shrink_data_url(s)
            if shrunk:
                logging.warning("[resolve_images] 内联图超限 %d 字符，已压缩至 %d", len(s), len(shrunk))
                out.append(shrunk)
            else:
                logging.warning("[resolve_images] 内联图超限 %d 字符且无法压缩，已丢弃", len(s))
            continue
        name = Path(s).name
        if not _IMG_NAME_RE.match(name):
            continue
        p = config.UPLOAD_DIR / name
        if not p.is_file():
            continue
        mime = _IMG_EXT_MIME.get(p.suffix.lower(), "image/png")
        out.append(f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}")
    return out


# ⚠️ 知识库检索的 query 必须截断：前端「附加日志文件」会把日志片段拼在用户描述之后，
# 不再限制大小（最长约 10 万字符）。若整段丢给 BM25，唯一 token 数暴涨
# （knowledge.search 里 df 是 O(词数×文档数) 计算），既卡顿又会被日志噪声带偏检索结果。
RETRIEVAL_QUERY_MAX = 2000

# ⚠️ 但**光截断不够**（2026-09-16 踩实）：用户自己的描述往往只有两百来字（报障卡），
# 截断到 2000 字等于把**日志正文 1700 字**放进了检索 query。日志是密集的英文/JSON token，
# 会把 BM25 打分整个带跑偏，而它提供不了任何「该查哪篇知识」的信息。
# 实测同一问题（工行设备交易失败 + 附加日志）：
#   · 日志进检索  → 43 段 / 4999 字，最高分 147，还额外灌进《活动预售&POS》13 段；
#   · 日志不进检索 → 30 段 / 4601 字，命中构成明显更贴题。
# 所以：**检索只用「问题部分」，日志原样交给大模型当一手证据**（见 llm.answer(log=…)），
# 两边各取所需 —— 检索要干净，模型要完整。
_LOG_MARK_RE = re.compile(r"（附：报错日志片段[^\n]*）")
# 兜底：用户没走「附加日志文件」而是直接往输入框里粘了一大段日志时，按日志行特征识别。
_LOG_LINE_RE = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}:\d{2}")


def _split_attached_log(text: str) -> tuple[str, str]:
    """把提问文本切成 (问题, 附加日志)。没有附加日志时后半段为空串。

    两条识别路径：
    1. 前端「📎 附加日志文件」拼进来的固定分隔行「（附：报错日志片段（时间窗口：…））」——
       从它开始往后全是日志；
    2. 用户直接粘贴日志的兜底：正文里出现 ≥5 行 `2026-09-16 11:04:37,239 …` 形态的行时，
       从第一行这样的行切开（只在文本够长时启用，避免把一两行示例误判成日志）。
    """
    if not text:
        return text, ""
    m = _LOG_MARK_RE.search(text)
    if m:
        return text[: m.start()].strip(), text[m.end():].strip()
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if _LOG_LINE_RE.match(ln)]
    if len(hits) >= 5 and len(text) > 600:
        i = hits[0]
        return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
    return text, ""


def _check_key(x_api_key: str):
    # 仅当配置了 GATEWAY_API_KEY 时才校验；本机直连(不带头)不受影响
    if config.GATEWAY_API_KEY and x_api_key != config.GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _check_admin(x_auth_user: str = ""):
    """管理类接口的权限校验（2026-09-24 P1）。

    背景：原来前端硬编码着一个网关密钥，任何使用者 F12 就能复制走，
    于是能调 /cases（拉全量案例库）、/coach（写教练记录 = 改助手行为）这类管理接口。

    ⚠️ 现在只认**网关注入的 X-Auth-User**（值取自验签通过的登录令牌）：
    - 网关**从不透传**客户端传来的同名头 → 前端伪造无效；
    - **本机直连**（没有该头）视为管理员 —— 只有本机和网关持有 ORIGIN key，
      能通过 `_check_key` 的本机调用天然可信（本机开发/运维场景）。

    ⚠️ 2026-11-02：`KB_ADMIN_UIDS` 里除了 uid（`u1`），**也可以写员工编号**（`zuoyuan1`）。
       理由：uid 是内部编号，人记不住也容易在不同库里变；员工编号 是用户自己填的、
       跨身份稳定。两种写法同时支持 ⇒ 白名单写 `u1,zuoyuan1` 双保险。
       ⚠️ **绝不能用 name**：用户能自己改名（/auth/name），改叫「左源」就白拿管理员了。
    """
    if not auth.enabled():
        return                       # 访问认证关闭时（本机开发）不做额外限制
    u = (x_auth_user or "").strip()
    if not u:
        return                       # 本机直连：交给 _check_key 把关
    allow = {a.strip() for a in config.ADMIN_UIDS if a.strip()}
    if u in allow:
        return
    if u:
        staff = ""
        for x in auth.load_users():
            if x.get("id") == u:
                staff = auth.norm_staff_id(x.get("staff_id") or "")
                break
        if staff and staff.lower() in {a.lower() for a in allow}:
            return
    raise HTTPException(status_code=403, detail="该操作仅限管理员")


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "ask-ops",
        "time": int(time.time()),
        "docs": knowledge.count(),
        # ⚠️ 2026-10-23 加：**篇数**（去重文件数），供页头副标题写「知识 N 篇」——
        #    `docs` 是切块数（7000+），当"篇"用会差两个数量级。两字段都留。
        "kb_files": knowledge.file_count(),
        "code_lines": codesearch.count_lines(),
        "provider": config.LLM_PROVIDER,
        # 2026-09-28 主人要求：页头状态行「AI 在线 · 知识 X 段」后面加「数据在线」（内网通道状态）。
        # ⚠️ 内网通道状态在本开源版恒为关闭，这里不做任何同步探活：
        #    /healthz 同时是云端网关的探活口（2.5 秒超时），同步等 5 秒会被误判成「服务离线」。
        # 内网取数通道在本开源版中未启用（原实现见 docs/PLUGINS.md）
        "intranet": {"enabled": False, "online": False},
    }


@app.get("/")
def root():
    return {"name": "ask-ops", "usage": "POST /ask {question}",
            "docs_count": knowledge.count(),
            "code_lines": codesearch.count_lines()}


def _with_queue(fn):
    """给 /ask 加排队：领票 → 等空位 → 执行 → 归还。

    ⚠️ 用装饰器而不是改函数体：/ask 的函数体很长，整体缩进会引入大量无意义 diff（也易改错）。
    functools.wraps 会保留原签名，FastAPI 的参数解析照常工作。

    ⚠️ 2026-10-10 修（评审 ⑧）：等待有上限（`KB_QUEUE_WAIT_TIMEOUT`，默认 100s）。
    超上限**快速回一条带位次的人话**，而不是让用户白等到前端 150s 假超时 ——
    那样后端还在跑、结果永远到不了用户手里（还占着并发位）。
    100s 的取值依据：deep 中位 53s / p90 79s，排 2 位 ≈ 最坏 160s —— 诚实告知
    「等太久」比假装在处理好。Worker 侧同步加了转发 AbortSignal（三层超时对齐）。
    """
    wait_cap = int(getattr(config, "QUEUE_WAIT_TIMEOUT", 100))

    def _rid_of(a: tuple, k: dict) -> str:
        """从 wrapper 收到的实参里安全取出 rid（前端用它查"我排第几"）。

        ⚠️ 用 *args/**kw 之后不能再直接 `body.rid` —— FastAPI 可能按位置或关键字传参，
        两种都要罩住；取不到就返回空串（`queue.enter("")` 是合法输入，只是查不了位次）。
        """
        try:
            b = k.get("body") or (a[0] if a else None)
            return str(getattr(b, "rid", "") or "")
        except Exception:                       # noqa: BLE001
            return ""

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        """⚠️ 2026-10-12（评审第二轮 1.1）：**不再手写参数列表**。

        原写法是 `def wrapper(body, x_api_key="", x_auth_user="")` + `return fn(body, ...)` ——
        与 `ask()` 的签名靠"人记得同步"维持。**已经炸过一次**：给 `/ask` 加 `x_auth_user` 时
        漏改 wrapper ⇒ 线上所有 `/ask` 500（`TypeError: unexpected keyword argument`）。
        更隐蔽的是：`x_api_key` 与 `x_auth_user` 都是 `str`，写反了**不会报错**，
        只会把所有请求的鉴权判成失败 —— 比 500 更难查。
        改成 `*args/**kw` 一次性消除该隐患：FastAPI 仍按 `ask()` 的签名做参数解析
        （`functools.wraps` 会设 `__wrapped__`，`inspect.signature` 会跟随它）。
        """
        ticket = queue.enter(_rid_of(args, kw))
        try:
            got = queue.acquire(timeout=wait_cap)
            if not got:
                ahead = queue.ahead_of(ticket)
                print(f"[queue] 等待超上限（{wait_cap}s），前面还有 {ahead} 位 → 快速回绝",
                      flush=True)
                return JSONResponse(
                    status_code=200,
                    content={
                        "answer": (
                            f"现在用的人比较多，你前面还有 {ahead} 位在排队，"
                            f"预计还要等 {ahead * 60} 秒左右 —— 这条不用重发，"
                            "稍等一两分钟后**再问一次**即可（现在继续排队也轮不到你，"
                            "只会白等一场）。急着处理的话，可以先把报障信息发给人工。"
                        ),
                        "queued_out": True,
                        "agent": {"steps": 0, "depth": "queue"},
                    },
                )
            try:
                # ⚠️ 2026-10-12：改成 *args/**kw 透传 —— 不再需要手动同步参数列表
                #   （原写法漏一个形参就 500，已踩过）。
                return fn(*args, **kw)
            finally:
                queue.release()
        finally:
            queue.leave(ticket)
    return wrapper


@app.get("/queue-status")
@app.get("/queue-status/{rid}")
def queue_status(rid: str = ""):
    """排队快照（免鉴权，只回数字 —— 不含任何业务数据）。

    ⚠️ 走 path 而不是 query：网关转发 GET 时用的是 `origin + url.pathname`，
    **query 会被丢掉**（worker.js），所以 rid 必须放路径里。
    """
    return queue.snapshot_by_rid(rid)


class AdminUidIn(BaseModel):
    uid: str = ""
    reason: str = ""


class AdminRenameIn(BaseModel):
    """改名请求（2026-11-02）：把老邀请的匿名名改成人名。"""
    uid: str = ""
    name: str = ""


class AdminUsageIn(BaseModel):
    days: int = 0        # 0 = 用默认（KB_USAGE_DAYS，默认 15）


@app.post("/admin/usage")
def admin_usage(body: AdminUsageIn | None = None,
                x_api_key: str = Header(default=""),
                x_auth_user: str = Header(default="")):
    """管理面板「使用统计」：最近 N 天（默认 15）的用量与解决情况。

    ⚠️ 与 /admin/status 同样用 POST（GET 拿不到网关注入的 X-Auth-User，会被当本机直连放行）。
    ⚠️ 非管理员 → 403。
    ⚠️ 2026-10-12（第六轮评审）：**补上 `_check_key`** —— 原先它只有 `_check_admin`，
       而 `_check_admin` 对"无身份头"（本机直连）是**直接放行**的，注释里写着
       「本机直连：交给 `_check_key` 把关」—— **但这里根本没调**，等于那条防线不存在。
       实测（本机无 Key 无身份）：`/admin/status`、`/admin/usage` 返回 **200**，
       而 `/cases`、`/coach` 返回 401 —— 同一批管理接口鉴权口径不一致。
       （实际暴露面有限：9000 只监听 127.0.0.1，公网打不到；但**口径必须统一**，
         否则将来一旦监听地址或反代配置变化，就会直接漏数据。）
    """
    _check_key(x_api_key)
    _check_admin(x_auth_user)
    days = int(getattr(body, "days", 0) or 0) if body else 0
    try:
        return {"ok": True, **cases.usage_stats(days=days, users=auth.load_users())}
    except Exception as e:                       # noqa: BLE001
        print(f"[admin] usage 统计失败：{type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"统计失败：{e}")


@app.post("/admin/status")
def admin_status(x_api_key: str = Header(default=""),
                 x_auth_user: str = Header(default="")):
    """管理员面板数据（内网通道 / 排队 / 系统 / 用户列表）。

    ⚠️ 用 POST 而不是 GET：网关只在 **POST 转发时注入 `X-Auth-User`**（worker.js），
    GET 不带这个头，后端会当成"本机直连"直接放行 —— 那就等于把管理数据公开了。

    ⚠️ 非管理员 → 403（由 `_check_admin` 抛）。前端据此判断要不要显示「⚙ 管理」按钮：
    **前端隐藏只是体验，真正的权限在这一层。**

    ⚠️ 2026-10-12（第六轮评审）：**补上 `_check_key`**（原因同 `/admin/usage`）——
    原先只有 `_check_admin`，而它对"无身份头"是直接放行的，等于注释里承诺的
    「本机直连交给 _check_key 把关」这条防线**不存在**。与 `/cases`、`/coach` 统一口径。
    """
    _check_key(x_api_key)
    _check_admin(x_auth_user)
    users = []
    for u in auth.load_users():
        dev = auth.device_summary(u)
        users.append({
            "uid": u.get("id") or "",
            "name": u.get("name") or "",
            # 员工编号（只记录不校验）：管理页靠它认人 —— 名字可能是匿名邀请名，
            # 而 staff_id 是用户**自己填的**，一眼能看出是谁在用。
            "staff_id": auth.norm_staff_id(u.get("staff_id") or ""),
            "auth_via": u.get("auth_via") or ("invite" if u.get("from_invite") else ""),
            "enabled": bool(u.get("enabled")),
            "revoked": auth.is_revoked(u.get("id") or ""),
            "last_used_at": u.get("last_used_at") or 0,
            "use_count": u.get("use_count") or 0,
            # ⭐ 2026-11-02：设备统计（按 fp8 去重）。ids 给前端做「同名分组去重」用，
            #   否则同一台设备被两条记录记到时会被算两次。
            "devices": dev["count"],
            "device_last_at": dev["last_at"],
            "device_ids": dev["ids"],
            # 名字是不是老流程自动生成的（`灰度第一批-01`）—— 前端据此提示「这个名字要补」
            "name_auto": auth.looks_anonymous(u.get("name") or "") or u.get("name_source") == "auto",
        })
    users.sort(key=lambda x: (not x["enabled"], -(x["last_used_at"] or 0)))
    return {
        "ok": True,
        "me": x_auth_user or "",
        "intranet": {"enabled": False, "online": False},
        "queue": queue.snapshot(),
        "system": {
            "docs": knowledge.count(),
            "code_lines": codesearch.count_lines(),
            "provider": config.LLM_PROVIDER,
            "model": getattr(config, "OPENAI_MODEL", ""),
        },
        "users": users,
        "revoked": auth.load_revoked(),
        "invites": [
            {
                "id": it.get("id") or "",
                "note": it.get("note") or "",
                "exp": it.get("exp") or 0,
                "enabled": bool(it.get("enabled", True)),
                "claimed": len(it.get("claimed") or []),
                # ⚠️ 2026-10-23：把类型暴露给管理端 —— 否则后台列表里
                #   分不清"哪条是一次性的"，管理员会拿一次性链接去群发。
                "kind": auth.invite_kind(it),
                # 一次性邀请的可用性一眼可见（省得点开才知道被领了）
                "usable": (bool(it.get("enabled", True))
                           and int(it.get("exp") or 0) > int(time.time())
                           and not (auth.invite_kind(it) == "single"
                                    and (it.get("claimed") or []))),
            }
            for it in auth.load_invites()
        ],
    }


@app.post("/admin/user/disable")
def admin_user_disable(body: AdminUidIn, x_auth_user: str = Header(default="")):
    """停用用户（写吊销名单 → 其令牌立即失效）。"""
    _check_admin(x_auth_user)
    uid = (body.uid or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="缺少 uid")
    # ⚠️ 不许把自己关在门外：一旦停用自己，管理员面板就再也进不去了
    if uid == (x_auth_user or "").strip():
        raise HTTPException(status_code=400, detail="不能停用自己")
    auth.revoke(uid, body.reason or "管理员停用")
    auth.audit("admin_disable", uid=uid, by=x_auth_user)
    return {"ok": True, "uid": uid}


@app.post("/admin/reload")
def admin_reload(x_auth_user: str = Header(default="")):
    """知识库热重载（KB 文件增删改后点一下即可，不用重启服务）。

    ⚠️ 为什么单独开一个 /admin/reload：原来的 `/reload` 要求网关密钥 + 管理员，
    **且不在网关的 POST 白名单里**（公网根本转发不到）。这个走 /admin/ 前缀，
    由网关注入身份、后端判管理员，前端按钮即可触发。
    """
    _check_admin(x_auth_user)
    knowledge.reload()
    auth.audit("admin_reload", by=x_auth_user, docs=knowledge.count())
    return {"ok": True, "docs": knowledge.count()}


class AdminInviteIn(BaseModel):
    note: str = ""
    ttl_hours: int = 0
    invite_id: str = ""
    # ⚠️ 2026-10-23 新增：邀请类型
    #   · "multi"（默认，缺省即老行为）—— 多人共用，到期失效；
    #   · "single"                       —— **一次性**：一个人领取后立即失效。
    # 默认值刻意选 "multi"：老调用方（不传这个字段）行为完全不变。
    kind: str = "multi"


@app.post("/admin/invite/create")
def admin_invite_create(body: AdminInviteIn, x_auth_user: str = Header(default="")):
    """生成一条邀请（返回可直接发出去的链接）。"""
    _check_admin(x_auth_user)
    r = auth.new_invite((body.note or "").strip(), body.ttl_hours or None, body.kind or "multi")
    if not r.get("ok"):
        raise HTTPException(status_code=400, detail=r.get("error") or "生成失败")
    # 前端只需要能复制出去的链接；token 本身不回给别处
    r["url"] = "/auth/join/" + r.get("token", "")
    auth.audit("admin_invite_create", invite=r.get("id"), by=x_auth_user, kind=r.get("kind"))
    return r


@app.post("/admin/invite/revoke")
def admin_invite_revoke(body: AdminInviteIn, x_auth_user: str = Header(default="")):
    """作废一条邀请（只影响尚未领取的链接，已领过的人不受影响）。"""
    _check_admin(x_auth_user)
    iid = (body.invite_id or "").strip()
    if not iid:
        raise HTTPException(status_code=400, detail="缺少 invite_id")
    if not auth.revoke_invite(iid):
        raise HTTPException(status_code=404, detail="没找到这条邀请")
    auth.audit("admin_invite_revoke", invite=iid, by=x_auth_user)
    return {"ok": True, "invite_id": iid}


@app.post("/admin/invite/reset")
def admin_invite_reset(body: AdminInviteIn, x_auth_user: str = Header(default="")):
    """**重新启用**一条邀请：清空「已领」名单，让它能再服务一个人。

    用于「一次性邀请被领走了，但我还要再发一次」—— 比作废重生成顺手。
    已领过的人不受影响（他们的身份早就独立生成了）。
    """
    _check_admin(x_auth_user)
    iid = (body.invite_id or "").strip()
    if not iid:
        raise HTTPException(status_code=400, detail="缺少 invite_id")
    r = auth.reset_invite_claims(iid)
    if not r.get("ok"):
        raise HTTPException(status_code=404, detail=r.get("error") or "没找到这条邀请")
    auth.audit("admin_invite_reset", invite=iid, by=x_auth_user, cleared=r.get("cleared"))
    return r


@app.post("/admin/user/rename")
def admin_user_rename(body: AdminRenameIn, x_api_key: str = Header(default=""),
                      x_auth_user: str = Header(default="")):
    """管理员给某个身份改名（2026-11-02）。

    主要用途：把老邀请流程自动生成的 `灰度第一批-01` 改成人名 —— 改完之后，
    管理面板会把**同名**的多条身份归成一组、设备数合并显示（见 /admin/status）。
    ⚠️ 只改显示名，**不合并身份、不动密钥** —— 合并必须由本人在邀请页输动态码确认，
       否则等于给任何人一个「把别人身份并过来」的入口。
    """
    _check_key(x_api_key)
    _check_admin(x_auth_user)
    uid = (body.uid or "").strip()
    new_name = auth.norm_name(body.name)
    if not uid:
        raise HTTPException(status_code=400, detail="缺少 uid")
    if len(new_name) < 2:
        raise HTTPException(status_code=400, detail="名字至少 2 个字")
    u = auth.rename_user(uid, new_name)
    if not u:
        raise HTTPException(status_code=404, detail="没找到该用户")
    auth.audit("admin_rename", uid=uid, name=new_name, by=x_auth_user)
    return {"ok": True, "uid": uid, "name": new_name}


@app.post("/admin/user/enable")
def admin_user_enable(body: AdminUidIn, x_auth_user: str = Header(default="")):
    """恢复用户（移出吊销名单）。"""
    _check_admin(x_auth_user)
    uid = (body.uid or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="缺少 uid")
    auth.unrevoke(uid)
    auth.audit("admin_enable", uid=uid, by=x_auth_user)
    return {"ok": True, "uid": uid}


@app.post("/ask")
@_with_queue
def ask(body: AskIn, x_api_key: str = Header(default=""),
        x_auth_user: str = Header(default="")):
    _check_key(x_api_key)
    # ⭐ 角色分级：从网关注入的 X-Auth-User（验签令牌里的 uid）取角色。
    #    只影响回答形态，不影响权限（管理员判定仍走 _check_admin）。本机直连按默认角色。
    _role = auth.role_of((x_auth_user or "").strip() or None)
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    history = body.history or []

    # 报障卡 + 附加日志要**分头使用**：日志不进检索（会把 BM25 带偏、还灌进无关文档）。
    # 闭环模式下它也不再是「整段塞进 prompt」，而是变成模型可**按需检索**的证据对象
    # （见 app/tools.py）。详见 _split_attached_log 的注释。
    question, log_text = _split_attached_log(question)

    # 多轮场景：检索 / 取证都要用「最完整的那条用户描述」做主 query——
    # 追问轮用户往往只回字段（如单号/版本），单独用它检索会丢背景；
    # 首轮/最长描述的【功能+现象+版本】才是知识库与源码的最佳入口。
    # ⚠️ 顺带把「最长的那份日志」也捡出来：前端不保证每轮都重传日志，
    #    日志只出现在历史轮时，闭环仍要能查到它。
    primary = question
    best_log = log_text or ""
    for h in history:
        if h.get("role") == "user" and h.get("content"):
            q2, l2 = _split_attached_log(h["content"])
            if len(q2) > len(primary):
                primary = q2
            if len(l2) > len(best_log):
                best_log = l2

    # 人工教练记录：与本次提问相似的历史纠正，优先级高于知识库。
    # 在取证之前算好 —— 它只影响「怎么答」，不参与检索打分。
    # ⭐ 2026-10-29：先识别意图（规则 0ms；判不出且库里有带场景的教练才调一次模型），
    #   再按意图精确匹配，避免"换个说法/换个号码就匹配不上"。
    _intent = intent.resolve(primary[:RETRIEVAL_QUERY_MAX],
                             has_intent_coach=any(c.get("intents") for c in cases.list_coach(active_only=True)))
    coach_hits = cases.match_coach(primary[:RETRIEVAL_QUERY_MAX], intent=_intent)

    agent_meta: dict = {}
    if getattr(config, "AGENT_ENABLED", True):
        # 取证闭环（2026-09-16）：查什么由模型自己决定。工具在 app/tools.py，
        # 轮数上限与末轮强制收口在 llm.answer_agentic。溯源字段从 meta 回填。
        answer_text, agent_meta = llm.answer_agentic(
            question, history=history, images=_resolve_images(body.images),
            coach=coach_hits, log=best_log or None, mode=body.mode or "",
            role=_role)
        # mode 由前端首屏入口决定："report"=走了报障卡；其余=自由提问（形状自适应）。
        # ⚠️ 下面的单轮回退路径（KB_AGENT_ENABLED=0）暂不区分 mode —— 它是应急开关，默认不走。
        hits, weak, rare = [], False, {}
        code_hits, code_clues = [], []
    else:
        # —— 回退路径：原单轮流程，一行未改（KB_AGENT_ENABLED=0 时走这里）——
        res = knowledge.search_ex(primary[:RETRIEVAL_QUERY_MAX])
        hits, weak = res["hits"], res["weak"]
        rare = res.get("rare") or {}

        # 源码查询模块：排障/定位/实现原理类问题自动检索源码证据
        code_hits = []
        code_clues = []
        # ⚠️ 弱命中时刻意「不查源码」：这时知识片段只是词面沾边，从中提的代码线索同样不可靠，
        # 搜回来的无关代码会被放大，反倒挤占答案、误导判断。
        # 真正的新问题（hits 为空）不受影响，仍然会去源码里找一轮。
        # 线索提取也一律用「问题部分」：日志里有成千上万个类名 / URL / traceId，
        # 拿它去提线索会把源码检索冲成一片噪声。
        if not weak and codesearch.need_code(question + " " + primary[:400], hits):
            code_clues = codesearch.extract_clues(question, hits)
            # 若追问轮没贡献新线索，则用主问题再提一次（覆盖首轮关键词）
            if not code_clues:
                code_clues = codesearch.extract_clues(primary, hits)
            if code_clues:
                code_hits = codesearch.search(code_clues)

        answer_text = llm.answer(question, hits, history, code_hits,
                                 images=_resolve_images(body.images), coach=coach_hits,
                                 weak=weak, rare=rare, log=log_text, role=_role)

    # 溯源字段：闭环下检索发生在工具内部，改从取证轨迹回填（同一文档去重），
    # 保证前端「复制给人工」与 /review 仍拿得到来源。
    src = [{"file": h["file"], "idx": h["idx"], "score": h["score"]} for h in hits]
    if not src and agent_meta.get("kb_hits"):
        seen, src = set(), []
        for h in agent_meta["kb_hits"]:
            key = (h.get("file"), h.get("idx"))
            if key in seen:
                continue
            seen.add(key)
            src.append({"file": h.get("file"), "idx": h.get("idx"),
                        "score": h.get("score") or 0})
    resp = {"answer": answer_text, "sources": src}
    # ⭐ 2026-10-12 使用统计：把「本次是谁在用」回给前端，前端落案例时写进 `user` 字段。
    #   身份由 **Worker 注入**（取自验签通过的门禁令牌，客户端伪造无效），
    #   前端落库时只做搬运，不做判断。无鉴权（本机直连/未配 AUTH_SECRET）时为空串。
    if x_auth_user:
        resp["user"] = x_auth_user
    if weak:
        # 供页面/日志判断这次回答是不是「没听懂」——弱命中时答案是「没找到」而非排障步骤
        resp["weak"] = True
        resp["rare"] = rare
    if coach_hits:
        # 只回 id 与分数：教练正文属于「内部规则」，不在页面上暴露给一线用户
        resp["coached"] = [{"id": c["id"], "score": c["score"]} for c in coach_hits]
    if code_clues or code_hits:
        resp["code_sources"] = [{"file": h["file"], "line": h["line"], "score": h["score"]}
                                for h in code_hits]
        resp["code_clues"] = code_clues
    if agent_meta.get("diagram"):
        # 本轮模型画的简图（时间线 / 链路 / 判定分支）：前端显示在答案最上方。
        # 正常走 url（落盘后按 /diagram/<name> 取）；落盘失败时带回 svg 源码由前端内联。
        resp["diagram"] = agent_meta["diagram"]
    if agent_meta.get("trace") and getattr(config, "AGENT_EXPOSE_TRACE", True):
        # 本次取证轨迹：调了哪些工具、各返回多少字、共花多少 token ——
        # 运维复核「这个结论是怎么查出来的」，也便于盯成本。
        # ⚠️ 2026-10-05 追加 depth / plan / selfcheck：分档上线后要能看出
        #    「这次走的哪一档、规划轮写了什么、自检发现了哪些缺口」，否则无法调参与追责。
        resp["agent"] = {"steps": agent_meta.get("steps"),
                         "depth": agent_meta.get("depth"),
                         "tools": agent_meta.get("trace") or [],
                         "plan": agent_meta.get("plan"),
                         "selfcheck": agent_meta.get("selfcheck"),
                         "clipped": agent_meta.get("clipped"),      # 被折叠的历史工具结果条数
                         "early_stop": agent_meta.get("early_stop"),  # 提前收口发生在第几轮
                         # ⭐ 2026-10-12：各收敛判据各命中几次（决定"哪条判据可以删"的唯一依据）
                         "stall_kinds": agent_meta.get("stall_kinds") or {},
                         # ⭐ 2026-10-12（主人定"动态超时"）：本次的时间预算与是否触发
                         #   —— 前端可据此展示"本次任务预期最多 N 秒"（预期管理），
                         #   排障时能看出"答案是不是被预算截停的"
                         "time_budget": agent_meta.get("time_budget"),
                         "time_budget_hit": agent_meta.get("time_budget_hit", False),
                         "usage": agent_meta.get("usage") or {}}
    return resp


@app.post("/ask-stream")
def ask_stream(body: AskIn, x_api_key: str = Header(default=""),
               x_auth_user: str = Header(default="")):
    """SSE 版 /ask：绕开 Cloudflare 的 100 秒 524。

    【问题】
    CF 判 524 的依据是「origin 在 100 秒内**没有开始**返回响应」。而 deep 档的
    取证 + 自检 + 精简正常要 60~180 秒，一旦越过 100s：
      · 用户在边缘侧收到 524 /「服务不在线」
      · **后端其实还在跑，跑完的答案永远送不出去**（白烧 token、白占并发位）

    【解法】
    立刻返回 `text/event-stream`（首字节几毫秒），之后每 8 秒发一个 SSE 注释心跳：
      · CF 看到响应已开始，就不再按 100s 掐；
      · 心跳保证中间设备（及浏览器）不会把"长时间无数据"判成断连。
    算完后推 `event: done` + 完整 JSON，前端拿到即渲染。整条链时长不再受 100s 限制。

    【兼容】
    /ask 原样保留：前端可随时改回旧路径。两者共用同一套业务逻辑（这里直接调用
    ask()，**不复制代码** —— 复制一份必然会随时间漂移）。
    """
    def _run(box: dict) -> None:
        try:
            res = ask(body, x_api_key, x_auth_user)
            if isinstance(res, JSONResponse):
                box["status"] = res.status_code
                box["payload"] = json.loads(res.body.decode("utf-8"))
            else:
                box["payload"] = res
        except HTTPException as e:
            box["status"] = e.status_code
            box["payload"] = {"detail": e.detail}
        except Exception as e:                       # noqa: BLE001
            # 与 ask 同口径：原始异常只进日志，不出现在用户可见文案里
            box["status"] = 500
            box["payload"] = {"detail": "服务内部错误，请稍后重试"}
            print(f"[ask-stream] 未预期异常：{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            box["done"] = True
            box["ev"].set()

    def gen():
        box = {"done": False, "payload": None, "status": 200, "ev": threading.Event()}
        threading.Thread(target=_run, args=(box,), daemon=True).start()
        # 立刻吐一段：让 CF 与浏览器都认定"响应已经开始"，这行就是解 524 的关键
        yield ": start\n\n"
        while not box["ev"].wait(timeout=8):
            yield ": ping\n\n"
        payload = box["payload"] if box["payload"] is not None else {}
        yield "event: done\ndata: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",     # 将来若在中间加 nginx，这行能关掉它的缓冲
            "Connection": "keep-alive",
        },
    )


# ---------------- 访问认证：共享口令 → 设备令牌（2026-09-22 主人定）
# 动机：助手是总部管理员权限，内部不需要分级，但要挡住外部人员嗅探数据。
# 原网关密钥硬编码在前端 JS（F12 可见）= 等于没鉴权，这一层补上。
# ⚠️ 这三个路由**必须免 Key**：还没验证的人也得能访问它们。网关白名单同步加。
# ⚠️ 令牌的**校验**在 Cloudflare Worker 层做（全站拦，含页面本身），
#    这里只负责**签发**；Worker 用同一个 AUTH_SECRET 离线验签，不必回源。
class LoginIn(BaseModel):
    code: str = ""        # 6 位 TOTP 动态码（老路径）**或**共享长期授权码（2026-10-09 新增）
    staff_id: str = ""      # 员工编号：授权码路径**必填**，TOTP 路径选填（只记录不校验）
    fp: str = ""          # 浏览器/设备标识（前端生成的随机 ID，非指纹）


# ---- 登录限流（2026-09-25）----
# 动机：TOTP 只有 6 位、`totp` 的 window=1（同时有 3 个码有效）、且 verify_code 要**遍历全部用户**
#      去比对；而 /auth/login 免 Key、原来**没有任何限流或锁定** →
#       按 10^6/(3·N) 量级尝试即可撞中某个人的码。这是当时最现实、最便宜的入口风险。
# 实现：按「设备指纹 + 来源 IP」计数，连续失败达到上限后锁定一段时间。
# ⚠️ 内存态（进程重启即清空）：当前是单机部署，够用；要跨重启保留得落盘，不值得为它引复杂度。
# ⚠️ 被锁后**不重置**首次计数时间，避免"打满就换锁"的绕法。
_LOGIN_FAILS: dict = {}          # key -> [连续失败次数, 锁定截止时间戳]
_LOGIN_MAX_FAILS = 8             # 连续失败上限
_LOGIN_LOCK_SECS = 300           # 触发后锁 5 分钟


def _login_key(fp: str, request: Request) -> str:
    ip = (request.client.host if request.client else "") or "?"
    return f"{(fp or '')[:16]}|{ip}"


def _login_wait_secs(key: str) -> int:
    """还需等待多少秒才能再试；0 = 可以试。"""
    rec = _LOGIN_FAILS.get(key)
    return max(0, int(rec[1] - time.time())) if rec else 0


def _login_note_fail(key: str) -> None:
    rec = _LOGIN_FAILS.get(key) or [0, 0.0]
    rec[0] = int(rec[0]) + 1
    if rec[0] >= _LOGIN_MAX_FAILS:
        rec[1] = time.time() + _LOGIN_LOCK_SECS
        rec[0] = 0               # 锁定期满后重新计数
    _LOGIN_FAILS[key] = rec


def _login_guard(fn):
    """把 `/auth/login` 的**未预期异常**变成「有审计 + 有人话」的 500。

    ⚠️ 2026-10-09 加：当天在 service.log 里看到 8 次连续 `POST /auth/login 500`，
       而 `auth_audit.jsonl` 里**一条对应记录都没有** —— 说明异常发生在写审计之前，
       等于「线上有人就是登不进来，而我们事后完全查不到为什么」。内测要扩，
       这种"看不见的失败"最致命。
    这一层只做两件事：① 写一条 `login_error` 审计 + 打堆栈；② 回一句用户能照着做的话。
    ⚠️ **仍然以 500 抛出** —— 不吞异常、不假装成功，该报警还得报警。
    ⚠️ 我们自己主动抛的 HTTPException（401/400/429/503）原样放行，不改写成 500。
    """
    @functools.wraps(fn)
    def wrapper(body, request):
        try:
            return fn(body, request)
        except HTTPException:
            raise
        except Exception as e:                       # noqa: BLE001
            auth.audit("login_error", error=f"{type(e).__name__}: {str(e)[:200]}",
                       fp=(getattr(body, "fp", "") or "")[:8])
            print(f"[auth] /auth/login 未预期异常：{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise HTTPException(status_code=500,
                                detail="服务内部错误，请稍后重试；若持续失败请联系管理员")
    return wrapper


@app.post("/auth/login")
@_login_guard
def auth_login(body: LoginIn, request: Request):
    """校验凭据 → 签发设备令牌（HttpOnly Cookie，默认 **30 天**）。

    两条入口（2026-10-09 主人定：放开内测）：
      · **6 位纯数字** → 走 TOTP 动态码（老路径，已绑密钥的人不受影响）；
      · **其它** → 走「共享长期授权码」，这时**必须**带员工编号（只记录不校验，
        当身份标识用；同一个人换设备也是同一条记录）。
    判据是"长度 + 是否纯数字"，不是"先试 TOTP 再试授权码" —— 后者会让
    6 位数字撞上 30 秒窗口时产生难以解释的行为。
    """
    if not auth.enabled():
        return {"ok": True, "disabled": True, "message": "访问认证未启用（未配置 KB_AUTH_SECRET）"}

    key = _login_key(body.fp, request)
    wait = _login_wait_secs(key)
    if wait > 0:
        auth.audit("login_locked", wait_sec=wait, fp=(body.fp or "")[:8])
        raise HTTPException(status_code=429,
                            detail=f"尝试过于频繁，请 {wait} 秒后再试")

    code_v = (body.code or "").strip()
    staff = auth.norm_staff_id(body.staff_id)
    u = None
    via = ""

    if code_v.isdigit() and len(code_v) == totp.DIGITS:
        # ---- ① TOTP 老路径 ----
        # 用户表空的时候给一句能照做的提示，别让人对着"动态码不对"猜是哪里出问题
        if not any(x.get("enabled") for x in auth.load_users()):
            raise HTTPException(status_code=503,
                                detail="还没有配置任何用户，请先在本机运行 scripts/manage_users.py add --name 你的名字")
        u = auth.verify_code(code_v)
        via = "totp"
    else:
        # ---- ② 共享长期授权码路径 ----
        if not auth.access_code_enabled():
            raise HTTPException(status_code=503,
                                detail="共享授权码未启用（管理员未配置 KB_ACCESS_CODE）；"
                                       "已绑动态令牌的用户请改输 6 位动态码")
        if not auth.verify_access_code(code_v):
            _login_note_fail(key)
            auth.audit("login_fail", via="access_code", code_len=len(code_v), fp=(body.fp or "")[:8])
            left = _LOGIN_MAX_FAILS - int((_LOGIN_FAILS.get(key) or [0])[0])
            tip = "授权码不对，请核对管理员发的授权码（注意别多复制空格）"
            if 0 < left <= 3:
                tip += f"（再错 {left} 次将锁定 {_LOGIN_LOCK_SECS // 60} 分钟）"
            raise HTTPException(status_code=401, detail=tip)
        got = auth.find_or_create_by_staff(staff)
        if not got.get("ok"):
            # 员工编号 没填 / 该身份被停用 —— 不算"验证失败"，不累加失败计数
            raise HTTPException(status_code=400, detail=got.get("error") or "LDAP 账号有误")
        u = got["user"]
        via = "access_code"
        if got.get("created"):
            auth.audit("user_created", uid=u["id"], staff_id=staff, via="access_code")

    if not u:
        _login_note_fail(key)
        auth.audit("login_fail", code_len=len(code_v), fp=(body.fp or "")[:8])
        left = _LOGIN_MAX_FAILS - int((_LOGIN_FAILS.get(key) or [0])[0])
        tip = "动态码不对或已过期，请核对手机上的当前 6 位码"
        if 0 < left <= 3:
            tip += f"（再错 {left} 次将锁定 {_LOGIN_LOCK_SECS // 60} 分钟）"
        raise HTTPException(status_code=401, detail=tip)

    # TOTP 老用户顺手填了员工编号 就补记（**只补空的**，见 auth.set_staff_id 注释）
    if via == "totp" and staff:
        auth.set_staff_id(u["id"], staff)
        u["staff_id"] = u.get("staff_id") or staff     # 让下面回给前端的 user 与落盘结果一致

    _LOGIN_FAILS.pop(key, None)          # 成功即清零
    res = auth.issue(body.fp, u["id"])
    if not res.get("ok"):
        # ⚠️ 正常流程走不到这里：前端 worker.js 的 getDeviceId() 有三级降级
        #    （randomUUID → getRandomValues → Math.random），**保证一定带得上设备标识**。
        #    能走到这里说明请求不是从正常页面发出的（页面脚本被拦/被改写，或有人直接调接口）。
        #    所以① 文案要说人话、给可操作动作（旧文案「缺少浏览器指纹」用户完全看不懂，
        #    只会反复重输动态码 —— 2026-10-09 就这么踩了一回）；
        #    ② 状态码用 400 而不是 500：这是请求侧的问题，不是服务器故障，
        #    免得监控里把它算成服务端错误。
        raise HTTPException(status_code=400, detail=res.get("error") or "签发令牌失败")

    auth.touch_user(u["id"], body.fp)      # 2026-11-02：顺带记录「这台设备」（按 fp8 去重）
    auth.audit("login_ok", uid=u["id"], name=u.get("name"), via=via,
               staff_id=auth.norm_staff_id(u.get("staff_id") or ""), fp=(body.fp or "")[:8])
    resp = JSONResponse({"ok": True, "ttl_days": res["ttl_days"], "expires_at": res["expires_at"],
                         "via": via,
                         "user": {"id": u["id"], "name": u.get("name"),
                                  "staff_id": auth.norm_staff_id(u.get("staff_id") or "")}})
    for c in auth.set_cookie_headers(res["token"], body.fp, res["max_age"]):
        resp.headers.append("set-cookie", c)      # ⚠️ 两个 Set-Cookie 必须 append，不能塞进 dict
    return resp


def _qr_svg(text: str, px: int = 232) -> str:
    """服务端生成二维码 SVG —— **不依赖任何前端库 / CDN / canvas**。

    ⚠️ 2026-09-22 这段踩了三次才走对，记下来别再退回去：
      ① 引 cdnjs 的 qrcodejs → 国内加载不出来，二维码区一片空白（其它部分正常，极易误判成代码写错）；
      ② 改成内联 JS 库 → 模板里写了 `{{ }}`（f-string 的习惯），但这是普通字符串，原样输出 → JS 语法错；
      ③ 语法修好后 canvas 确实画了，可 qrcodejs 要「canvas → toDataURL → img.onload」才显示，
         无头浏览器里等不到那一步，**连验证都做不了**。
    → 服务端直接出 SVG 最稳：浏览器只负责显示，一个脚本都不需要。
    生成失败时降级成一行提示（页面下方仍有手动密钥，用户照样能完成绑定）。
    """
    try:
        import io
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage,
                          box_size=10, border=2)
        buf = io.BytesIO()
        img.save(buf)
        svg = buf.getvalue().decode("utf-8")
        svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)                   # 内联进 HTML，不需要 XML 声明
        svg = re.sub(r'width="[\d.]+mm"\s+height="[\d.]+mm"',
                     f'width="{px}" height="{px}"', svg, count=1)     # 物理尺寸(mm) → 像素
        return f'<div style="display:flex;justify-content:center">{svg}</div>'
    except Exception as e:
        return ('<div style="color:#b91c1c;font-size:13px;line-height:1.6">'
                f'二维码生成失败（{e}）<br>请改用下方的手动密钥绑定</div>')


_BIND_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>绑定动态令牌</title>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>
 body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
      background:#f6f8fc;font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#0f172a}
 .card{width:min(420px,93vw);background:#fff;border:1px solid #dbeafe;border-radius:16px;padding:26px 24px;
       box-shadow:0 10px 30px rgba(37,99,235,.10);text-align:center}
 h1{font-size:17px;margin:0 0 5px} .sub{font-size:12.5px;color:#64748b;margin-bottom:16px}
 #q{display:flex;justify-content:center;margin:6px 0 14px;min-height:232px}
 .key{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:#f1f5f9;border-radius:8px;
      padding:9px 11px;word-break:break-all;user-select:all}
 .warn{margin-top:14px;font-size:12px;color:#b45309;background:#fffbeb;border:1px solid #fde68a;
       border-radius:9px;padding:9px 11px;text-align:left;line-height:1.65}
 .bad{font-size:15px;color:#b91c1c;background:#fef2f2;border:1px solid #fecaca;border-radius:10px;
      padding:14px;line-height:1.7;text-align:left}
 /* ── 2026-10-23 新增：下载验证器 App 的入口 ── */
 .step{display:flex;align-items:center;gap:7px;font-size:12.5px;color:#475569;margin:0 0 6px;text-align:left}
 .step i{flex:none;width:18px;height:18px;border-radius:50%;background:#2563eb;color:#fff;
         font-style:normal;font-size:11.5px;font-weight:700;display:flex;align-items:center;justify-content:center}
 .hint{font-size:12px;color:#475569;background:#f8fafc;border:1px solid #e2e8f0;border-radius:9px;
       padding:9px 11px;text-align:left;line-height:1.7;margin:0 0 11px}
 .hint b{color:#0f172a}
 .dlbtn{display:inline-flex;align-items:center;gap:6px;border:1px solid #bfdbfe;background:#eff6ff;color:#1d4ed8;
        border-radius:9px;padding:7px 13px;font-size:13px;font-weight:600;cursor:pointer;
        font-family:inherit;margin:0 0 10px}
 .dlbtn:hover{background:#dbeafe}
 /* 弹窗 */
 .mask{position:fixed;inset:0;background:rgba(15,23,42,.5);display:none;align-items:center;
       justify-content:center;padding:18px;z-index:99}
 .mask.on{display:flex}
 .modal{background:#fff;border-radius:16px;padding:20px 18px 16px;width:min(360px,92vw);
        box-shadow:0 18px 50px rgba(15,23,42,.28);text-align:center}
 .modal h2{font-size:15px;margin:0 0 3px}
 .modal .m-sub{font-size:12px;color:#64748b;margin-bottom:12px}
 .modal img{width:min(240px,70vw);border:1px solid #e2e8f0;border-radius:10px;display:block;margin:0 auto}
 .modal .m-tip{font-size:12px;color:#1d4ed8;background:#eff6ff;border:1px solid #bfdbfe;border-radius:9px;
               padding:8px 10px;margin-top:12px;text-align:left;line-height:1.6}
 .modal .m-warn{font-size:12px;color:#b45309;background:#fffbeb;border:1px solid #fde68a;border-radius:9px;
                padding:8px 10px;margin-top:8px;text-align:left;line-height:1.6}
 .mlink{display:flex;gap:8px;margin-top:9px;align-items:stretch}
 .mlink a{flex:1;display:flex;align-items:center;justify-content:center;font-size:12.5px;color:#1d4ed8;
          background:#eff6ff;border:1px solid #bfdbfe;border-radius:9px;padding:7px 6px;text-decoration:none}
 .mlink a:hover{background:#dbeafe}
 .mlink button{flex:none;border:1px solid #cbd5e1;background:#fff;color:#475569;border-radius:9px;
               padding:7px 12px;font-size:12.5px;cursor:pointer;font-family:inherit}
 .mclose{margin-top:13px;width:100%;border:0;background:#0f172a;color:#fff;border-radius:10px;
         padding:9px;font-size:13.5px;font-weight:600;cursor:pointer;font-family:inherit}
</style></head><body>
<div class="card">
  <h1>🔐 绑定动态令牌</h1>
  <!-- ⚠️ 名字必须留着（2026-09-22 主人指出）：绑定链接是"谁拿到谁就能绑"，
       本人打开时得能确认「这条是给我的、没发错人」。去掉它就少了一道自检。 -->
  <div class="sub"><b>{name}</b> · POS 智能报障助手</div>
  <div class="step"><i>1</i><span>用<b>验证器 App</b>扫描下面的码（已有这类 App 的直接扫，不用再装）</span></div>
  <div class="hint">
    支持的 App：<b>腾讯身份验证器</b>（应用宝能搜到，国内最好下）、
    <b>微软 Authenticator</b>、<b>Google Authenticator</b>、<b>1Password</b> 等，<b>任选其一</b>。<br>
    ⚠️ <b>已经有了就别再装</b> —— 下面这个按钮只是给"一个都没有"的人用的，
    且<b>仅提供「腾讯身份验证器」</b>一个的下载。
  </div>
  <button type="button" class="dlbtn" id="dlOpen">📱 没有验证器 App？下载「腾讯身份验证器」</button>
  <div class="step"><i>2</i><span>用 App 扫下面这个码完成绑定</span></div>
  <div id="q">{qrsvg}</div>
  <div class="key">{secret}</div>
  <div class="warn">
    用任一支持「扫描二维码」的验证器 App 扫码，例如
    <b>腾讯身份验证器</b>（应用宝搜得到，国内最好下）、
    <b>微软 Authenticator</b>、<b>Google Authenticator</b>、<b>1Password</b>；<br>
    扫不出来就手动输入上面那串密钥（App 里选「手动输入 / 无法扫码」）。<br>
    绑定完成后，打开 <b>ask.example.com</b> 输 App 里的 6 位动态码，验证一次管 {days} 天。<br>
    {warn_extra}<br>
    ⚠️ 这类 App 大多<b>不云同步</b>，手机丢了要重新绑定 —— 建议绑定后自行备份密钥。
  </div>
</div>
<div class="mask" id="dlMask">
  <div class="modal" role="dialog" aria-modal="true" aria-label="下载腾讯身份验证器">
    <h2>📱 腾讯身份验证器</h2>
    <div class="m-sub">扫码下载到手机</div>
    <img src="{appqr}" alt="腾讯身份验证器 下载二维码">
    <div class="m-tip">
      <b>手机就在手上？</b>不用扫自己屏幕 —— <b>长按上面的二维码</b>，
      选「识别图中二维码」即可直接打开下载页。
    </div>
    <div class="mlink">
      <a href="{appurl}" target="_blank" rel="noopener">直接打开下载页 ↗</a>
      <button type="button" id="dlCopy">复制链接</button>
    </div>
    <div class="m-warn">
      ⚠️ 装好后<b>回到本页</b>，用 App 扫的是<b>页面里那个</b>绑定二维码
      （就是编号 <b>2</b> 下面那个）。这两个码别搞混：上面这个只用来<b>装 App</b>，
      装完就没用了。
    </div>
    <button type="button" class="mclose" id="dlClose">知道了</button>
  </div>
</div>
<script>
(function(){
  var mask=document.getElementById('dlMask');
  var open=document.getElementById('dlOpen');
  var close=document.getElementById('dlClose');
  if(!mask||!open) return;
  function show(){ mask.classList.add('on'); }
  function hide(){ mask.classList.remove('on'); }
  open.addEventListener('click', show);
  if(close) close.addEventListener('click', hide);
  // 点遮罩空白处关闭（点弹窗内部不关）
  mask.addEventListener('click', function(e){ if(e.target===mask) hide(); });
  document.addEventListener('keydown', function(e){ if(e.key==='Escape') hide(); });
  // 复制下载链接：兜底 navigator.clipboard 不可用（http 页面 / 老浏览器）
  var cp=document.getElementById('dlCopy');
  if(cp) cp.addEventListener('click', function(){
    var u='{appurl}';
    function done(){ cp.textContent='已复制 ✓'; setTimeout(function(){ cp.textContent='复制链接'; },1500); }
    if(navigator.clipboard && navigator.clipboard.writeText){
      navigator.clipboard.writeText(u).then(done, function(){ fallback(); });
    } else { fallback(); }
    function fallback(){
      var t=document.createElement('textarea'); t.value=u; t.style.position='fixed'; t.style.opacity='0';
      document.body.appendChild(t); t.select();
      try{ document.execCommand('copy'); done(); }catch(e){ cp.textContent='请手动复制'; }
      document.body.removeChild(t);
    }
  });
})();
</script>
</body></html>"""

_BIND_ERR = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>绑定链接不可用</title>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#f6f8fc;
 font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#0f172a}
 .card{width:min(420px,93vw);background:#fff;border:1px solid #fecaca;border-radius:16px;padding:26px 24px;
 box-shadow:0 10px 30px rgba(239,68,68,.08)}
 h1{font-size:17px;margin:0 0 10px} .bad{font-size:14px;color:#b91c1c;line-height:1.7}</style></head><body>
<div class="card"><h1>⛔ 绑定链接不可用</h1><div class="bad">{why}</div></div></body></html>"""


# ── 一次性邀请的「确认领取」页（2026-10-28）──────────────────────
# 事故背景：/auth/join/<token> 原先「打开页面即领取」，于是
#   · 管理员生成后自己点开看一眼（最常见的那个）；
#   · 微信/企微对链接做预取或卡片预览；
#   · 手机误触、重复点开
# 任何一次 GET 都会把**唯一名额**吃掉，真到收件人手里就只剩"已被领取"。
# 实测证据：inv28 生成 10:30:56 → 10:31:03 被领（7 秒），而领取者从未登录过。
# 改法：一次性邀请的 GET 只渲本页（**不消耗名额**），点了按钮才 POST 真领。
_JOIN_CSS = """
 body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#f6f8fc;
      font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#0f172a}
 .card{width:min(460px,93vw);background:#fff;border:1px solid #dbeafe;border-radius:16px;padding:26px 24px;
       box-shadow:0 10px 30px rgba(37,99,235,.08)}
 h1{font-size:17px;margin:0 0 6px} .sub{font-size:13px;color:#64748b;line-height:1.7;margin-bottom:14px}
 label{display:block;font-size:13px;color:#334155;margin:14px 0 6px;font-weight:600}
 input{width:100%;box-sizing:border-box;padding:11px 12px;border:1px solid #cbd5e1;border-radius:10px;
       font-size:15px;font-family:inherit;outline:none;background:#fff}
 input:focus{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
 .hint{font-size:12px;color:#94a3b8;margin-top:6px;line-height:1.6}
 .tip{margin-top:14px;padding:10px 12px;background:#f1f5f9;border-radius:10px;font-size:12.5px;color:#64748b;line-height:1.65}
 .err{margin:12px 0 0;padding:10px 12px;background:#fef2f2;border:1px solid #fecaca;border-radius:10px;
      color:#b91c1c;font-size:13px;line-height:1.6}
 button{margin-top:18px;width:100%;padding:13px;border:0;border-radius:12px;background:#2563eb;color:#fff;
        font-size:15px;font-weight:600;cursor:pointer;font-family:inherit}
 button:active{opacity:.85}
 .meta{margin-top:10px;font-size:12px;color:#94a3b8}
 a{color:#2563eb}
"""

# 邀请页第一步：填名字（2026-11-02）
# ⚠️ 原来的名字是 `<邀请备注>-01` 自动生成，管理员看不出是谁、也没法按人合并；
#    现在由用户自己填，同一个人多台设备填同一个名字。
_JOIN_FORM = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>加入 POS 报障助手</title>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>""" + _JOIN_CSS + """</style></head><body>
<div class="card">
  <h1>👋 加入 POS 报障助手</h1>
  <div class="sub">先填一下你的名字，再领取属于你的动态令牌。</div>
  <form method="post" action="/auth/join/{token}">
    <label for="nm">你的名字 <span style="color:#dc2626">*</span></label>
    <input id="nm" name="name" maxlength="24" required autocomplete="name"
           placeholder="例：张三" value="{nm_value}">
    <div class="hint">这个名字会显示在管理员面板上。同一个人的多台设备请填<b>同一个名字</b>，
      系统会让你验证一次动态码，然后把它们合并成一个人。</div>
    <label for="hr">LDAP账号 <span style="color:#94a3b8;font-weight:400">（选填）</span></label>
    <input id="hr" name="staff" maxlength="40" autocomplete="username"
           placeholder="例：zuoyuan1" value="{staff_value}">
    <div class="hint">填了就能跨设备自动认出你（换手机也不用再验证）。</div>
    <input type="hidden" name="fp" id="fp" value="{fp_value}">
    {err}
    <button type="submit">{btn}</button>
  </form>
  <div class="tip">{tip}</div>
  <div class="meta">邀请 {iid} · {note} · 最晚 {exp} 过期</div>
</div>
<script>
/* 设备标识：与登录页同口径（随机十六进制，不依赖任何新 API，老内核也能跑） */
(function(){try{
  var f=document.getElementById('fp');
  if(!f||f.value) return;
  var s='',c='abcdef0123456789',i;
  try{var a=new Uint8Array(8);(window.crypto||window.msCrypto).getRandomValues(a);
      for(i=0;i<a.length;i++) s+=c[a[i]&15];}catch(e){}
  if(!s){for(i=0;i<16;i++) s+=c[Math.floor(Math.random()*16)];}
  f.value=s;
}catch(e){}})();
</script>
</body></html>"""

# 邀请页第二步：名字已存在 → 证明是本尊（否则填个名字就能拿到别人的密钥）
_JOIN_MERGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>合并到已有身份</title>
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>""" + _JOIN_CSS + """</style></head><body>
<div class="card">
  <h1>🔗 名字「{nm}」已经有了</h1>
  <div class="sub">如果你就是<b>{nm}</b>本人：输入验证器 App 里当前的 <b>6 位动态码</b>，
    系统会把<b>这台设备</b>加进同一个身份（不会新建人，也不占新名额）。</div>
  <form method="post" action="/auth/join/{token}">
    <input type="hidden" name="name" value="{nm_value}">
    <input type="hidden" name="staff" value="{staff_value}">
    <input type="hidden" name="fp" value="{fp_value}">
    <label for="cd">6 位动态码</label>
    <input id="cd" name="code" inputmode="numeric" pattern="[0-9]{6}" maxlength="6" required
           placeholder="000000" autocomplete="one-time-code"
           style="letter-spacing:6px;font-size:20px;text-align:center">
    {err}
    <button type="submit">是我本人，加入这台设备</button>
  </form>
  <div class="tip">不是你本人？<a href="/auth/join/{token}">返回上一步换个名字</a>
    （例如「{nm}-财务」「{nm}-门店B」），同名但不同人会各算一个人。</div>
</div>
</body></html>"""

# 合并成功：直接登录这台设备（meta refresh，不依赖 JS）
_JOIN_DONE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>已加入</title>
<meta http-equiv="refresh" content="1;url=/">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>""" + _JOIN_CSS + """</style></head><body>
<div class="card" style="text-align:center">
  <h1>✅ 已加入「{nm}」</h1>
  <div class="sub" style="margin:8px 0 0">{msg}</div>
  <div class="meta" style="margin-top:16px"><a href="/">没自动跳转？点这里进入</a></div>
</div>
</body></html>"""


@app.get("/auth/bind/{token}", response_class=HTMLResponse)
def auth_bind(token: str):
    """限时绑定页：本人扫码把 TOTP 密钥装进 Authenticator。

    ⚠️ 这条路由必须**既免门禁又免 Key** —— 扫码的人此刻还没有任何凭据。
       安全性由「链接不可猜（HMAC 签名）+ 默认 24 小时过期」保证：链接本身就是一次性凭证。
    ⚠️ 页面里含**明文密钥**，所以别把链接贴到公开群里；发完本人即可。
    """
    uid, why = auth.verify_bind_token(token)
    if not uid:
        return HTMLResponse(_BIND_ERR.replace("{why}", why), status_code=410)
    u = next((x for x in auth.load_users() if x.get("id") == uid), None)
    if not u:
        return HTMLResponse(_BIND_ERR.replace("{why}", "该用户不存在或已停用"), status_code=410)
    auth.audit("bind_view", uid=uid, name=u.get("name"))
    # 专属绑定：链接只发给这一个人 —— 警告要**点名**，让人确认"这是给我的、没发错人"
    warn = ('⚠️ <b>此链接等同于密码</b>：只发给本人，请勿转发 —— '
            f'如果你不是 <b>{u.get("name")}</b>，请勿扫码并告知管理员。<br>'
            f'⚠️ 本链接 {config.AUTH_BIND_TTL_HOURS} 小时后失效；'
            '失效后找管理员重新生成即可（<b>已绑定的不受影响</b>）。')
    # ⚠️ `{qrsvg}` 必须**最后**替换：Python 的 str.replace 是全局替换，
    #    生成的 SVG 里有大量 `{...}` 路径数据，先注进去的话后面几个 replace 可能误伤它。
    url = totp.otpauth_url(str(u.get("secret") or ""), str(u.get("name") or uid))
    html = (_BIND_HTML
            .replace("{name}", str(u.get("name") or uid))
            .replace("{secret}", str(u.get("secret") or ""))
            .replace("{days}", str(config.AUTH_TTL_DAYS))
            .replace("{hours}", str(config.AUTH_BIND_TTL_HOURS))
            .replace("{warn_extra}", warn)
            .replace("{qrsvg}", _qr_svg(url))
            .replace("{appqr}", TOTP_APP_QR)
            .replace("{appurl}", TOTP_APP_URL))
    return HTMLResponse(html)


def _esc(s) -> str:
    """HTML 转义（名字/备注都是用户填的，必须转义后再拼进页面）。"""
    return htmllib.escape(str(s or ""), quote=True)


def _join_warn(invite: dict) -> str:
    """绑定页顶部的警告文案（按邀请类型分开口径）。

    ⚠️ 2026-10-23：一次性邀请**绝不能**再说"仅限本群成员、请勿外传"（它不是群发的），
       否则会诱导管理员把它丢进群里，而它只能给一个人。
    ⚠️ 2026-10-28：文案里的 **xx** 一律改成 <b> —— 它被直接塞进 HTML，
       星号会原样显示出来（主人截图里就看到了"**一次性**"）。
    """
    _exp_txt = time.strftime("%Y-%m-%d %H:%M", time.localtime(invite.get("exp") or 0))
    if auth.invite_kind(invite) == "single":
        return ('⚠️ 这是<b>一次性专属邀请</b>（只发给一个人）：你领取之后，<b>这条链接立即失效</b>，'
                '别人再打开会提示"已被领取"。<br>'
                '⚠️ 你领到的这份令牌只属于你，请勿把自己的密钥转给别人。<br>'
                f'⚠️ 邀请 {_esc(invite.get("id"))}（{_esc(invite.get("note"))}）的链接'
                f'最晚 {_exp_txt} 过期；若你还没领而它已失效，找管理员要新的。')
    return ('⚠️ 这是群发的邀请，<b>仅限本群成员领取，请勿外传</b>；'
            '你领到的这份令牌只属于你，与其他人的互不干扰。<br>'
            f'⚠️ 邀请 {_esc(invite.get("id"))}（{_esc(invite.get("note"))}）将于 '
            f'{_exp_txt} 失效，之后找管理员要新的邀请即可。')


def _join_bind_page(u: dict, invite: dict, token: str, fresh: bool) -> HTMLResponse:
    """把某个身份的密钥以二维码形式给本人看（领取成功后的那一步）。

    ⚠️ `{qrsvg}` 必须**最后**替换：Python 的 str.replace 是全局替换，
       生成的 SVG 里有大量 `{...}` 路径数据，先注进去的话后面几个 replace 可能误伤它。
    """
    url = totp.otpauth_url(str(u.get("secret") or ""), str(u.get("name") or u["id"]))
    html = (_BIND_HTML
            .replace("{name}", _esc(u.get("name") or u["id"]))
            .replace("{secret}", _esc(u.get("secret") or ""))
            .replace("{days}", str(config.AUTH_TTL_DAYS))
            .replace("{hours}", str(config.AUTH_BIND_TTL_HOURS))
            .replace("{warn_extra}", _join_warn(invite))
            .replace("{qrsvg}", _qr_svg(url))
            .replace("{appqr}", TOTP_APP_QR)
            .replace("{appurl}", TOTP_APP_URL))
    resp = HTMLResponse(html)
    if fresh:      # 30 天有效，够他以后回来重扫（换手机时还能再看一眼二维码）
        resp.headers.append("set-cookie",
                            f"kb_claim={u['id']}.{auth.claim_sig(u['id'])}"
                            "; Path=/; Secure; SameSite=Lax; Max-Age=2592000; HttpOnly")
    return resp


def _join_form_page(invite: dict, token: str, name: str = "", staff: str = "",
                    fp: str = "", err: str = "") -> HTMLResponse:
    """邀请页第一步：让用户填**自己的名字**（2026-11-02）。

    老流程是给每人自动起名 `<邀请备注>-01` —— 管理员看不出是谁，也没法按人合并。
    """
    is_single = auth.invite_kind(invite) == "single"
    tip = ("⚠️ 只是打开本页<b>不会占用名额</b>，点下面的按钮才算领取；领取后这条链接立即失效。"
           if is_single else
           "⚠️ 领取后你会拿到一份<b>只属于你</b>的动态令牌，请勿转给别人。"
           "同一个人的多台设备填<b>同一个名字</b>即可合并成一个人。")
    btn = "确认领取我的令牌" if is_single else "继续"
    html = (_JOIN_FORM
            .replace("{token}", token)
            .replace("{nm_value}", _esc(name))
            .replace("{staff_value}", _esc(staff))
            .replace("{fp_value}", _esc(fp))
            .replace("{err}", f'<div class="err">{_esc(err)}</div>' if err else "")
            .replace("{btn}", btn)
            .replace("{tip}", tip)
            .replace("{iid}", _esc(invite.get("id")))
            .replace("{note}", _esc(invite.get("note") or "未命名邀请"))
            .replace("{exp}", time.strftime("%Y-%m-%d %H:%M",
                                            time.localtime(invite.get("exp") or 0))))
    return HTMLResponse(html)


def _join_merge_page(invite: dict, token: str, nm: str, hr: str,
                     fp: str, err: str = "") -> HTMLResponse:
    """邀请页第二步：名字已存在 → 让本人输 6 位动态码证明身份，再把本设备并进去。"""
    html = (_JOIN_MERGE
            .replace("{token}", token)
            .replace("{nm}", _esc(nm))
            .replace("{nm_value}", _esc(nm))
            .replace("{staff_value}", _esc(hr))
            .replace("{fp_value}", _esc(fp))
            .replace("{err}", f'<div class="err">{_esc(err)}</div>' if err else ""))
    return HTMLResponse(html)


def _join_done_page(u: dict, fp: str) -> HTMLResponse:
    """合并成功：直接签发这台设备的令牌（拿不到设备标识时降级为「回首页用动态码登录」）。"""
    fp = (fp or "").strip()
    res = auth.issue(fp, u["id"]) if fp else {"ok": False}
    if res.get("ok"):
        auth.touch_user(u["id"], fp)
        msg = "这台设备已经和你原来的身份绑定，正在进入助手…"
    else:
        msg = ("这台设备已合并到该身份。<b>请回到首页，用你验证器里的 6 位动态码登录</b>"
               "（当前浏览器没能生成本机设备标识）。")
    html = (_JOIN_DONE
            .replace("{nm}", _esc(u.get("name") or u["id"]))
            .replace("{msg}", msg))
    resp = HTMLResponse(html)
    if res.get("ok"):
        for c in auth.set_cookie_headers(res["token"], fp, res["max_age"]):
            resp.headers.append("set-cookie", c)
    resp.headers.append("set-cookie",
                        f"kb_claim={u['id']}.{auth.claim_sig(u['id'])}"
                        "; Path=/; Secure; SameSite=Lax; Max-Age=2592000; HttpOnly")
    return resp


def _join_page(token: str, kb_claim: str, allow_claim: bool,
               name: str = "", staff: str = "", code: str = "", fp: str = "",
               err: str = "") -> HTMLResponse:
    """邀请页的公共实现（GET / POST 共用）。

    流程（2026-11-02 改）：
      ① 这台浏览器领过（`kb_claim` 签名 Cookie）→ 直接给绑定页，刷新不会重复建档；
      ② 否则先出**填名字**的表单（GET 只渲染，不占一次性名额）；
      ③ 提交后：**已经有同名/同员工编号 的身份** → 要求输 6 位动态码证明是本尊，
         通过则把本设备并入该身份（不新建、设备数 +1）；
      ④ 没有 → 用**用户填的名字**建一个新身份，出二维码。

    ⚠️ 为什么合并要验动态码：名字不是唯一标识。若只凭"填了同一个名字"就把身份并过去，
       任何人填「张三」都能看到张三的密钥（u1 是管理员 ⇒ 直接提权）。
       验码用的是 `verify_code_for(uid, code)` —— **只认这一个身份的码**，
       不能用遍历全表的 `verify_code()`（那样撞上别人的码也算过）。
    ⚠️ 2026-10-28 的约束仍然成立：一次性邀请**打开页面不领取**，
       必须 POST（点提交）才消耗名额，否则 IM 预取/误触会吃掉唯一名额。
    """
    known = auth.read_claim_cookie(kb_claim)
    ok, why, invite = auth.verify_invite_token(token, known_uid=known)
    if not ok:
        return HTMLResponse(_BIND_ERR.replace("{why}", why), status_code=410)

    # ① 已经领过 → 幂等
    if known:
        u = next((x for x in auth.load_users()
                  if x.get("id") == known and x.get("enabled")), None)
        if u:
            return _join_bind_page(u, invite, token, fresh=False)

    # ② GET（以及一次性邀请的第一步）→ 只出表单，不领取
    if not allow_claim:
        return _join_form_page(invite, token, name, staff, fp, err)

    nm = auth.norm_name(name)
    if len(nm) < 2:
        return _join_form_page(invite, token, name, staff, fp,
                               err="请填写你的名字（至少 2 个字）")
    hr = auth.norm_staff_id(staff)
    cd = (code or "").strip()

    # ③ 找「同一个身份」：员工编号 优先（唯一），其次同名
    users = [x for x in auth.load_users() if x.get("enabled")]
    cands: list[dict] = []
    if hr:
        cands = [x for x in users if auth.norm_staff_id(x.get("staff_id") or "") == hr]
    if not cands:
        cands = [x for x in users if auth.norm_name(x.get("name")) == nm]

    if cands:
        hit = next((x for x in cands if auth.verify_code_for(x["id"], cd)), None) if cd else None
        if not hit:
            return _join_merge_page(
                invite, token, nm, hr, fp,
                err=("动态码不对或已过期，请填验证器里当前的 6 位码" if cd else ""))
        if not auth.mark_claim(invite.get("id"), hit["id"]):
            return HTMLResponse(
                _BIND_ERR.replace("{why}",
                    "这条邀请刚刚已经被别人领走了（或已作废）。请找管理员要一条新的。"),
                status_code=410)
        auth.audit("invite_merge", uid=hit["id"], name=hit.get("name"),
                   invite=invite.get("id"), fp=(fp or "")[:8])
        return _join_done_page(hit, fp)

    # ④ 没有同一个身份 → 用用户填的名字建一个新的
    u = auth.claim_identity(invite, nm)
    # ⚠️ 2026-10-23：一次性邀请**可能在锁内被别人抢走** ——
    #    verify 时还是空的，到这里已经不是了。这种情况必须给明确提示，
    #    不能继续往下走（否则会拿 None 当用户用，直接 500）。
    if not u:
        return HTMLResponse(
            _BIND_ERR.replace("{why}",
                "这条邀请是<b>一次性</b>的，刚刚已经被别人领走了（或已被作废）。<br>"
                "请找管理员要一条新的 —— 一次性邀请每条只能给一个人。"),
            status_code=410)
    if hr:
        auth.set_staff_id(u["id"], hr)
        u["staff_id"] = u.get("staff_id") or hr
    auth.audit("invite_claim", uid=u["id"], name=u.get("name"),
               invite=invite.get("id"), fp=(fp or "")[:8])
    return _join_bind_page(u, invite, token, fresh=True)


@app.get("/auth/join/{token}", response_class=HTMLResponse)
def auth_join(token: str, kb_claim: str = Cookie(default="")):
    """**邀请链接**：一条链接发到群里，群成员各自点开、自助领取一份动态令牌。

    与 `/auth/bind/<uid>`（发给指定的人）的区别：这条**不含身份**，谁点谁领 ——
    每人为自己拿到一份**独立**的 TOTP 密钥，彼此不干扰、可单独停用。

    ⚠️ 幂等靠**签名 Cookie** `kb_claim`：刷新页面不能又建一个身份，
       否则一次群发会在库里堆出一串没人用的垃圾用户。
       （`/auth/join/` 是免鉴权的，这个 Cookie 是唯一依据，所以它必须签名 ——
        否则伪造一个 uid 就能认领别人的身份。）
    ⚠️ 2026-10-28：**一次性邀请的 GET 不再领取**，只渲染表单 —— 见 `_join_page`。
    ⚠️ 2026-11-02：先出**填名字**的表单（名字从此由用户自己填，不再用邀请备注自动编号）。
    """
    return _join_page(token, kb_claim, allow_claim=False)


async def _read_form(request: Request) -> dict:
    """读表单字段。

    ⚠️ 网关转发 POST 时会把 `Content-Type` **统一改成 application/json**（worker.js），
       但 body 仍是 `name=xx&staff=yy` 这种 urlencoded —— **不能信 header**，
       按内容判断：以 `{` 开头当 JSON，否则按 query string 解析。
    """
    try:
        raw = (await request.body() or b"").decode("utf-8", "replace").strip()
    except Exception:
        return {}
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    try:
        return {k: (v[0] if v else "") for k, v in parse_qs(raw, keep_blank_values=True).items()}
    except Exception:
        return {}


@app.post("/auth/join/{token}", response_class=HTMLResponse)
async def auth_join_confirm(token: str, request: Request, kb_claim: str = Cookie(default="")):
    """表单提交 → 真正领取 / 合并（一次性邀请只在这一步消耗名额）。"""
    f = await _read_form(request)
    return _join_page(token, kb_claim, allow_claim=True,
                      name=str(f.get("name") or ""),
                      staff=str(f.get("staff") or ""),
                      code=str(f.get("code") or ""),
                      fp=str(f.get("fp") or ""))



@app.get("/auth/whoami")
def auth_whoami(kb_device: str = Cookie(default=""), kb_fp: str = Cookie(default="")):
    """当前设备是否已通过验证。**故意免 Key**：前端拿它决定要不要弹验证框。

    ⚠️ 2026-11-02：多返回 `needs_name` —— 老邀请流程自动起的名字（`灰度第一批-01`）
       对管理员没有意义，前端据此在页面上问一次真名（见 index.html 的「补充名字」条）。
    """
    if not auth.enabled():
        return {"enabled": False, "ok": True, "reason": "认证未启用"}
    ok, why, uid = auth.verify(kb_device, kb_fp)
    name, staff = "", ""
    role, role_set = auth.ROLE_DEFAULT, False
    if uid:
        for x in auth.load_users():
            if x.get("id") == uid:
                name = str(x.get("name") or "")
                staff = auth.norm_staff_id(x.get("staff_id") or "")
                role = auth.norm_role(x.get("role"))     # 老用户没该字段 → 默认角色
                role_set = bool(x.get("role"))           # 前端据此决定要不要弹「选角色」
                break
    return {"enabled": True, "ok": ok, "reason": why, "uid": uid, "name": name, "staff_id": staff,
            "role": role, "role_label": auth.ROLE_LABELS.get(role, ""), "role_set": role_set,
            "needs_name": bool(uid) and auth.looks_anonymous(name)}


class SelfNameIn(BaseModel):
    name: str = ""


@app.post("/auth/name")
def auth_set_name(body: SelfNameIn, kb_device: str = Cookie(default=""),
                  kb_fp: str = Cookie(default="")):
    """本人补充/修改自己的名字（2026-11-02）。

    ⚠️ 只改**自己**这一条（凭设备令牌定位 uid），所以免 Key ——
       与 `/auth/whoami` 同口径；没有有效令牌一律 401。
    ⚠️ 管理员的 `/admin/user/rename` 是另一条路（能改任何人），两者别混。
    """
    ok, why, uid = auth.verify(kb_device, kb_fp)
    if not ok or not uid:
        raise HTTPException(status_code=401, detail=why or "请先完成验证")
    n = auth.norm_name(body.name)
    if len(n) < 2:
        raise HTTPException(status_code=400, detail="名字至少 2 个字")
    u = auth.rename_user(uid, n)
    if not u:
        raise HTTPException(status_code=404, detail="没找到你的身份")
    auth.audit("self_rename", uid=uid, name=n)
    return {"ok": True, "name": n}


class SelfRoleIn(BaseModel):
    role: str = ""


@app.post("/auth/role")
def auth_set_role(body: SelfRoleIn, kb_device: str = Cookie(default=""),
                  kb_fp: str = Cookie(default="")):
    """本人设置角色（回答按角色组织）。

    ⚠️ 角色只决定**回答的段落形态**，不决定权限：
       「能删案例 / 能改名 / 能看管理面板」仍由 config.ADMIN_UIDS 判定。
       所以允许用户自选、随时改 —— 选错最坏是多看到或少看到几段，不存在提权。
    ⚠️ 免 Key，与 `/auth/whoami` 同口径：凭设备令牌定位 uid，无令牌一律 401。
    """
    ok, why, uid = auth.verify(kb_device, kb_fp)
    if not ok or not uid:
        raise HTTPException(status_code=401, detail=why or "请先完成验证")
    r = auth.norm_role(body.role)
    u = auth.set_role(uid, r)
    if not u:
        raise HTTPException(status_code=404, detail="没找到你的身份")
    auth.audit("self_role", uid=uid, role=r)
    return {"ok": True, "role": r, "role_label": auth.ROLE_LABELS.get(r, "")}


class SelfStaffIn(BaseModel):
    staff: str = ""


@app.post("/auth/staff")
def auth_set_staff_self(body: SelfStaffIn, kb_device: str = Cookie(default=""),
                      kb_fp: str = Cookie(default="")):
    """本人补充/修改自己的 LDAP 账号（原「员工编号」，2026-11-04）。

    背景：老邀请流程的用户没填过 LDAP 账号，管理页只能看到自动编号名，认不出是谁。
    ⇒ 前端在每次新会话时问一次（见 index.html 的「补全资料」弹窗）。

    ⚠️ 只改**自己**这一条（凭设备令牌定位 uid），所以免 Key ——
       与 `/auth/name`、`/auth/whoami` 同口径；没有有效令牌一律 401。
    ⚠️ 管理员的 `/admin/user/rename` 是另一条路（能改任何人），两者别混。
    """
    ok, why, uid = auth.verify(kb_device, kb_fp)
    if not ok or not uid:
        raise HTTPException(status_code=401, detail=why or "请先完成验证")
    h = auth.norm_staff_id(body.staff)
    if len(h) < 2:
        raise HTTPException(status_code=400, detail="请填写你的 LDAP 账号")
    if not auth.set_staff_id_self(uid, h):
        raise HTTPException(status_code=404, detail="没找到你的身份")
    auth.audit("self_set_staff", uid=uid, staff=h)
    return {"ok": True, "staff_id": h}


@app.post("/auth/logout")
def auth_logout():
    """清掉设备令牌 —— 换设备、借人用、怀疑泄露时点一下。"""
    resp = JSONResponse({"ok": True})
    for c in auth.clear_cookie_headers():
        resp.headers.append("set-cookie", c)
    return resp


# ---------------- 会话：跨设备续聊（2026-09-22，设计见 docs/设计-会话ID与减少追问-20260922.md）
# 场景：在这台电脑答到一半，换台电脑或转人工要接着看 —— 凭「会话 ID」把整段拉回来。
# ⚠️ 轻量定位（主人定）：**7 天硬删、不做归档**；拉不到就如实回「不存在或已过期」，不引导找回。

class SessionNewIn(BaseModel):
    title: str | None = None


class SessionSyncIn(BaseModel):
    sid: str
    title: str | None = None
    messages: list[dict] = []


@app.post("/session/new")
def session_new(body: SessionNewIn | None = None, x_api_key: str = Header(default="")):
    """建会话，返回不可猜的会话 ID（70bit 随机）。"""
    _check_key(x_api_key)
    sid = sessions.new_sid()
    sessions.save(sid, {"title": (body.title if body else "") or "", "messages": []})
    return {"sid": sid, "ttl_days": config.SESSION_TTL_DAYS}


@app.post("/session/sync")
def session_sync(body: SessionSyncIn, x_api_key: str = Header(default="")):
    """覆盖写入会话。前端每次对话变更后**静默**调用 —— 失败绝不影响问答主流程。"""
    _check_key(x_api_key)
    if not sessions.valid(body.sid):
        raise HTTPException(status_code=400, detail="会话 ID 格式不正确")
    res = sessions.save(body.sid, {"title": body.title, "messages": body.messages})
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail="保存失败")
    return res


@app.get("/session/{sid}")
def session_get(sid: str):
    """按会话 ID 拉取整段会话（含消息文本、日志、图片与简图的引用）。

    ⚠️ 免 Key：sid 是 70bit 不可猜凭证，与 `/img/` 同一思路（浏览器直接跳转带不了鉴权头）。
    """
    doc = sessions.load(sid)
    if not doc:
        raise HTTPException(status_code=404, detail=sessions.NOT_FOUND_MSG)
    return doc


@app.post("/gc")
def gc_now(x_api_key: str = Header(default=""), x_auth_user: str = Header(default=""),
           dry: bool | None = None):
    """手动清理过期会话与无主图片（不传 dry 则按 KB_GC_DRY_RUN 决定，默认真删）。"""
    _check_key(x_api_key)
    _check_admin(x_auth_user)          # 2026-09-24 P1：清理属管理操作，仅管理员
    return sessions.gc(dry)


def _gc_loop():
    """启动时跑一次，之后每 SESSION_GC_HOURS 跑一次。

    ⚠️ 用守护线程而不是事件循环任务：本项目没有 lifespan，守护线程最省事、也不怕 uvicorn reload。
    ⚠️ 无论清理出什么错都吞掉 —— 磁盘清理绝不能影响问答服务。
    """
    while True:
        try:
            res = sessions.gc()
            if res.get("sessions_expired") or res.get("uploads_expired"):
                print(f"[gc] 会话过期 {res['sessions_expired']} / 图片过期 {res['uploads_expired']}"
                      f"（仍被引用的图片 {res['referenced_uploads']}）")
        except Exception as e:
            print(f"[gc] 清理异常（已忽略）：{e}")
        time.sleep(max(1, config.SESSION_GC_HOURS) * 3600)


threading.Thread(target=_gc_loop, daemon=True).start()


@app.post("/upload")
def upload(body: UploadIn, x_api_key: str = Header(default="")):
    """报障截图存本机，返回随机 id；前端把 id 放进 /ask 的 images 里送模型读图。

    走「先存后引用」而不是每次内联 base64：图片要能跨刷新保留、要能随
    「复制给人工」转交，而且内联大图会把 /ask 请求体撑大。
    """
    _check_key(x_api_key)
    m = re.match(r"^data:(image/[a-zA-Z+.-]+);base64,([A-Za-z0-9+/=\s]+)$",
                 (body.dataUrl or "").strip(), re.S)
    if not m:
        raise HTTPException(status_code=400, detail="dataUrl 需为 data:image/...;base64,... 格式")
    mime = m.group(1).lower()
    ext = _IMG_MIME.get(mime)
    if not ext:
        raise HTTPException(status_code=400, detail=f"不支持的图片类型：{mime}（仅支持 jpg/png/webp/gif）")
    try:
        raw = base64.b64decode(re.sub(r"\s+", "", m.group(2)), validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 解码失败，请重试")
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(raw) > config.UPLOAD_MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"图片超过 {config.UPLOAD_MAX_MB}MB，请压缩后再传")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + ext
    (config.UPLOAD_DIR / name).write_bytes(raw)
    return {"id": name, "url": f"/img/{name}", "bytes": len(raw)}


# ---------------- 排查日志包（zip）----------------
# 一线现在常直接甩一个日志包过来：`<SN>_<YYYY-MM-DD>.zip`，里面通常两个 txt：
#   · `YYYY-MM-DD.txt`   POS 日志
#   · `bankposMMDD.txt`  嘉利程序日志（可能没有）
# 本接口只做「解开 + 按文件返回全文」，**不落盘、不截取**：
# 截取基准是前端的「发生时间」（只有前端知道），这里截了等于用错尺子。
# 前端拿到全文后仍走 `sliceLogByTime`（±1 小时），与单文件 txt 完全同一套口径。
ZIP_MAX_MB = 5            # 压缩包本体上限（与图片/txt 统一为 5MB；实测一份日志包 ~33KB）
ZIP_MAX_UNPACK_MB = 60    # 解压后总量上限（防 zip bomb）
ZIP_MAX_FILES = 20        # 最多解几个文件
ZIP_TEXT_EXT = (".txt", ".log", ".md", ".err", ".json", ".csv")
_IMPOS_NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.(txt|log)$", re.I)
_BANKPOS_RE = re.compile(r"bankpos", re.I)
_ZIP_NAME_RE = re.compile(r"^(?P<sn>[A-Za-z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})", re.I)


class ZipIn(BaseModel):
    dataUrl: str                 # data:application/zip;base64,... 或裸 base64
    name: str | None = None      # 原始文件名，用来解析 SN 与日期


def _decode_text(raw: bytes) -> str:
    """日志编码兜底：UTF-8 → GBK（老版嘉利程序常见）→ latin-1 无损兜底。"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def _classify_log(name: str) -> tuple[str, str]:
    """按文件名判来源，返回 (kind, 中文标签)，供前端分组展示、给模型分辨。"""
    if _BANKPOS_RE.search(name):
        return "bankpos", "嘉利程序日志"
    if _IMPOS_NAME_RE.match(name):
        return "pos", "POS 日志"
    return "other", "其他日志"


@app.post("/upload-zip")
def upload_zip(body: ZipIn, x_api_key: str = Header(default="")):
    """解开排查日志包，返回其中每个文本文件的内容。

    ⚠️ 只返回文本，**不存盘**：日志正文最终会随 /ask 一起提交，
    tools.build_context 会再落一份供人工复核，没必要在这里存两遍。
    """
    _check_key(x_api_key)
    s = (body.dataUrl or "").strip()
    m = re.match(r"^data:[^;]*;base64,(.*)$", s, re.S)
    if m:
        s = m.group(1)
    try:
        raw = base64.b64decode(re.sub(r"\s+", "", s), validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="日志包内容不是合法的 base64")
    if not raw:
        raise HTTPException(status_code=400, detail="日志包内容为空")
    if len(raw) > ZIP_MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"日志包超过 {ZIP_MAX_MB}MB")
    buf = io.BytesIO(raw)
    if not zipfile.is_zipfile(buf):
        raise HTTPException(status_code=400, detail="不是有效的 zip 文件")

    zip_name = (body.name or "").strip() or "日志包.zip"
    sn = date = ""
    mm = _ZIP_NAME_RE.match(Path(zip_name).name)
    if mm:
        sn, date = mm.group("sn"), mm.group("date")

    files: list[dict] = []
    total = 0
    try:
        with zipfile.ZipFile(buf) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                base = Path(info.filename).name          # 只取文件名，防 ../ 穿越
                if not base or base.startswith("."):
                    continue
                if not base.lower().endswith(ZIP_TEXT_EXT):
                    continue
                if len(files) >= ZIP_MAX_FILES:
                    break
                total += max(0, info.file_size)
                if total > ZIP_MAX_UNPACK_MB * 1024 * 1024:
                    raise HTTPException(status_code=413,
                                        detail=f"日志包解压后超过 {ZIP_MAX_UNPACK_MB}MB")
                with zf.open(info) as fh:
                    data = fh.read()
                text = _decode_text(data)
                if not text.strip():
                    continue
                kind, label = _classify_log(base)
                files.append({
                    "name": base, "kind": kind, "label": label,
                    "text": text, "lines": text.count("\n") + 1, "bytes": len(data),
                })
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="zip 已损坏，无法解开")
    except Exception as exc:                                # noqa: BLE001
        logging.warning("[upload-zip] 解压失败: %s", exc)
        raise HTTPException(status_code=400, detail="zip 解压失败")

    if not files:
        raise HTTPException(status_code=400,
                            detail="日志包里没有可读的文本文件（支持 .txt/.log/.md/.json）")
    # 大文件排前面：POS 日志通常更大、也更关键
    files.sort(key=lambda x: -x["bytes"])
    return {"zip": zip_name, "sn": sn, "date": date, "files": files}


@app.get("/img/{name}")
def get_img(name: str):
    """读回报障截图。id 是 32 位随机 hex，等同一次性的不可猜凭证，故不再校验 Key
    （<img src> 带不了鉴权头）。"""
    safe = Path(name).name
    if not _IMG_NAME_RE.match(safe):
        raise HTTPException(status_code=404, detail="not found")
    p = config.UPLOAD_DIR / safe
    if not p.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/diagram/{name}")
def get_diagram(name: str):
    """读回回答里的简图（tools.draw_diagram 生成的 SVG）。

    文件名是 SVG 内容的 sha1 前 16 位，等同不可猜的随机名，故与 /img 一样不再校验 Key
    （<img src> 带不了鉴权头）。内容是本地模板产出的、文本已 XML 转义，不含脚本。
    """
    safe = Path(name).name
    if not _DIAG_NAME_RE.match(safe):
        raise HTTPException(status_code=404, detail="not found")
    p = config.DIAGRAM_DIR / safe
    if not p.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p, media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.post("/summarize")
def summarize(body: SummarizeIn, x_api_key: str = Header(default="")):
    """把用户勾选的对话压成「转人工交接简报」，随原文一起复制给人工。

    summary 为空串 = 摘要不可用（模型调用失败 / echo 模式 / 未勾选有效消息），
    前端据此降级为「只复制原始对话」，而不是让复制整体失败。
    """
    _check_key(x_api_key)
    msgs = [m for m in (body.messages or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and str(m.get("content") or "").strip()]
    if not msgs:
        raise HTTPException(status_code=400, detail="messages 不能为空")
    return {"summary": llm.summarize(msgs)}


@app.post("/reload")
def reload_kb(x_api_key: str = Header(default=""), x_auth_user: str = Header(default="")):
    """知识库内容更新后调用，或直接重启服务。

    ⭐ 三层分离（2026-10-09）：本接口是「**数据层**」的热更新入口 ——
      · 数据层 `data/kb/*.md`    → 调本接口即可，**不需要重启**
      · 工具层 `scripts/*.js`    → 被 subprocess 调用，**每次现读文件**，push 完即生效、也不用重启
      · 主体层 `app/*.py` + `env.sh` → **必须重启**（Python 不支持热换代码；env 只在进程启动时读一次）
    ⇒ `scripts/deploy-to-ops.command` 会比对「主体指纹」，只有主体层真变了才重启服务。
    """
    _check_key(x_api_key)
    _check_admin(x_auth_user)          # 2026-09-24 P1：重建索引属管理操作，仅管理员
    knowledge.reload()
    return {"status": "ok", "docs": knowledge.count()}


@app.post("/code-rebuild")
def code_rebuild(x_api_key: str = Header(default=""), x_auth_user: str = Header(default="")):
    """源码索引强制重建（首次部署或源码目录大改后调用；一般由 codesearch 自动构建）。"""
    _check_key(x_api_key)
    _check_admin(x_auth_user)          # 2026-09-24 P1：重建源码索引属管理操作，仅管理员
    st = codesearch.build()
    return {"status": "ok", **st}


# ---------------- 案例审查 & 教练 ----------------
# 三个接口都是 POST：Worker 网关只对白名单 POST 路径做「鉴权 + 转发」，
# 复用同一套逻辑比再加一条带鉴权的 GET 分支更省事、更不容易漏放行。

@app.post("/feedback")
def feedback(body: CaseIn, x_api_key: str = Header(default="")):
    """上报/回写一个案例。

    前端在两处调用：① 用户点「已解决/未解决」（verdict + via=click）；
    ② 点「复制给人工」成功后自动回写 verdict=unsolved（via=copy）。
    ⚠️ 用户点了按钮就立刻返回，标记失败不能反过来影响用户操作本身。
    """
    _check_key(x_api_key)
    rec = cases.upsert_case(body.model_dump(exclude_none=True))
    # 2026-09-27 主人要求：点「未解决」时自动生成对话摘要并推 webhook。
    # 异步执行（见 alerts.py 的四条纪律），排队动作不影响这里的返回速度。
    # ⚠️ 2026-10-28：via=copy（复制给人工）**不再**推未解决通知 ——
    #   复制往往只是"转给同事看"，不代表答案没解决。前端已同步不再自动把 verdict 设成 unsolved，
    #   这里再挡一道：旧版前端缓存 / 其它调用方也不会把通知打出去。
    #   事故实证：pos-231y820f3fvtd8 先复制后标已解决 ⇒ 运维收到"未解决"通知、页面却是"已解决"。
    if (body.verdict or "") == "unsolved" and (body.via or "") != "copy":
        alerts.notify_unsolved(body)
    return {"status": "ok", "case": rec}


@app.post("/case/delete")
def case_delete(body: CaseDelIn, x_api_key: str = Header(default=""),
                x_auth_user: str = Header(default="")):
    """删除一条案例（连同它的教练记录）。

    ⚠️ 这是**物理删除**，但会先把原行归档到 `data/cases_deleted.jsonl`，误删可捞回。
    仅管理员：案例库是内部调优数据。
    """
    _check_key(x_api_key)
    _check_admin(x_auth_user)
    try:
        r = cases.delete_case(body.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", **r}


@app.post("/cases")
def cases_list(x_api_key: str = Header(default=""), x_auth_user: str = Header(default="")):
    """案例审查页的数据源：案例列表 + 已有教练记录 + 汇总统计。"""
    _check_key(x_api_key)
    _check_admin(x_auth_user)          # 2026-09-24 P1：全量案例库属内部数据，仅管理员可拉
    return {"cases": cases.list_cases(), "coach": cases.list_coach(),
            "stat": cases.count_cases(), "intent_labels": intent.INTENTS}


@app.post("/coach")
def coach_add(body: CoachIn, x_api_key: str = Header(default=""),
              x_auth_user: str = Header(default="")):
    """写一条教练记录：人工告诉助手「这个问题真正的问题点在哪、正确该怎么处理」。

    记录落盘后即刻生效——此后与本问题相似的提问都会带上它（优先级高于知识库）。
    """
    _check_key(x_api_key)
    _check_admin(x_auth_user)          # 2026-09-24 P1：写教练记录会改变助手回答口径，仅管理员
    data = body.model_dump(exclude_none=True)
    try:
        rec = cases.add_coach(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "coach": rec}


@app.post("/coach/active")
def coach_active(body: CoachActiveIn, x_api_key: str = Header(default=""),
                 x_auth_user: str = Header(default="")):
    _check_key(x_api_key)
    _check_admin(x_auth_user)          # 2026-09-24 P1：启停教练记录，仅管理员
    if not body.id:
        raise HTTPException(status_code=400, detail="id 不能为空")
    cases.set_coach_active(body.id, body.active)
    return {"status": "ok", "coach": cases.list_coach()}


@app.post("/retry")
def retry(body: RetryIn, x_api_key: str = Header(default=""),
          x_auth_user: str = Header(default="")):
    """用教练记录重跑同一个问题，让人在审查页直接看到「教完有没有变好」。

    coach_id 指定时强制带上那条记录（即使相似度没过门槛）——刚写完纠正就复测，
    不该因为算法阈值把自己刚教的东西挡在外面。
    """
    _check_key(x_api_key)
    _check_admin(x_auth_user)          # 2026-09-24 P1：复测属审查页功能，仅管理员
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    question, log_text = _split_attached_log(question)   # 与 /ask 同口径：日志进模型、不进检索
    res = knowledge.search_ex(question[:RETRIEVAL_QUERY_MAX])
    hits, weak, rare = res["hits"], res["weak"], res.get("rare") or {}
    code_hits = []
    if not weak and codesearch.need_code(question, hits):
        clues = codesearch.extract_clues(question, hits)
        if clues:
            code_hits = codesearch.search(clues)
    _intent = intent.resolve(question[:RETRIEVAL_QUERY_MAX],
                             has_intent_coach=any(c.get("intents") for c in cases.list_coach(active_only=True)))
    coach_hits = cases.match_coach(question[:RETRIEVAL_QUERY_MAX], intent=_intent)
    if body.coach_id:
        want = [c for c in cases.list_coach(active_only=False) if c["id"] == body.coach_id]
        if want and not any(c["id"] == body.coach_id for c in coach_hits):
            coach_hits = [{**want[0], "score": 1.0}] + coach_hits
    answer = llm.answer(question, hits, history=None, code_hits=code_hits,
                        coach=coach_hits, weak=weak, rare=rare, log=log_text)
    return {"answer": answer,
            "sources": [{"file": h["file"], "score": h["score"]} for h in hits],
            "coached": [{"id": c["id"], "score": c["score"]} for c in coach_hits]}

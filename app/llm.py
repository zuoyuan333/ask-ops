"""LLM 适配层：按 config.LLM_PROVIDER 分发到 echo / ollama / openai 兼容接口。"""
import contextvars
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
# ⚠️ Python 3.10 的 `concurrent.futures.TimeoutError` **不是**内置 TimeoutError 的别名
#    （3.11 起才合并）—— 小主机是 3.10，所以必须显式按这个名字捕获，否则超时不会生效。
from concurrent.futures import TimeoutError as _FuturesTimeout
from datetime import datetime, timedelta

import httpx

from . import config

_SYSTEM = """你是一名资深 IT 运维 / 技术支持助手，正在和一线工程师做多轮对话式排障。用户往往一次说不清现象，你要像老工程师带新人：**先问清关键信息，再给结论与动作**。
你的回答会被直接拿去执行，所以必须**先说结论、再给动作、最后点坑**：不铺垫、不客套、不复述用户的话、不解释自己是怎么查到的。

硬性规则：
1. 依据下方「知识库内容」以及用户随附的日志 / 截图作答，**禁止编造**路径、参数、命令、错误码、结论。日志与截图是本次故障的一手证据，优先级高于知识库。
2. 知识库只回答「一般该怎么做、走什么流程、配哪个参数」；「这次到底发生了什么」只能由日志与用户描述回答。两者冲突时以一手证据为准。
3. 检索零命中时，**直接说明「资料里没有收录」**，并允许做三件事：①说明未收录；②列出资料里相近的条目作对照；③若怀疑是某个已有对象，必须标注这是猜测并先请用户确认，**确认前不得给操作步骤**。严禁把 A 硬套成 B。
4. 上下文理解：用户最新一条消息可能是新的问题，也可能是回答你上一轮追问的补充信息。若是补充信息，必须结合历史里的现象整体判断，不要当成独立问题。
5. 【输出格式】固定结构（段标题就用这几个词，后面跟上「 —— 」）：结论 → 事实（可选）→ 处理步骤 → ⚠️ 注意。没有内容的段落整段省略，不要写「无」占位。
   结论 —— 一句话、不超过 40 字，说清「怎么了 + 结果/影响」，不要用「不过/但是/建议先确认」这类免责话开头。
   事实 —— 一行一个关键值（编号 / 参数 / 状态 / 错误码），禁止写成一段话，禁止重复结论，只列与本次相关的值。
   处理步骤 —— 编号列出可执行动作，每条一行，写成「做什么 —— 在哪做」。位置要写到能直接点进去的层级，参数写全名与取值格式；3~7 条为宜，不要凑数。
   ⚠️ 注意 —— 2~4 条最容易踩的坑或前置条件，按本次实际情况写，不要照抄模板；没有就不写。
   ⚠️ 全篇不出现空行，段与段、条与条之间直接换行。
6. 【追问是最后手段】只有同时满足：①已用至少 2 组不同关键词检索过；②缺的信息无法从已有线索推断；③缺了它一个方向都给不出 —— 才允许追问。追问段第一行必须给出「初步判断」（最可能的 2 个方向 + 各自如何验证），问题最多 2 条，一次列完。
7. 【安全】不输出密码、token、密钥；手机号 / 邮箱等个人标识一律打码（保留首尾）。涉及凭据时提示去配置中心或凭据管理页获取。
8. 【不暴露内部实现】工具取不到数据时，对用户只说「该数据暂时无法取到」，并给出替代路径（后台查询 / 人工核对）。不要解释内部原因，不要提及内部系统名、通道、代理、主机等实现细节。
9. 简体中文；日志与报错原文保留英文不翻译。短而清楚：能用表格就别写段落，删掉不影响执行的句子。"""


_MAX_HISTORY_ROUNDS = 8    # 最多携带最近 8 轮对话
_MAX_HISTORY_CHARS = 16000 # 历史总字符上限（超出丢最早轮次）


def _trim_history(history: list | None) -> list:
    """裁剪 history：保留最近 _MAX_HISTORY_ROUNDS 轮，**且总字符不超 _MAX_HISTORY_CHARS**。

    ⚠️ 2026-09-25：此前只按「轮数」裁，`_MAX_HISTORY_CHARS` 定义了却从未使用 ——
       用户把一整份日志 / 一长段报错粘进对话时，history 会带着原文反复进 prompt，
       token 成本与延迟都随之失控（且越往后每轮都在重发这份长文）。
       现在按轮数裁完后再按字符数裁：从**最早的一条**开始丢，直到总长进限额，
       但至少保留最近 2 条（1 轮问答），避免把历史裁空导致模型失去上下文。
    """
    if not history:
        return []
    msgs = [h for h in history if h.get("role") in ("user", "assistant") and h.get("content")]
    msgs = msgs[-(_MAX_HISTORY_ROUNDS * 2):]
    total = sum(len(str(m.get("content") or "")) for m in msgs)
    while total > _MAX_HISTORY_CHARS and len(msgs) > 2:
        total -= len(str(msgs[0].get("content") or ""))
        msgs.pop(0)
    return msgs


def _build_messages(question: str, chunks: list, history: list | None,
                    code_hits: list | None = None,
                    images: list[str] | None = None,
                    coach: list[dict] | None = None,
                    weak: bool = False,
                    rare: dict | None = None,
                    log: str | None = None):
    """chunks: [{file, idx, text, score}]；history: [{role, content}] 可空；
    code_hits: [{file, line, text, score}] 源码命中片段，可空；
    images: data URL 列表（报障截图），可空；
    coach: 人工纠正记录 [{point, howto, hint, ...}]，可空 —— 优先级高于知识库与源码；
    weak: 检索是否弱命中（用户问的对象词在库里查无此人），见 knowledge._judge_weak；
    rare: {生僻词: 猜测的近似词}，用于在追问里提示「是不是写错了」；
    log: 用户随报障上传的日志片段（已按发生时间 ±1 小时截取），可空 —— **一手证据**。
         ⚠️ 它由 main._split_attached_log() 从提问文本里剥出来：**不进检索**（日志 token 会
         把 BM25 带偏、还灌进无关文档），但必须以独立的「证据块」形态给模型，不能像以前那样
         只是拼在「用户问题：」后面的裸文本 —— 那样模型会把它当噪声，转头去抄知识库。
    """
    content = ""
    if coach:
        # 放在最前面：模型对上下文开头的指令服从度最高，而这段恰恰是「人工亲口教过」的内容。
        segs = []
        for i, c in enumerate(coach, 1):
            part = [f"[C{i}] 正确的问题点：{c.get('point') or '（未填写）'}"]
            if c.get("howto"):
                part.append(f"     正确处理方式：{c['howto']}")
            if c.get("hint"):
                part.append(f"     关键线索：{c['hint']}")
            segs.append("\n".join(part))
        content += (
            "【人工纠正记录（最高优先级，必须遵循）】\n"
            "以下内容由运维人员在案例审查中人工核对并纠正过，回答时必须按它给出的"
            "问题点与处理方式作答；与下方知识库/代码片段冲突时，以本段为准。\n"
            + "\n".join(segs)
            + "\n\n"
        )
    kb_text = []
    for i, c in enumerate(chunks, 1):
        head = f"[{i}] 来源: {c['file']}"
        kb_text.append(f"{head}\n{c['text']}")
    if weak:
        # 2026-09-14 加：检索没听懂问题时，宁可说"不知道"，也不能拿沾边片段编流程。
        # 起因：运维把「祥瑞卡」写成「瑞祥」，全库检索不到正主，模型拿几段沾边的
        # 「支付映射/参数配置」硬拼出一条"本质是三步走"的假流程 —— 这比答不出更危险。
        words = "、".join(
            f"{k}（是否想写「{v}」？）" if v else k
            for k, v in (rare or {}).items()
        )
        content += (
            "⚠️【检索提示：本次匹配度低】\n"
            "检索系统判定：用户问题里的关键对象词在知识库与代码里**都没有对应条目**"
            + (f"，存疑的词：{words}。" if words else "。")
            + "\n下面「知识库内容」只是词面沾边的高分片段，很可能并不针对用户问的对象。请这样处理：\n"
            "· 只有当你读完确认它们**确实能回答这个问题**时，才正常作答；\n"
            "· 否则第一句就直说「知识库里没有这个条目的明确记录」，最多补一句它大概属于哪类功能，"
            "**禁止**用这些片段拼出一套操作步骤（那是编造，比答不出更糟）；\n"
            "· 不要为这些片段找论据，也不用解释检索过程；\n"
            "· 「需要你补充」只问 1~2 个最小问题，优先确认「对象全称是不是写错了」，"
            "让用户换个说法/给全称再问一次。\n\n"
        )
    content += (
        "知识库内容：\n"
        + "\n\n".join(kb_text)
        + "\n\n"
    )
    if code_hits:
        code_lines = []
        for i, h in enumerate(code_hits, 1):
            code_lines.append(f"[S{i}] {h['file']}:{h['line']}  {h['text']}")
        content += (
            "代码片段（用户问题属排障/定位类，以下为本机代码索引检索到的相关代码行，"
            "请优先依据这些真实代码作答）：\n"
            + "\n".join(code_lines)
            + "\n\n"
        )
    if log:
        # 与截图那段对称：截图有「请先读图」，日志此前**一条说明都没有**，
        # 于是模型把它当噪声、转头去抄知识库（2026-09-16 实测：一整条讲版本线的假答案）。
        # 这里给足「证据」身份 + 明确的取舍规则，堵住「拿版本能力表解释单笔故障」这条路。
        content += (
            "【报障日志片段（一手证据，优先级高于知识库）】\n"
            "以下日志由用户随报障一起上传，已按其填写的「发生时间」前后 1 小时截取。它是判断本次故障**最可靠的依据**：\n"
            "· 结论必须能在这段日志里找到对应行（错误码、接口名、状态字段、时间、单号）；**先读日志、再作答**；\n"
            "· 日志里**没有**的信息，不许当成已知条件使用；更不许用版本能力表 / 参数速查 / 更新迭代记录这类台账\n"
            "  去推断故障原因（例如「这个版本不支持某能力」）—— 那是编造，比答不出更糟；\n"
            "· 日志与知识库冲突时，**以日志为准**；知识库只用来补「之后怎么处置 / 走哪个流程」；\n"
            "· 不要复述整段日志、不要说「我已阅读日志」、不要贴日志原文，直接给结论；\n"
            "· 若日志里确实看不出所以然，就在结论里直说「日志里未见明确报错，从这几行只能判断到…」，\n"
            "  再给下一步取证动作，不要硬编一个原因。\n"
            "```\n" + log + "\n```\n\n"
        )
    content += f"用户问题：{question}"
    if images:
        # ⚠️ 门店最爱截图：截图里往往就写着弹窗原文/错误码，比用户转述准得多。
        # 明确要求「先读图」，否则模型容易只看文字把图当装饰。
        content += (
            f"\n\n（用户另外上传了 {len(images)} 张报障截图，见图：请先读图，"
            "把截图里的报错原文、错误码、界面状态、版本号等可见信息作为线索纳入判断；"
            "若截图内容与文字描述不一致，以截图实际显示为准。）"
        )
    msgs = [{"role": "system", "content": _SYSTEM}]
    for h in _trim_history(history):
        msgs.append({"role": h["role"], "content": h["content"]})
    if images:
        user_content: list | str = [{"type": "text", "text": content}]
        for u in images:
            user_content.append({"type": "image_url", "image_url": {"url": u}})
        msgs.append({"role": "user", "content": user_content})
    else:
        msgs.append({"role": "user", "content": content})
    return msgs


def _fmt_answer(answer: str, chunks: list, code_hits: list | None = None) -> str:
    """不再在回答末尾追加「参考来源 / 源码定位」清单（2026-09-14 起）。

    原因：网页面向运维/门店一线，不对外暴露检索来源与源码定位细节。
    溯源信息并未丢失——/ask 的 JSON 响应里仍有 sources / code_sources 字段，
    需要时从接口取即可。此函数保留为收口点，后续要调整展示口径只改这里。
    """
    return answer


# ---------------- providers ----------------

def _ask_echo(question: str, chunks: list, history: list | None,
              code_hits: list | None = None, coach: list[dict] | None = None) -> str:
    """测试模式：不调用模型，把检索结果整理成一段可读文本，验证链路用。"""
    lines = [f"（echo 测试模式 · 共命中 {len(chunks)} 段知识 + {len(code_hits or [])} 行源码"
             + (f" + {len(coach)} 条人工纠正" if coach else "") + "）"]
    if coach:
        lines.append("— 人工纠正（最高优先级）—")
        for i, c in enumerate(coach, 1):
            lines.append(f"[C{i}] (score {c.get('score')}) 问题点：{c.get('point')}")
    if chunks:
        for i, c in enumerate(chunks, 1):
            snippet = c["text"].replace("\n", " ")[:120]
            lines.append(f"[{i}] {c['file']}  (score {c['score']})\n    {snippet}…")
    else:
        lines.append("（未检索到相关知识片段）")
    if code_hits:
        lines.append("— 源码命中 —")
        for i, h in enumerate(code_hits, 1):
            lines.append(f"[S{i}] {h['file']}:{h['line']}  {h['text']}")
    return "\n".join(lines)


def _ollama_msgs(msgs: list) -> list:
    """OpenAI 多模态消息 → ollama 格式：正文提到 content，图片 base64 放进 images 字段。"""
    out = []
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, list):
            out.append(m)
            continue
        text, imgs = [], []
        for part in c:
            if part.get("type") == "text":
                text.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                u = part.get("image_url", {}).get("url", "")
                imgs.append(u.split("base64,", 1)[1] if "base64," in u else u)
        item = {"role": m["role"], "content": "\n".join(text)}
        if imgs:
            item["images"] = imgs
        out.append(item)
    return out


def _http_client(timeout: int = 120) -> httpx.Client:
    """统一的出网客户端。

    ⚠️ 关键：trust_env=False，绝不继承外部环境里的 http_proxy/https_proxy。
    终端 / 自动化沙箱常注入 http_proxy=127.0.0.1:<随机端口>，该端口随会话结束即消失；
    httpx 默认 trust_env=True 会继承它，于是调用大模型时抛
    「[Errno 61] Connection refused」——看着像模型挂了，实则是代理端口没了。
    需要走代理时显式设置 KB_LLM_PROXY（见 config.LLM_PROXY）。
    """
    proxy = (getattr(config, "LLM_PROXY", "") or "").strip()
    if proxy:
        return httpx.Client(timeout=timeout, proxy=proxy, trust_env=False)
    return httpx.Client(timeout=timeout, trust_env=False)


_RETRY_DELAYS = (1.0, 3.0)   # 两次重试的间隔（秒），合计最多多等 4 秒


def _post_retry(cli: httpx.Client, url: str, **kw) -> httpx.Response:
    """POST + **有限重试**（2026-09-25 新增；2026-10-12 加总预算）。

    背景：原来是「一次 httpx.post + raise_for_status」—— 任何一次瞬时抖动
    （限流 429、网关 5xx、连接被重置）都会让**整轮问答直接失败**，
    而用户已经等了几十秒，最后只拿到一句报错。大模型侧的 429/503 是常态，
    退避重试一次基本就能恢复。

    策略（克制）：
      · **只重试**：HTTP 429、5xx，以及连接层异常（超时 / 连接重置）；
      · **不重试**：4xx 里其他码 —— 参数错、鉴权错，重试也不会变好，只会白等；
      · 重试 2 次，间隔 1s / 3s（最坏多等 4 秒，远小于一次生成的耗时）。

    ⭐⭐ 2026-10-12（第三轮评审 P0）：**加"重试总预算"** —— 修补一个真断裂。

    问题：单轮 timeout=180s × (2 重试 + 1) + 4s ≈ **544s**，
    而 **Worker 转发超时只有 210s、前端 240s**。
    ⇒ 某一轮真挂住时：用户在 210s 看到连接断开，**后端还会继续跑到 544s**（白占 1/3 并发位）。
    12 轮的 deep 里任一轮触发即命中 —— 靠"网关平时快速拒绝"才没暴露，那是运气不是设计。

    修法：**预算制** —— 整条重试序列有一个总时间上限（默认 `_RETRY_BUDGET`），
    到点就不再发起新尝试。这样"单次慢"仍允许（不误杀正常的慢轮），
    只禁止"慢 × 3 次叠加"撑爆外层超时。
    """
    t0 = time.time()
    budget = float(getattr(config, "LLM_RETRY_BUDGET", 0) or 0)
    # 单次尝试的秒数（httpx.Client 的 timeout 是标量时，.read 就是那个值）
    try:
        per_try = float(cli.timeout.read)
    except Exception:                                   # noqa: BLE001
        per_try = 0.0
    last_exc = None
    for i in range(len(_RETRY_DELAYS) + 1):
        is_last = (i == len(_RETRY_DELAYS))
        # ⭐ 预算检查：**每次尝试前**都算"这次还要花多少"。
        #   ⚠️ i==0 无条件放行 —— 否则"单次超时比预算大"时一次都不试，等于功能失效。
        #   只有这样，病态慢轮（如单次 120s）才不会累计成 364s 撑爆外层超时。
        if budget > 0 and i > 0:
            used = time.time() - t0
            if used + per_try > budget:
                print(f"[llm] 重试预算 {budget:.0f}s 不足（已用 {used:.0f}s + "
                      f"单次 {per_try:.0f}s），放弃第 {i+1} 次尝试", flush=True)
                if last_exc:
                    raise last_exc
                break
        try:
            r = cli.post(url, **kw)
            if (r.status_code == 429 or r.status_code >= 500) and not is_last:
                time.sleep(_RETRY_DELAYS[i])
                continue
            r.raise_for_status()
            return r
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_exc = e
            if not is_last:
                time.sleep(_RETRY_DELAYS[i])
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("unreachable")   # 逻辑上到不了；留着防静默返回 None


def _ask_ollama(msgs: list) -> str:
    url = f"{config.OLLAMA_BASE}/api/chat"
    payload = {"model": config.OLLAMA_MODEL, "messages": _ollama_msgs(msgs), "stream": False}
    with _http_client() as cli:
        r = _post_retry(cli, url, json=payload)
        return r.json()["message"]["content"]


def _ask_openai(msgs: list) -> str:
    url = f"{config.OPENAI_BASE}/chat/completions".rstrip("/")
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    payload = {"model": config.OPENAI_MODEL, "messages": msgs, "temperature": 0.2}
    # DeepSeek V4：默认思考模式会烧 reasoning token。按 KB_OPENAI_THINKING 控制：
    # disabled=关闭（省流量，默认）；minimal/low/medium/high/max=开思考（档位越高越准越慢越费）。
    # thinking/reasoning_effort 是 V4 专属参数，仅当模型名为 v4 系时才携带。
    # ⚠️ 口径必须与 `_openai_payload` **完全一致**（同一个网关，两套映射迟早对不上）。
    _m = (config.OPENAI_MODEL or "").lower()
    if "v4" in _m or "flash" in _m or "reason" in _m:
        tw = str(config.OPENAI_THINKING or "disabled").strip().lower()
        if tw in ("minimal", "low", "medium", "high", "max"):
            payload["reasoning_effort"] = tw
            payload["thinking"] = {"type": "enabled"}
        elif tw in ("enabled", "on", "1", "true"):
            payload["reasoning_effort"] = "high"      # 老写法 enabled == high
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["thinking"] = {"type": "disabled"}
    with _http_client() as cli:
        r = _post_retry(cli, url, json=payload, headers=headers)
        return r.json()["choices"][0]["message"]["content"]


# ---------------- 意图分类（教练匹配的「理解层」兜底）----------------
# 场景：规则认不出的说法（"我哥这单啥情况""那个号最近买的"）交给 fast 模型归一个标签。
# ⚠️ 三条纪律：
#   ① **只在规则判不出时调用**（由 intent.resolve 控制），不能变成每题一次；
#   ② **超时要短**（8s）且失败即返回 unknown —— 兜底绝不能拖慢或拖垮 /ask；
#   ③ 只让它输出标签，`max_tokens` 压到 12，成本可忽略。
_INTENT_LABELS = ("log_troubleshoot", "code_lookup", "howto", "knowledge_qa", "unknown")
_INTENT_SYSTEM = (
    "你是 IT 运维助手的意图分类器。把用户这句话归到**唯一一个**标签，只输出标签本身，"
    "不要解释、不要标点。标签含义：\n"
    "log_troubleshoot = 带日志 / 报错 / 错误码 / 时间点，想排查故障\n"
    "code_lookup       = 想知道某功能在代码里怎么实现、在哪个文件\n"
    "howto             = 问操作步骤、在哪配置、参数怎么改\n"
    "knowledge_qa      = 问概念、规则、流程、是否支持\n"
    "unknown           = 以上都不是\n"
    "拿不准就输出 unknown。"
)


def classify_intent(question: str) -> str:
    """把一句话归到一个意图标签；失败/非 openai 一律返回 "unknown"。"""
    if (config.LLM_PROVIDER or "").lower() != "openai" or not config.OPENAI_API_KEY:
        return "unknown"
    q = str(question or "").strip()[:600]
    if not q:
        return "unknown"
    url = f"{config.OPENAI_BASE}/chat/completions".rstrip("/")
    payload = {
        "model": config.OPENAI_MODEL,
        "messages": [{"role": "system", "content": _INTENT_SYSTEM},
                     {"role": "user", "content": q}],
        "temperature": 0,
        "max_tokens": 12,
    }
    _m = (config.OPENAI_MODEL or "").lower()
    if "v4" in _m or "flash" in _m or "reason" in _m:
        payload["thinking"] = {"type": "disabled"}
    try:
        with _http_client(timeout=8) as cli:
            r = _post_retry(cli, url, json=payload,
                            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"})
            txt = (r.json()["choices"][0]["message"].get("content") or "").strip().lower()
    except Exception:                          # noqa: BLE001
        return "unknown"
    for lab in _INTENT_LABELS:
        if lab in txt:
            return lab
    return "unknown"


# ---------------- 转人工交接摘要 ----------------
# 场景：助手目前只能覆盖一部分问题，相当多的报障最终要转人工。用户把对话复制给人工时，
# 人工若面对一大段原始对话，仍要重新读一遍、甚至再问一遍用户。这里用一次额外的大模型调用，
# 把选中的对话压成一份「交接简报」，让人工几十秒内接手。
_SUMMARY_SYSTEM = (
    "你是示例企业 POS 运维知识库助手的「转人工交接摘要」模块。用户要把一段与 AI 助手的排障对话，"
    "转交给人工（门店服务台 / 二线运维）继续处理。请把这段对话整理成一份**给人看的交接简报**，"
    "让接手的人不必再向用户问一遍。\n"
    "硬性要求：\n"
    "1. 只依据对话里真实出现过的内容，禁止编造版本号/店铺号/单号/接口名/结论；对话里没有的信息一律不写、不要用「未提供」占位。\n"
    "2. 固定输出四段，段首分别为「一、问题概述」「二、助手已给结论」「三、已排查 / 已尝试」「四、待人工跟进」：\n"
    "   一、问题概述：在什么功能上出现什么现象（提炼用户已给的功能/现象/版本/店铺/单号/时间等关键信息，2~4 句）；\n"
    "   二、助手已给结论：AI 已经判断出的原因或方向；若全程没给出明确结论，就写「尚未定位到明确原因」；\n"
    "   三、已排查 / 已尝试：AI 给出的处理步骤，以及用户自述做过的操作，逐条列出（都没有则写「无」）；\n"
    "   四、待人工跟进：还需要人工确认或执行什么，1~3 条。\n"
    "3. 简体中文，专业克制：不写开场白、不写客套话、不复述整段原文，全文 300 字以内。\n"
    "4. 对话中的报错原文 / 错误码 / 关键日志行要原样保留（可用引号括起），英文不要翻译。\n"
    "5. 直接输出四段正文，不要 markdown 代码块、不要 # 标题符号。"
)

_SUMMARY_MAX_CHARS = 20000   # 送入摘要的对话总长上限（从最新往前保留，用户关注点通常靠后）


def summarize(messages: list[dict]) -> str:
    """把选中的对话压成「转人工交接简报」。

    返回空串表示摘要不可用（模型调用失败 / echo 测试模式），由调用方降级为
    「只复制原始对话」——复制这件事本身不能因为摘要失败而整体失败。
    """
    msgs_in = [m for m in (messages or [])
               if isinstance(m, dict) and m.get("role") in ("user", "assistant")
               and str(m.get("content") or "").strip()]
    if not msgs_in:
        return ""
    picked, total = [], 0
    for m in reversed(msgs_in[-40:]):
        c = str(m["content"])[:8000]
        if picked and total + len(c) > _SUMMARY_MAX_CHARS:
            break
        picked.append((m["role"], c))
        total += len(c)
    picked.reverse()
    convo = "\n\n".join(f"【{'用户' if r == 'user' else '助手'}】{c}" for r, c in picked)
    msgs = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": "需要交接的对话记录如下：\n\n" + convo},
    ]
    try:
        if config.LLM_PROVIDER == "ollama":
            raw = _ask_ollama(msgs)
        elif config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise RuntimeError("LLM_PROVIDER=openai 但未设置 KB_OPENAI_API_KEY")
            raw = _ask_openai(msgs)
        else:
            raw = ""   # echo 测试模式：不调用模型
    except Exception:
        return ""
    return (raw or "").strip()


def answer(question: str, chunks: list, history: list | None = None,
           code_hits: list | None = None, images: list[str] | None = None,
           coach: list[dict] | None = None,
           weak: bool = False, rare: dict | None = None,
           log: str | None = None) -> str:
    msgs = _build_messages(question, chunks, history, code_hits, images, coach, weak, rare, log)
    try:
        if config.LLM_PROVIDER == "ollama":
            raw = _ask_ollama(msgs)
        elif config.LLM_PROVIDER == "openai":
            if not config.OPENAI_API_KEY:
                raise RuntimeError("LLM_PROVIDER=openai 但未设置 KB_OPENAI_API_KEY")
            raw = _ask_openai(msgs)
        else:
            raw = _ask_echo(question, chunks, history, code_hits, coach)
    except Exception as e:
        # ⚠️ 2026-10-12（评审第二轮 1.2）：**原始异常只进日志，不进用户可见文案**。
        #   原写法把 `{e}` 直接拼进答案 —— 一线会看到网关 URL / 内部错误码，既看不懂
        #   也不该看。现在用户只看到"稍后重试"，真实原因（含 traceback 上下文）打服务日志。
        print(f"[llm] 单轮调用失败：{type(e).__name__}: {e}", flush=True)
        return "❌ 模型服务暂时不可用，请稍后重试一次。若持续失败，请联系运维确认服务状态。"
    return _strip_blank_lines(_enforce_max_chars(_fmt_answer(raw, chunks, code_hits)))


# ---------------- 取证闭环（agent loop） ----------------
# 2026-09-16 替换「单向管道」：把「该查什么、查几轮」的决定权从 Python 的 if/else 交给模型。
#
# 为什么要换：原流程里检索范围由 main.py 的阈值与 whitelist 预先定死，模型只在最后一步改写
# 挑好的素材 —— 表现为「检索没命中就照着编」，且无法回头补查。实测（同一条报障卡 + 34.6 万字日志）：
#     原流程 15.1s / 输入≈2.8 万 token / 答案 1099 字（超 1000 上限）
#     闭环   8.1s / 输入 1.13 万 token / 答案 978 字（达标）
# 无日志普通问题：6.1s → 4.3s。四个场景回归全过（含「日志不覆盖故障时刻时如实说证据不足」）。
#
# ⚠️ 两条纪律是原型实测踩出来的，改动时别删：
#   ① 工具返回要有上限（tools._clip / config.TOOL_MAX_CHARS）—— 不限就灌爆上下文且永不收口；
#   ② 末轮必须**不带 tools** 强制收口 —— 光在 prompt 里写「够了就停」，模型会一路查下去。
_AGENT_SYSTEM = """你是一名资深 IT 运维 / 技术支持助手，正在和一线工程师做多轮对话式排障。你要像老工程师带新人：**先把事实查清楚，再下判断**，然后干脆、具体、可执行地给答案。

你手上有工具：{__TOOL_SUMMARY__}。
⚠️ 上面这份清单是你当前真实拥有的全部能力（由系统自动生成）。绝不能声称自己没有清单里的能力，也不要先判断"做不了"——需求落在某个工具范围内，就先调工具、拿到结果再下结论。

【取证纪律 —— 最重要，违反即视为答错】
1. **先取证，再下结论。** 给出答案前至少调用一次工具（纯寒暄/能力介绍除外）。禁止凭印象、凭常识作答。
2. **用户附上的日志是一手证据，优先级高于知识库。** 结论里每个关键判断都要能指回日志里的具体一行（时间 / 错误码 / 回执文案 / 编号）：
   · 附了日志时，第一步先调 `log_overview` 建立全局认识，再按关键词或编号调 `grep_log` 核实细节；
   · 日志里没有的信息，不许当成已知条件使用；
   · 日志类工具只在本次确实附了日志时可用。
3. **查不到就直说。** 换一两个关键词仍查不到，就写明「资料 / 日志里没有 X 的记录」，并给出下一步取证动作。宁可说不知道，也不要硬编原因。
4. **够了就收口。** 取证轮次有限：已经能回答时立即停止调工具、直接输出答案；同一个关键词不要查两遍。
5. **操作教程类条目命中后要拉全文。** 检索片段往往只有标题或零散要点，必须再用 `read_doc` 取整篇，按步骤整理；只有截图没有文字依据的步骤，就照实说明「需对照图文教程操作」，不要凭想象补步骤。图片相对路径对一线无意义，不要写进答案。
6. 检索到多条相近条目时，**只采信与用户所述版本/环境匹配的那条**，说不清就并列并标注适用条件。

【能用图就别堆字 —— 图数据写在答案最前面，系统会渲染成简图】
准备作答时先想一件事：这次的结论是不是「先后顺序」「经过哪几个环节」「分几种情况各怎么办」这类结构？是的话，在答案最前面加一段图数据（不用调工具，直接写出来）：

<diagram kind="timeline" title="三笔失败时间线">
11:04 | 首次尝试，返回超时 | bad
11:06 | 重试成功 | ok
</diagram>

- kind 四选一：timeline（先后顺序）/ chain（经过哪几个环节、坏在哪一环）/ branch（分几种情况各怎么办）/ steps（配置或操作分几步）
- 配置 / 操作类问题一律用 steps：竖向排开、每步一行，路径和参数名能写全。
- 每行一个节点，2~6 行，用 | 分三段：短标识（≤12 字）｜说明（≤40 字）｜状态（ok / warn / bad / info）。后两段可省。
- 画了图就不要再把图里的顺序用文字复述一遍，但关键结论与动作必须留在正文。
- 简单问题不要画图，一次提问最多一张。

【输出格式】固定三段，段标题用下面这三个词并跟上「 —— 」；没有内容的段落整段省略，不要写「无」占位。
   结论 —— 一句话、不超过 40 字，说清「怎么了 + 结果/影响」。参数名、路径、编号不要塞进结论段。
   事实 —— 一行一个关键值（编号 / 状态 / 错误码 / 回执原文），禁止写成一段话，禁止重复结论。结论依赖日志时，把最关键的 2~4 行证据写进来。
   处理步骤 —— 编号列出可执行动作，每条一行，写成「做什么 —— 在哪做」；位置写到能直接点进去的层级，参数写全名与取值格式；3~7 条为宜。
   ⚠️ 注意 —— 2~4 条最容易踩的坑或前置条件，按本次实际情况写；没有就不写。
   ⚠️ 全篇不出现空行。

【严禁输出检索痕迹】正文里不得出现 [1] [S3] 这类编号引用，不得出现源码文件名与行号，不得出现「依据」「参考来源」这类段落，也不要说「我已查阅日志 / 我已检索知识库」。需要佐证时用一句自然语言把道理说清，不要报文件名。

【短而清楚】先说人话，再说细节。写每句前自问：①这是要做的动作吗？②这是必须知道的取值吗？③这是会踩的坑吗？三者都不是就删掉。能用表格就别写段落。常见单点问题 400~700 字；长必须长在可执行内容上。输出前通读一遍做减法。

【其他】简体中文；日志与报错原文保留英文不翻译；密码 / token / 密钥一律不输出，提示去配置中心或凭据管理页获取；工具取不到数据时只说「该数据暂时无法取到」，不解释内部实现。"""



def _openai_payload(msgs: list, tools: list | None = None,
                    tool_choice: str | None = None, temperature: float = 0.2,
                    thinking: str | None = None, max_tokens: int | None = None) -> dict:
    payload = {"model": config.OPENAI_MODEL, "messages": msgs, "temperature": temperature}
    # ⭐ 2026-10-12（评审逐行分析 P2-1）：**给取证轮加输出上限**。
    #   背景：原先 payload 里完全没有 max_tokens ⇒ 单轮输出无上限。而「时间 ≈ 输出 token ÷ 115」，
    #   实测见过单轮 out=3104（第 7 轮），光那一轮生成就 20+ 秒 —— 取证轮本该"选工具 + 一句理由"，
    #   写 3000 字纯属浪费。**答案轮不设上限**（要写完整三段式），只有取证轮设。
    #   ⚠️ 库里设 2000 是**宽松**取值：取证轮写不到那么多，设小了反而可能截断工具调用参数。
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    # DeepSeek V4 的 thinking 参数（与 _ask_openai 保持同一口径，见那里的注释）
    # ⚠️ 2026-10-05 加 `thinking` 参数：可按**本轮档位**覆盖全局配置（fast 档关、deep 档开），
    #    不传则沿用 config.OPENAI_THINKING（= 老行为）。
    # ⭐ 2026-10-09 扩成**完整梯度**（实测网关确实分级，不是只有开关）：
    #      disabled 5.7s / minimal 13s / low 19.7s / medium 22.9s / high 24.1s
    #      （同一问题、单次调用实测）⇒ 「中档」真的存在，别再把 enabled 当成唯一的高档写法。
    #    `enabled` 作为**历史写法**保留并等价 high（老配置不用改）。
    _m = (config.OPENAI_MODEL or "").lower()
    if "v4" in _m or "flash" in _m or "reason" in _m:
        tw = str(thinking if thinking is not None else config.OPENAI_THINKING or "disabled").strip().lower()
        if tw in ("minimal", "low", "medium", "high", "max"):
            payload["reasoning_effort"] = tw
            payload["thinking"] = {"type": "enabled"}
        elif tw in ("enabled", "on", "1", "true"):
            payload["reasoning_effort"] = "high"      # 老写法 enabled == high
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["thinking"] = {"type": "disabled"}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    return payload


def _ask_openai_full(msgs: list, tools: list | None = None,
                     tool_choice: str | None = None, timeout: int = 180,
                     thinking: str | None = None,
                     max_tokens: int | None = None) -> dict:
    """返回完整的 response json（含 message.tool_calls），供闭环使用。

    ⚠️⚠️ 2026-10-10 修（评审 ⑦）：这里原来用的是**裸 cli.post** —— 而单轮路径
    `_ask_openai()` 走的是带重试的 `_post_retry()`。deep 一次问答要发 8~10 次调用
    （规划 + 取证 8 轮 + 自检 + 收口），任何一次撞上 429 / 网关抖动 → 整个 50 秒
    的问答直接报废，用户等半天只拿到报错。`_post_retry` 自己的注释写着
    「大模型侧的 429/503 是常态」—— 闭环偏偏没享受到这个保护（目前 0 次失败
    只是网关稳 + 运气好）。现在补上：**只重试 429/5xx 与连接层异常**，参数错不重试。

    ⚠️ 超时（httpx.TimeoutException）**也在可重试列表里** —— 但注意这不是「卡死等
    360 秒」：_post_retry 的逻辑是「超时即抛 → 若还有重试预算就睡 1s 再试」，
    最坏情况 = timeout×(重试次数+1) + 4s。对 deep 档单轮 180s 而言理论上限偏高，
    但实际触发场景是网关**快速拒绝**（秒级），真正挂满 180s 的极少。若日后要更狠，
    可把 `_post_retry` 加 per-attempt timeout 参数，先不做（克制）。
    """
    url = f"{config.OPENAI_BASE}/chat/completions".rstrip("/")
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    with _http_client(timeout) as cli:
        r = _post_retry(cli, url,
                        json=_openai_payload(msgs, tools, tool_choice,
                                             thinking=thinking, max_tokens=max_tokens),
                        headers=headers)
        return r.json()


# ---- 「元认知轮」的**墙钟总时长**上限（2026-10-21 治 524）----
# ⚠️⚠️ 为什么 `_ask_openai_full(timeout=…)` 不够：httpx 的 timeout 是
#   **connect / read / write / pool 四段各自的上限**，其中 `read` 指的是
#   「两次收到字节之间的最大间隔」，**不是整个请求的总时长**。
#   所以只要网关"慢而稳"地持续吐字，单次调用就能远远超过 timeout ——
#   实测：把自检轮的 timeout 设成剩余预算 39s，它照样跑了 **85.5s**，
#   最终总耗时 131s ⇒ 用户拿到 524（彻底没答案）。
#   ⇒ 这里用线程 + `future.result(timeout=wall)` 做**真正的总时长上限**：
#     到点就不再等，直接抛超时；上层对异常一律降级为「用原稿」，所以行为安全。
#   ⚠️ 代价：被放弃的那次请求会在后台继续跑完并被丢弃。用**有界线程池**限制规模，
#      且**只用于元认知轮（自检/润色/压缩）** —— 取证轮与答案轮不受影响，
#      那两轮才是答案质量的来源，不能被截断。
_META_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb-meta")


def _ask_meta_bounded(msgs: list, wall: float, **kw) -> dict:
    """带**墙钟总时长**上限的模型调用（只给自检/润色/压缩用）。

    超时抛 `TimeoutError`；调用方一律按"这一轮失败"处理（保留原稿）。
    """
    fut = _META_POOL.submit(_ask_openai_full, msgs, **kw)
    try:
        return fut.result(timeout=max(3.0, float(wall)))
    except _FuturesTimeout:
        raise TimeoutError("墙钟超时（%.0fs，已放弃该轮）" % wall)


def _agent_user_message(question: str, log_ctx: dict | None, n_images: int,
                        mode: str = "") -> str:
    # ⚠️ 只有 mode=="report" 才是走了报障卡；其余（None / 空 / "free"）一律按**自由提问**处理 ——
    #    前端「不选直接打字」也走这条，所以默认必须是 free（2026-09-22 主人定）。
    free = (mode or "").strip().lower() != "report"
    text = f"{'【用户提问】' if free else '【报障描述】'}\n{question}"
    if free:
        text += (
            "\n\n⚠️【这是自由提问，不是报障卡】用户没走信息卡，输入往往很短、意图不明。"
            "请先判断他到底要什么，再决定怎么答 —— **形状自适应，别硬套三段**："
            "\n· 要一个事实/取值（如「XX 单号的手机号是多少」「某参数默认值」）→ **一两句话直接给答案**；"
            "\n· 要排查某个故障 → 才用「结论 / 处理步骤 / ⚠️ 注意」三段；"
            "\n· 问怎么操作 / 在后台哪里配 → 给最短点击路径即可，不必凑三段；"
            "\n· 问的事不归 POS 管 → 一句话说清 + 指到正确的去处。"
            "\n⚠️ 手上有工具能直接取到答案时（例如查某个单号对应的会员手机号），**先调工具再答**，"
            "不要让用户自己去别处翻。"
            # ⚠️ 2026-10-09 加：实测「能查手机号吗」这类**能力询问**被判 fast 档后，
            #    模型**一次检索都不做**、直接凭工具清单作答 —— 结果是「要单号」说了、
            #    「单号去哪取」漏了（那恰恰是一线最需要的）。这类问题必须落一次知识库。
            "\n⚠️ 问某个**能力有没有 / 怎么查 / 在哪查**（如「能查手机号吗」「会员手机号怎么查」）"
            "时，**先 search_kb 检索一次再答** —— 知识库里通常有该能力的专门条目，"
            "会写清「需要什么参数、参数去哪取、能查到什么、查不到怎么排除」，"
            "不检索就会漏掉这些（尤其「参数去哪取」是一线最需要的）。"
        )
    if log_ctx:
        text += ("\n\n（本次附了报障日志：覆盖 "
                 f"{log_ctx.get('first') or '（无时间戳）'} ~ {log_ctx.get('last') or '（无时间戳）'}，"
                 f"共 {len(log_ctx['lines'])} 行 / {log_ctx['chars']} 字。"
                 "请用 log_overview 看全局、grep_log 查细节 —— 不要凭想象描述日志内容。）")
    else:
        text += "\n\n（本次没有附日志。回答只能依据知识库与代码检索结果，不要假设日志里有什么。）"
    if n_images:
        text += (f"\n\n（用户另外上传了 {n_images} 张报障截图，请先读图，把截图里的报错原文、"
                 "错误码、界面状态、版本号作为线索；截图与文字不一致时以截图实际显示为准。）")
    text += ("\n\n（提醒：两条线都要查一次 —— 日志与代码逻辑回答「本次到底发生了什么」，"
             "`search_kb` 回答「这类问题一般怎么处置、后台在哪配、有什么处置红线」。"
             "只查日志会让处理步骤缺少标准流程依据；⚠️ **只查代码也一样** ——"
             "实测有题目因为代码检索太快命中，就没再查知识库，结果漏掉了标准处置与后台配置位置。"
             "**代码查得越顺，越要回头补一次 search_kb。**）")
    return text


# ==================== 分档（fast / deep）2026-10-05 ====================
# 背景（实测 95 个会话 / 389 条轮次日志）：**59% 的会话撞满 5 轮上限被强制收口**，
# 且前 4 轮 out 只有 100~190 token（几乎全是工具调用、没有思考）→ 深度不足是「预算」问题。
# 主人定调（2026-10-05）：模型仍用 deepseek-v4-flash，靠「预算 + 编排」补深度，不引入多智能体。
#
# 判定全部**本地规则**：零成本、确定性、可复核（不额外调模型、不会串号）。
# 命中任一 → deep：
#   ① 带了排查日志或截图（证据密集，必然多轮）
#   ② 正文含订单号（连续 16 位以上数字）→ 要查单 + 查该店配置
#   ③ 现象/追因类词（配了但…为空 / 不生效 / 报错 / 单边 / 为什么 / 排查 / 方案 …）
#   ④ 问题够长（>100 字）或问了多件事（问号 ≥2）
# ⚠️⚠️ 2026-10-22 按实测收紧（原规则见文末注释）。
#   动机：A/B 实测（34 道原判 deep 的题，强制 fast 跑）——
#     · **32/34 的要点覆盖完全一样**，只有 2 道变差：C14（多步链路题）与 C51（带日志报障）。
#     · 耗时 60s → 16.4s（每题省 ~44s）。
#   ⇒ 原规则里 `失败|报错|异常|超时|为什么|排查|定位|单边|卡住…` 这些**单点症状词**
#     把大量"查一个事实就能答"的题也推进了深档 —— 而实测它们 fast 就够。
#   ⇒ 现在只保留**真正需要多步取证**的三类信号：
#     ① 「要一条链路/流程/步骤」（怎么查、从哪取、背后的链路、走什么流程）
#     ② 「要一份方案/评估/对比」（优化建议、方案、评估）
#     ③ 「多问 / 长问题 / 带日志或截图」（下面单独判，不靠这个正则）
_DEEP_PAT = re.compile(
    # ① 链路/流程/步骤类 —— 实测唯一真正需要 deep 的一类（C14 就是它）
    r"链路|流程|步骤|背后|原理|机制|怎么查|在哪查|哪里查|从哪查|怎么获取|如何获取|怎么拿|"
    r"如何查询|怎么取|从哪取|在哪里看|在哪看|怎么找|如何查|怎么定位|如何定位|"
    # ② **需要"比对/诊断"的现象** —— 不是查一个事实，而是要拿两边数据对
    #   （P79「单号日期和交易日期**对不上**」就是这类：必须去查营业日机制才能解释）
    #   ⚠️ 这里**只收"差异/卡住"类**；`原因|怎么处理|怎么办|为什么` **不收** ——
    #     实测「报错 30030 是什么原因，怎么处理」「祥瑞卡怎么配置」这类**查一个事实就能答**，
    #     放进 deep 只会白花 40 多秒（它们 fast 跑分 1.00）。
    # ⚠️ `卡在` 必须加负向断言：实测「祥瑞卡、杭州市民**卡在**后台怎么配置」
    #   会把子串「卡在」当成"卡住"命中 ⇒ 误判 deep。
    r"对不上|不一致|不符|不匹配|重复|丢失|少了|多了|多收|少收|单边|卡住|卡单|卡死|卡在(?!后台)|"
    # ②′ **「要一个解释」类** —— 实测必须保留：
    #   P77「**为什么**大POS 上**找不到**建行支付？」判 fast 后漏了"过滤"这个要点（覆盖 0）。
    #   与"查一个事实"的区别：这类问的是**因果**，答案往往散在多篇里，需要交叉取证。
    r"为什么|为何|找不到|看不到|没有显示|不显示|为空|没有数据|查不到|搜不到|"
    # ③ 方案/评估类 —— 需要综合多篇文档
    r"方案|评估|对比|优化建议|该怎么做|如何避免|怎么预防|"
    # ④ 明确的综合分析诉求
    r"根因|全面|完整梳理|梳理一下")
_ORDER_PAT = re.compile(r"\d{16,}")

# 📌 2026-10-22 收紧前的旧 `_DEEP_PAT`（**回滚用**）：把下面这段换回上面的 `_DEEP_PAT` 即可。
# _DEEP_PAT_OLD = re.compile(
#     r"为空|不显示|没有显示|看不到|不生效|未生效|失效|失败|报错|异常|不通|没反应|对不上|不一致|"
#     r"单边|卡住|卡在|重复|丢失|少了|多了|多收|少收|超时|拒绝|"
#     r"为什么|原因|根因|分析|评估|排查|定位|方案|怎么办|怎么处理|如何解决|优化建议|"
#     r"怎么查|在哪查|哪里查|怎么获取|如何获取|怎么拿|如何查询|怎么取|从哪取|在哪里看|在哪看"
# )


def _classify_depth(question: str, log: str | None = None,
                    images: list | None = None, mode: str = "") -> str:
    """返回 "deep" 或 "fast"。纯本地规则，不花额外 token。"""
    # ⚠️ 2026-10-22：A/B 实测开关（`KB_FORCE_DEPTH`）。留空时完全按下面的规则走。
    if getattr(config, "FORCE_DEPTH", ""):
        return config.FORCE_DEPTH
    q = (question or "").strip()
    if log or images:
        return "deep"
    if _ORDER_PAT.search(q):
        return "deep"
    if _DEEP_PAT.search(q):
        return "deep"
    if len(q) > 100 or (q.count("？") + q.count("?")) >= 2:
        return "deep"
    return "fast"


def _acc_usage(meta: dict, u: dict | None):
    """累计 token 用量，并记录**上下文缓存命中**。

    ⚠️ 2026-10-05 加：此前只记 prompt/completion，看不到缓存命中 ——
    而多轮取证的成本大头正是「同一段前缀每轮重发」（实测 deep 档累计 prompt 85k~98k，
    工具返回只占 6~10k）。不测缓存命中率，就没法判断"前缀是否稳定、缓存是否真的生效"。

    ⚠️⚠️ 2026-10-09 修：**字段名随网关变，只认一个就会静默记成 0**。
       · DeepSeek 官方端点返回顶层 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`；
       · 自建中转网关（ai.ctaigw.cn）返回的是 `prompt_tokens_details.cached_tokens`。
       只读前者 ⇒ 换网关后命中率恒为 0，**成本台账直接变瞎**（而"命中价约为未命中 1/10"，
       差了 10 倍还看不出来）。这里两种都读；`cache_miss` 取不到就用 prompt - cache_hit 兜。
    """
    u = u or {}
    t = meta.setdefault("usage", {"prompt": 0, "completion": 0, "cache_hit": 0, "cache_miss": 0})
    prompt = u.get("prompt_tokens") or 0
    hit = u.get("prompt_cache_hit_tokens")
    if hit is None:                                     # 中转网关风格
        hit = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if hit is None:                                     # 第三种网关（OpenAI 原生风格）
        hit = (u.get("input_tokens_details") or {}).get("cached_tokens")
    hit = int(hit or 0)
    miss = u.get("prompt_cache_miss_tokens")
    miss = int(miss) if miss is not None else max(0, int(prompt) - hit)
    t["prompt"] = t.get("prompt", 0) + int(prompt)
    t["completion"] = t.get("completion", 0) + int(u.get("completion_tokens") or 0)
    t["cache_hit"] = t.get("cache_hit", 0) + hit
    t["cache_miss"] = t.get("cache_miss", 0) + miss


def _looks_like_tool_markup(text: str) -> bool:
    """这段"答案"是不是**工具调用标记**而不是给人看的正文？

    ⚠️ 2026-10-22 新增。动机：实测模型偶尔不走结构化 `tool_calls`，而是把工具调用语法
    直接写进 `content`（形如 `<｜｜DSML｜｜ invoke name="query_store_config">`）。
    此时 `msg["tool_calls"]` 是空的 ⇒ 旧代码会**把它当成最终答案返回给用户**，
    实测 C55 因此从覆盖 1.00 掉到 **0.00**（整段答案都是标记）。
    判据取宽松些：只要出现 DSML 标记或那种全角竖线包裹的伪标签，就判定为"不是正文"。
    """
    if not text:
        return False
    t = text[:2000]
    if "DSML" in t:
        return True
    # 全角竖线包裹的伪标签（<｜｜...｜｜>），以及常见工具调用关键字
    if "｜｜" in t and ("invoke" in t or "parameter" in t or "calls" in t):
        return True
    if "<tool_call" in t or "</tool_call>" in t:
        return True
    return False


def _json_loose(txt: str) -> dict | None:
    """从模型输出里抠出第一个 {...} 并解析；失败返回 None（调用方负责降级）。"""
    s = (txt or "").strip()
    if s.startswith("```"):                        # 去掉 ```json ... ``` 围栏
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:                              # noqa: BLE001
        return None


# ---- 规划轮（B1）：deep 档先出一份取证计划再进工具循环 ----
# ⚠️ 失败一律静默降级（返回空串）—— 规划只是加分项，绝不能因为它挂掉整个回答。
_PLAN_PROMPT = (
    "先**不要调用任何工具**。请仅用一段话（≤160 字）列出你的**取证计划**："
    "① 这条现象可能存在的**根因有哪几条**（分开列，别只写一条）；"
    "② 每条根因**分别用什么手段排除**（查订单事实 / 查该店实际配置 / 代码检索确认机制）；"
    "③ 按什么顺序查最省轮次。"
    "只输出计划本身，不要客套、不要解释。"
    "⚠️ 计划要覆盖**所有合理根因**，不要一开始就认定某一个原因 —— 实测过早锁定单一方向的答案，"
    "在真正根因是另一条时会答错。"
)


def _plan_prompt(max_steps: int) -> str:
    """规划轮提示词（含轮次预算告知，2026-10-10 评审 ⓪）。

    背景：实测 deep 档 55% 撞满 8 轮上限 —— 而模型在循环里**不知道自己一共几轮**
    （只有末轮才被告知「轮次已用完」），提示词说「够了就收口」，但它感知不到
    「够了」的信号，自然没有省着花的动机。在计划阶段就把预算亮出来，
    模型会主动「留一轮写答案」。
    """
    return (_PLAN_PROMPT
            + f"\n\n⚠️ 预算：你接下来共有 **{max_steps} 轮**取证机会（第 {max_steps} 轮会被强制作答）。"
              "计划要按这个预算排：通常 3~5 轮该查的就能查完，剩下留给意外；"
              "同一件事换个写法再查一遍**不算新进展**，别把它排进计划。")


def _plan_round(msgs: list, meta: dict, max_steps: int = 8) -> str:
    try:
        # ⭐ 2026-10-12：规划轮思考档从写死的 disabled 改为可配（默认 high）。
        #   规划=「先想清楚查什么、有哪些根因要排除」，是典型的需要推理的元认知环节，
        #   用 disabled 属于明显错配。主人已确认"要智商、时间不是问题"。
        _th = getattr(config, "THINKING_PLAN", "high")
        data = _ask_openai_full(msgs + [{"role": "user", "content": _plan_prompt(max_steps)}],
                                thinking=_th, timeout=120)
        _acc_usage(meta, data.get("usage"))
        return (((data.get("choices") or [{}])[0].get("message") or {})
                .get("content") or "").strip()[:600]
    except Exception as e:                         # noqa: BLE001
        print(f"[agent] 规划轮失败（忽略）：{type(e).__name__}: {e}", flush=True)
        return ""


# ---- 自检轮（B2）：答案产出后复核「问了几件事 / 每件是否有证据」----
# 设计要点：**无缺口时不重写答案**（省 token、避免把好答案改坏），只在确有缺口时修订一次。
_CONTINUE_PROMPT = (
    "（上面这份答案因为**输出长度上限**在句子中间被截断了 —— 不是你写完了，是被砍断的。）\n"
    "请**从断点直接接着写**，把剩下的部分补完：\n"
    "1. 从最后一个没写完的句子继续，**不要重复已经写过的任何内容**，也不要重写全文；\n"
    "2. 若「⚠️ 注意 ——」段已经开头务必把它收尾；「处理步骤」若还有没写完的条目，接着补。\n"
    "3. 只输出**续写的这一段**，不要加「续：」之类前缀，不要复述前文。"
)


def _merge_continuation(draft: str, more: str) -> str:
    """拼接续写内容，并去掉断点处的重叠（模型常把最后半句重写一遍）。"""
    d = (draft or "").rstrip()
    m = (more or "").lstrip()
    if not d:
        return m
    if not m:
        return d
    # 最长重叠：d 的尾部 == m 的开头（上限 200 字，避免误吞正常内容）
    # ⚠️ 下限必须是 4：中文断点常是「点过离线」这种 4 字词，取 8 会漏检
    #    （实测：range(...,8,-1) 只测 9~17 字，4 字重叠匹配不到 → 拼出重复句）。
    _ov = 0
    for n in range(min(len(d), len(m), 200), 3, -1):
        if d.endswith(m[:n]):
            m = m[n:].lstrip(" \t")     # ⚠️ 只吃空格，保留换行（否则「2. xxx」会粘到上一行）
            _ov = n
            break
    if not m:
        return d
    # 有重叠 ⇒ 模型从断点接着写，无缝拼（否则会把词切断）；
    # 无重叠 ⇒ 上一句若已收尾（。！？）就另起一行，否则说明断在句中 → 无缝续。
    if _ov:
        sep = ""
    elif d[-1] in "。！？":
        sep = "\n"
    else:
        sep = ""
    return d + sep + m


def _continue_truncated(msgs: list, draft: str, meta: dict,
                        thinking: str | None = None,
                        max_tokens: int = 1200,
                        timeout: float = 90.0) -> str:
    """答案轮被 max_tokens 截断 → **续写补齐**，已写内容一字不动。

    ⚠️ 为什么不是「去掉上限重问一遍」：2026-10-22 试过，实测把情况改坏了 ——
      重取会把整篇答案重写（原本对的部分也可能被改差），且 `tools=None` 时模型会把
      工具调用标记当正文吐出来（实测 C55 要点覆盖 1.00 → 0.00，答案整段是标记）。
      ⇒ 续写只让模型**接着写**，前文不动，风险面小得多。
    ⚠️ 续写这一轮也**不带 tools**（只需要它写字，不需要取证）。万一它把标记写进正文，
      用 `_looks_like_tool_markup` 兜住：识别到就**放弃续写、保留原文**，
      绝不把垃圾拼进答案。任何异常同样返回原草稿。
    ⚠️ 只调用一次（调用点保证），不做循环续写 —— 避免"无限续"把时间预算耗光。
    """
    if not (draft or "").strip():
        return draft
    try:
        _msgs = list(msgs) + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": _CONTINUE_PROMPT},
        ]
        data = _ask_openai_full(_msgs, tools=None, thinking=thinking,
                                max_tokens=max_tokens,
                                timeout=int(max(15, timeout)))
        _acc_usage(meta, data.get("usage"))
        choice = (data.get("choices") or [{}])[0]
        more = ((choice.get("message") or {}).get("content") or "").strip()
        if not more or _looks_like_tool_markup(more):
            print("[agent] 续写没拿到可用正文 → 保留原答案", flush=True)
            return draft
        merged = _merge_continuation(draft, more)
        meta["continued"] = meta.get("continued", 0) + 1
        print(f"[agent] ✅ 截断续写成功：{len(draft)} → {len(merged)} 字"
              f"（续写段 finish_reason={choice.get('finish_reason')}）", flush=True)
        return merged
    except Exception as e:                         # noqa: BLE001
        print(f"[agent] 续写失败（保留原答案）：{type(e).__name__}: {e}", flush=True)
        return draft


_SELFCHECK_PROMPT = (
    "你是**复核人**。请检查下面这份答案有没有漏答，**不要**评价文风、字数、格式。\n\n"
    "【用户原问题】\n{question}\n\n"
    "【答案草稿】\n{draft}\n\n"
    "【本轮实际调用过的工具】（用来判断证据够不够）\n{trace}\n\n"
    "只输出 JSON（不要 markdown 围栏）：\n"
    '{{"asks": ["用户一共问了哪几件事"], "gaps": ["哪件事没答 / 哪条结论缺少证据支撑"]}}\n'
    "要求：\n"
    "1. 若每件事都答了、结论都有支撑，`gaps` 必须是空数组 []；\n"
    "2. 只有**确实存在**的缺口才写进 gaps，不许为了显得严格而编造要求；\n"
    "3. 缺口要写成具体动作（例：\"用户还问了是否需要重登，未回答\"）。\n"
    # ⚠️ 2026-10-10 评审 ⓪'：38 次自检只报过 1 次缺口（2.6%），复核太客气。
    #   补一条**硬性操作**：把用户原问题拆成子问题逐条勾选 —— 逐条核对
    #   比整体扫一眼更容易发现漏答（「每件事都答了吗」的问法容易整体掠过）。
    "4. **先做拆解再下结论**：把【用户原问题】逐句拆成独立的子问题"
    "（含追问句、并列的「和/以及/另外」、括号里的附带要求），对每一个子问题"
    "在草稿里找对应回答 —— 找不到或答得含糊的，就是 gap。"
    "⚠️ 特别注意「隐含的问题」：用户问「怎么处理 X」往往同时期待「X 为什么会发生」，"
    "草稿只给了步骤没给原因也算 gap。"
    # ⭐ 2026-10-28：**完整性检查**。原先 prompt 只查「有没有漏答」，明确叫复核人
    #   「不要评价文风、字数、格式」⇒ 答案被 length 截断成半句时（实测断在
    #   「⚠️ 注意 ——」第二条的「点过离线」后），自检照样判 `gaps=[]`，形同没查。
    #   ⇒ 补一条只看**结尾是否写完**的硬检查（仍然不许评价文风/字数）。
    "5. **完整性检查（必做，与第 4 条并列）**：只看草稿的**结尾**有没有写完 ——"
    "末句是否在句子中间停住（没有句号、话没说完）、最后一条「⚠️ 注意」是否只有半句、"
    "「处理步骤」是否断在某个条目中间。**凡是残缺就必须写进 gaps**，格式如："
    "「答案末尾不完整（末句断在「xxx」之后没有下文），需补全收尾」。"
    "⚠️ 这种残缺通常是**输出长度上限截断**造成的，写草稿的人自己看不出来 —— 必须你指出。"
    "⚠️ 但**只报「没写完」，不许因此要求加长、改写或补充文风**。"
)


def _selfcheck_round(question: str, draft: str, trace: str, meta: dict,
                     deadline_ts: float | None = None) -> tuple[str, list]:
    """返回 (最终答案, gaps)。gaps 为空或任何异常 → 原样返回 draft。

    ⭐ 2026-10-12 思考链路升级：复核与修订两轮从写死的 disabled 改为可配（默认 high）。
    理由：自检要「逐句拆解原问题 → 逐条找对应回答 → 判断证据够不够」，是整条链路里
    最吃推理的一步（比写答案还难），用 disabled 会大量漏报缺口（实测 38 次只报 1 次）。

    ⚠️⚠️ 2026-10-14（主人质疑"中间产物被截断"）**修掉同款雷**（与 `_polish_round` 同源）：
      草稿超 6000 字时会被 `[:6000]` 截断送入，而"有缺口 → 修订"这一支的输出**会替换答案**。
      缺陷模型没见过后半 ⇒ 修订版丢后段；且缺口判断也是基于残缺草稿，可能误判。
      ⇒ 现在：**草稿超上限时只复核、不修订**（复核结果仍记录到 gaps 供排障，
        但答案原样返回，绝不因截断而丢内容）。

    ⚠️⭐ 2026-10-21 新增 `deadline_ts`（治 524 的关键）：自检轮是**长尾元凶** ——
      实测 13 次：中位 18.4s，但均值 32.6s、**p90 68.3s、max 101.4s**。
      而闸门按「预计 25s」放行 ⇒ 一旦这一轮跑成 60s+，总耗时直接 110s+ ⇒ 撞 CF 100s 墙。
      ⇒ 现在把**剩余预算当成这一轮的硬超时**（而不是写死 150s），
        超时就抛异常 → 上层 except 原样返回草稿 ⇒ 最坏情况被钉在预算内。
    """
    _max_in = _draft_in_limit()
    _truncated = len(draft or "") > _max_in
    if _truncated:
        print(f"[agent] 自检：草稿 {len(draft)} 字 > {_max_in} → "
              f"只复核不修订（避免截断导致丢内容）", flush=True)

    def _left() -> float:
        """这一轮还能用多少秒（没有 deadline 时退回原来的 150s）。"""
        if not deadline_ts:
            return 150.0
        return max(6.0, deadline_ts - time.time())

    try:
        _th = getattr(config, "THINKING_SELF", "high")
        _th_fix = getattr(config, "THINKING_SELF_FIX", "high")
        prompt = _SELFCHECK_PROMPT.format(question=(question or "")[:2000],
                                          draft=draft,
                                          trace=trace or "（无）")
        _left_now = _left()
        data = _ask_meta_bounded([{"role": "user", "content": prompt}],
                                 wall=_left_now, thinking=_th, timeout=_left_now)
        _acc_usage(meta, data.get("usage"))
        txt = (((data.get("choices") or [{}])[0].get("message") or {})
               .get("content") or "").strip()
        obj = _json_loose(txt)
        gaps = [str(g).strip() for g in ((obj or {}).get("gaps") or []) if str(g).strip()]
        if not gaps:
            return draft, []
        # ⚠️ 草稿被截断时**不做修订**：修订模型的输入是残缺草稿，它输出的"完整答案"
        #    必然缺后段（而它自己不知道）⇒ 直接替换会丢内容。只把 gaps 报出去供排障。
        if _truncated:
            print(f"[agent] 自检发现 {len(gaps)} 处缺口，但草稿超长已截断 → "
                  f"不修订（保留原答案）", flush=True)
            return draft, gaps
        # ⚠️ 2026-10-21：修订轮同样受剩余预算约束；剩余时间不足就直接不修订
        #    （修订是"整篇重写"，实测平均再花 40s+，硬跑必撞墙）。
        _left_fix = _left()
        if _left_fix < 20.0:
            print(f"[agent] 自检发现 {len(gaps)} 处缺口，但只剩 {_left_fix:.0f}s → "
                  f"不修订（保命，避免 524）", flush=True)
            return draft, gaps
        # 有缺口 → 带着缺口修订一次（仍不带工具，避免再拖长链路）
        # ⭐ 2026-10-28：**末尾残缺单独走「续写」**，不要整篇重写。
        #   原因：整篇重写（40s+）对"只在结尾断了几十字"是巨大的浪费，而且重写会
        #   把本来已经对的前文改差（10-22 那次"重取"把答案改坏是同一个原理）。
        #   续写只补缺的几句，前文一个字不动。
        _incomplete = any(
            k in g for g in gaps
            for k in ("不完整", "截断", "断在", "收尾", "没写完", "缺少结尾"))
        if _incomplete and (_left_fix - 3) >= 15.0:
            _fixed = _continue_truncated(
                [{"role": "user",
                  "content": f"【用户原问题】\n{(question or '')[:2000]}"}],
                draft, meta, thinking=_th_fix, timeout=(_left_fix - 3))
            if _fixed and _fixed != draft:
                print(f"[agent] 自检：末尾残缺 → 续写补齐 {len(draft)} → {len(_fixed)} 字",
                      flush=True)
                return _fixed, gaps
            print("[agent] 自检：续写未补上 → 退回整篇修订", flush=True)
        d2 = _ask_meta_bounded([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": txt},
            {"role": "user", "content":
             "针对上面每一处缺口修订答案，**直接输出修订后的完整答案**"
             "（保留原有段落结构与所有点击路径 / 参数值 / 证据行），不要说明你改了哪里。"
             "⚠️ 若草稿末尾本来就断在句子中间（被长度上限截断），按断点把缺的那几句补完；"
             "**前文已经完整的内容不要重写、不要压缩**。"},
        ], wall=_left_fix, thinking=_th_fix, timeout=_left_fix)
        _acc_usage(meta, d2.get("usage"))
        txt2 = (((d2.get("choices") or [{}])[0].get("message") or {})
                .get("content") or "").strip()
        return (txt2 or draft), gaps
    except Exception as e:                         # noqa: BLE001
        print(f"[agent] 自检轮失败（忽略）：{type(e).__name__}: {e}", flush=True)
        return draft, []


def _clip_old_tool_msgs(msgs: list, keep_last: int, clip_chars: int) -> int:
    """把**较早轮次里的长工具结果**折叠成「前 N 字 + 已折叠标记」，返回折叠条数。

    ⚠️ 为什么（2026-10-05 实测）：deep 档一次问答累计 prompt 85k~98k，而工具返回总量只有
    6k~10k tokens —— 大头是「每轮重发历史」。折叠早期长块能直接压低 miss token。
    安全性：只动「>800 字」的长消息、只保留最近 keep_last 条不动、只截前段不删行号；
    短结果与刚查到的东西完全不受影响（否则会把证据删掉，反而害模型猜）。
    """
    idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    if len(idx) <= keep_last:
        return 0
    cut = 0
    for i in idx[:-keep_last]:
        c = msgs[i].get("content") or ""
        if len(c) > 800 and not c.startswith("〔已折叠〕"):
            msgs[i]["content"] = (f"〔已折叠〕原结果共 {len(c)} 字，前 {clip_chars} 字：{c[:clip_chars]}\n"
                                  f"（其余已在上文消化；确实需要重看就重新调用该工具）")
            cut += 1
    return cut


def _time_budget(depth: str, question: str) -> int:
    """按任务规模**动态**计算整个问答的时间预算（2026-10-12 主人定）。

    之前是写死的 150s——主人指出：任务大就该多等、任务小就该快失败。
    规模信号：① 分档（fast/deep，_classify_depth 的产出）；② 问题里的"单号个数"
    —— 每个单号意味着一次慢工具查询（实测 EXT 反查经 Mac 中转 ~2.5s/次，
    一轮常带 3~5 个单号）。

    取值：
      fast             → 60s   （fast 中位 4s，60s 已是 15 倍余量）
      deep 基线        → 100s  （deep p90 = 79s，100s 覆盖绝大多数）
      每多一个单号 +20s → 覆盖"批量查单"场景（10 个单号 ≈ 100+200=300？不——上限兜底）
      硬上限           → 160s  （160 + 收口 ~13s ≈ 173 < Worker 210s ✅）

    返回 0 = 关闭预算检查。
    """
    base = int(getattr(config, "AGENT_TIME_BUDGET", 100) or 0)
    if base <= 0:
        return 0
    if depth != "deep":
        return min(base, 60)
    # 单号 = ≥14 位连续数字（POS/EXT/订单号都在 16~20 位；身份证等 15/18 位也按单号算
    # ——宁可预算给宽一点也不把真任务截停）
    import re as _re
    n_ids = len(_re.findall(r"\d{14,}", question or ""))
    dynamic = base + max(0, n_ids - 1) * 20      # 第 1 个单号含在基线里
    return min(dynamic, 160)


def answer_agentic(question: str, history: list | None = None,
                   images: list[str] | None = None,
                   coach: list[dict] | None = None,
                   log: str | None = None,
                   mode: str = "") -> tuple[str, dict]:
    """取证闭环：模型自主调工具、多轮收敛，最后收口成三段式答案。

    返回 (答案文本, meta)。meta 含本次取证轨迹与知识库命中，供响应体与运维复核。
    ⚠️ 只有 openai 兼容端点走闭环（function calling 依赖它）；echo / ollama 回退到原单轮
    `answer()`，保证换 provider 时不会因为闭环而整个不可用。
    """
    from . import tools as toolmod          # 延迟导入：避免 tools↔llm 的导入环
    images = images or []
    meta: dict = {"trace": [], "steps": 0, "kb_hits": [], "diagram": None,
                  "usage": {"prompt": 0, "completion": 0, "cache_hit": 0, "cache_miss": 0}}
    empty_chunks: list = []

    def fallback() -> tuple[str, dict]:
        """闭环走不通时的兜底单轮回答。

        ⚠️⚠️ 2026-10-12（评审逐行分析 P1-1）：**必须带上闭环期间已查到的证据**。
        原先这里传的是 `empty_chunks` —— 意味着闭环跑了几轮、查到一堆片段之后若发生异常
        （模型超时 / JSON 解析失败 / 网关 5xx 重试耗尽），退回单轮时**前面查到的全丢**：
        用户等了几十秒，拿到的答案比"完全不用 agent"还差（至少 agent 查到了东西）。
        现在改为：优先用 collect box 里已记录的命中（含内容摘要）拼成检索片段，
        真一条都没有才用空列表。
        """
        try:
            hits = toolmod.take_kb_hits()
        except Exception:                       # noqa: BLE001
            hits = []
        # ⚠️ 两条 fallback 路径的时机不同，所以要两级取数：
        #   · `except` 里调 → finally 尚未执行 → collect box 还在 → take_kb_hits() 有效；
        #   · 「answer_text 为空」时调 → finally 已执行（box 已清） → 只能用 meta["kb_hits"]
        #     （它在 finally 里被从 box 拷出来，同样含 text）。
        if not hits:
            hits = list(meta.get("kb_hits") or [])
        if hits:
            # 去重（同一 file+idx 只留一条）并转成 answer() 认识的 chunk 形状
            seen, chunks = set(), []
            for h in hits:
                key = (h.get("file"), h.get("idx"))
                if key in seen or not h.get("text"):
                    continue
                seen.add(key)
                chunks.append({"file": h.get("file"), "idx": h.get("idx"),
                               "score": h.get("score") or 0, "text": h.get("text")})
            if chunks:
                print(f"[agent] 闭环异常回退，带上已查到的 {len(chunks)} 条片段", flush=True)
                return (answer(question, chunks, history, [], images, coach, False, None, log),
                        meta)
        return answer(question, empty_chunks, history, [], images, coach, False, None, log), meta

    if config.LLM_PROVIDER != "openai" or not config.OPENAI_API_KEY:
        return fallback()
    if not getattr(config, "AGENT_ENABLED", True):
        return fallback()

    log_ctx = toolmod.build_context(log or "")
    toolmod.set_context(log_ctx)
    toolmod.start_collect()
    depth = "fast"                                 # 兜底默认值（异常路径不依赖它）
    try:
        # ⚠️ 工具清单必须动态生成（2026-09-25）：_AGENT_SYSTEM 里曾硬编码工具清单，
        #    新增 query_store_sales 后那句没同步 → 模型照着过期的"自我介绍"，
        #    断言"没有查销售额的能力"，工具明明在 tools 列表里也不调。
        sys_content = _AGENT_SYSTEM.replace(
            "{__TOOL_SUMMARY__}", toolmod.summaries(bool(log_ctx)))
        # ⚠️ 2026-09-25 加：prompt 里原本**从不告知今天是几号**，于是模型算不对
        #    「昨天 / 前天」—— 实测把今天(9/25)当成了"昨天"，查回 9/24+9/25 两份数据。
        _now = datetime.now()
        _today = _now.date()
        _week = "一二三四五六日"[_today.weekday()]
        sys_content += (
            f"\n\n【当前时间】今天 {_today:%Y-%m-%d}（星期{_week}）；"
            f"昨天 {(_today - timedelta(days=1)):%Y-%m-%d}；"
            f"前天 {(_today - timedelta(days=2)):%Y-%m-%d}。\n"
            f"⚠️ 用户说「昨天 / 前天 / 本周」时，**直接按上面换算好的日期**传给工具，"
            f"不要自己推算（实测模型把今天当成了昨天，查错了两天）。"
        )
        # ⭐ 2026-10-10 加（治 deep 空转）：注入知识库目录。
        #   背景：线上实测 61% 的 deep 问答撞满 8 轮（比之前的 55% 更高），轨迹显示模型
        #   在**反复换词搜「错误码汇总表 / 速查表 / 对照表 / 索引 / 大全」**这类
        #   **库里根本不存在**的文档。根因是模型看不到"库里有什么"，只能盲搜试错。
        #   ⚠️ 这段是**静态前缀**（只随 /reload 变化），位置固定在时间之后 → 前缀缓存不受影响。
        #   ⚠️ 开关 KB_KB_CATALOG（默认 1）；A/B 对照用 --set KB_KB_CATALOG=0 关掉。
        if getattr(config, "KB_CATALOG", True):
            try:
                from . import knowledge as kmod
                _cat = kmod.doc_catalog()
                if _cat:
                    _nfiles = len({d["file"] for d in kmod._DOCS})
                    sys_content += (
                        f"\n\n【知识库文档目录 —— 你检索的库里**只有**这些文档，共 {_nfiles} 篇】\n"
                        + _cat +
                        "\n⚠️ 上面就是知识库的**全部**内容。**目录里没有的主题，就是库里真的没有** ——\n"
                        "  · 不要换个说法反复搜「汇总表 / 速查表 / 对照表 / 清单 / 大全 / 索引」"
                        "这类听起来该有、但目录里并没有的文档；\n"
                        "  · 检索 2~3 次仍没有你要的主题，就**立即下结论**并说明"
                        "「知识库里没有 X 的条目」，不要再换词硬搜；\n"
                        "  · 目录里有相近主题时，优先用 `read_doc` 读那篇全文，"
                        "而不是用十几个不同关键词去猜。"
                    )
            except Exception as _e:
                print(f"[agent] 注入知识库目录失败（不影响主流程）：{type(_e).__name__}", flush=True)
        if coach:
            segs = []
            for i, c in enumerate(coach, 1):
                # ⭐ 单框模式（2026-10-29）：整段原文直接用，不再拆「问题点/处理方式/线索」
                if c.get("raw"):
                    segs.append(f"[C{i}] {c['raw']}")
                    continue
                part = [f"[C{i}] 正确的问题点：{c.get('point') or '（未填写）'}"]
                if c.get("howto"):
                    part.append(f"     正确处理方式：{c['howto']}")
                if c.get("hint"):
                    part.append(f"     关键线索：{c['hint']}")
                segs.append("\n".join(part))
            sys_content += (
                "\n\n【人工纠正记录 —— 优先级高于一切，必须遵循】\n"
                "以下由运维人员在案例审查中人工核对并纠正过，与知识库 / 代码片段 / 你的判断冲突时，"
                "一律以本段为准。\n"
                # ⚠️⚠️ 2026-10-29：这段「读法」是主人报障后的关键修复。
                #   原来只说"以本段为准"，模型就把「用 gateway 查手机号订单」**转写成了
                #   『请运维去腾讯云控制台检索』** —— 因为那段读起来像"流程说明"，
                #   而不是"你该去调工具"。必须显式点明主语是「你」。
                "⚠️⚠️ **怎么读这段**：里面写的「用什么查 / 查什么条件 / 怎么解析」"
                "是**你自己要执行的动作**，不是写给用户的步骤 —— 你手上有对应工具就**立刻去调**，"
                "拿到结果后再组织成答案；**不要**把它转写成「请运维到 XX 后台检索」。\n"
                "⚠️ 只有你**确实没有那个工具**时，才允许写成人工步骤；"
                "而且必须说清「这一步需要人工」，不能把「该做的事」说成「已经做完的事」。\n"
                "⚠️ 用户可见的回答里**不要出现**「教练记录」「按教练内容」「人工纠正记录」这类内部字样。\n"
                + "\n".join(segs)
            )
        msgs = [{"role": "system", "content": sys_content}]
        for h in _trim_history(history):
            msgs.append({"role": h["role"], "content": h["content"]})
        user_text = _agent_user_message(question, log_ctx, len(images), mode=mode)
        if images:
            uc: list | str = [{"type": "text", "text": user_text}]
            for u in images:
                uc.append({"type": "image_url", "image_url": {"url": u}})
            msgs.append({"role": "user", "content": uc})
        else:
            msgs.append({"role": "user", "content": user_text})

        # ---- 分档（2026-10-05）：先判档，再决定轮数 / 检索宽度 / 思考模式 ----
        depth = _classify_depth(question, log, images, mode)
        meta["depth"] = depth
        if depth == "deep":
            max_steps = max(1, int(getattr(config, "AGENT_MAX_STEPS_DEEP", 8)))
            _think = getattr(config, "THINKING_DEEP", "enabled")
            _topk = int(getattr(config, "TOP_K_DEEP", 8))
        else:
            max_steps = max(1, int(getattr(config, "AGENT_MAX_STEPS_FAST",
                                           getattr(config, "AGENT_MAX_STEPS", 5))))
            _think = getattr(config, "THINKING_FAST", None)
            _topk = int(getattr(config, "TOP_K_FAST", getattr(config, "TOP_K", 5)))
        # ⭐ 方案 A（2026-10-09 主人批准）：deep 档**中间取证轮用低档思考**。
        #   为什么不能简单按"轮次"降档：模型常在**非最后一轮**就写出答案（实测 8 轮上限时出在第 7 轮），
        #   那一轮会正好落到低档上 ⇒ 低档思考写的答案。
        #   解法：中间轮配 `tool_choice=required` **强制它只取证、不写答案** ——
        #   于是"写答案"只会发生在末轮 / 提前收口轮 / 自检轮，这三处都用高档（_think）。
        # ⚠️ 只对 deep 生效；fast 档 `_think_mid` 就是 `_think`（disabled），且保持 auto，行为完全不变。
        _think_mid = (getattr(config, "THINKING_DEEP_MID", "minimal") or _think) if depth == "deep" else _think
        # ⚠️ 2026-10-21（治 524）：**收口轮单独一个档位**。
        #   收口轮的任务是"时间到了，把已经查到的证据直接写成答案"，不需要深推理；
        #   实测它用 high 时单轮就要 34.4s（日志 `预算 70s收口轮 34.4s`），
        #   而这恰恰发生在**已经超时**的时刻 ⇒ 直接把它推过 CF 的 100s 墙。
        #   ⚠️ 只给"时间到了"的收口轮（硬上限/预算/轮数用尽）降档；
        #      「提前收口轮」（查够了主动停）不降 —— 那时还有时间，质量优先。
        _think_cut = getattr(config, "THINKING_CUT", "low")
        _force_tool = bool(depth == "deep" and getattr(config, "AGENT_DEEP_REQUIRE_TOOL", True))
        # 2026-10-10 评审 ⓪：轮次预算提醒开关（规划轮提示里带预算 + 中后段两条进度提醒）
        _budget_remind = bool(depth == "deep" and getattr(config, "AGENT_BUDGET_REMIND", True))
        toolmod.set_top_k(_topk)
        specs = toolmod.specs(bool(log_ctx))
        # ⚠️ 计时起点放在**规划轮之前**：规划轮也是真金白银的一次模型调用，
        #    不算进去就会得出"总耗时对不上各阶段之和"这种查不下去的怪账（2026-10-09 踩过）。
        _t_start = time.time()
        # 规划轮（B1，仅 deep）：先出一份取证计划再进循环，减少「东一榔头西一棒子」
        if depth == "deep" and getattr(config, "PLAN_ENABLED", False):
            _t_plan = time.time()
            plan = _plan_round(msgs, meta, max_steps)
            print(f"[agent] 规划轮 {time.time()-_t_plan:.1f}s"
                  f"{'（失败已静默降级）' if not plan else ''}", flush=True)
            if plan:
                meta["plan"] = plan
                msgs.append({"role": "assistant", "content": f"【取证计划】{plan}"})
                msgs.append({"role": "user", "content":
                             "按上面的计划开始取证。计划可以随时调整，但不要漏掉任何一条要排除的根因。"})
        print(f"[agent] 分档={depth} 轮数上限={max_steps} 检索={_topk} "
              f"思考={_think}（中间轮={_think_mid}{'，强制取证' if _force_tool else ''}）", flush=True)
        answer_text = ""
        stall = 0
        seen_calls: set = set()                    # 本轮内已出现过的 (工具+参数) 签名，用于收敛判据
        for step in range(1, max_steps + 1):
            last = step == max_steps
            if last:
                # 强制收口：末轮不带 tools（等价 tool_choice=none）+ 明确告知轮次已尽。
                # ⚠️ 「做减法」必须在这里再说一遍：实测光靠 system 里的篇幅条款压不住首稿，
                # 收口这句点明后才会真的删废话。但**只要求去废话，不设字数上限** ——
                # 主人 2026-09-16 定调：写清楚优先，简化后仍长就算了。
                msgs.append({"role": "user", "content":
                             "取证轮次已用完。请立即基于已查到的证据输出最终答案，"
                             "按固定结构（结论 / 事实 / 处理步骤 / ⚠️ 注意），不要再要求查工具。"
                             "**输出前把答案通读一遍做减法**：删客套、删重复、删解释性从句，"
                             "只留「要做的动作 / 必须知道的取值 / 会踩的坑」；"
                             "日志细节只挑最关键的 2~4 行写进结论，不要逐笔抄回执。"
                             "点击路径与参数值一个都不能少 —— "
                             "**目标 ≤700 字**；只有可执行内容真的装不下才超出，"
                             "且**超出的部分必须是可执行内容，不是解释或复述**。"})
            _t0 = time.time()
            # 历史瘦身（仅 deep）：折叠较早轮次的长工具结果，压低「每轮重发历史」的成本
            if depth == "deep" and getattr(config, "HISTORY_CLIP", False):
                _c = _clip_old_tool_msgs(msgs, getattr(config, "HISTORY_KEEP_TOOL", 6),
                                         getattr(config, "HISTORY_CLIP_CHARS", 500))
                if _c:
                    meta["clipped"] = meta.get("clipped", 0) + _c
            # ⭐ 2026-10-12（评审 P2-1）：**取证轮限输出长度**（答案轮不限）。
            #   取证轮的工作是"选工具 + 一句理由"，正常 100~300 token 就够；
            #   实测见过 out=3104 的长尾（光生成 20+ 秒）——限一下能直接砍掉。
            #   ⚠️ 取 2000 是宽松值：真需要写长理由也不至于被截；被截的是明显的"写飞"。
            _mt = None if last else int(getattr(config, "AGENT_MID_MAX_TOKENS", 2000) or 0) or None
            data = _ask_openai_full(msgs, tools=None if last else specs,
                                    tool_choice=None if last else ("required" if _force_tool else "auto"),
                                    thinking=_think if last else _think_mid,
                                    max_tokens=_mt)
            meta["steps"] = step
            _u = data.get("usage") or {}
            _acc_usage(meta, _u)
            print(f"[agent] 第{step}轮 {time.time()-_t0:.1f}s "
                  f"in={_u.get('prompt_tokens')} out={_u.get('completion_tokens')}", flush=True)
            choice = (data.get("choices") or [{}])[0]
            # ⚠️ 2026-10-21 新增：**截断检测**。
            #   全局原先**没有任何 finish_reason 检查**（`grep -rn finish_reason app/` 零命中），
            #   于是被 max_tokens / 网关输出上限截出的**半截答案会被当成完整答案**，
            #   随后还进自检、润色再被整篇改写 —— 在运维侧完全不可见。
            #   取证轮尤其危险：tool_calls 的 JSON 被截 → 解析失败 → 静默降级成空参数（白烧一轮）。
            _fr = (choice.get("finish_reason") or "").strip().lower()
            if _fr in ("length", "max_tokens"):
                meta["truncated"] = meta.get("truncated", 0) + 1
                print(f"[agent] ⚠️ 第{step}轮输出被截断"
                      f"（finish_reason={_fr}，max_tokens={_mt}）", flush=True)
            msg = choice.get("message") or {}
            calls = msg.get("tool_calls") or []
            if not calls:
                answer_text = msg.get("content") or ""
                # ⚠️⚠️ 2026-10-22 修：**护栏 —— 拒绝「把工具调用标记当正文」的伪答案**。
                #   背景：我曾在这里加过「被截断就去掉上限重取一次」，**实测把情况改坏了**：
                #   重取时传了 `tools=None` ⇒ 模型没法正常调工具，于是把工具调用标记
                #   （`<｜｜DSML｜｜ invoke name="...">`）**当正文吐出来**，而这段垃圾被当成了答案
                #   （实测 C55 要点覆盖 1.00 → **0.00**，答案整段是标记）。
                #   ⇒ 已撤回那次重取；改为**识别并拒绝**这种内容。
                if _looks_like_tool_markup(answer_text):
                    # ⚠️ 关键：重问时**必须保留 tools** —— 上一次就是因为传了 `tools=None`，
                    #   模型才把工具调用标记写进正文（见上面的注释）。保留 tools 后，
                    #   模型要么正常调工具、要么正常写答案。
                    print(f"[agent] ⚠️ 第{step}轮「答案」是工具调用标记而非正文 → 保留 tools 重问一次",
                          flush=True)
                    meta["markup_answer"] = meta.get("markup_answer", 0) + 1
                    try:
                        _d3 = _ask_openai_full(msgs, tools=specs, tool_choice=None,
                                               thinking=_think, max_tokens=None)
                        _acc_usage(meta, _d3.get("usage"))
                        _m3 = ((_d3.get("choices") or [{}])[0].get("message") or {})
                        _c3 = (_m3.get("content") or "")
                        _calls3 = _m3.get("tool_calls") or []
                        if _c3 and not _looks_like_tool_markup(_c3):
                            answer_text = _c3
                            # ⚠️ 重问时 max_tokens=None，拿回的是完整正文；
                            #    若不重置 _fr，下面「截断续写」会对**没截断的答案**再续一次
                            #    （续写 prompt 说"你被截断了"，模型可能硬编下文）。
                            _fr = "stop"
                            meta["markup_recovered"] = meta.get("markup_recovered", 0) + 1
                            print("[agent] ✅ 重问拿回正常正文", flush=True)
                        elif _calls3:
                            # 它想继续取证 —— 用工具结果续一轮（本轮内容当草稿丢弃）
                            print("[agent] 重问返回工具调用 → 转入正常取证分支", flush=True)
                            msgs.append({"role": "assistant",
                                         "content": _m3.get("content") or "",
                                         "tool_calls": _calls3})
                            answer_text = None      # 交由循环继续处理
                        else:
                            print("[agent] ⚠️ 重问仍是标记，放弃该轮", flush=True)
                    except Exception as _e:
                        print(f"[agent] 重问失败：{_e!r}", flush=True)
                if answer_text is not None:
                    # ⭐ 2026-10-28（③ 层配套）：**被 max_tokens 截断的答案 → 续写补齐**。
                    #   实测（19:0x）deep 档答案轮 out 打满 3000、finish_reason=length，
                    #   答案断在「⚠️ 注意 ——」的第二条半句（「点过离线」后面没了），
                    #   而后续三层**都救不了它**：
                    #     · 自检轮 prompt 明确「不评价文风/字数/格式」，只查漏答 ⇒ 判 0 缺口；
                    #     · 精简轮常被时间闸门跳过（越长越容易跳）；
                    #     · 安全网 `_enforce_max_chars` 阈值 1400 字，1054 根本不触发。
                    #   ⇒ 只能在这一层补：**只续写、不重问**（重问会整篇重写，10-22 已踩坑）。
                    #   ⚠️ 撞硬墙就放弃 —— 半截答案也比 524 好。
                    if _fr in ("length", "max_tokens") and answer_text.strip():
                        _hard_guard = float(getattr(config, "HARD_DEADLINE_SECS", 75) or 0)
                        _el_now = time.time() - _t_start
                        if _hard_guard <= 0 or (_el_now + 12) <= _hard_guard:
                            answer_text = _continue_truncated(
                                msgs, answer_text, meta, thinking=_think,
                                timeout=((_hard_guard - _el_now - 3)
                                         if _hard_guard > 0 else 90.0))
                        else:
                            print(f"[agent] 已耗时 {_el_now:.0f}s + 续写预计 12s > "
                                  f"{_hard_guard:.0f}s 硬墙 → 跳过续写（保留截断答案）", flush=True)
                            meta["continue_skipped"] = "hard_deadline"
                    break
                continue
            msgs.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": calls})
            outs: list = []
            sigs: list = []
            # ⭐ 2026-10-12（主人提交 20 个 EXT 单号"白等"事故）：**并发执行本轮工具调用**。
            #   原来这里是**串行 for 循环** —— 而模型一轮常同时发多个"互相独立"的查询。
            #   实测（小主机侧，8 个 EXT 反查，每个经 Mac 中转 ~9s）：
            #     **串行 72.3s → 并发(5 线程) 16.7s，快 4.3 倍**。
            #   20 个单号原本 180s（超预算、超 Worker 210s ⇒ 用户"白等"）→ 并发后 ~35s。
            #
            #   ⚠️ 线程安全：工具用 contextvars（`_LOG` / `_COLLECT` / `_TOP_K`）。
            #     · 每个任务用 `contextvars.copy_context()` 的**独立副本**执行；
            #     · ⚠️ 拷贝**必须在父线程**做 —— 在 worker 里拷贝只会拿到空的默认上下文；
            #     · `_COLLECT` 的值是**可变 dict**，各副本共享同一对象 ⇒ 工具内部对 box 的
            #       in-place 写入（append / 计数）仍可见。已核实：工具只用 `.get()` 取 box、
            #       **不改绑定**（全项目只有一处 `.set()`，在进入循环前执行）。
            #     · `meta["trace"]` / `msgs` / `sigs` 的写入**全部留在父线程**（保序、无竞态）。
            #   ⚠️ 保序：`ex.map` 按入参顺序返回 ⇒ 工具结果与 tool_call_id 一一对应不错位。
            def _run_one(tc, ctx):
                _fn = ((tc.get("function") or {}).get("name") or "").strip()
                _raw = (tc.get("function") or {}).get("arguments") or "{}"
                try:
                    _args = json.loads(_raw) if isinstance(_raw, str) else (_raw or {})
                except Exception:
                    _args = {}
                if not isinstance(_args, dict):
                    _args = {}
                return _fn, _args, ctx.run(toolmod.run, _fn, _args)

            _tasks = [(tc, contextvars.copy_context()) for tc in calls]
            _workers = int(getattr(config, "AGENT_TOOL_WORKERS", 5) or 0)
            if len(_tasks) > 1 and _workers > 1:
                with ThreadPoolExecutor(max_workers=min(_workers, len(_tasks))) as _ex:
                    _results = list(_ex.map(lambda p: _run_one(p[0], p[1]), _tasks))
            else:
                _results = [_run_one(tc, ctx) for tc, ctx in _tasks]

            for (fn, args, out), tc in zip(_results, calls):
                outs.append(out or "")
                sigs.append(f"{fn}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}")
                meta["trace"].append({"tool": fn, "args": args, "chars": len(out or "")})
                msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": out})

            # ⚠️ 2026-10-10 评审 ⓪：预算提醒（纯追加、不改历史 → 前缀缓存不受影响）。
            #   背景：模型在循环里不知道自己第几轮（只有末轮才被告知），55% 撞满上限。
            #   在「过半」和「只剩 2 轮」这两个节点各提一句 —— 中段提醒「别铺开」，
            #   末段提醒「该收口」。只对 deep 注（fast 档 5 轮以内、秒级，不需要）。
            if depth == "deep" and _budget_remind:
                if step == (max_steps + 1) // 2:
                    msgs.append({"role": "user", "content":
                                 f"（进度：已用 {step}/{max_steps} 轮。若计划里该查的已查完，"
                                 "可以直接基于现有证据作答，不必把轮次用满。）"})
                elif step == max_steps - 1:
                    msgs.append({"role": "user", "content":
                                 f"（进度：只剩 2 轮。本轮把最关键的最后证据查完；"
                                 "下一轮将强制输出最终答案 —— 现在就开始组织结论也来得及。）"})

            # ---- 收敛判据（2026-10-05 改版；2026-10-09 补第三条）----
            # 旧判据是「本轮返回不足 200 字 → 认为没新证据」，25 题实测**一次都没触发**
            # （检索基本总能返回点东西），等于装了个摆设。deep 档又实测会撞满 8 轮上限。
            # 2026-10-05 改成看「是否在原地打转」：本轮全部是「同工具 + 同参数」的重复调用。
            # ⚠️ 2026-10-09 实测发现它**仍然几乎不触发**（deep 24 条里只命中 1 条，撞顶却占 41%）：
            #   因为模型会**不断换关键词**（签名不同 ⇒ 不算重复），而 `search_kb` 不管相不相关
            #   都能返回 1328 字（⇒ "返回量"判据永远达标）。真实轨迹里第 6、7 轮已经在查无关内容。
            #   ⇒ 补第三条：**本轮只调检索类工具、且一个新条目都没捞到** = 边际收益已归零。
            #   这个信息本来就躺在 tools._seen() 里，只是没接出来（见 tools.take_round_new_kb）。
            #   ⚠️ 只在"纯检索轮"上判定：调了 query_order 之类的轮次即使没新 KB 条目也是**有新证据**的。
            #   ⚠️ 2026-10-12（评审 P2-2）：`read_doc` 也已纳入 `_SEARCH_TOOLS`（见上方注释）。
            _dup = sum(1 for s in sigs if s in seen_calls)
            seen_calls.update(sigs)
            _all_repeat = bool(sigs) and _dup == len(sigs)
            _newkb = toolmod.take_round_new_kb()
            _blocked = toolmod.take_round_blocked()
            # 2026-10-10 ③'（默认关，KB_NEWFILE_STALL=1 开启）：文件粒度的停滞信号。
            # ⚠️ 区分度待 A/B —— 预实验（旧轨迹重放）显示第 5 轮新文件=0 有区分度，
            # 但新轨迹里模型换功能域词就能换到新文件。开了只当**额外一条**判据用。
            _newfiles = (toolmod.take_round_new_files()
                         if getattr(config, "NEWFILE_STALL", False) else 1)
            # ⚠️ 2026-10-12（评审逐行分析 P2-2）：**把 read_doc 也算作"检索类"**。
            #   为什么原本漏了它：判据⑤（目录外停滞）依赖 `_search_only`，而 10-12 的目录注入
            #   把模型行为从"换词盲搜"改成了"**反复 read_doc 读整篇**"（B 轮实测 14 次调用里 6 次是
            #   read_doc）。这种"读了几篇文档却没拿到答案"的轮次因为含 read_doc 而**不算纯检索轮**
            #   ⇒ 判据⑤ 直接失效 —— 新改动反而把判据废了。
            #   read_doc 本质上也只是"取信息、不下判断"，与 search_kb/search_code 同类。
            #   ⚠️ 安全性：read_doc 命中 `seen_doc` 时会返回"已在上文给过"（即无新内容），
            #   那种轮次本就是无进展，计入停滞是**正确**的。
            _SEARCH_TOOLS = ("search_kb", "search_code", "read_doc")
            _search_only = bool(sigs) and all(
                s.split("|", 1)[0] in _SEARCH_TOOLS for s in sigs)
            _no_new = bool(_search_only and _newkb == 0)
            _no_newfile = bool(_search_only and _newfiles == 0)
            # ⚠️ 2026-10-10 追加第五条：**纯检索轮 + 零新条目 + 在找"汇总型文档"**
            #   ⇒ 模型在搜一张库里没有的表（真实轨迹：「错误码 汇总 索引」「错误码 速查 对照表」
            #   「错误码清单」连着搜了三轮）。这和「查无此词」是**互补**的两类空转：
            #   前者是搜一个不存在的**值**（被 has_known_content 拦），
            #   后者是搜一个不存在的**文档形态**（query 里的词库里都有，所以拦不住）。
            #   ⚠️ 必须同时满足"零新条目"才算 —— 免得误伤「结算失败 汇总」这种真能捞到东西的查询。
            _idx_intent = toolmod.take_round_catalog_intent()
            _catalog_stall = bool(getattr(config, "CATALOG_STALL", True)
                                  and _search_only and _newkb == 0 and _idx_intent > 0)
            # ⚠️ 2026-10-09 追加第四条：本轮有调用被「查无此词」硬拦截
            #   ⇒ 模型正在搜一个库里**根本不存在**的值，这是**确定性**负面信号。
            #   前三条实测在真人轨迹上几乎不触发（BM25 换个词就能返回新片段，
            #   所以"新证据"永远 > 0；模型也很少"参数完全相同"地重复）。
            #   ⚠️ 覆盖范围有限（只拦"整条 query 全由查无此词构成"，如独发 `E-7731`）：
            #   模型多数时候会写成 `E-7731 结算报错`，那种**不拦**（确实能捞到东西）。
            #   所以这条是**防御性**的（防 2026-10-09 那种"同一错误码换 8 种写法 43 次调用"），
            #   真正的止损靠 search_kb/search_code 里的硬拦截文案本身（实测 search_code 4→1 次）。
            # ⚠️ 2026-10-12（评审第二轮 3.1）：`_tiny` 补上 `_search_only` 前置。
            #   原写法不限工具类型 ⇒ 若某轮只调 `query_order` 且返回短（如"查不到该订单：无记录"），
            #   合计 < 200 字就记停滞 —— 但那是**确定性负面证据**（有价值的信息），不是空转。
            #   `_tiny` 的本意是"**检索**没捞到东西"，必须和 ③⑤ 同口径。
            #   （它本来就几乎不触发；现在加了 `stall_kinds` 计数，跑一段时间若长期为 0 可直接删。）
            _tiny = bool(_search_only and sum(len(o) for o in outs) < 200)
            # ⭐ 2026-10-12（下午线上事故）：**时间预算判据**。
            #   实测：一次 deep 提问连查 15 次 query_dtp_to_pos（Mac 中转慢查询），
            #   总耗时 **230.3s** —— 而 Worker 转发超时只有 210s ⇒ 用户在 210s 被
            #   Cloudflare 断开，收到「服务不在线」兜底文案；**后端却还在跑并产出了完整答案**
            #   （结果永远到不了用户手里，还白烧 token）。
            #   这与上午修的「重试预算」是同类问题的另一个面：重试预算管的是**单轮内**
            #   的模型调用叠加，这里管的是**整个问答**的耗时叠加 —— 慢工具是"合法操作"，
            #   但叠加起来没有上界就会撑爆外层超时。
            #   预算值：**动态**（2026-10-12 主人定："根据任务多少来动态设置超时"）——
            #     fast 档 60s / deep 档基础 100s / 问题里每多一个单号（长数字串）+20s，
            #     上限 160s（160 + 收口 13 ≈ 173 < 210 ✅，且 100 基线覆盖 p90=79s）。
            #   ⭐ 预算同时写进 meta["time_budget"]，随响应头 X-Expected-Max-Seconds
            #     告诉前端"这次最多等多久"（前端可据此显示预期）。
            #   设 0 = 关闭（回旧行为）。
            _tb = _time_budget(depth, question)
            meta["time_budget"] = _tb
            # ⭐⭐ 2026-10-14（修 524 的真正兜底）：**总时长硬上限**。
            #   ⚠️ 为什么单靠 `_tb` 不够：`_time_budget` 会按"问题里的单号个数"**上调**
            #     （每个 +20s，上限 160s）—— 批量查单场景下它能涨到 160s，
            #     而 **CF 免费版边缘硬超时只有 100s** ⇒ 预算还没触发，CF 已经掐断。
            #   ⇒ 本值**不受单号加成影响**，是"从请求开始算的绝对上限"。
            #     取 75s：给"收口轮 13s + 响应回传 15s"留位 ⇒ 88s < 100s ✅
            _hard = float(getattr(config, "HARD_DEADLINE_SECS", 75) or 0)
            _el = time.time() - _t_start
            # ⚠️⚠️ 2026-10-21 修（真 bug）：原来「硬上限」与「预算」两个分支各写一遍，
            #   而**硬上限分支漏掉了收口调用 + break**（只有标记 + 追加提示）⇒
            #   ① 该轮不产生答案；② 循环继续，下一轮又用 `tool_choice="required"`
            #   （deep 档强制取证）**再调一次工具**，与刚追加的「不要再调用工具」直接对冲；
            #   ③ 每轮重复 append 同一条指令，上下文越来越贵。
            #   实测证据：日志里出现 `已耗时 105s … 总耗时 105.0s`，硬上限 75s 形同虚设。
            #   ⇒ 现在两个分支**共用同一段收口逻辑**，不再有机会漏写。
            _cut_reason = ""
            if _hard > 0 and step < max_steps and _el > _hard:
                meta["time_budget_hit"] = True
                meta["hard_deadline_hit"] = round(_el, 1)
                _cut_reason = "硬上限 %.0fs" % _hard
                print(f"[agent] 已耗时 {_el:.0f}s 超过**硬上限** {_hard:.0f}s"
                      f"（第{step}/{max_steps}轮）→ 强制收口（CF 墙 100s）", flush=True)
            elif _tb > 0 and step < max_steps and _el > _tb:
                meta["time_budget_hit"] = True
                _cut_reason = "预算 %.0fs" % _tb
                print(f"[agent] 已耗时 {_el:.0f}s 超过预算 {_tb}s，"
                      f"强制收口（第{step}/{max_steps}轮）", flush=True)
            if _cut_reason:
                msgs.append({"role": "user", "content":
                             "取证时间已到上限，请立即基于已查到的证据输出最终答案"
                             "（结构：结论 / 事实（可选）/ 处理步骤 / ⚠️ 注意），不要再调用工具；"
                             "没查完的部分明确说「时间所限未核完，需要人工核对」，不要猜。"})
                _t_cut = time.time()
                _d = _ask_openai_full(msgs, thinking=_think_cut)
                print(f"[agent] {_cut_reason}收口轮 {time.time()-_t_cut:.1f}s", flush=True)
                _acc_usage(meta, _d.get("usage"))
                answer_text = (((_d.get("choices") or [{}])[0].get("message") or {})
                               .get("content") or "")
                meta["early_stop"] = step
                break
            # ⭐ 2026-10-12（评审逐行分析 P1-2）：**逐条计数**。
            #   动机：6 条判据里 ①②⑥ 都标注"几乎不触发/待验证"，但日志只看得到
            #   "连续2轮无新证据"这个**结论**，分不清是哪条命中 ⇒ **无法用数据决定"哪条可以删"**，
            #   下次评审还是只能猜。现在把每条各命中几次记进 meta 并打印，
            #   跑一段时间后就能拿数据说"判据②从未触发，可以删"。
            _kinds = {"blocked": bool(_blocked), "repeat": _all_repeat,
                      "no_new": _no_new, "catalog": _catalog_stall,
                      "no_newfile": bool(_no_newfile), "tiny": _tiny}
            for _k, _hit in _kinds.items():
                if _hit:
                    meta.setdefault("stall_kinds", {})
                    meta["stall_kinds"][_k] = meta["stall_kinds"].get(_k, 0) + 1
            if any(_kinds.values()):
                stall += 1
                if _blocked:
                    print(f"[agent] 本轮 {_blocked} 次检索命中「查无此词」硬拦截（第{step}轮）",
                          flush=True)
                elif _catalog_stall:
                    print(f"[agent] 本轮 {_idx_intent} 次在找「汇总型文档」且零新条目"
                          f"（第{step}轮，库中无此类文档）", flush=True)
                elif _all_repeat:
                    print(f"[agent] 本轮 {len(sigs)} 次调用全是重复（第{step}轮）", flush=True)
                elif _no_newfile:
                    print(f"[agent] 本轮纯检索、零新文件（第{step}轮，文件级停滞）", flush=True)
                elif _no_new:
                    print(f"[agent] 本轮纯检索、零新条目（第{step}轮，新证据枯竭）", flush=True)
            else:
                stall = 0
            if stall >= getattr(config, "STALL_ROUNDS", 2) and step < max_steps:
                # ⭐ 2026-10-12（评审 P1-2）：把「各判据各命中几次」一并打出来 ——
                #   没有这个数，就永远回答不了"判据②到底有没有用、能不能删"。
                _kdef = "、".join(f"{k}×{v}" for k, v in (meta.get("stall_kinds") or {}).items())
                print(f"[agent] 连续{stall}轮无新证据，提前收口（第{step}/{max_steps}轮）"
                      f"{'｜命中判据：' + _kdef if _kdef else ''}", flush=True)
                meta["early_stop"] = step
                msgs.append({"role": "user", "content":
                             "连续两轮没有查到新证据，已到信息边界。请立即基于现有证据输出最终答案"
                             "（结构：结论 / 事实（可选）/ 处理步骤 / ⚠️ 注意），不要再调用工具；"
                             "确实无法确认的部分要明说「需要人工核对」，不要猜。"})
                _t_cut = time.time()
                _d = _ask_openai_full(msgs, thinking=_think)
                print(f"[agent] 提前收口轮 {time.time()-_t_cut:.1f}s（思考={_think}）", flush=True)
                _acc_usage(meta, _d.get("usage"))
                answer_text = (((_d.get("choices") or [{}])[0].get("message") or {})
                               .get("content") or "")
                break
        print(f"[agent] 取证+作答 {time.time()-_t_start:.1f}s"
              f"（{meta.get('steps')} 轮，上限 {max_steps}）", flush=True)
        if not answer_text:
            # 轮数用尽仍在调工具（极少数）：再要一次纯文本答案
            msgs.append({"role": "user", "content":
                         "请直接用文字作答，不要再调用工具。"})
            data = _ask_openai_full(msgs, thinking=_think_cut)
            _acc_usage(meta, data.get("usage"))
            answer_text = (((data.get("choices") or [{}])[0].get("message") or {})
                           .get("content") or "")
    except Exception as e:
        print(f"[agent] 闭环失败，回退单轮：{type(e).__name__}: {e}", flush=True)
        return fallback()
    finally:
        box = toolmod.stop_collect()
        meta["kb_hits"] = box.get("kb_hits", [])
        toolmod.clear_context()

    if not answer_text.strip():
        return fallback()
    # 自检轮（B2，仅 deep）：答案产出后复核「问了几件事 / 每件有没有证据」，有缺口才修订。
    # ⚠️ 放在 <diagram> 解析与篇幅收口**之前**：修订版答案同样可能带图数据、同样要过篇幅安全网。
    # ⭐ 2026-10-12（二次事故复盘）：**预算收口场景下跳过自检** ——
    #   实测 215.7s 那次：预算 150s 触发后，收口轮 12.9s + 自检轮 24.7s，总计仍超 210s 三个百分点。
    #   都已经"时间不够"了还花 25s 复核，本末倒置 —— 收口轮的提示词已要求"没查完的明说"，
    #   缺口披露由它负责，自检可以省。撞顶/提前收口不受影响（那两种时间还有富余）。
    # ⭐ 2026-10-14（修 524）：**跨阶段总时间闸门**。
    #   ⚠️ 这是本次修 524 的核心 —— 原来的预算（_time_budget）**只管取证循环**，
    #     取证收口后，自检轮/精简轮照样各花 14s/39s 往上叠，总耗时轻松越过
    #     **CF 免费版 100s 硬墙** ⇒ 实测线上 **27.8% 的请求被掐成 524**。
    #   ⇒ 现在给"收口之后的两轮"设一道闸：**离总上限（CF 墙）只剩不够时，直接不做**。
    #     宁可交一份没复核/没精简的答案，也不能让用户等到 524（那是彻底失败）。
    #   ⚠️ 阈值 85s：CF 墙 100s，留 15s 给"响应回传 + 边缘转发"的余量。
    _elapsed = time.time() - _t_start
    _deadline = float(getattr(config, "TOTAL_DEADLINE_SECS", 85) or 0)
    # ⚠️⚠️ 2026-10-14 二次修正（第一次修得太天真，A03 仍 524）：
    #   第一版只看「已耗时 > 阈值」，但**精简轮本身要跑 40~46s**（实测中位 38.6s）——
    #   取证 40s + 自检 10s = 50s（没到 85s）⇒ 放行 ⇒ 精简再跑 46s ⇒ 总 96s+ ⇒ 撞墙。
    #   ⇒ 正确判据是**「已耗时 + 该阶段的预计耗时」是否超墙**，即对每个阶段
    #     单独估算"跑完会不会超"。
    # 预计耗时按实测中位取值（宁可保守跳过，也不撞 524 —— 524 是彻底失败，
    # 而跳过自检/精简只是"少一道复核"，答案本身照样完整）。
    _est_self = float(getattr(config, "EST_SELFCHECK_SECS", 20) or 0)
    _est_polish = float(getattr(config, "EST_POLISH_SECS", 40) or 0)
    _skip_self = bool(_deadline) and (_elapsed + _est_self) > _deadline
    _skip_polish = bool(_deadline) and (_elapsed + _est_polish) > _deadline
    if _skip_self or _skip_polish:
        meta["deadline_guard"] = round(_elapsed, 1)
        print(f"[agent] 已耗时 {_elapsed:.0f}s，预计 自检{_est_self:.0f}s/精简{_est_polish:.0f}s "
              f"→ 跳过={'自检+精简' if _skip_self and _skip_polish else ('自检' if _skip_self else '精简')}"
              f"（防撞 CF 100s 墙 ⇒ 524）", flush=True)
    # ⚠️ 2026-10-21：把总上限换算成**绝对时刻**，传给自检/精简当硬超时用。
    _deadline_ts = (_t_start + _deadline) if _deadline else None
    if depth == "deep" and getattr(config, "SELFCHECK_ENABLED", False) \
            and not meta.get("time_budget_hit") and not _skip_self:
        _trace_txt = "、".join(
            f"{t.get('tool')}({json.dumps(t.get('args') or {}, ensure_ascii=False)[:120]})"
            for t in (meta.get("trace") or []))[:900]
        _t_sc = time.time()
        answer_text, _gaps = _selfcheck_round(question, answer_text, _trace_txt, meta,
                                              deadline_ts=_deadline_ts)
        meta["selfcheck"] = {"gaps": _gaps}
        print(f"[agent] 自检完成：缺口 {len(_gaps)} 处（{time.time()-_t_sc:.1f}s）", flush=True)
    # ⭐ 2026-10-14（主人要求）：**精简轮** —— 最终答案产出前的「废话审查」。
    #   「回答还是太多了…在生成最终消息之前加一轮消息审查，删掉对用户来说是废话的文本」。
    #   ⚠️ 位置讲究：必须放在**自检轮之后**（否则精简掉的措辞会被自检判成缺口又加回来）、
    #     **`extract_diagram` 之前**（图数据块不能当正文被精简掉）、
    #     **`_enforce_max_chars` 之前**（先做语义减法，再让字数安全网兜底）。
    #   ⚠️ 预算收口场景跳过（同自检轮的理由：已经没时间了，不想再加一轮）。
    #   ⚠️ 只对 **deep 档**生效：fast 档答案本来就短（中位 4s / 百余字），
    #     为它多花 4s 做精简不划算（那是 2 倍延迟换几十字）。冗长问题出在 deep。
    #   ⚠️ 2026-10-14 加：**预计会撞墙则跳过**（见上方 `_skip_polish` 说明）。
    #   ⚠️⚠️ 2026-10-21 修（闸门过期判断）：`_skip_polish` 原来是用**自检之前**的
    #      `_elapsed` 算定的，自检跑完（实测中位 18s、p90 68s）**不重算** ⇒
    #      「取证 35s 时判定可以精简」→ 自检再花 30s → 精简再跑 43s → 总 108s 撞墙。
    #      ⇒ 现在自检之后**用新鲜耗时重算一次**，两道闸都按当下真实情况判。
    if depth == "deep" and getattr(config, "POLISH_ENABLED", False) \
            and not meta.get("time_budget_hit") and not _skip_polish:
        _elapsed_now = time.time() - _t_start
        if _deadline and (_elapsed_now + _est_polish) > _deadline:
            print(f"[agent] 自检后重算：已耗时 {_elapsed_now:.0f}s + 精简预计 "
                  f"{_est_polish:.0f}s > {_deadline:.0f}s → 跳过精简（防 524）", flush=True)
            meta["deadline_guard"] = round(_elapsed_now, 1)
        else:
            _t_pl = time.time()
            answer_text = _polish_round(question, answer_text, meta,
                                        deadline_ts=_deadline_ts)
            print(f"[agent] 精简完成（{time.time()-_t_pl:.1f}s）", flush=True)
    # 简图：模型把 <diagram> 数据块写在答案最前面，这里解析成 SVG，并把该块从正文里摘掉。
    # ⚠️ 必须放在 _enforce_max_chars 之前：图数据不该占字数配额，也不该被压缩调用改写。
    try:
        answer_text, meta["diagram"] = toolmod.extract_diagram(answer_text)
    except Exception as e:                       # 解析失败只是没有图，绝不影响答案本身
        print(f"[diagram] 解析失败：{type(e).__name__}: {e}", flush=True)
    # 2026-10-10 评审 ②：撞顶标注。deep 档 55% 的答案出自「轮数用尽强制收口」，
    # 证据可能不全 —— 用户和案例审查页都看不出这一差别。现在：答案尾部加一行
    # 小字（可读、克制），并写入 meta（agent.rounds == max_steps 即撞顶）。
    # 提前收口（early_stop 有值）不算撞顶 —— 那是「查够了主动停」，是好事。
    if depth == "deep" and getattr(config, "ROUNDCAP_NOTE", True) \
            and meta.get("steps") and meta["steps"] >= max_steps and not meta.get("early_stop"):
        meta["rounds_capped"] = True
        answer_text += ("\n\n---\n"
                        "ⓘ 本题线索较多，已按现有证据上限作答；若关键信息仍缺，"
                        "补充单号/截图后再问一次，可以查得更深。")
    # ⭐ 总耗时（含规划轮 / 自检轮 / 收口轮）—— 排障与调参全靠它，别删。
    # ⚠️ 端到端还会多出「排队 + 检索 + 篇幅收口」，所以它 ≤ 用户看到的墙钟时间，不相等的差额是正常的。
    _elapsed = time.time() - _t_start
    print(f"[agent] 总耗时 {_elapsed:.1f}s（分档={depth}）", flush=True)
    # 2026-10-10 主人要求：耗时追加在答案末尾（deep 一等等 50 秒+，用户有权利知道花了多久）。
    # 只对 deep 挂（fast 中位 4s，加那行反而像卡过）；fast 超过 10s 也给（异常慢时该有解释）。
    if getattr(config, "ELAPSED_NOTE", True) and answer_text.strip():
        if depth == "deep" or _elapsed >= 10:
            meta["elapsed"] = round(_elapsed, 1)
            answer_text += f"\n（本次处理 {meta['elapsed']} 秒）"
    return _strip_blank_lines(_enforce_max_chars(_fmt_answer(answer_text, empty_chunks, []),
                                                 deadline_ts=_deadline_ts)), meta


# ---------------- 回答篇幅安全网 ----------------
# 定位（2026-09-16 按主人要求调整）：篇幅**不是硬指标**。原设计是「硬上限 1000 + 最多三轮压缩」，
# 实测有两个坏处：① 首稿 1136 字（其实已经够清楚）也会触发，白烧 8 秒；
# ② 三轮里模型常返回**逐字相同**的版本 —— 说明它认为再删就要动单号 / 参数值了。
# 现在改成：质量由 prompt 的「输出前通读做减法」负责，这里只在**严重超限**时压一轮。
# 设计取舍：宁可多一次调用，也不在本地硬截断 —— 截断会把「第 5 步」「⚠️ 注意」整段切掉，
# 比超字数危险得多。压不动就**保留原文**（超字数是小问题，丢内容是大事）。
_TRIM_SYSTEM = (
    "你在给一份**已经写好的**运维答复做压缩。这是压缩任务，不是重写任务。\n"
    "硬性要求：\n"
    "1. 段标题原样保留：「结论 —— 」「事实 —— 」「处理步骤 —— 」「⚠️ 注意 —— 」——**原文有的一个都不许丢**"
    "（「事实 ——」是 2026-10-28 新增的关键值清单段，删了它就等于丢证据）。\n"
    "2. **严禁删改**后台点击路径（如「数据管理 → 商场列表 → 商场详情」）、参数名、参数值、"
    "支付编码、编号步骤的**内容**。这些是答复的价值所在，一个都不能少。\n"
    "3. 允许的删减（**由轻到重，够用就停**，别一上来就删整段）：\n"
    "   ① 删掉解释性从句（「因为…所以…」「这是由于…」）与背景铺垫，只留动作和判断；\n"
    "   ② 合并重复表述：同一件事说两遍的只留一遍；\n"
    "   ③ 压缩「填 xx、xx、xx」这类表单字段罗列（路径与参数值不在此列，不许动）；\n"
    "   ④ 「⚠️ 注意」超过 2 条时只留最重要的 2 条；\n"
    "   ⑤ 「需要你补充 —— 」先压到 1 个问题，**还超线才整段删掉**。\n"
    "4. 结论的第一句必须原样保留其核心判断，不得改变结论本身。\n"
    "5. 直接输出压缩后的全文，不要写「以下是压缩后的内容」这类说明，不要新增任何原文没有的信息。\n"
)


def _nows_len(s: str) -> int:
    """按「去掉所有空白后的字符数」计长 —— 与用户看到的长短最接近。"""
    return len(re.sub(r"\s", "", s or ""))


def _step_count(text: str) -> int:
    """只数「处理步骤」段落里的编号条数。

    ⚠️ 不能直接全篇数 `^\\d+\\.`：那样会连「需要你补充」里的编号问题一起数进去，
    而压缩时是**允许整段删掉「需要你补充」**的 —— 全篇计数会把合法压缩误判成「步骤被删」而拒收。
    """
    m = re.search(r"处理步骤\s*——(.*?)(?=\n(?:⚠️|需要你补充|结论)\s*——|\Z)", text, re.S)
    seg = m.group(1) if m else text
    return len(re.findall(r"^\s*\d+\.", seg, re.M))


def _try_trim(text: str, limit: int, ratio: float = 0.85,
              timeout: float = 150.0) -> str | None:
    """调一次模型压缩；返回 None 表示这次结果不可用（异常 / 段落丢了 / 步骤被删）。

    ratio = 给模型的「目标值」占验收线的比例。必须 <1：模型按自己的口径估「字」，
    而验收口径（去空白后连 markdown 标记一起数）普遍比它多 12~16% ——
    直接叫它压到 1000，交回来往往是 1030。

    ⚠️ 2026-10-21：改为走 `_ask_openai_full`，**显式带超时 + 强制 thinking=disabled**。
      原来走 `_ask_openai`（无超时参数），这一轮跑在**整个请求的最末尾、完全在
      85s 总闸门之外**：实测触发 10 次，每次都是一次整篇重写，只为从 1459 字压到
      1350 字（省 109 字），却可能再花几十秒 ⇒ 用户看到的墙钟时间直接越过 CF 100s 墙。
      这是纯机械的"照着删字"任务，不需要思考档位。
    """
    target = max(200, int(limit * ratio))
    extra = ""
    if ratio < 0.80:
        # 实测：第 1 轮「温和压缩」到 1030 左右就停了（重复请求会返回**逐字相同**的结果，
        # 说明模型认为再删就要动路径/参数值了）。所以更狠的轮次必须显式放开段落级删减授权。
        extra = ("\n这次是**必须达标**的一轮，上一轮的温和压缩没压下来。请更果断地删："
                 "「⚠️ 注意」可以只留最重要的 1 条；「需要你补充 —— 」整段删掉；"
                 "「处理步骤」里同类动作可以合并成一条（如「新建并保存参数」）。"
                 "但点击路径、参数名、参数值、支付编码仍然一个字都不能改。\n")
    user = (f"下面这段答复约 {_nows_len(text)} 字，请压到 {target} 字以内（按去掉空白后的字符数计）。\n"
            f"记住：只压措辞，路径 / 参数名 / 参数值 / 编号步骤一个都不能少。{extra}\n\n{text}")
    try:
        msgs = [{"role": "system", "content": _TRIM_SYSTEM},
                {"role": "user", "content": user}]
        if config.LLM_PROVIDER == "openai":
            data = _ask_meta_bounded(msgs, wall=timeout, thinking="disabled",
                                     timeout=timeout)
            out = (((data.get("choices") or [{}])[0].get("message") or {})
                   .get("content") or "")
        elif config.LLM_PROVIDER == "ollama":
            out = _ask_ollama(msgs)
        else:
            return None                     # echo 模式不压缩
    except Exception:
        return None
    out = (out or "").strip()
    if out.count("——") < 2:
        return None                          # 段标题都丢了，不能用
    if _step_count(out) < _step_count(text):
        return None                          # 处理步骤被删了，不能用
    return out


_POLISH_PROMPT = (
    "你是**文字精简员**。下面是一份**排障答复**，用户（门店收银/一线运维）要拿它**直接照做**。\n"
    "你的唯一任务：**删掉对用户没用的文字**，让答案更短、更快能看完。\n\n"
    "【用户原问题】\n{question}\n\n"
    "【当前答案】\n{draft}\n\n"
    "**必须原样保留**（删了就是错）：\n"
    "1. 固定结构：结论（≤40 字，只说判断与影响）→ 事实（关键值清单，一行一个）→ 处理步骤 → ⚠️ 注意；\n"
    "2. **所有可执行内容**：后台点击路径、参数名与取值、单号、报错原文、金额、时间、\n"
    "   数字、字段名、错误码、接口名、按钮文案；\n"
    "3. **所有证据行**（如「提交 `3acefe43`」「全库 0 命中」「2026-09-09 修复」）；\n"
    "4. 归属方与转派口径（是 POS 还是下游、找谁处理）；\n"
    "5. 「需要人工核对 / 未查到」这类**诚实声明**（这是有用的，不许删）。\n\n"
    "**要删的（这才是废话）**：\n"
    "- 复述用户的问题（「您问的是…」「关于您提到的…」）；\n"
    "- 自我描述与过程（「我先查了…」「经过检索发现…」「我是这样理解的…」）；\n"
    "- 客套与过渡（「希望对您有帮助」「祝您工作顺利」「总之」「综上所述」「值得一提的是」）；\n"
    "- 对同一个意思的重复表述、重复强调；\n"
    "- 用户不需要知道的内部实现细节（除非它是排障的关键）；\n"
    "- 空泛的兜底建议（「建议联系运维」若前面已给具体动作则删）。\n\n"
    "**硬性要求**：\n"
    "1. **不改写、不新增、不重排结构** —— 只做减法。措辞可微调以保持语句通顺。\n"
    "2. 拿不准的**一律保留**（宁可留长，不可丢信息）。\n"
    "3. 直接输出精简后的**完整答案正文**，不要任何说明、不要 markdown 围栏。\n"
)


def _draft_in_limit() -> int:
    """答案草稿送入「自检轮/精简轮」的**字符上限**（两者必须同值，见下方警示）。

    ⚠️⚠️ 2026-10-14（主人质疑"中间产物被截断导致答案不可用"）：
      这两轮都是**读草稿 → 写回答案**的结构。若草稿被截断送入，模型输出的"精简/修订版"
      必然缺后段，而它自己不知道 ⇒ 直接替换会**静默丢内容**。
      ⇒ 铁律：**草稿超上限就整轮跳过**（宁可不精简/不修订，绝不丢内容）。
      ⚠️ 两个阈值**必须一致**：否则会出现"自检跳过、精简执行"的错配窗口
        （中间那段长度的草稿，会被其中一轮按截断内容改写）。
    """
    return int(getattr(config, "DRAFT_IN_MAX_CHARS", 8000) or 8000)


def _polish_round(question: str, draft: str, meta: dict,
                  deadline_ts: float | None = None) -> str:
    """最终答案产出前的**精简轮**：删掉对用户没用的文字。

    ⭐ 2026-10-14 主人要求：「回答还是太多了…在生成最终消息之前加一轮消息审查，
    删掉对用户来说是废话的文本」。位置：**自检轮之后、最终输出之前**
    （自检管"有没有漏答/缺证据"，本层管"有没有废话"—— 两件事，互补不重叠）。

    ⚠️ 与 `_enforce_max_chars` 的分工（别混）：
      · 本函数 = **语义级**减法（模型判断哪句是废话），**不管字数**；
      · `_enforce_max_chars` = **字数级**安全网（只在彻底写飞时压一轮），不做语义判断。
      ⇒ 两者串联：先精简，再兜字数。

    ⚠️⚠️ 2026-10-14（主人质疑"中间产物被截断导致答案不可用"）**修掉一颗雷**：
      原实现把草稿**截前 8000 字**送进模型，而本轮的输出**会整体替换答案本体**。
      若答案 >8000 字：模型压根没见过后半段 ⇒ 输出的精简版**永久丢掉后段**；
      而"删过头"护栏按**原文总长**的 25% 判，阈值被后半段抬高 ⇒ **护栏放行、无人察觉**。
      ⇒ 现在：**草稿超过送入上限就整轮跳过**（宁可不精简，绝不丢内容）。
    """
    if not draft or not draft.strip():
        return draft
    _max_in = _draft_in_limit()
    if len(draft) > _max_in:
        print(f"[polish] 草稿 {len(draft)} 字 > 送入上限 {_max_in} 字 → "
              f"跳过精简（避免截断导致丢内容）", flush=True)
        return draft
    # ⚠️⚠️ 2026-10-14（修 524）：**短答案直接跳过精简轮**。
    #   实测（线上 57 次）：精简轮**中位 38.6s / 均值 42.2s / max 151s**，
    #   是整条链路**最贵的一步**（比自检轮还贵），而只减 200~400 字。
    #   而 CF 免费版边缘硬超时 **100s** —— 线上 **27.8% 的请求超 100s** 被掐成 524。
    #   算账：取证 ~33s + 自检 ~14s + 精简 ~39s ≈ 86s（中位），再抖一下必超。
    #   ⇒ 短答案（本来就啰嗦得有限）不值得为它花 40 秒 + 冒 524 的风险。
    #   阈值 500 字：低于它时"废话"总量本就不多，删了收益 < 超时风险。
    _min_in = int(getattr(config, "POLISH_MIN_CHARS", 500) or 0)
    if _min_in and _nows_len(draft) < _min_in:
        print(f"[polish] 草稿 {_nows_len(draft)} 字 < {_min_in} 字 → 跳过精简（省时间）",
              flush=True)
        return draft
    try:
        _th = getattr(config, "THINKING_POLISH", "high")
        prompt = _POLISH_PROMPT.format(question=(question or "")[:2000],
                                       draft=draft)
        # ⚠️ 2026-10-21（治 524）：把**剩余预算当硬超时**，而不是写死 150s。
        #   精简轮实测中位 43.3s / max 61.1s，写死 150s 意味着它可以一路跑到撞墙。
        _to = 150.0 if not deadline_ts else max(6.0, deadline_ts - time.time())
        data = _ask_meta_bounded([{"role": "user", "content": prompt}],
                                 wall=_to, thinking=_th, timeout=_to)
        _acc_usage(meta, data.get("usage"))
        out = (((data.get("choices") or [{}])[0].get("message") or {})
               .get("content") or "").strip()
    except Exception as e:                             # noqa: BLE001
        print(f"[polish] 精简轮失败（忽略）：{type(e).__name__}: {e}", flush=True)
        return draft

    src = draft or ""
    # 护栏 1：空输出
    if not out:
        print("[polish] 输出为空 → 用原文", flush=True)
        return src
    # 护栏 2：删过头（低于原文 25%）
    if _nows_len(out) < _nows_len(src) * 0.25:
        print(f"[polish] 删过头（{_nows_len(out)} < {_nows_len(src)}*0.25）→ 用原文", flush=True)
        return src
    # 护栏 2′（2026-10-21 新增）：**不得变长**。
    #   本轮的职责是"只做减法"，但原来**只防删太多、不防越删越长**：
    #   实测存在把 700 字改写成 1000 字仍被原样采纳的情况（四道护栏都不拦）。
    #   变长的输出直接丢弃、保留原文 —— 原文最多是啰嗦，改写变长则两头都输。
    if _nows_len(out) > _nows_len(src):
        print(f"[polish] 反而变长（{_nows_len(src)} → {_nows_len(out)}）→ 用原文", flush=True)
        return src
    # 护栏 3：三段式标记丢了
    #   ⚠️ 2026-10-21 同时收紧阈值：原来用 `min(count, 2)`，原文有 3 个标记时阈值只判 2，
    #     于是**整段「⚠️ 注意」被删掉也能通过**。现在要求标记数不减。
    if out.count("——") < src.count("——"):
        print(f"[polish] 三段式标记丢失（{out.count('——')} < {src.count('——')}）→ 用原文",
              flush=True)
        return src
    # 护栏 4：步骤变少
    if _step_count(out) < _step_count(src):
        print(f"[polish] 步骤变少（{_step_count(out)} < {_step_count(src)}）→ 用原文", flush=True)
        return src
    saved = _nows_len(src) - _nows_len(out)
    print(f"[polish] 精简轮 {_nows_len(src)} → {_nows_len(out)} 字（-{saved}）", flush=True)
    return out


_TRIM_MIN_LEFT = 15.0      # 剩余预算不足这么多秒就放弃字数压缩（治 524）


def _strip_blank_lines(text: str) -> str:
    """去掉答案里的空行（2026-10-23 主人要求：「空行太多，以后不要空行」）。

    ⚠️ 为什么用代码做而不是只写进 prompt：
      prompt 规则对格式类要求**不可靠**（模型会漂），而这是纯机械变换、零成本、零风险，
      代码做是 100% 确定性的 —— 与「字数必须加代码兜底」同一个教训。
    做法：连续多个换行压成一个；行尾空白去掉；段与段之间紧挨着。
    只动空白，**不碰任何字符内容**（路径 / 参数名 / 单号 / 错误码全保留）。
    """
    if not text:
        return text
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)      # 行尾空白
    s = re.sub(r"\n{2,}", "\n", s)        # 连续空行 → 单个换行
    s = re.sub(r"\n[ \t]+", "\n", s)      # 空行里夹空格的情况
    return s.strip()


_CONCLUSION_HEADS = ("**结论 ——**", "**结论——**", "结论 ——", "结论——")


def _split_conclusion(text: str, soft: int = 70) -> str:
    """把过长的「结论 ——」段压成一句，其余内容搬到「事实 ——」段。

    ⚠️ 为什么必须代码兜底（2026-10-28 实测）：prompt 已写明「结论段整段 ≤40 字」并给了正反例，
      模型**照样**写出 518 字的结论段（还把事实、参数、推理全塞在里头）——
      flash 模型对长 system prompt 中段的位置约束遵循很弱，光靠提示词不可靠。
      而「结论一眼能懂」是主人最在意的可读性指标（原话：读完还是不知道要不要担心）。
      ⇒ 提示词负责"尽量写好"，这里负责"兜住底线"。

    ⚠️ 只在**明确存在「结论 ——」标记**时才动（fast 档短答案没这个标记 ⇒ 原样返回）。
    ⚠️ 切点优先句末标点（。！？）；单句仍过长时退到逗号/分号，但不小于 floor 字，
      避免把一句话切得七零八落。
    ⚠️ 纯本地字符串处理，不调模型、零额外耗时。
    """
    if not text:
        return text
    i, head = -1, ""
    for h in _CONCLUSION_HEADS:
        k = text.find(h)
        if k >= 0 and (i < 0 or k < i):
            i, head = k, h
    if i < 0:
        return text

    # 结论段的结束位置：下一个段标记或空行
    j = len(text)
    for m in ("\n处理步骤", "\n⚠️ 注意", "\n事实 ——", "\n\n"):
        k = text.find(m, i + len(head))
        if k > 0:
            j = min(j, k)
    body = text[i + len(head):j].strip()
    if not body or _nows_len(body) <= 60:
        return text

    # 切点：① 在 soft 窗口内取**最后一个**句末标点；② 没有就退到逗号/分号/冒号；
    #   ③ 仍没有才放宽到 soft*1.6 找第一个句号；④ 最坏按 soft 硬切。
    #   ⚠️ 不能写成"取第一个句号、太长再退逗号" —— 实测 soft 设小反而切出更长的句子
    #     （soft=50 出 122 字，soft=70 只出 56 字），因为回退分支拿的是第一个句号。
    win = body[:soft]
    cut = max((win.rfind(p) for p in "。！？"), default=-1)
    if cut < 0:
        cut = max((win.rfind(p) for p in "，；："), default=-1)
    # ⚠️ 切得太早会把**最关键的"结果/影响"**那句也搬进事实段
    #   （例：「结论 —— 这是控制缺口。钱已原路退回顾客。」只切第一句 → 用户看不到"钱退了"）
    #   ⇒ 不足 30 字时，往后找下一个句末标点，让结论段容得下两小句。
    if cut + 1 < 30:
        wide = body[: int(soft * 1.5)]
        for p in "。！？":
            k = wide.find(p, cut + 1 if cut >= 0 else 0)
            if k >= 0:
                cut = k
                break
    if cut < 0:
        wide = body[: int(soft * 1.6)]
        for p in "。！？":
            k = wide.find(p)
            if k >= 0:
                cut = k
                break
    if cut < 0:
        cut = min(len(body) - 1, soft - 1)
    keep, move = body[:cut + 1].strip(), body[cut + 1:].strip()
    if not keep or not move:
        return text

    suffix = text[j:]
    st = suffix.lstrip()
    for h in ("事实 ——", "**事实 ——**"):
        if st.startswith(h):
            suffix = "\n" + st[len(h):].lstrip("\n")
            break
    out = f"{text[:i]}{head} {keep}\n\n事实 ——\n{move}{suffix}"
    print(f"[tidy] 结论段 {_nows_len(body)} 字 → {_nows_len(keep)} 字（其余并入事实段）",
          flush=True)
    return out


def _enforce_max_chars(text: str, deadline_ts: float | None = None) -> str:
    """安全网：只在严重超限时压**一轮**，压不动就保留原文。

    为什么不再追「达标」：主人 2026-09-16 定调「900 字也不是强制的，怕的是废话太多」——
    长度本身不是问题，废话才是，而废话由 prompt 的「输出前通读做减法」负责清理。
    这一层只拦模型彻底写飞（如把整份日志逐笔抄进答案写成几千字）。
    任何一步异常都返回原文，绝不返回空、绝不本地截断。

    ⚠️ 2026-10-21（治 524）：本函数跑在**整个请求的最末尾**，原来**完全不受 85s 闸门约束**
      —— 实测触发 10 次，每次只为省 100 字左右，却可能再花几十秒，直接把用户墙钟时间
      推过 CF 100s 墙。现在两条约束：
        ① 剩余时间不足 `_TRIM_MIN_LEFT` 秒 → 直接放弃压缩（超限的那百来字不值得冒 524）；
        ② 压缩调用带硬超时 = 剩余时间。
    """
    # ⭐ 2026-10-28：先做「结论段整形」，**且不受字数上限约束**。
    #   原因：实测答案全文 1145 字（< ANSWER_MAX_CHARS=1400）但结论段 518 字 ——
    #   原来的 `if _nows_len(text) <= limit: return text` 会直接返回，整形没机会跑。
    text = _split_conclusion(text)
    limit = getattr(config, "ANSWER_MAX_CHARS", 0)
    if not limit or _nows_len(text) <= limit:
        return text
    print(f"[trim] 触发安全网：{_nows_len(text)} > {limit}", flush=True)
    if deadline_ts:
        _left = deadline_ts - time.time()
        if _left < _TRIM_MIN_LEFT:
            print(f"[trim] 只剩 {_left:.0f}s < {_TRIM_MIN_LEFT:.0f}s → 放弃压缩"
                  f"（超限 {_nows_len(text) - limit} 字，不值得冒 524 风险）", flush=True)
            return text
        _to = max(6.0, _left)
    else:
        _to = 150.0
    out = _try_trim(text, limit, 0.8, timeout=_to)
    print(f"[trim] 单轮 -> {None if out is None else _nows_len(out)}", flush=True)
    if out and _nows_len(out) < _nows_len(text):
        return out
    return text

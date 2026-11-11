"""取证工具层：把 知识库 / 源码 / 日志 包装成 function-calling 工具，交给模型自主取证。

为什么要有这一层（2026-09-16）：
    此前 /ask 是「单向管道」—— 查什么由 main.py 里的 if/else 与阈值预先决定，模型只能在最后
    一步改写挑好的素材。于是「检索没命中 → 它照样照着编」几乎是必然：它没有取证权，也没法
    回头补查。把工具交给模型后，它会在 4 轮内自己收口，且比原流程更快更省
    （同一条报障：15.1s/1099字 → 8.1s/978字，输入 token 2.8万 → 1.13万）。

⚠️ 三条工程纪律（都是原型实测踩出来的，动手改之前先读）：
    1. **返回必须精炼**：单次返回上限 config.TOOL_MAX_CHARS。原型 v1 每次返回 3500 字 ×10 次，
       输入 token 冲到 5.86 万（比整段灌日志更贵），且 6 轮仍不收口 —— 模型会一直读下去。
       压到 1300 字后降到 1.05 万，4 轮收口。
    2. **日志检索必须「诊断行优先」**：一份 1490 行的日志里含错误关键词的通常只有 8~20 行，
       其余是每笔交易都有的常规流程行（发起支付 / 状态更新 / 扫码解锁）。全量返回 = 把噪声
       灌给模型，挤掉真正有信息量的那几行。
    3. **按场景裁剪工具**：本次没有日志就**不要暴露**日志类工具，否则模型会去查不存在的日志，
       白白烧掉一整轮（它还可能在结论里编出"日志未见异常"）。
"""
import contextvars
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import codesearch, config, knowledge

# 本次请求的日志。用 contextvars 而非全局变量：/ask 是并发的，全局变量会串号。
_LOG: contextvars.ContextVar[dict | None] = contextvars.ContextVar("kb_agent_log", default=None)

# 「诊断行」关键词：命中其一才算证据行，其余只给行号摘要（见纪律 2）
DIAG_WORDS = (
    "RSP_NO", "RESULT", "DESCRIPTION", "errcode", "errmsg", "returnMsg",
    "失败", "异常", "错误", "拒绝", "超时", "撤销", "冲正",
    "ERROR", "Error", "error", "FAIL", "Fail", "fail", "EXCEPTION", "timeout",
)

_TIME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")


# ---------------- 日志上下文 ----------------
def build_context(log_text: str) -> dict | None:
    """把附加日志文本变成可查对象（行列表 + 元信息），并落盘一份供人工复核。"""
    text = (log_text or "").strip()
    if not text:
        return None
    lines = text.splitlines()
    stamps = []
    for ln in lines:
        m = _TIME_RE.match(ln.lstrip())
        if m:
            stamps.append(m.group(0))
    ctx = {
        "lines": lines,
        "chars": len(text),
        "first": stamps[0] if stamps else "",
        "last": stamps[-1] if stamps else "",
        "path": _persist(text),
    }
    return ctx


def _persist(text: str) -> str:
    """按内容 hash 落盘（同一份日志重复提交只存一份），顺带清理过期文件。

    ⚠️ 为什么落盘：模型的 log 工具是「按需查」，日志就得是能独立存在的对象；
    另外运维复核一条回答时，常需要回看当时那份原始日志。
    落盘失败不影响问答 —— 返回空串即可，工具仍走内存里的 lines。
    """
    try:
        d = Path(config.AGENT_LOG_DIR)
        d.mkdir(parents=True, exist_ok=True)
        p = d / (hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16] + ".log")
        if not p.exists():
            p.write_text(text, encoding="utf-8")
        # 顺手清理：低频动作（只在有新日志时才可能触发新增），不值得单开定时任务
        cutoff = time.time() - max(1, config.AGENT_LOG_KEEP_DAYS) * 86400
        for f in d.glob("*.log"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
        return str(p)
    except Exception:
        return ""


def set_context(ctx: dict | None):
    _LOG.set(ctx)


def get_context() -> dict | None:
    return _LOG.get()


def clear_context():
    _LOG.set(None)


def has_log() -> bool:
    ctx = _LOG.get()
    return bool(ctx and ctx.get("lines"))


# 本轮取证中命中过的知识库片段。闭环下不再预检索，sources 只能从工具调用里收集，
# 否则响应体的 sources 会空掉（前端「复制给人工」靠它做溯源）。
_COLLECT: contextvars.ContextVar[dict | None] = contextvars.ContextVar("kb_collect", default=None)

# 本轮检索宽度（2026-10-05 分档）：deep 档多喂几个语义段给模型。
# ⚠️ 同样必须用 contextvars —— /ask 是并发的，全局变量会串号（同 _LOG 的教训）。
# 未设置时回退到 config.TOP_K（= 老的单档行为）。
_TOP_K: contextvars.ContextVar[int | None] = contextvars.ContextVar("kb_agent_top_k", default=None)


def set_top_k(n: int | None):
    """设置本轮检索返回的片段数；传 None 回退到 config.TOP_K。"""
    try:
        _TOP_K.set(int(n) if n else None)
    except Exception:                             # noqa: BLE001
        _TOP_K.set(None)


def get_top_k() -> int:
    v = _TOP_K.get()
    return int(v) if v else config.TOP_K


def start_collect():
    # ⚠️ 2026-10-05 追加三个「本轮已给过」集合：同一请求里模型常拿不同关键词反复查，
    # 重复片段既占上下文又稀释注意力。见 `_seen()`。
    # ⚠️ 2026-10-09 追加 `new_kb` 计数器：给**收敛判据**用。
    #    背景：deep 档实测 41% 撞满 8 轮上限，而收敛判据几乎不触发 ——
    #    因为它看的是「本轮工具返回的总字数」，可 **search_kb 不管相不相关都能返回 1328 字**，
    #    于是"返回量"永远达标。真正该看的是「**这轮有没有捞到新条目**」，
    #    而这个信息本来就躺在 `_seen()` 里，只是没接出来。
    _COLLECT.set({"kb_hits": [], "new_kb": 0, "blocked": 0,
                  "seen_kb": set(), "seen_code": set(), "seen_doc": set(),
                  "seen_files": set(), "round_new_files": 0})


def take_round_new_kb() -> int:
    """取走并清零「本轮到目前新增的知识库命中数」。

    ⚠️ 每轮取证后由 llm.py 调用（判定本轮的边际新证据）。`search_kb` 命中重复条目时
    只回占位符、**不计入**这个数 —— 所以它天然就是"新证据量"。
    """
    box = _COLLECT.get()
    if box is None:
        return -1                      # 不在收集态 → 返回 -1，让调用方**不要**据此判收敛
    n = int(box.get("new_kb") or 0)
    box["new_kb"] = 0
    return n


def take_round_new_files() -> int:
    """取走并清零「本轮命中的**新文件**数」（2026-10-10 评审 ③'，默认关，待 A/B）。

    背景：chunk 粒度的 new_kb 无区分度（BM25 换个词就返回新 chunk、哪怕全无关），
    但**文件**粒度粗一档 —— 「本轮命中的文件一个都没出现过」= 在同一批文档里打转。
    ⚠️ 注意区分度有限（模型换功能域词就能换文件，预实验数据只覆盖旧轨迹），
    所以默认关闭（KB_NEWFILE_STALL=0），开 1 后由 STALL_ROUNDS 配合判定。
    """
    box = _COLLECT.get()
    if box is None:
        return 0
    seen_files = box.setdefault("seen_files", set())
    newf = box.get("round_new_files") or 0
    box["round_new_files"] = 0
    return int(newf)


def take_round_blocked() -> int:
    """取走并清零「本轮被『查无此词』硬拦截的次数」。
    ⚠️ 这是比 `new_kb` 可靠得多的收敛信号（2026-10-09 实测）：
    BM25 **换个词就能返回不同的片段**（哪怕全是无关内容），所以"有没有新命中"几乎永远 > 0；
    而"模型去搜了一个库里根本不存在的值"是**确定性**的负面信号 ——
    它意味着这一轮问的这个问题，问多少次都不会有答案。
    """
    box = _COLLECT.get()
    if box is None:
        return 0
    n = int(box.get("blocked") or 0)
    box["blocked"] = 0
    return n


def _bump_blocked() -> None:
    """记一次「查无此词」拦截（由 search_kb 在硬拦截分支调用）。"""
    box = _COLLECT.get()
    if box is not None:
        box["blocked"] = int(box.get("blocked") or 0) + 1


# 「目录外检索意图」词 —— 这些是**听起来该有、但知识库里并没有**的文档形态。
# ⚠️ 2026-10-10 加（治 deep 空转）：线上实测 61% 的 deep 问答撞满 8 轮，真实轨迹显示
#   模型在反复换词搜「错误码 汇总 索引」「错误码 速查 对照表」「错误码清单」
#   「后台菜单路径速查大全」这类**库里不存在的汇总型文档**（库里错误码类只有孤例
#   《错误码-YBLQSMP3079...》）。它每换一个词，BM25 都能返回点沾边片段，
#   于是收敛判据永远不触发 —— 这是「查无此词」硬拦截**覆盖不到**的另一类空转。
#   判据：本轮 search_kb 的 query **全部**含这类"汇总型"意图词、且零新条目 ⇒ 记 1 次停滞。
#   ⚠️ 只在**纯检索轮 + 零新条目**时才算（有别的正常词参与检索时不算，避免误伤
#   「结算失败 汇总」这种正常查询）。
_INDEX_INTENT_RE = re.compile(
    r"汇总表|汇总|速查表|速查|对照表|清单|大全|全表|索引|一览|总表|汇总索引|表$")


def _is_catalog_intent(query: str) -> bool:
    """判断一次 search_kb 的 query 是否属于「找汇总型文档」的意图。"""
    return bool(_INDEX_INTENT_RE.search(query or ""))


def _bump_catalog_intent() -> None:
    box = _COLLECT.get()
    if box is not None:
        box["catalog_intent"] = int(box.get("catalog_intent") or 0) + 1


def take_round_catalog_intent() -> int:
    """取走并清零「本轮『目录外检索意图』次数」（见 _INDEX_INTENT_RE 的注释）。"""
    box = _COLLECT.get()
    if box is None:
        return 0
    n = int(box.get("catalog_intent") or 0)
    box["catalog_intent"] = 0
    return n


def _seen(bucket: str, key) -> bool:
    """本轮内 (bucket, key) 是否**已经给过**模型。返回 True = 重复，调用方应只回占位符。

    ⚠️ 放在 contextvars 的 collect 盒子里（同 `_LOG` 的道理）：/ask 是并发的，
    用全局集合会串号 —— 甲用户的重合片段会害乙用户看不到内容。
    """
    box = _COLLECT.get()
    if box is None:
        return False
    s = box.setdefault(bucket, set())
    if key in s:
        return True
    s.add(key)
    if bucket == "seen_kb":            # 新条目 → 计入"本轮新证据"（见 take_round_new_kb）
        box["new_kb"] = int(box.get("new_kb") or 0) + 1
        # 2026-10-10 ③'：新 chunk 所属文件也记录（take_round_new_files 用，默认不启用）
        if isinstance(key, tuple) and key[0] not in box.get("seen_files", set()):
            box.setdefault("seen_files", set()).add(key[0])
            box["round_new_files"] = int(box.get("round_new_files") or 0) + 1
    return False


def stop_collect() -> dict:
    return _COLLECT.get() or {"kb_hits": []}


def _record(file: str, idx, score, text: str = ""):
    """记录一条知识库命中。

    ⚠️ 2026-10-12（评审逐行分析 P1-1）：**必须连内容一起存**。
    原因：闭环（answer_agentic）跑了几轮、查到了一堆片段，此时若发生异常 → 走 fallback
    退回单轮 `answer()`。原先 fallback 传的是 `empty_chunks`（空列表）⇒ **前面查到的证据全丢**，
    用户等了几十秒拿到一个"没有检索"的答案，比不用 agent 还差。
    现在把内容摘要也存进 collect box，fallback 时可以用它兜底（见 llm.answer_agentic 的 fallback）。
    `ctx_txt` 是给 fallback 复用的**上下文文本**（不需要太长，够模型判断方向即可）。
    """
    box = _COLLECT.get()
    if box is not None:
        rec = {"file": file, "idx": idx, "score": score}
        if text:
            rec["text"] = text[:900]        # 摘要即可：fallback 只求"别丢证据"，不求全文
        box["kb_hits"].append(rec)


def take_kb_hits() -> list:
    """取走本轮的命中（含内容摘要），供闭环异常回退时兜底。

    ⚠️ 与 `stop_collect()` 不同：这里**不清空** box（调用方通常在 finally 里取）。
    """
    box = _COLLECT.get()
    if box is None:
        return []
    return list(box.get("kb_hits") or [])


def _clip(s: str, n: int | None = None) -> str:
    n = n or config.TOOL_MAX_CHARS
    if len(s) <= n:
        return s
    return s[:n] + f"\n…（已截断，原始 {len(s)} 字；换更精确的关键词再查）"


# ---------------- 日志工具 ----------------
def log_overview() -> str:
    """把整份日志压成一屏：时间范围、失败回执归类、后台查询返回、订单状态、成功交易对照。

    这是排查的第一步，也是「按需查」相对「整段灌」最省的地方：34.6 万字 → 约 1600 字，
    且保住了所有有诊断价值的行。
    """
    ctx = _LOG.get()
    if not ctx:
        return "本次提问没有附日志。"
    lines = ctx["lines"]
    out = [f"日志覆盖：{ctx['first'] or '（无时间戳）'} ~ {ctx['last'] or '（无时间戳）'}，"
           f"共 {len(lines)} 行 / {ctx['chars']} 字"]

    # 1) 银行回执失败：RSP_NO≠00 的按 DESCRIPTION 归类，并给出涉及的流水号区间
    fails: dict[str, list] = {}
    for ln in lines:
        m = re.search(r'"RSP_NO":"(\d+)".*?"EXT_ORDER_NO":"(\d+)".*?"DESCRIPTION":"([^"]*)"', ln)
        if m and m.group(1) != "00":
            desc = m.group(3).strip() or f"RSP_NO={m.group(1)}"
            it = fails.setdefault(desc, [0, m.group(2), m.group(2), []])
            it[0] += 1
            it[2] = m.group(2)
            ts = _TIME_RE.search(ln)
            if ts and len(it[3]) < 12:
                it[3].append(ts.group(0)[11:])
    if fails:
        out.append("\n【银行回执失败（RSP_NO≠00）按原因归类】")
        for desc, (n, first, last, tss) in sorted(fails.items(), key=lambda kv: -kv[1][0]):
            out.append(f"  ×{n}  「{desc}」  流水 {first} → {last}"
                       + (f"  时间 {'、'.join(tss)}" if tss else ""))

    # 2) 后台支付查询返回（转义与非转义两种写法都兼容）
    q: dict[str, int] = {}
    for ln in lines:
        if "queryOrgPay" not in ln and "returnMsg" not in ln:
            continue
        m = re.search(r'returnMsg\\?":\\?"([^"\\]{1,80})', ln)
        if m:
            key = m.group(1)[:70]
            q[key] = q.get(key, 0) + 1
    if q:
        out.append("\n【后台支付查询返回】")
        for k, n in sorted(q.items(), key=lambda kv: -kv[1]):
            out.append(f"  ×{n}  {k}")

    # 3) 订单中间状态：能看出「反复重试后关单」还是「从未进入支付」
    st: dict[tuple[str, str], int] = {}
    for ln in lines:
        for m in re.finditer(r"(\d{16,22})\s+(PAYING|PAYERROR|CLOSE|COMPLETE|WAITPAY\w*)", ln):
            st[(m.group(1), m.group(2))] = st.get((m.group(1), m.group(2)), 0) + 1
    if st:
        out.append("\n【订单状态出现次数】")
        byoid: dict[str, list[str]] = {}
        for (oid, s), n in sorted(st.items()):
            byoid.setdefault(oid, []).append(f"{s}×{n}")
        for oid, segs in list(byoid.items())[:24]:
            out.append(f"  {oid}  {' '.join(segs)}")
        if len(byoid) > 24:
            out.append(f"  …另有 {len(byoid)-24} 个订单未列出…")

    # 4) 成功交易对照：判断「是设备整体故障还是时段性故障」最有力的证据
    ok = []
    for ln in lines:
        if "type=SALE" not in ln or "status=COMPLETE" not in ln:
            continue
        ts = _TIME_RE.search(ln)
        amt = re.search(r"totalAmt=(\d+)", ln)
        if ts:
            ok.append(f"{ts.group(0)[11:16]}¥{int(amt.group(1))/100:g}" if amt
                      else ts.group(0)[11:16])
    if ok:
        out.append(f"\n【销售成功交易 {len(ok)} 笔】" + "、".join(ok[:24])
                   + ("…" if len(ok) > 24 else ""))
    return _clip("\n".join(out), max(config.TOOL_MAX_CHARS, 2000))


def grep_log(kw: str) -> str:
    """在日志里按关键词/单号检索；命中多时只给「诊断行」全文，常规流程行只给行号。"""
    ctx = _LOG.get()
    if not ctx:
        return "本次提问没有附日志。"
    kw = (kw or "").strip().strip('"')
    if not kw:
        return "kw 不能为空。"
    lines = ctx["lines"]
    idx = [i for i, ln in enumerate(lines) if kw in ln]
    if not idx:
        return (f"日志中没有出现「{kw}」。可以换个关键词（例如订单号、错误码、"
                f"DESCRIPTION 里的文案片段）再查一次。")
    diag = [i for i in idx if any(w in lines[i] for w in DIAG_WORDS)]
    other = [i for i in idx if i not in set(diag)]
    out = [f"「{kw}」命中 {len(idx)} 行（日志共 {len(lines)} 行）；"
           f"其中 {len(diag)} 行含错误/回执关键词，优先列出："]
    for i in diag[:14]:
        out.append(f"L{i+1}: {lines[i][:260]}")
    if len(diag) > 14:
        out.append(f"…另有 {len(diag)-14} 行诊断命中未列出…")
    if not diag and other:
        out.append(f"（{len(other)} 行均为常规流程行，不含错误关键词，行号："
                   f"{'、'.join('L'+str(i+1) for i in other[:12])}"
                   + ("…" if len(other) > 12 else "") + "）")
    return _clip("\n".join(out))


# ---------------- 知识库 / 源码工具 ----------------
def search_kb(query: str) -> str:
    """检索 POS 运维知识库。库里没有时明确告知，避免模型拿沾边片段拼流程。"""
    query = (query or "").strip()
    if not query:
        return "query 不能为空。"
    # ⚠️ 2026-10-09 硬拦截：**整条 query 全由"库里查无此词"的专名构成**时，
    #   直接返回"库里没有"，**不给任何检索片段**。
    #   为什么必须硬拦：实测只提示不拦时，模型仍会继续换写法（`E-7731`→`E-77`→`E-7731 结算报错`
    #   →`E-7710 E-7730 错误码 前缀`…共 10 次 search_kb），因为**每次换词 BM25 都能返回片段**
    #   —— 它据此以为"还有东西可挖"。不给片段，这条路才真的断。
    #   ⚠️ 只在"全未知"时拦：`E-7731 结算报错` 这类混着正常词的不拦（那确实能捞到东西）。
    if not knowledge.has_known_content(query):
        unk = knowledge.unknown_tokens(query)
        _bump_blocked()
        shown = "、".join(f"「{t}」" for t in unk) or f"「{query}」"
        # 长得像「单号 / 店铺号 / 商场号」的值，八成是「该放参数里却拿来当检索词了」
        looks_like_id = any(re.fullmatch(r"[0-9]{6,}", t) or re.fullmatch(r"[A-Za-z]{2,}\d{2,}[A-Za-z0-9]*", t)
                            for t in unk)
        extra = ("⚠️ 看着像**单号 / 店铺号 / 商场号**：它不该当检索词 —— 后台路径、"
                 "处置流程用**功能名**（如「支付方式配置」「结算失败」）查；"
                 "要按单号取数据请用对应的**查询工具**，不是检索。\n"
                 if looks_like_id else
                 "⚠️ 看着像**错误码 / 外部透传值**：这类值渠道或银行才有，"
                 "库里不会收录，换个写法同样没有。\n")
        return (f"⛔ 知识库里没有 {shown} 的任何条目 —— 这是**全库检索的结论，不是本次没命中**。\n"
                "不要再换写法 / 拆词 / 加同义词重试了：换任何写法都同样没有，只会白烧轮次。\n"
                + extra +
                "请立刻改走这两条：\n"
                "1) 用**功能名 + 现象**（如「结算失败」「支付方式配置」）再查一次，看有没有"
                "讲该功能的条目；\n"
                "2) 直接基于已查到的证据作答，并在结论里如实说明"
                "「知识库中没有该错误码/该值的条目」，必要时建议联系对应渠道或服务台。")
    r = knowledge.search_ex(query, top_k=get_top_k())
    hits = r.get("hits") or []
    # ⚠️ 2026-10-10：记录「找汇总型文档」的检索意图（给收敛判据用，见 _INDEX_INTENT_RE）。
    if _is_catalog_intent(query):
        _bump_catalog_intent()
    # ⚠️ 2026-10-09：把「库里的关键词」当成硬信号摆出来。
    #   实测最贵的浪费是**反复换写法搜一个库里根本不存在的东西**（典型是错误码 `E-7731`：
    #   全库出现 0 次，模型却换了 8 种写法、21 次调用；每换个词 BM25 都返回"新片段"，
    #   于是收敛判据永远不触发）。这里直接明说"这个词库里没有"，掐掉"再换个写法试试"。
    unk = knowledge.unknown_tokens(query)
    tip = ""
    if unk:
        shown = "、".join(f"「{t}」" for t in unk)
        tip = (f"⚠️ {shown} 在**整个知识库里查无此词**（不是本次没命中，是库里根本没有）。\n"
               "→ 若这是**错误码 / 报错文案 / 外部系统透传的值**：别再换写法搜了"
               "（实测同一错误码会烧掉十几次调用），直接说明「库里没有该错误码的条目」。\n"
               "→ 若这是**单号 / 店铺号这类参数**：它本来就该出现在别的工具的**参数**里，"
               "不是检索词，换用对应工具查。\n\n")
    if not hits:
        return tip + "知识库无命中（这个条目库里没有）。"
    segs = []
    for h in hits:
        _record(h["file"], h.get("idx"), h.get("score"), h.get("text") or "")
        head = f"[{h['file']} #{h.get('idx')}]"
        if _seen("seen_kb", (h["file"], h.get("idx"))):
            segs.append(f"{head}\n（该片段本轮已给过，内容同上，不再重复）")
        else:
            segs.append(f"{head}\n{h['text'][:700]}")
    txt = "\n---\n".join(segs)
    if r.get("weak"):
        rare = r.get("rare") or {}
        hint = "、".join(f"{k}（是否想写「{v}」？）" if v else k for k, v in rare.items())
        txt = ("⚠️ 弱命中：提问里的关键对象词在库中查无此人"
               + (f"，存疑词：{hint}" if hint else "")
               + "。下面只是词面沾边的高分片段，**若确认答不了这个问题，就直说库里没有该条目**，"
                 "禁止用它们拼一套操作步骤。\n\n" + txt)
    return _clip(tip + txt)


def search_code(kw: str, path: str = "") -> str:
    """在 POS 源码索引里检索，用于核对编码取值 / 参数名 / 判断逻辑。

    ⚠️ 2026-10-05 升级（修一个「概率性零命中」的硬 bug）：
      · **检索词大小写不敏感**，且可用 `pos.pay.digital.force` 这种带点的全限定名 ——
        内部会规范化（整串小写 → 去分隔符 → 需要时再按分隔符/驼峰拆词）后再查。
        索引里标识符是按**小写**存的，所以以前传 `PayMethodsKeyBoard` 是 **0 命中**、
        传 `paymethodskeyboard` 是 38 命中 —— 模型只能一次次换词试（实测一轮 22 次调用）。
      · **命中行合并成连续块**并带上下文（旧行为是零散单行，if/else 分支被拆开，只能猜）。
      · `path` 可把范围缩到某仓库/模块（如 `a8_new_ui`、`login-service`）。
    """
    kw = (kw or "").strip()
    if not kw:
        return "kw 不能为空。"
    # ⚠️ 2026-11-11：**索引没加载 ≠ 源码里没有**。
    #   原逻辑在索引为空时直接走"未检索到"分支，还告诉模型「该词不在源码里，
    #   别再用同义词重试」—— 这是**错误结论**，会让它放弃一个本来能查到的东西
    #   （实测：索引未加载时，连文件名都报"源码中没有"）。
    if not codesearch.count_lines():
        return ("代码索引**尚未就绪**（不是「源码里没有」）。"
                "请改用知识库检索；或稍后重试本工具。")
    hits = codesearch.search([kw], path=(path or "").strip() or None)
    if not hits:
        tail = f"（已限定路径含「{path}」）" if (path or "").strip() else ""
        # ⚠️ 2026-10-09：零命中时**必须明确叫停**，不能只说"可以换个更短的词"——
        #   实测模型会一直换写法重试（同一个错误码换了 7 种写法、共 43 次调用）。
        #   这里直说"该词不在源码里 + 下一步该干嘛"，把"继续撞墙"这条路堵上。
        return (f"源码中未检索到「{kw}」{tail}。\n"
                "**说明该词不在 POS 源码里，不要再用同义词 / 加词 / 改大小写重试**"
                "（同类反复重试实测要烧掉十几次调用）。请改走这两条：\n"
                "1) 用 search_kb 查有没有关于它（或它所属功能）的知识条目；\n"
                "2) 直接基于已查到的证据作答，并在结论里如实说明「源码中未见该标识」。\n"
                "⚠️ 若你搜的是**错误码 / 报错文案**（如 E-7731）—— 那是渠道或银行透传的，"
                "源码里本来就没有，别再搜了。")
    blocks = codesearch.merge_blocks(hits, gap=3, ctx=1)
    segs, budget = [], 0
    for b in blocks:
        rng = f"{b['line']}" + (f"-{b['end']}" if b["end"] != b["line"] else "")
        if _seen("seen_code", (b["file"], b["line"], b["end"])):
            seg = f"[{b['file']}:{rng}]\n（该代码块本轮已给过，内容同上，不再重复）"
        else:
            seg = f"[{b['file']}:{rng}]\n{b['text']}"
        if budget and budget + len(seg) > config.TOOL_MAX_CHARS:
            break                                  # 宁可少给一块，也不把块切一半
        segs.append(seg)
        budget += len(seg)
    return "\n---\n".join(segs)


def read_doc(name: str) -> str:
    """按文件名片段读取整篇知识库文档。"""
    name = (name or "").strip()
    if not name:
        return "name 不能为空。"
    try:
        files = [p for p in Path(config.KB_DIR).rglob("*.md") if name in p.name]
    except Exception as e:
        return f"读取失败：{e}"
    if not files:
        return (f"没有找到文件名含「{name}」的知识库文档。\n"
                "⚠️ **不要换同义词重试** —— 文件名是确定的。改走这两条：\n"
                "1) 用 search_kb 按**功能名 + 现象**检索内容（文件名里未必有你要的词）；\n"
                "2) 直接基于已查到的证据作答，并如实说明「知识库里没有这个条目」。")
    # ⚠️ 2026-10-12（评审逐行分析 P2-3）：**多篇匹配时必须回列表，不能静默只取第一篇**。
    #   旧行为 `p = files[0]`：模型传 name="银行接口规范" 会匹配 **8 篇**却只读到第 1 篇，
    #   而且**完全不知道自己读错了**（没有任何提示）—— 它会拿 A 银行的手册去答 B 银行的问题。
    #   这个隐患在 10-12 目录注入后**被放大**：read_doc 从低频变高频（实测单题 6 次）。
    #   现在多匹配就回列表 + 让模型自己挑更精确的名字（列表带"按银行/主题挑"的提示）。
    if len(files) > 1:
        # 按文件名排序，保证输出稳定（可复现、不随文件系统顺序变）
        names = sorted(p.name for p in files)
        shown = "\n".join(f"  · {n}" for n in names[:12])
        more = f"\n  …（另有 {len(names) - 12} 篇）" if len(names) > 12 else ""
        return (f"「{name}」匹配到 **{len(names)} 篇**文档，请用**更精确的文件名**重读其中一篇：\n"
                f"{shown}{more}\n"
                "⚠️ 不要把这一篇的内容当成全部 —— 挑一篇最贴合本次问题的读。")
    p = files[0]
    if _seen("seen_doc", p.name):
        # 全文已在本轮上文给过 —— 再回一遍纯属重复占位（多轮取证里很常见）
        return (f"《{p.name}》的全文本轮已在上文给过（不再重复）。"
                "请直接引用上文内容；若只是要定位某一节，改用 search_kb 取片段更省。")
    body = p.read_text(encoding="utf-8", errors="ignore")
    return _clip(f"《{p.name}》全文：\n{body}", config.TOOL_MAX_CHARS * 2)


# ---------------- 简图工具（draw_diagram）----------------
# 为什么要让模型「画图」（2026-09-16 主人定调）：一线多半在忙，一张图比一段字好读得多 ——
# 时间顺序、链路定位、判定分支这三类信息，画出来 5 秒看懂，写成文字要读 30 秒。
#
# 分工是这套东西的关键：**模型只决定画什么（图型 / 节点 / 哪个环节不正常），
# SVG 由本地模板渲染**。两个理由：
#   · 不让模型直接产出 SVG / HTML —— 语法错一处就是整块空白，比没有图更糟；
#   · 也不把 SVG 源码回给模型 —— 那是几百行噪声，白占它的上下文（工具只回一句短确认，
#     这句确认还顺带告诉它「图里画过的别在正文再讲一遍」，帮它把答案写短）。


# ───────── 按手机号查订单列表（2026-10-29 新增）─────────
# 隐私口径（主人 2026-10-29 定）：**手机号 + 会员姓名脱敏**，会员号 / 会员ID 保留原值。
#   · 手机号 → 138****8057（前 3 + **** + 后 4）
#   · 姓名   → 日志里若已打过码（「江*玮」）原样用；没打码则「江**」
# 为什么不在这里就返回明文：手机上/群里截图传阅时，明文手机号+姓名等于把会员档案外泄。








# ---------------- 订单查询与店铺配置（2026-09-25 加，主人拍板的「选项 A」）----------------
# 背景：支付列表为空的排障实测暴露「事实缺口」—— 模型会 search_code 反推了，但没有任何
# 渠道知道「该店的某个配置项当前是什么值 / 这笔单当时长什么样」，只能凭先验猜。
# 这两个工具补的就是事实层：单号 → 订单全量（CLS）、单号 → 该店终端配置（api/config）。
# 两者都**只读**；查配置走 Mac 内网代理（token 在代理内存里，不经过本层）。
#
# ⚠️⚠️ 目录变量名有两套历史写法（2026-10-09 踩过）：
#     · 本模块用 **KB_WORKTOOL_DIR**
#     · member.py 与工具箱 JS 用 **WORKTOOL_DIR**
#   线上最初只设了 `WORKTOOL_DIR`，这里读 `KB_WORKTOOL_DIR` 读不到 → 落到 Mac 默认路径
#   → subprocess 的 `cwd=` 是个不存在的目录 → **FileNotFoundError**（未捕获时会把整个
#   /ask 打成 500）。⇒ 现在**两种名字都认**，且 cwd 不存在时不传、并捕获 OSError 兜底。

_WORKTOOL_DIR = (os.environ.get("KB_WORKTOOL_DIR")
                 or os.environ.get("WORKTOOL_DIR")
                 or "/Users/mac/工作/示例企业/code/mySpace/toolkit")
_WORKTOOL_CLI = os.environ.get(
    "KB_WORKTOOL_CLI",
    os.path.expanduser("~/.workbuddy/skills/pos-worktool/scripts/worktool.js"))

# 配置项里这些词的**值**不回给模型（键名可以看，值只确认「已配置」）
_SENSITIVE_KEY = re.compile(r"secret|token|password|passwd|private", re.I)

# 2026-10-10 新增：EXT 反查 POS 的 CLI 路径。
# ⚠️ 放在技能目录（pos-worktool/scripts/）—— 部署脚本 ③-4 只自动同步技能 scripts，
#    放项目 scripts/ 不会跟走（踩过：第一次就放错位置）。本机默认路径兜底 KB_EXT_REVERSE_CLI。
_EXT_REVERSE_CLI = os.environ.get(
    "KB_EXT_REVERSE_CLI",
    os.path.expanduser("~/.workbuddy/skills/pos-worktool/scripts/dtp-reverse.js"))

# Mac 中转地址（同 KB_MAC_PROXY 模式）：小主机到不了 els（零信任 fakeip），
# 由 Mac 上的 mac-proxy /dtp-reverse 端点代查。没配就退回本地 CLI（Mac 上开发用）。
_MAC_PROXY = (os.getenv("KB_MAC_PROXY") or "").strip().rstrip("/")
_MAC_PROXY_KEY = (os.getenv("KB_MAC_PROXY_KEY") or "").strip()
_EXT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))














_DIAG_STATE = {                      # state -> (主色, 底色)；红=异常、橙=可疑、绿=正常
    "ok":   ("#16a34a", "#f0fdf4"),
    "warn": ("#d97706", "#fffbeb"),
    "bad":  ("#ef4444", "#fef2f2"),
    "info": ("#94a3b8", "#f8fafc"),
}
_DIAG_KIND_CN = {"timeline": "时间线", "chain": "链路", "branch": "判定分支", "steps": "操作步骤"}
_DIAG_KINDS = tuple(_DIAG_KIND_CN)
_FONT = "-apple-system,BlinkMacSystemFont,PingFang SC,Microsoft YaHei,sans-serif"


def _xesc(s) -> str:
    """XML 转义 —— label / sub 是模型给的自由文本，不转义会把 SVG 撑坏。"""
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))


def _cut(s, n: int) -> str:
    """截断并按 SVG 的空白规则压成单行（换行在 SVG text 里只会变成空格）。"""
    s = " ".join(str(s if s is not None else "").split())
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def _w(s, fs: float = 13) -> float:
    """粗略估算文本像素宽：全角算 1 个字宽、半角 0.55 —— 只为排版不撞车，不必精确。"""
    n = sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in str(s or ""))
    return n * fs


def _norm_nodes(nodes, sub_max: int = 46, label_max: int = 14) -> list:
    """把模型给的 nodes 归一成 [{label, sub, state}]，顺手挡住畸形输入。

    模型偶尔会写成纯字符串数组，或用 name / desc 等别名，这里一并兼容 ——
    宁可宽容地收下，也不要让它因为参数格式白烧一轮。

    sub_max / label_max：截断上限。steps 图型装的是完整路径，需要放宽（见 render_diagram）。
    """
    out = []
    for it in (nodes or [])[: 8]:
        if isinstance(it, str):
            out.append({"label": _cut(it, label_max), "sub": "", "state": "info"})
        elif isinstance(it, dict):
            st = str(it.get("state") or "info").strip().lower()
            out.append({
                "label": _cut(it.get("label") or it.get("name") or it.get("title") or "", label_max),
                "sub": _cut(it.get("sub") or it.get("desc") or it.get("note") or "", sub_max),
                "state": st if st in _DIAG_STATE else "info",
            })
    return [n for n in out if n["label"] or n["sub"]]


def _svg_head(h: int, title: str) -> list:
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 {h}" width="100%" '
         f'font-family="{_FONT}">',
         f'<rect x="0.5" y="0.5" width="679" height="{h - 1}" rx="12" fill="#ffffff" stroke="#e2e8f0"/>']
    if title:
        o.append(f'<text x="20" y="29" font-size="14" font-weight="600" fill="#0f172a">'
                 f'{_xesc(_cut(title, 34))}</text>')
    return o


def _svg_timeline(title: str, nodes: list) -> str:
    """时间线：每行 = 彩点 + 左侧标识（时间 / 单号 / 步骤）+ 右侧说明。"""
    rh, top = 40, (46 if title else 20)
    h = top + len(nodes) * rh + 14
    o = _svg_head(h, title)
    dx = 26
    if len(nodes) > 1:
        o.append(f'<line x1="{dx}" y1="{top + rh / 2}" x2="{dx}" '
                 f'y2="{top + (len(nodes) - 1) * rh + rh / 2}" stroke="#e2e8f0" stroke-width="2"/>')
    for i, nd in enumerate(nodes):
        cy = top + i * rh + rh / 2
        fg, _ = _DIAG_STATE[nd["state"]]
        o.append(f'<circle cx="{dx}" cy="{cy}" r="5.5" fill="{fg}"/>')
        x = 44.0
        if nd["label"]:
            o.append(f'<text x="{x:.0f}" y="{cy + 4.5:.0f}" font-size="13" font-weight="600" '
                     f'fill="#0f172a">{_xesc(nd["label"])}</text>')
            x += _w(nd["label"], 13) + 14
        if nd["sub"]:
            x = max(x, 158.0)
            o.append(f'<text x="{x:.0f}" y="{cy + 4.5:.0f}" font-size="12.5" fill="#64748b">'
                     f'{_xesc(_cut(nd["sub"], max(8, int((662 - x) / 12.5))))}</text>')
    o.append("</svg>")
    return "".join(o)


def _svg_chain(title: str, nodes: list) -> str:
    """链路：横向方框 + 箭头，框色标出哪一环不正常（如 收银台→支付中心→工行→微信）。"""
    n = len(nodes)
    pad, gap, bh = 20.0, 30.0, 52.0
    bw = (680 - pad * 2 - gap * (n - 1)) / n
    top = 46 if title else 22
    o = _svg_head(int(top + bh + 14), title)
    for i, nd in enumerate(nodes):
        x = pad + i * (bw + gap)
        fg, bg = _DIAG_STATE[nd["state"]]
        o.append(f'<rect x="{x:.1f}" y="{top}" width="{bw:.1f}" height="{bh}" rx="8" '
                 f'fill="{bg}" stroke="{fg}" stroke-width="1.5"/>')
        lmax = max(4, int((bw - 14) / 12.5))
        lw = _w(nd["label"], 12.5)
        lx = x + max(6.0, (bw - lw) / 2)
        ly = top + (23 if nd["sub"] else 31)
        o.append(f'<text x="{lx:.1f}" y="{ly}" font-size="12.5" font-weight="600" '
                 f'fill="#0f172a">{_xesc(_cut(nd["label"], lmax))}</text>')
        if nd["sub"]:
            smax = max(4, int((bw - 6) / 11))
            sw = _w(_cut(nd["sub"], smax), 11)
            o.append(f'<text x="{x + max(3.0, (bw - sw) / 2):.1f}" y="{top + 41}" font-size="11" '
                     f'fill="#64748b">{_xesc(_cut(nd["sub"], smax))}</text>')
        if i < n - 1:
            ay = top + bh / 2
            x0, x1 = x + bw + 5, x + bw + gap - 11
            o.append(f'<line x1="{x0:.1f}" y1="{ay}" x2="{x1 - 6:.1f}" y2="{ay}" '
                     f'stroke="#cbd5e1" stroke-width="1.5"/>')
            o.append(f'<path d="M{x1 - 6:.1f} {ay - 4} L{x1:.1f} {ay} L{x1 - 6:.1f} {ay + 4} Z" '
                     f'fill="#cbd5e1"/>')
    o.append("</svg>")
    return "".join(o)


def _svg_branch(title: str, nodes: list) -> str:
    """判定分支：顶上一个条件框，下面列出各分支的结论（节点 1 是条件，其余是分支）。"""
    cond, brs = nodes[0], nodes[1:]
    rh, top = 40, (46 if title else 20)
    h = top + 54 + 20 + len(brs) * rh + 12
    o = _svg_head(h, title)
    o.append(f'<rect x="20" y="{top}" width="640" height="54" rx="9" fill="#f1f5f9" stroke="#cbd5e1"/>')
    o.append(f'<text x="34" y="{top + 23}" font-size="13" font-weight="600" fill="#0f172a">'
             f'{_xesc(cond["label"])}</text>')
    if cond["sub"]:
        o.append(f'<text x="34" y="{top + 41}" font-size="11.5" fill="#64748b">'
                 f'{_xesc(_cut(cond["sub"], 48))}</text>')
    stem = top + 54
    last = stem + 20 + (len(brs) - 1) * rh if brs else stem
    if brs:
        o.append(f'<line x1="44" y1="{stem}" x2="44" y2="{last}" stroke="#cbd5e1" stroke-width="1.5"/>')
    for i, nd in enumerate(brs):
        cy = stem + 20 + i * rh
        fg, _ = _DIAG_STATE[nd["state"]]
        o.append(f'<line x1="44" y1="{cy}" x2="58" y2="{cy}" stroke="#cbd5e1" stroke-width="1.5"/>')
        o.append(f'<circle cx="62" cy="{cy}" r="4" fill="{fg}"/>')
        x = 76.0
        if nd["label"]:
            o.append(f'<text x="{x:.0f}" y="{cy + 4:.0f}" font-size="12.5" font-weight="600" '
                     f'fill="#0f172a">{_xesc(nd["label"])}</text>')
            x += _w(nd["label"], 12.5) + 12
        if nd["sub"]:
            o.append(f'<text x="{x:.0f}" y="{cy + 4:.0f}" font-size="12.5" fill="#475569">'
                     f'{_xesc(_cut(nd["sub"], max(8, int((646 - x) / 12.5))))}</text>')
    o.append("</svg>")
    return "".join(o)


def _svg_steps(title: str, nodes: list) -> str:
    """操作步骤：竖向编号清单，每步一行，**路径/参数名完整不截断**（必要时折行）。

    ⚠️ 为什么单独做一种（2026-10-22 加）：
      配置类答案（「祥瑞卡配置五步」「改店铺参数的路径」）原本被塞进 chain 图型 ——
      5 个方框横排在 680px 里，每格只有 130px，路径被砍成「商场详情→支付…」、
      参数名砍成「pos.s…」。实测线上该类图 **11 段文字里 8 段被截断**，
      而路径和参数名恰恰是配置场景最不能丢的信息 ⇒ 图等于白做。
      steps 走竖向布局：编号 + 主标题一行，说明文字另起一行可容纳 ~44 字，必要时再折行。
    """
    rh_head, rh_sub = 25.0, 20.0          # 主行 / 说明行 的高度
    pad, top = 20.0, (46 if title else 20)
    steps = nodes[:8]

    def sub_lines(nd) -> list:
        """把说明文字折行。⚠️ 不要硬切 —— 英文词/参数名被切开（如 fa|lse）会让人看不懂。"""
        s = nd["sub"]
        if not s:
            return []
        per = 44
        if len(s) <= per:
            return [s]
        out, rest = [], s
        while rest and len(out) < 2:
            if len(rest) <= per:
                out.append(rest)
                break
            head = rest[:per]
            # 从后往前找可断点：空格 / 标点 / 箭头
            cut = -1
            for k in range(len(head) - 1, int(per * 0.55), -1):
                if head[k] in " ，,、。；;）)】》→/|":
                    cut = k + 1
                    break
            if cut <= 0:
                cut = per
            out.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip()
        if rest and len(out) < 2:
            out.append(rest)
        return out[:2]

    h = top
    for nd in steps:
        h += rh_head + len(sub_lines(nd)) * rh_sub + 8
    h += 8
    o = _svg_head(int(h), title)

    y = top
    for i, nd in enumerate(steps):
        fg, bg = _DIAG_STATE[nd["state"]]
        blk = rh_head + len(sub_lines(nd)) * rh_sub
        # 左侧编号圆（序号由本函数统一给，故把模型 label 里可能自带的序号前缀剥掉）
        label = re.sub(r"^\s*\d+[.、)\s]+", "", nd["label"] or "").strip() or nd["label"]
        cy = y + rh_head / 2
        o.append(f'<circle cx="{pad + 11:.0f}" cy="{cy:.0f}" r="11" fill="{bg}" '
                 f'stroke="{fg}" stroke-width="1.5"/>')
        o.append(f'<text x="{pad + 11:.0f}" y="{cy + 4.5:.0f}" font-size="12.5" font-weight="600" '
                 f'text-anchor="middle" fill="{fg}">{i + 1}</text>')
        tx = pad + 32
        o.append(f'<text x="{tx:.0f}" y="{y + 17:.0f}" font-size="13" font-weight="600" '
                 f'fill="#0f172a">{_xesc(_cut(label, 34))}</text>')
        for j, ln in enumerate(sub_lines(nd)):
            o.append(f'<text x="{tx:.0f}" y="{y + rh_head + 12 + j * rh_sub:.0f}" '
                     f'font-size="12" fill="#64748b">{_xesc(ln)}</text>')
        y += blk + 8
        if i < len(steps) - 1:
            o.append(f'<line x1="{pad + 11:.0f}" y1="{y - 8:.0f}" x2="{pad + 11:.0f}" '
                     f'y2="{y:.0f}" stroke="#e2e8f0" stroke-width="1.5"/>')
    o.append("</svg>")
    return "".join(o)


def _persist_diagram(svg: str) -> str:
    """简图落盘，返回文件名（前端按 /diagram/<name> 取）。失败返回空串，由调用方退回内联。"""
    try:
        d = Path(config.DIAGRAM_DIR)
        d.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha1(svg.encode("utf-8")).hexdigest()[:16] + ".svg"
        p = d / name
        if not p.exists():
            p.write_text(svg, encoding="utf-8")
        # 顺手清理过期图（低频动作，不另开定时任务）
        cutoff = time.time() - max(1, config.AGENT_LOG_KEEP_DAYS) * 86400
        for f in d.glob("*.svg"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
        return name
    except Exception:
        return ""


_DIAG_BLOCK_RE = re.compile(r"<diagram\b([^>]*)>(.*?)(?:</diagram>|\Z)", re.S | re.I)
_DIAG_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def render_diagram(kind: str, title: str, nodes: list) -> dict | None:
    """渲染简图并落盘，返回 {kind, title, url, svg}；节点不够或渲染失败返回 None。"""
    if kind not in _DIAG_KINDS:
        # 模型没给或不认识 kind：按节点数猜一个（少节点多半是判定，多节点多半是顺序）
        kind = "branch" if len(nodes) <= 3 else "timeline"
    # ⚠️ steps 是「操作步骤」，sub 里装的是完整后台路径 / 参数名 —— 截到 46 字会把
    #    「数据管理 → 商场列表 → 商场详情 → 支付键盘」这类长路径砍掉，正是最不能丢的部分。
    #    其余图型仍是横向紧凑布局，必须短。故这里分区对待。
    nds = _norm_nodes(nodes, sub_max=(88 if kind == "steps" else 46),
                      label_max=(34 if kind == "steps" else 14))
    if len(nds) < 2:
        return None
    nds = nds[:6] if kind == "chain" else nds[:8]
    svg = {"timeline": _svg_timeline, "chain": _svg_chain,
           "branch": _svg_branch, "steps": _svg_steps}[kind](title, nds)
    name = _persist_diagram(svg)
    return {"kind": kind, "title": _cut(title, 40),
            "url": f"/diagram/{name}" if name else "",
            "svg": "" if name else svg}


def extract_diagram(text: str) -> tuple[str, dict | None]:
    """从答案里抽出 <diagram> 图数据块：返回（去掉该块后的正文, 图信息或 None）。

    ⚠️ 为什么不用工具调用（2026-09-16 实测后改的）：给模型一个 draw_diagram 工具时，
    它会把有限的取证轮次全花在查日志上，**两轮实测都没轮到画图** —— 工具路径在 5 轮预算里
    等于永远不触发。改成让它把图数据直接写在答案最前面（写答案时它已把信息组织好了），
    后端解析后渲染：不占轮次、零额外调用，格式写错也只是没有图，无害降级。

    行格式（宽容解析）：`短标识 | 说明 | state`，后两段可省；state 取 ok/warn/bad/info。
    """
    m = _DIAG_BLOCK_RE.search(text or "")
    if not m:
        return text, None
    attrs = dict(_DIAG_ATTR_RE.findall(m.group(1) or ""))
    nodes = []
    for ln in (m.group(2) or "").splitlines():
        parts = [p.strip() for p in ln.strip().lstrip("-*• ").split("|")]
        if not parts or not parts[0]:
            continue
        nodes.append({"label": parts[0],
                      "sub": parts[1] if len(parts) > 1 else "",
                      "state": parts[2] if len(parts) > 2 else "info"})
    info = render_diagram(attrs.get("kind", "").strip().lower(), attrs.get("title", ""), nodes)
    cleaned = (text[: m.start()] + text[m.end():]).strip()
    print(f"[diagram] {len(nodes)} 节点 -> {info and (info['url'] or '内联') or '丢弃（不足 2 个）'}",
          flush=True)
    return cleaned, info


# ⭐ 2026-10-23 新增：生成「修单报文」（**只生成、不提交**）
#   背景：主人要求让助手能"帮到修单那一步"，但**不推翻"写操作人工执行"的边界**。
#   做法：复用技能脚本 `pos-repair-gen/scripts/pos-repair.js` —— 它把工具箱
#        「订单修复-自动」页面逻辑放进沙箱驱动，并在加载前用白名单把
#        uploadOrderToImpos / deleteImposOrder 等写方法**堵死**，结构上无法提交。
#   ⇒ 助手产出报文，运维复制到工具箱页面自行核对后上送。
_REPAIR_CLI = os.environ.get("KB_REPAIR_CLI", "/opt/ask-ops/scripts/pos-repair.js")




# ---------------- 工具注册与分发 ----------------
_SPECS = {
    "log_overview": (
        {"type": "function", "function": {
            "name": "log_overview",
            "description": ("本次报障日志的整体概览：时间范围、银行失败回执按原因归类计数、"
                            "后台支付查询返回、各订单状态变化次数、当日成功交易对照。"
                            "用户附了日志时，这是排查的第一步，务必先调它建立全局认识。"),
            "parameters": {"type": "object", "properties": {}}}},
        lambda **kw: log_overview(), True),
    "grep_log": (
        {"type": "function", "function": {
            "name": "grep_log",
            "description": ("在本次报障日志里按关键词或订单号精确检索，返回命中的原始日志行。"
                            "用于核实具体某一笔、某个错误码、某个时间点前后的细节。"
                            "关键词要具体（订单号 / 错误码 / 回执文案片段），查不到就换一个词。"),
            "parameters": {"type": "object", "properties": {
                "kw": {"type": "string", "description": "关键词或订单号"}},
                "required": ["kw"]}}},
        lambda kw="", **kw2: grep_log(kw), True),
    "search_kb": (
        {"type": "function", "function": {
            "name": "search_kb",
            "description": ("检索 POS 运维知识库，返回最相关的知识片段。"
                            "用于查后台配置路径、处置流程、参数取值、已知问题。"
                            "注意：知识库讲的是「一般怎么做」，不能用来解释「本次具体发生了什么」。"),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "检索词，用功能名+现象，不要写整句话"}},
                "required": ["query"]}}},
        lambda query="", **kw: search_kb(query), False),
    "search_code": (
        {"type": "function", "function": {
            "name": "search_code",
            "description": ("在 POS 源码索引里检索，返回 文件:行号-结束行 + 连续代码块（带上下文）。"
                            "用于核对某个编码取值、参数名、默认值、判断逻辑到底怎么写的。"
                            "⚠️ 检索词**大小写不敏感**，可直接传 `PayMethodsKeyBoard` 这类标识符，"
                            "也可传 `pos.pay.digital.force` 这种带点的全限定名（会拆词再查）；"
                            "返回零命中时，优先换**更短的核心词**，别一次次改大小写或加词。"
                            # ⚠️ 2026-10-09 加：实测 226 次 search_code 里 **82 次零命中（36%）**，
                            #   其中 52% 是在反复搜同一个错误码（'7731' 12 次 / 'E-7731' 8 次 /
                            #   'E-7' 6 次 / 'E-77' 5 次 / 'E7731' 4 次 / '"E-"' 3 次 / 'E-' 3 次…）。
                            #   根因：工具描述只说了"换更短的词"，**没说清"这类东西源码里压根没有"**，
                            #   于是模型一直换写法撞墙。明确划出禁区比"换个词试试"有效得多。
                            "⚠️⚠️ **不要用它搜「错误码 / 报错文案 / 界面提示语」**"
                            "（如 E-7731、30030、弹窗原文）—— 那类码与文案是**收单渠道 / 银行 / 第三方透传**的，"
                            "POS 源码里不会有，搜了必然零命中。这类线索要用 search_kb 查有没有对应条目，"
                            "或让用户提供弹窗原文再判断。"),
            "parameters": {"type": "object", "properties": {
                "kw": {"type": "string", "description": "参数名 / 方法名 / 编码等精确词"},
                "path": {"type": "string",
                         "description": "可选：只在该路径子串内检索，如 a8_new_ui / login-service"}},
                "required": ["kw"]}}},
        lambda kw="", path="", **kw2: search_code(kw, path), False),
    "read_doc": (
        {"type": "function", "function": {
            "name": "read_doc",
            "description": ("按文件名片段读取整篇知识库文档全文。"
                            "已知文档名（或 search_kb 结果里出现过文件名）时才用，"
                            "需要完整步骤清单时比 search_kb 的片段更全。"),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "文件名片段，如「支付方式配置」"}},
                "required": ["name"]}}},
        lambda name="", **kw: read_doc(name), False),
    # ── 内网数据查询（2026-09-25 主人要求：「page 放到 1000，助手直接回答某店某天销售」）──
    # ── 订单查询 / 店铺配置（2026-09-25 加：补「现象类排障」的事实层）──
    # 注：曾经有过一个 draw_diagram 工具，2026-09-16 实测后撤掉 —— 模型会把有限的取证轮次
    # 全花在查日志上，两轮都没轮到画图。现在改成「答案里内嵌 <diagram> 块 + 后端解析」
    # （见 extract_diagram），不占轮次、零额外调用。
    # ⭐ 2026-10-23 新增：生成修单报文（只生成、不提交）
}


# 日志类工具只在本次确实有日志时暴露（纪律 3）
LOG_TOOLS = {"log_overview", "grep_log"}


def specs(has_attached_log: bool) -> list:
    """按场景返回可暴露的工具定义列表。"""
    return [spec for name, (spec, _, need_log) in _SPECS.items()
            if (not need_log) or has_attached_log]


# 一句话用途摘要。⚠️ 2026-09-25 加：llm.py 的 _AGENT_SYSTEM 里原本**硬编码**了工具清单
# （"你手上有查询工具（知识库检索、代码检索、读文档…）"），新增工具忘了同步那一句，
# 结果模型拿那句过期的"自我介绍"当权威清单 —— 实测 deepseek-v4-flash 因此断言
# "没有查销售额的工具"，明明 query_store_sales 就在 tools 列表里也没调。
# → 改成动态生成后，以后新增工具会自动出现在 system prompt 里。
# ⚠️ 新增工具请务必在这里补一条（漏了不影响调用，但会影响模型"知道有这个能力"）。
_HINTS: dict = {
    "search_kb": "检索运维知识库（文档/FAQ/流程）",
    "search_code": "检索源码（定位实现与调用关系）",
    "read_doc": "读取指定知识库文档全文",
    "log_overview": "查看用户附带的报障日志概览",
    "log_grep": "在用户附带的日志里按关键词检索",
}


def summaries(has_attached_log: bool = False) -> str:
    """给 system prompt 用的一句话工具清单（随 _SPECS 动态变化）。"""
    parts = []
    for spec in specs(has_attached_log):
        name = spec["function"]["name"]
        parts.append(f"{name}（{_HINTS.get(name, '查询工具')}）")
    return "、".join(parts)


def run(name: str, args: dict | None) -> str:
    """执行工具。任何异常都转成文本回给模型（让它自己换招），不向上抛。"""
    # 统一调用日志（2026-09-25 加）：排查「模型到底调了什么工具」必须看得到，
    # 参数截断且不打印完整手机号/单号以外的敏感值。
    _safe_args = {k: (str(v)[:24] + "…" if isinstance(v, str) and len(str(v)) > 24 else v)
                  for k, v in (args or {}).items()}
    print(f"[tools] 调用 {name} args={_safe_args}", flush=True)
    item = _SPECS.get(name)
    if not item:
        return f"没有名为 {name} 的工具。"
    _, impl, need_log = item
    if need_log and not has_log():
        return "本次提问没有附日志，该工具不可用。"
    try:
        return impl(**(args or {}))
    except TypeError:
        try:                                  # 模型偶尔会把参数包一层
            return impl(**(args or {}).get("kwargs") or {})
        except Exception as e:
            return f"工具参数有误（{e}），请按参数说明重试。"
    except Exception as e:
        # ⚠️ 不要把 traceback 抛给模型：它看到 KeyError 这类内部错误只会重复调用同一个工具，
        # 白烧一轮（实测过一次：log_overview 连续调用两次都拿到同一段错误文本）。
        print(f"[tools] {name} 失败：{type(e).__name__}: {e}", flush=True)
        return f"该工具本次执行失败（{type(e).__name__}），请换用其他工具或换个关键词。"

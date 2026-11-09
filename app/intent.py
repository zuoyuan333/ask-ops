"""输入理解层：把用户随口一句话归一成「内部意图标签」，供教练（人工纠正记录）匹配使用。

为什么需要：
    教练匹配如果只做 2-gram 字面相似度，「报错 E1024 怎么处理」与「E1024 报错怎么解决」
    这类同义换序就会低于门槛；同一类问题换个说法就匹配不上。
    **不能要求一线背固定话术**，所以先"听懂"再匹配。

两级识别（A 规则 / B 模型兜底）：
    ① 规则（0ms）：关键词与结构化特征（错误码、时间、路径）先判，命中率高且确定；
    ② 模型兜底（+1~2s）：规则判不出时才问一次 fast 模型 —— 由调用方用
       `resolve(question, has_intent_coach=True)` 决定是否值得花这个钱。

⚠️ 本模块**不依赖** cases / llm（避免循环导入）：LLM 调用在 `resolve()` 里惰性 import。
"""

import re

# 意图标签（顺序即优先级，越靠前越具体）
INTENTS = {
    "log_troubleshoot": "看日志排查故障",
    "code_lookup": "定位代码/实现",
    "howto": "操作步骤/配置方法",
    "knowledge_qa": "概念/流程/规则咨询",
    "unknown": "未识别",
}

# ---- 结构化实体 ----
_RE_ERRCODE = re.compile(r"\b(?:E|ERR|0x)[0-9A-Fa-f]{3,8}\b|[A-Z]{2,}[_-]?\d{3,6}")
_RE_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_RE_FILE = re.compile(r"\b[\w./-]+\.(?:py|js|ts|java|go|json|ya?ml|sql)\b")

# ---- 关键词（同义说法归一）----
# ⚠️ 词表刻意收窄：只放**强特征**词，否则任何问题都会被归成同一类，教练会串味。
_KW = [
    ("log_troubleshoot", ("日志", "报错", "异常", "堆栈", "traceback", "超时", "失败", "崩溃",
                          "复现", "排查", "定位不到")),
    ("code_lookup", ("代码", "源码", "实现", "哪个文件", "函数", "接口定义", "调用链", "怎么写的")),
    ("howto", ("怎么配", "在哪配", "怎么设置", "步骤", "操作", "开通", "配置项", "参数怎么改")),
    ("knowledge_qa", ("是什么", "为什么", "原理", "规则", "流程", "区别", "能不能", "支持吗")),
]


def detect(question: str) -> str:
    """规则识别（0ms，纯本地）。判不出返回 "unknown"。"""
    q = str(question or "")
    if not q.strip():
        return "unknown"
    low = q.lower()

    # ① 带具体错误码 / 时间点 / 文件名 → 基本是拿证据来排查
    if _RE_ERRCODE.search(q) or _RE_TIME.search(q) or _RE_FILE.search(q):
        return "log_troubleshoot"
    # ② 关键词（强特征词）
    for name, words in _KW:
        if any(w.lower() in low for w in words):
            return name
    return "unknown"


def infer_text(question: str, raw: str = "") -> str:
    """给教练记录猜「适用场景」：先看它挂在哪个案例问题上，再看教练正文。

    返回单个意图标签（unknown 表示猜不出，此时不限制场景 = 走原来的字面匹配）。
    宁可猜 unknown 也不硬套：套错场景会让这条教练在无关问题上被注入。
    """
    for text in (str(question or ""), str(raw or "")[:400]):
        if not text.strip():
            continue
        r = detect(text)
        if r != "unknown":
            return r
    return "unknown"


# ---------------- B：模型兜底 ----------------
_cache: dict[str, str] = {}
_CACHE_MAX = 512


def resolve(question: str, has_intent_coach: bool = True) -> str:
    """识别的总入口：规则优先，判不出且"库里有带场景的教练"时才调一次模型。

    :param has_intent_coach: 库里有没有声明了 `intents` 的教练。
        没有的话，识别出意图也用不上（match_coach 仍按字面匹配）⇒ **不必花这次调用**。
    """
    r = detect(question)
    if r != "unknown" or not has_intent_coach:
        return r

    key = re.sub(r"\s+", "", str(question or ""))[:120]
    if key in _cache:
        return _cache[key]

    try:
        from . import llm                      # 惰性导入：避免与 llm → cases 的依赖打架
        r2 = llm.classify_intent(question)
    except Exception:                          # noqa: BLE001
        r2 = "unknown"                         # 兜底失败就当没识别出来，绝不拖垮 /ask
    if r2 not in INTENTS:
        r2 = "unknown"
    if len(_cache) < _CACHE_MAX:
        _cache[key] = r2
    return r2

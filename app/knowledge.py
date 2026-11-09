"""知识库加载与检索（纯 Python BM25 + 别名/错写扩展，零外部服务依赖）。

2026-09-14 检索加固（起因：运维把「祥瑞卡」说成「瑞祥」，助手全库检索不到正主条目，
反而拿一堆弱相关片段编了个似是而非的流程 —— 比答不出更危险）。三层：
  ① 别名组：从 KB 里的「关键词 / 别名」声明 + 别名表格自动建同义词组，
     查询命中组内任一词，就把整组词一起拿去召回（含字符 2-gram，容得下词序颠倒/漏字）。
  ② 罕见词模糊扩展：查询里 df 极低的中文词（疑似错字）与 KB 词表做字符集 Dice 匹配，
     "瑞祥" → "祥瑞" 这样一字不差只是颠倒的，能被自动救回，无需人工登记。
  ③ 弱命中判据：报出「这个对象库里没人认识」——上层据此禁止模型硬凑答案。
"""
import math
import os
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

try:
    import jieba
    jieba.setLogLevel(60)  # 关掉 jieba 初始化日志
except Exception:  # pragma: no cover
    jieba = None

from . import config

_lock = threading.Lock()
_DOCS = []          # [{file, idx, text, kw}]
_INDEX = None       # [{token: count}]
_AVGDL = 0.0
_K1, _B = 1.5, 0.75
# ⚠️⚠️ 2026-10-26 新增：**单块最多计几个「扩展词」的命中**（`KB_EXP_TERM_CAP`，默认 3）。
#   动机（实测踩到的真 bug）：查询里的一个词经别名组会被展开成**几十个同义词**，
#   而知识库里恰好有「别名清单块」——它短（96 字）、又把这一组词几乎全列了一遍。
#   于是该块能一次性命中几十个扩展词，再叠加 BM25 的长度归一化（块越短，每个命中词
#   的权重越大）⇒ 实测「接驳订单号怎么生成」：这篇 96 字的别名块拿 **110.8** 分，
#   把真正回答该问题的文档（46.8 分）压到第 5。
#   修法：**把「用户原词」和「别名扩展词」分开计分** —— 原词照常全额累加（它才是
#   用户真正问的东西），扩展词最多只取贡献最大的 N 个，从根上堵住"几十个同义词叠分"。
#   ⚠️ 取 3 而不是 1：错写救援（用户把词写错、只有扩展词能命中）通常也就 1~3 个词，
#      砍到 1 会让正常的同义词召回变钝。要调就改环境变量，别改这个默认值。
try:
    _EXP_TERM_CAP = int((os.getenv("KB_EXP_TERM_CAP", "") or "").strip() or 3)
except ValueError:
    _EXP_TERM_CAP = 3
if _EXP_TERM_CAP < 0:
    _EXP_TERM_CAP = 0          # 0 = 完全不计扩展词（排障用，别在生产开）
_loaded_files = set()

_ALIAS = {}         # {词: frozenset(同组全部词)}  别名 / 常见错写
# ⚠️ 2026-10-21 修：**别名表必须按「整文件」解析，不能按块解析**。
#   背景：`_split_chunks` 是「按空行分段 + 超 400 字硬切」，不认 Markdown 表格。
#   实测《POS-术语别名与常见错写.md》被切成 8 块，而 `_parse_alias_table` 要求
#   「表头与数据行同块」⇒ 只有第 1 块（含表头）被解析，**24 行里只有前 7 行生效**：
#   祥瑞卡（第 1 行）一直好用，而「隔日退 / 示例门店付 / 收单参数 / 强制补登 / 券资产 /
#   数字化交易单号 / 版本更新 / A8 / ELO」等全部丢失，且被切断的半行还会被当成
#   碎片词登记（"付对""射关""利祥"）。这里把别名表文件的**原文**存下来，
#   交给 `_build_alias` 整文件解析。
_FULL_TEXT = {}     # {相对路径: 整文件原文}，只存文件名像别名表的文件
# ⚠️ 2026-10-22 新增：**每篇文档的标题词集合**（{相对路径: set(标题分词)}），供标题命中加成用。
#   动机：语料从 7.5k 块涨到 9.7k 块后，top1 在几篇都相关的文档间变成"抛硬币"（88.6% → 82.9%）——
#   实测「热更新后门」的 top1 被《版本升级建议》抢走（真正那篇标题就叫「ImPOS 热更新后门」，
#   却因为正文里"热更新"词频不如别人高而排第 2）；「祥瑞卡」同理（《祥瑞卡、杭州市民卡配置》
#   标题里就有"祥瑞卡"，却输给《后台配置参数速查表》）。
#   ⇒ 标题命中是**强相关性信号**："这篇就是讲这个的" ≠ "这篇提到过这个"。BM25 只看块内词频，
#     区分不出这两者，所以补一个标题加成。
_TITLE_TOKENS = {}
# 加成系数：查询词全部出现在标题里时，得分 ×(1+系数)。0 = 关闭。
try:
    _TITLE_BOOST = float((os.getenv("KB_TITLE_BOOST", "") or "").strip() or 0.5)
except ValueError:
    _TITLE_BOOST = 0.5
if not (0.0 <= _TITLE_BOOST <= 2.0):
    _TITLE_BOOST = 0.5
_VOCAB = {}         # {中文词: df}，仅保留 df≥3 的
_DF_ALL = {}        # {中文词: df}，全量（弱命中判据用）
# ⚠️ 2026-10-09 新增：**全 token** 的 df（不限中文，含错误码/编号/英文缩写）。
# `_DF_ALL` 只统计中文词，于是 `E-7731` 里的 `7731` 根本查不到 df —— 而它恰恰是
# 最需要判断"库里到底有没有"的那类词。见 `unknown_tokens()`。
_DF_TOK = {}
_TERM_POOL = set()  # 显式声明过的术语及其字符变体（形近词比对候选池）
_CHAR_IDX = {}      # {汉字: {含该字的术语}}，加速「找形近词」

# ⚠️ 2026-10-10 评审 ⑩：银行接口规范类文档的 BM25 衰减（**默认关**，等 A/B）。
#   动机：实测「收银台不能签到」top3 全是银行接口文档（字段名词频爆炸），FAQ/手册
#   排 top6 之后 —— 对一线「怎么操作」类问题这是纯噪声。衰减而非剔除：
#   问题里含「接口/报文/字段」类词（真要查接口）时不衰减。
#   KB_API_DOC_DECAY = 0~1 的小数 = 衰减系数；0 / 1 / 未设 = 关闭。
_API_DOC_RE = re.compile(
    r"银行接口规范|支付应用API|收单应用接口|URI接口|硬件调用接口|后台接口总表"
    # ⚠️ 2026-10-22 试过把《原生插件接口与返回码速查》也加进来（想让症状类提问
    #   「客人说刷不了卡」让位给排障文档），**实测反而更差**：
    #   top1 从 30/35 掉到 29/35 —— 因为 `_api_query()` 是按**分词后**的 token 判豁免，
    #   而 jieba 把「返回码」切成「返回」+「码」，「工行返回码」这类提问**认不出豁免词**，
    #   于是连本该命中的参考表也被衰减掉了。
    #   ⇒ **已撤回**。记下来避免以后再试同一条路：要动这个正则，先确认 `_api_query` 的
    #   分词豁免对目标提问真的生效。
    )
_decay_env = (os.getenv("KB_API_DOC_DECAY", "") or "").strip()
try:
    _API_DOC_DECAY = float(_decay_env) if _decay_env else 0.0
except ValueError:
    _API_DOC_DECAY = 0.0
_API_DOC_DECAY_ON = 0.0 < _API_DOC_DECAY < 1.0

# ⚠️ 2026-10-23 **试过并撤回**：目录/索引类文档降权（KB_META_DOC_DECAY）。
#   起因：「POS支持非租登录」的 top1 被《运维工具箱-工具索引》抢走（111.99:105.13）
#   —— 目录类文档把每个工具名逐条列在正文里，等于抄了查询原文。
#   实测（35 题检索评测，0.85 / 0.7 / 0.55 三档）：
#     0.85 → 修好「POS支持非租登录」，但**弄坏「强制补登」**（它本来就该靠速查类文档回答）
#     0.7  → 还多弄坏「报错 30030」    0.55 → 还弄坏「银行返回码速查」
#   ⇒ **净收益为零**（32/35 还是 32/35），且更低的档位明显变差，故**不留**。
#   📌 教训：「X 在哪」类问题**本来就需要目录文档**，一刀切降权是按下葫芦起了瓢。
_API_QUERY_WORDS = {"接口", "报文", "字段", "api", "uri", "sdk", "返回码", "报文格式"}

# ⚠️ 2026-10-23 新增：**查询停用词**（只作用于**查询**分词，不动索引与 IDF 统计）。
#   动机（实测）：问「接驳是干什么的」时，top1 竟然是《支付渠道与供应商全表》
#   （该文**一次"接驳"都没有**），与正确的《接驳接口联调与报错排查》只差 **32.87 : 32.01**。
#   根因：查询分词留下 `是 / 的 / 干什么` 这类**纯功能词**，它们 IDF 极低但会在
#   **超长块**里累积出可观的分数，把 `接驳` 这种高 IDF 的真信号淹没。
#   ⇒ 检索前剔掉它们。**只剔"没有检索价值的功能词"**，不含任何业务词
#     （卡 / 券 / 单 / 版 这类单字业务词必须保留）。
_QUERY_STOP = {
    # 单字功能词
    "是", "的", "了", "吗", "呢", "啊", "吧", "么", "呀", "嘛", "哦", "嗯", "哈",
    "被", "把", "就", "都", "也", "还", "又", "而", "与", "或", "及", "等",
    "我", "你", "他", "她", "它", "们", "咱",
    # 多字功能词 / 口头禅（在一线提问里高频出现但检索价值为零）
    "什么", "怎么", "怎样", "为什么", "为何", "如何", "哪些", "哪个", "哪里",
    "干什么", "干啥", "怎么办", "怎么样", "一下", "请问", "帮我", "我们", "你们",
    "他们", "事情", "情况", "时候", "一个", "这个", "那个", "可以", "能否", "是否",
}

# ⚠️ 2026-10-22 新增：**历史归档文档的打分衰减**。
#   识别方式：文档开头 1500 字内含 `🗄️`（我们在历史文档头部加的「🗄️ 历史文档」标记）。
#   为什么需要：标记本身不影响检索 —— 实测「A8终端」的 top1 被 2020 年的《系统管理员手册》
#   以 28.79 : 28.33 微弱压过。衰减后历史文档不再抢 top1，但仍能进 top-k（个别条目仍有效）。
#   设 0 / 1 / 未设 = 关闭本衰减。
_hd_env = (os.getenv("KB_HIST_DOC_DECAY", "") or "").strip()
try:
    _HIST_DECAY = float(_hd_env) if _hd_env else 0.55
except ValueError:
    _HIST_DECAY = 0.55
if not (0.0 < _HIST_DECAY < 1.0):
    _HIST_DECAY = 1.0        # 关掉


def _api_query(q_tokens: list) -> bool:
    """问题本身是不是在问接口细节 —— 是就不衰减（要的就是接口文档）。"""
    return any((t or "").lower() in _API_QUERY_WORDS for t in q_tokens)

_ZH_RE = re.compile(r"^[\u4e00-\u9fff]+$")
# 声明的两种类型，语义完全不同，不能混：
#   别名/错写 → 同义词，查询命中组内任一词就把整组拿去召回（跨词扩展）
#   关键词   → 该条目的检索标签，只让「这个条目」更容易被这些词命中，不外扩
#
# ⚠️ 必须要求声明词出现在**行首**（允许前置 # / - / 数字等 markdown 前缀）。
# 早期版本只做 contains 匹配，于是「翼支付证书别名」「xx 字段别名」这类正文词
# 也被当成了别名声明行，把整段文字切成同义词灌进词典 —— 检索因此全面跑偏。
# ⚠️⚠️ 2026-10-26 加 `>`（引用块）前缀。踩到的真实故障：
#   知识库里有 4 篇文档用 `> 别名：A、B、C…` 声明别名（人类看得懂），
#   但这个正则不认 `>` ⇒ ① 声明**完全没生效**（好在中央别名表
#   《POS-术语别名与常见错写》里也有同一批词，所以扩展仍能命中）；
#   ② 更糟的是**这一块自己还参与检索**：它短、又几乎把别名组几十个词全列了一遍，
#   BM25 长度归一化把每个词的权重放大 ⇒ 实测「接驳订单号怎么生成」这一问，
#   这篇 96 字的别名块拿 **110.8** 分，压过真正讲这事的文档（46.8 分）。
#   认了 `>` 之后，`_extract_decl` 能正确识别它，`_kw_only` 也会把它判成
#   「纯词表块」→ 按既有规则**从结果里剔除**（它的词已经通过别名组生效了）。
_DECL_LINE_RE = re.compile(
    r"^\s*(?:>{1,2}\s*|#{1,6}\s*|[-*+•]\s*|\d+[.、)]\s*)?"
    r"(关键词|关键字|别名|又名|同义词|常见错写|易错写法|别称|俗称)"
    r"\s*(?:[（(][^）)]*[）)])?\s*(?:[:：]\s*(.*))?$"
)
_ALIAS_HEADS = ("别名", "又名", "同义词", "常见错写", "易错写法", "别称", "俗称")


# ---------------- 分块与分词 ----------------

def _split_chunks(text: str, max_len: int = 400):
    """按空行切成独立段落块（不跨主题合并，避免串味）；
    单段超过 max_len 时，**优先按行边界切**，并保证 Markdown 表格块不被拦腰切断。

    ⚠️ 2026-10-21 改（原来是无条件 `text[i:i+400]` 硬切）：
       实测全库 **6.4%（498/7775）的块正好 400 字**，就是被硬切的痕迹；
       其中大量是表格行被切断 —— 别名表被切成 11 块、表头与数据行分离
       （旧解析器因此只认到前 7 行，是「答不准」的头号根因）。
       表格一旦断开，「键 | 值 | 说明」的行内对应关系就没了，检索到也用不了。
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for p in paras:
        if len(p) <= max_len:
            chunks.append(p)
        elif _is_table_block(p):
            if len(p) <= _TABLE_MAX:
                chunks.append(p)                  # 表格整块保留
            else:
                chunks.extend(_hard_cut_lines(p, _TABLE_MAX))
        else:
            chunks.extend(_hard_cut_lines(p, max_len))
    return chunks


_TABLE_MAX = 700           # 单个表格块的保留上限。
#   ⚠️ 为什么是 700：`tools.py` 投喂每个片段时截到 700 字（`h['text'][:700]`）。
#      表格块若超过 700，多出来的部分根本进不了模型上下文 —— 那还不如按行切成两块
#      （每块都带完整的行），至少内容不丢。表格的价值在于「行内 键|值|说明 的对应」，
#      按行切能保住这一点，硬切（原来 text[i:i+400]）会把一行劈成两半，对应关系就没了。
_TABLE_LINE_RE = re.compile(r"^\s*\|")


def _is_table_block(text: str) -> bool:
    """整段是否是一张 Markdown 表格（≥2 行、且每行都以 | 开头）。"""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    return all(_TABLE_LINE_RE.match(l) for l in lines)


def _hard_cut(text: str, max_len: int):
    out = []
    for i in range(0, len(text), max_len):
        out.append(text[i:i + max_len])
    return out


def _hard_cut_lines(text: str, max_len: int):
    """按**行**累积切块：尽量不把一行切断（表格行 / 代码行切一半就没用了）。
    单行本身超过 max_len 时，退回硬切。"""
    out, buf = [], ""
    for line in text.splitlines():
        if len(line) > max_len:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_hard_cut(line, max_len))
            continue
        if buf and len(buf) + 1 + len(line) > max_len:
            out.append(buf)
            buf = line
        else:
            buf = line if not buf else buf + "\n" + line
    if buf:
        out.append(buf)
    return out


def _tokenize(text: str):
    """中文用 jieba 分词，英文/数字按词切；全小写。"""
    text = text.lower()
    if jieba is not None:
        toks = jieba.lcut(text)
        toks = [t for t in toks if t.strip()]
    else:
        toks = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    # 过滤单字虚词之外的超短 token（长度1的中文保留，数字/英文长度1保留）
    return [t for t in toks if t.strip()]


def _is_zh_word(t: str, lo: int = 2, hi: int = 4) -> bool:
    return lo <= len(t) <= hi and bool(_ZH_RE.match(t))


# ---------------- 别名 / 常见错写 ----------------

def _split_kw_words(s: str) -> list:
    """把一行「词表」切成词：去掉括号说明，按空白/顿号/逗号/竖线切分。

    长度上限放到 80：POS 的参数名很长（`pos.sale.jl_electronic_gwk.open` 就 34 字符），
    早先卡在 30 会把它整条丢掉，反而只剩 `gwk` 这种碎片进词表。
    """
    s = re.sub(r"[（(][^）)]*[）)]", " ", s)
    out = []
    for p in re.split(r"[\s,，、；;|/／]+", s):
        p = p.strip().strip("：:·-—*`\"'")
        if 1 <= len(p) <= 80:
            out.append(p.lower())
    return out


def _dedup(words: list) -> list:
    seen, out = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _extract_decl(text: str) -> tuple:
    """抽取 chunk 里声明的关键词与别名，返回 (aliases, keywords)。

    兼容两种写法（写知识条目时不必记格式）：
        ## 关键词（放宽词面匹配）      ← 标题式：词在下一行
        祥瑞卡 瑞祥卡 祥瑞 ...
        - 别名：祥瑞卡、瑞祥卡、祥瑞   ← 冒号式：词在同一行

    ⚠️ 两者用途不同，别混（混了会出事）：把「关键词」也当同义词组，
    一条条目的 20 个标签词会被展开成一大坨互不相干的词，任何含「支付」
    的提问都会被拽到这个条目上。
    """
    aliases, kws, lines = [], [], text.splitlines()
    for i, ln in enumerate(lines):
        m = _DECL_LINE_RE.match(ln)
        if not m:
            continue
        head, inline = m.group(1), (m.group(2) or "").strip()
        words = _split_kw_words(inline) if inline else []
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            # ⚠️ 2026-10-26：`>` 开头也排除 —— 声明行写成 `> 别名：A、B` 时，
            #   下一行往往也是引用块（如 `> 被问「…」时用本文。`）。
            #   不排掉就会把「被问…时用本文」整句拆成词灌进别名组（实测会污染）。
            if nxt.strip() and not nxt.lstrip().startswith(("#", "|", ">")) \
                    and not _DECL_LINE_RE.match(nxt):
                words += _split_kw_words(nxt)
        words = _dedup([w for w in words if w])
        (aliases if head in _ALIAS_HEADS else kws).extend(words)
    return _dedup(aliases), _dedup(kws)


def _extract_kw(text: str) -> list:
    """兼容旧调用：返回该块声明的全部词（关键词 + 别名）。"""
    a, k = _extract_decl(text)
    return _dedup(a + k)


def _word_variants(w: str) -> set:
    """把一个词展开成自身 + 连续 2/3-gram。

    目的：容下「词序颠倒 / 多字少字 / 中间夹字」这类口语错写 ——
    「祥瑞卡」展开后含「瑞祥」，于是用户把正名说反了也能命中。
    ⚠️ 只对纯中文词做 n-gram：英文标识符（pos.sale.jl_electronic_gwk.open）
    展开会产生上百个碎片，把同义词组搅成一锅粥。

    ⚠️⚠️ 2026-10-26 加一道闸：**跨功能词的 n-gram 一律不要**。
    踩到的真 bug（「接驳订单号怎么生成」top1 跑到会员手机号那篇）：
      别名「会员手机号怎么查」→ n-gram 生成出 key `号怎么`（df=0，谁也没用过），
      而它在**别的领域**的提问「接驳订单**号怎么**生成」里正好是子串
      ⇒ `_alias_substr_expand` 整串命中 → 把整个「会员手机号」同义词组
      （含其 n-gram，共 45 个词）灌进这次查询 → 那篇的别名清单块命中 7 个扩展词、
      拿 72.9 分抢走 top1。
    根因：`号怎么` 这种「实词尾 + 功能词」的碎片**没有词义**，任何查询里都可能出现。
    ⇒ 生成 n-gram 时若含 `_QUERY_STOP` 里的功能词就丢弃。实测全库 787 个别名 key 里
      有 53 个是这种垃圾碎片（`单是`/`是谁`/`么查`/`的版`/`复的`…），全部清掉。
    ⚠️ 只过滤**生成的 n-gram**，人工声明的原词（如「刷不了卡」本身含「不了」）照留。
    """
    return {w} | _ngrams(w)


_ALIAS_HEAD_WORDS = ("正名", "别名", "常见错写", "易错写法", "别称")
_MAX_GROUP = 200     # 一组超过这么多词 = 解析跑偏，整组丢弃（宁缺毋滥）
_ALIAS_FILE_RE = re.compile(r"(术语别名|常见错写|别名与错写|别名词表)")


def _parse_alias_table(text: str) -> list:
    """解析 markdown 表格式的别名表（用于《POS-术语别名与常见错写》）：
        | 正名 | 别名 / 常见错写 |
        | 祥瑞卡 | 瑞祥、瑞祥卡、翔瑞卡 |

    ⚠️ 两道闸：
      1. 只在**文件名像别名表**的文件里解析（见 _build_alias）；
      2. 只在表头明确写着「正名/别名/错写」的表格里解析。
    知识库里有大量普通数据表（编码对照表、参数速查表），早期版本把它们每一行
    都当成了同义词组，导致「005/012/支付证书」全被并成一组、检索全面跑偏 ——
    这个坑必须靠这两道闸堵死，不能靠"看起来像表格"。
    """
    groups, in_table = [], False
    for ln in text.splitlines():
        s = ln.strip()
        if not s.startswith("|"):
            in_table = False
            continue
        if set(s) <= set("|-: \t"):     # 分隔行 |---|---|
            continue
        if not in_table:
            if any(w in s for w in _ALIAS_HEAD_WORDS):
                in_table = True
            continue
        words = []
        for cell in s.strip("|").split("|")[:2]:   # 只取「正名」「别名」两列，说明列的文字不是词
            words += _split_kw_words(cell)
        words = [w for w in words if w and w not in _ALIAS_HEAD_WORDS]
        if len(words) >= 2:
            groups.append(set(words))
    return [g for g in groups if len(g) <= _MAX_GROUP]


def _kw_only(chunk: str) -> bool:
    """该块是否只是「关键词/别名」词表（没有正文）？

    这种块自己是元数据、不含答案，但 token 密度高、长度短，BM25 会给它虚高分，
    把真正的正文块挤下去。
    ⚠️ 2026-10-26 修正描述：它**不是「降权」而是「不参与候选」** ——
       实际的处置在 `_search_ex_locked` 的挑选循环里（`if ... or kw_only ... continue`），
       直接跳过。原因：这些词本来就已通过别名组生效，本体再进上下文只会抢答。
       （打分阶段那行 `score *= 0.35` 因为随后就被跳过，实际是死代码。）
    """
    kw = _extract_kw(chunk)
    if not kw:
        return False
    body = chunk.lower()          # 声明词统一小写存，比对前也把小写化（JFK vs jfk）
    # ⚠️ 必须按长度倒序替换：先替换短词会把长词切碎（「购物卡」先吃掉「电子购物卡」
    # 的一部分，留下孤零零的「电子」），结果是纯标签块被误判成有正文。
    for w in sorted(kw, key=len, reverse=True):
        body = body.replace(w, "")
    body = re.sub(r"[#\s*:：（）()【】\[\]、,，.。|/`\"']", "", body)
    return len(body) < 20


_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_IMG_ONLY_MAX_BODY = 24


def _img_only(chunk: str) -> bool:
    """该块是否只是「截图占位」（正文只有「步骤N xxx：」+ ![](assets/…)）？

    这类块对模型毫无价值 —— 它看不到图。但块短、token 少，BM25 反而容易给高分，
    实测会把**同一篇文档里**真正写着「后台点击路径 / 参数值 / 常见坑」的正文块挤出 top5，
    症状就是「检索命中了正确的条目，答案却只有一句结论、没有可执行步骤」（2026-09-14 祥瑞卡）。
    识别出来后按「非知识」处理：不进模型上下文（图仍留在文档里给人看）。
    """
    if not _IMG_RE.search(chunk):
        return False
    body = _IMG_RE.sub("", chunk)
    body = re.sub(r"[\s#*>`\-—~=:：()（）\[\]|]", "", body)
    return len(body) <= _IMG_ONLY_MAX_BODY


def _ngrams(w: str, sizes=(2, 3)) -> set:
    """取词的连续 2/3-gram（不含词本身）。

    ⚠️⚠️ 2026-10-26 加一道闸：**含功能词的 n-gram 一律丢弃**。
    踩到的真 bug（「接驳订单号怎么生成」top1 跑到会员手机号那篇）：
      别名「会员手机号怎么查」→ 生成出 key `号怎么`（df=0，从没在任何正文里出现过），
      而它在**另一个领域**的提问「接驳订单**号怎么**生成」里正好是子串
      ⇒ `_alias_substr_expand` 整串命中 → 把整个「会员手机号」同义词组（连带其
      n-gram，共 40+ 个词）灌进这次查询 → 那篇的别名清单块命中好几个扩展词，
      拿 72.9 分抢走 top1。
    根因：`号怎么`/`单是`/`么查`/`的版` 这类「实词尾 + 功能词」碎片**没有词义**，
    任何一句中文里都可能凑巧出现。实测全库 787 个别名 key 里有 53 个是这种碎片。
    ⚠️ 只过滤**生成的 n-gram**；人工声明的原词照留（如「刷不了卡」本身含「不了」，
    那是人写进别名表的，不属于这里）。
    """
    out = set()
    if _ZH_RE.match(w):
        for n in sizes:
            for i in range(len(w) - n + 1):
                g = w[i:i + n]
                if any(s in g for s in _QUERY_STOP):
                    continue
                out.add(g)
    return out


def _build_alias(docs: list):
    """汇总全库声明的「别名 / 常见错写」，生成 {词 -> 同组词} 映射。

    ⚠️ 两个必须守住的细节（都是踩出来的）：

    1. **同义词组的成员是「人工声明的原词」，n-gram 只当查找用的 key。**
       把 n-gram 也当组成员，会让不同组互相"粘"起来 —— 「嘉利祥瑞卡」和
       「嘉利积分卡」共享 2-gram「嘉利」，两组一连通，最后合并成一个包含
       「积分 → 祥瑞卡」的巨型组，任何问「积分」的人都会被拽到祥瑞卡条目上。

    2. **一个 key 落在多个组里 = 有歧义，要作废**（而不是取并集）。
       用户输入经 jieba 切分后常只剩术语片段（「瑞卡」），所以 n-gram 当 key
       是必要的；但只有唯一的 key 才敢用来扩展。

    ⚠️ 2026-10-21 修（第 2 条的**分级**版）：旧实现「只要落在 ≥2 组就整体作废」
       把最重要的正名也误杀了。原因是 n-gram key 天然跨组：
       「示例门店付」既是 示例门店付组 的**人工声明正名**，又是「示例门店付单号」的 3-gram
       （属数字化交易单号组）⇒ 被判歧义作废，于是问「示例门店付」时别名组**完全不生效**
       （实测展开 0 词，top1 跑到无关文档）。「隔日退」同理。
       现在分两类处理：
         · **人工声明的原词**（正名/别名）：只被一个组声明就归该组，
           哪怕它同时是别组的 n-gram —— 声明是有意为之，优先采信；
         · **纯 n-gram key**：仍要求全局唯一，否则作废 ——
           保留原本「嘉利」把两个卡组连成一锅粥的保护。
    """
    global _ALIAS
    groups = []
    seen_alias_files = set()
    for d in docs:
        al = d.get("alias") or []
        if len(al) >= 2:
            groups.append(frozenset(al))
        f = d["file"]
        if not _ALIAS_FILE_RE.search(f) or f in seen_alias_files:
            continue
        # ⚠️ 2026-10-21 修：**整文件解析**，不再逐块解析。
        # 逐块解析会因为 400 字硬切丢表头，导致 24 行别名表只生效前 7 行。
        seen_alias_files.add(f)
        text = _FULL_TEXT.get(f)
        if not text:                      # 兜底：把该文件的块按 idx 拼回去
            parts = sorted((x.get("idx", 0), x.get("text") or "")
                           for x in docs if x["file"] == f)
            text = "\n".join(t for _, t in parts)
        for g in _parse_alias_table(text):
            groups.append(frozenset(g))

    declared, declared_multi = {}, set()
    ngram, ngram_multi = {}, set()
    for g in groups:
        for w in g:
            if w in declared and declared[w] != g:
                declared_multi.add(w)
            else:
                declared.setdefault(w, g)
            for k in _ngrams(w):
                if k in ngram and ngram[k] != g:
                    ngram_multi.add(k)
                else:
                    ngram.setdefault(k, g)
    keymap = {}
    for w, g in declared.items():                 # 声明词优先
        if w not in declared_multi:
            keymap[w] = g
    for k, g in ngram.items():                    # 纯 n-gram：要求全局唯一且不与声明词冲突
        if k in ngram_multi or k in keymap:
            continue
        keymap[k] = g
    _ALIAS = {k: v for k, v in keymap.items() if v}


def _build_vocab():
    """从倒排索引统计 token 的文档频率（弱命中判据 / 模糊匹配 / 查无此词 用）。"""
    global _VOCAB, _DF_ALL, _DF_TOK
    df = Counter()
    df_all = Counter()
    for cnt in (_INDEX or []):
        for t in cnt:
            df_all[t] += 1
            if _is_zh_word(t):
                df[t] += 1
    _DF_ALL = dict(df)
    _DF_TOK = dict(df_all)
    _VOCAB = {t: c for t, c in df.items() if 3 <= c <= 300}


def _build_terms(docs: list):
    """构建「术语池」：知识条目里显式声明过的关键词/别名及其字符变体。

    ⚠️ 为什么不能用 jieba 的词表当候选池：**术语往往被 jieba 切散**。
    「祥瑞卡」会被切成「祥瑞 / 卡」，于是"祥瑞卡"这个词在 token 统计里 df 极低、
    进不了词表 —— 而它恰恰是用户最可能写错的那个词（写成「翔瑞卡」）。
    所以候选池必须来自**显式声明过的词**（关键词行、别名表），那里才有完整术语。
    """
    global _TERM_POOL, _CHAR_IDX
    pool = set()
    for d in docs:
        for w in (d.get("kw") or []) + (d.get("alias") or []):
            if _is_zh_word(w, 2, 8):
                pool |= _word_variants(w)
        if _ALIAS_FILE_RE.search(d["file"]):
            for g in _parse_alias_table(d["text"]):
                for w in g:
                    pool |= _word_variants(w)
    _TERM_POOL = pool
    idx = {}
    for w in pool:
        for ch in set(w):
            idx.setdefault(ch, set()).add(w)
    _CHAR_IDX = idx


def _cand_pool() -> set:
    """可用于「形近词」比对的候选：术语池里在库中确实出现过的词。"""
    return {w for w in _TERM_POOL if _DF_ALL.get(w, 0) >= 2 or w in _ALIAS}


# ---------------- 加载 ----------------

def _iter_files(kb_dir: Path):
    exts = {".md", ".txt", ".markdown"}
    for p in sorted(kb_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def load():
    """扫描知识库目录，增量加载新增/变更文件。"""
    global _INDEX, _AVGDL, _loaded_files
    kb_dir = config.KB_DIR
    if not kb_dir.exists():
        kb_dir.mkdir(parents=True, exist_ok=True)
        return

    changed = False
    for p in _iter_files(kb_dir):
        stat = p.stat()
        key = (str(p), stat.st_mtime_ns, stat.st_size)
        if key in _loaded_files:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(p.relative_to(kb_dir))
        # 别名表文件：把整文件原文留一份，供 _build_alias 整表解析（见 _FULL_TEXT 注释）
        if _ALIAS_FILE_RE.search(rel):
            _FULL_TEXT[rel] = text
        # 移除旧版本该文件的块
        _DOCS[:] = [d for d in _DOCS if d["file"] != rel]
        _is_hist = "🗄️" in (text or "")[:1500]     # 历史归档标记（见 _HIST_DECAY）
        # 标题词集合（供 _TITLE_BOOST 用）：优先取文首的 `# 标题`，退化为文件名去前缀
        _tm = re.search(r"^#\s+(.+)$", text or "", re.M)
        _title = _tm.group(1) if _tm else rel.replace("POS-", "").replace(".md", "")
        _TITLE_TOKENS[rel] = set(_tokenize(_title))
        for i, chunk in enumerate(_split_chunks(text)):
            aliases, kws = _extract_decl(chunk)
            _DOCS.append({"file": rel, "idx": i, "text": chunk,
                          "alias": aliases, "kw": kws,
                          "kw_only": _kw_only(chunk),
                          "img_only": _img_only(chunk),
                          "hist": _is_hist})
        _loaded_files.add(key)
        changed = True

    if changed or _INDEX is None:
        # ⚠️⚠️ 2026-10-22 修（真 bug）：**清理「磁盘上已消失的文件」的块**。
        #   背景：本函数是**增量**加载 —— 上面只对「这次读到的文件」做
        #   `_DOCS[:] = [d for d in _DOCS if d["file"] != rel]`。
        #   于是**被删除 / 被移动的文件，它的块会永远留在内存里成为孤儿**。
        #   实测（把 8 篇银行规范移进 `bank-specs/` 后调 `POST /reload`）：
        #   运行中的服务块数从 **7641 涨到 12149** —— 旧路径的块没删、新路径又加一份，
        #   检索里同一篇出现两次、分数被抬高；而**新进程加载是干净的（7580）**。
        #   ⇒ 这个 bug 只在「长跑进程 + /reload」时暴露，重启一下就"好了"，所以极隐蔽。
        #   修法：按当前磁盘实际存在的相对路径集合，把不在集合里的块清掉。
        _alive = {str(p.relative_to(kb_dir)) for p in _iter_files(kb_dir)}
        _before = len(_DOCS)
        _DOCS[:] = [d for d in _DOCS if d["file"] in _alive]
        if len(_DOCS) != _before:
            print(f"[kb] 清理已消失文件的块：{_before} → {len(_DOCS)}", flush=True)
        # _FULL_TEXT 同样会留下已删别名表的残留
        for _k in list(_FULL_TEXT):
            if _k not in _alive:
                _FULL_TEXT.pop(_k, None)
        # _TITLE_TOKENS 同理（已删/已改名的文档不能留旧标题词，否则会错误加成）
        for _k in list(_TITLE_TOKENS):
            if _k not in _alive:
                _TITLE_TOKENS.pop(_k, None)
        _rebuild()


def _rebuild():
    global _INDEX, _AVGDL
    _INDEX = []
    total = 0.0
    for d in _DOCS:
        cnt = Counter(_tokenize(d["text"]))
        # ⚠️ 不要把「关键词」并进索引词频。试过，副作用很大：条目会被自己的标签
        # 撑成「万能匹配器」—— 祥瑞卡条目写了「积分卡/购物卡/支付键盘…」当标签，
        # 于是任何人问「会员积分怎么查」都被它抢答。写反的名字由别名组负责召回
        # （见 _alias_expand），标签只需老老实实呆在原文里。
        _INDEX.append(cnt)
        total += sum(cnt.values())
    _AVGDL = (total / len(_INDEX)) if _INDEX else 1.0
    _build_vocab()
    _build_alias(_DOCS)
    _build_terms(_DOCS)


def count() -> int:
    return len(_DOCS)


def file_count() -> int:
    """知识库**篇数**（去重后的文档文件数），区别于 `count()` 的**段数**（切块数）。

    ⚠️ 2026-10-23 加：页头要显示「知识 N 篇」，而 `/healthz.docs` 给的是
    `count()` = 切块数（线上实测 7000+ 段）—— 直接当"篇"用会差两个数量级
    （真实篇数只有 100 出头）。两个数都留着：段数给管理面板看索引规模，
    篇数给对外文案 —— 用户对"篇"有直觉，对"段"没有。
    """
    return len({d["file"] for d in _DOCS})


# 目录聚类规则：把 70 篇文档的标题按主题归组，给模型一张「库里到底有什么」的地图。
# ⚠️ 2026-10-10 加（治 deep 空转）：线上实测 61% 的 deep 问答撞满 8 轮，
#   轨迹显示模型在**反复换词搜「错误码汇总表 / 速查表 / 对照表 / 索引 / 大全」**
#   这类**库里根本不存在**的文档（错误码类只有孤例 YBLQSMP3079）。
#   根因：模型手里只有 search_kb，看不到「库里有什么」的全局视图 —— 只能盲搜试错。
#   解法：把文档目录注入 system prompt（静态前缀，缓存友好），并明说
#   「目录里没有的主题就是真的没有，不要换词硬搜」。
#   ⚠️ 两条纪律：① 规则**从精确到宽泛**排列（否则「接口配置-xxx教程」会被"教程"抢走）；
#              ② 输出顺序必须稳定（防前缀缓存失效）。
_CATALOG_RULES = [
    ("银行接口规范", ("银行接口规范",)),
    ("接口与协议", ("接口", "规范", "URI", "API", "OpenAPI", "接驳", "协议", "编码对照")),
    ("源码与架构", ("源码", "架构", "部署", "导航", "总纲", "数据资产", "工具索引", "工具箱")),
    ("排障与故障", ("故障", "异常", "排查", "失败", "报错", "错误码", "问题", "案例", "应急", "风险")),
    ("订单与交易", ("订单", "交易", "退货", "离线退", "补登", "券", "预售", "支付单", "单号", "分账")),
    ("配置与参数", ("配置", "参数", "设置", "激活", "映射", "键盘", "保活", "上线事项", "切换")),
    ("教程与手册", ("教程", "手册", "培训", "操作", "指引", "说明", "介绍", "应知应会")),
    ("术语与FAQ", ("术语", "别名", "错写", "FAQ", "速查", "汇总表")),
]


def doc_catalog() -> str:
    """生成「知识库文档目录」文本（给 system prompt 用）。

    按主题聚类输出全部文档名，让模型知道**库里有什么**，从而不再盲搜不存在的汇总表。
    ⚠️ 输出必须稳定有序（文件先 sorted，再按固定分组顺序），否则会破坏前缀缓存。
    """
    files = sorted({d["file"] for d in _DOCS})
    if not files:
        return ""
    # 去掉统一前缀（POS- / pos- / imPos- 大小写混杂），让目录更短更好读
    names = []
    for f in files:
        n = re.sub(r"\.(md|markdown|txt)$", "", f, flags=re.I)
        n = re.sub(r"^pos-", "", n, flags=re.I)
        names.append(n)

    groups: dict = {g: [] for g, _ in _CATALOG_RULES}
    groups["其他"] = []
    for n in names:
        for gname, kws in _CATALOG_RULES:      # 从精确到宽泛，首个命中即归类
            if any(k in n for k in kws):
                groups[gname].append(n)
                break
        else:
            groups["其他"].append(n)

    lines = []
    for gname, _ in _CATALOG_RULES:
        items = groups.get(gname) or []
        if items:
            lines.append(f"· {gname}（{len(items)}）：{'、'.join(items)}")
    if groups["其他"]:
        lines.append(f"· 其他（{len(groups['其他'])}）：{'、'.join(groups['其他'])}")
    return "\n".join(lines)




def reload():
    """强制全量重扫（start.command 启动时调用一次即可；知识库更新后 /reload 触发）。

    ⚠️ 2026-09-25 修复并发 500：原实现是
        `_loaded_files = set(); _DOCS[:] = []; with _lock: load()`
      —— 先**清空**再重建，而检索路径当时**不加锁** → 清空与重建之间的窗口里，
      并发进来的检索读到 `_DOCS` 与 `_INDEX` 长度不一致 → IndexError（表现为偶发 500）。
      现在：① **不再预清空**（load() 本身就会按文件替换旧块，不会重复累积）；
            ② 全程持 `_lock`，与已加锁的检索互斥。
    """
    global _loaded_files
    with _lock:
        _loaded_files = set()
        load()


# ---------------- 查询扩展 ----------------

# ⚠️ 2026-10-21 新增：**别名展开的「区分度闸门」**。
#   背景：修好别名表「整文件解析」后，24 行全部生效（登记词条 102 → 408），
#   但同时暴露一个潜伏问题 —— 长术语的 n-gram 会撞上常用词：
#     · 「数字化交易单号」的 2-gram 含「交易」
#     · 「版本更新」的 2-gram 含「更新」
#     · 「隔日退货」的 2-gram 含「退货」（df=504，占全部块 6.5%）
#   于是任意含「交易 / 退货」的提问都会展开出几十个无关词（实测
#   「工行聚合支付交易失败排查」展开 +69 词），把原本准确的排序带偏。
#   判据：**常用词不触发展开** —— key 的文档频率超过阈值就跳过。
#   阈值按块数占比给（KB_ALIAS_MAX_DF，默认 0.02；设 0 关闭本闸门）。
_amd_env = (os.getenv("KB_ALIAS_MAX_DF", "") or "").strip()
try:
    _ALIAS_MAX_DF = float(_amd_env) if _amd_env else 0.04
except ValueError:
    _ALIAS_MAX_DF = 0.04


def _alias_expand(q_tokens: list) -> dict:
    """别名/错写组扩展：命中组内任一词 → 整组词（含其字符变体）一起召回。

    为什么还要展开 n-gram：组里的原词往往是「祥瑞卡」这种 jieba 切不出来的术语，
    它在索引里的 tf 很低，单靠它拉不动排序；而它的 2-gram「祥瑞」是索引里的常客。
    两者一起给，才能真正把正主条目顶上来。

    ⚠️ 2026-10-21：加了「常用词不展开」闸门（见 _ALIAS_MAX_DF 注释）。
    """
    out = {}
    n = len(_DOCS) or 1
    for t in q_tokens:
        grp = _ALIAS.get(t) or _ALIAS.get(t.lower())
        if not grp:
            continue
        if _ALIAS_MAX_DF > 0 and _DF_TOK.get(t, 0) > _ALIAS_MAX_DF * n:
            continue          # 常用词（交易/退货/更新…）：不做组扩展，避免噪声
        for w in grp:
            for v in _word_variants(w):
                out.setdefault(v, 1.0)
    return out


_SUBSTR_MIN = 3          # 整串子串匹配的最小词长（中文 3 字起，避免「卡」「密码」这类泛词乱触发）
_SUBSTR_MAX_GROUPS = 3   # 一次查询最多命中几组（按词长优先，防止长问句被多组稀释）


def _alias_substr_expand(query: str) -> dict:
    """整串子串匹配的别名扩展（2026-10-21 新增）。

    为什么需要它：`_alias_expand` 是**按 jieba 切出的 token** 去查别名 key 的，
    而口语恰恰会被 jieba 切碎 —— 「刷不了卡」→ ['刷','不了','卡']，
    其中没有一个是登记过的别名，于是口语提问永远命不中别名组。
    这里改为在**整串**上找「登记过的别名」是否作为子串出现。

    三道闸（都必须过）：
      ① 词长 ≥ _SUBSTR_MIN：挡掉「卡」「密码」「支付」这类泛词；
      ② df 闸门：与 `_alias_expand` 同一套（常用词不触发）；
      ③ 按词长从长到短取，最多 _SUBSTR_MAX_GROUPS 组：长词更具体，优先采信。
    """
    out = {}
    q = (query or "").lower()
    if not q:
        return out
    n = len(_DOCS) or 1
    cands = []
    for w in _ALIAS:
        if len(w) < _SUBSTR_MIN or w not in q:
            continue
        if _ALIAS_MAX_DF > 0 and _DF_TOK.get(w, 0) > _ALIAS_MAX_DF * n:
            continue
        cands.append(w)
    cands.sort(key=len, reverse=True)
    seen = set()
    for w in cands:
        grp = _ALIAS.get(w)
        if not grp or grp in seen:
            continue
        if len(seen) >= _SUBSTR_MAX_GROUPS:
            break
        seen.add(grp)
        for x in grp:
            for v in _word_variants(x):
                out.setdefault(v, 1.0)
    return out


def _dice(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def _fuzzy_expand(q_tokens: list, min_dice: float = 0.67,
                  min_df: int = 3, max_cand: int = 2) -> dict:
    """罕见词模糊扩展：只对「库里几乎没有」的中文词动手，避免污染正常检索。

    典型救场：用户写「瑞祥」(df=1)，库里有「祥瑞」(df=17)，字符集完全一致 → 救回。
    只做长度差 ≤1、Dice ≥ 0.67 的匹配，宁缺毋滥。
    """
    out = {}
    pool = _cand_pool()
    if not pool:
        return out
    for t in q_tokens:
        if not _is_zh_word(t) or _DF_ALL.get(t, 0) > 2:
            continue  # 非中文，或这个词库里认识 → 不动
        ts = set(t)
        best = []
        for w in pool:
            if abs(len(w) - len(t)) > 1:
                continue
            d = _dice(ts, set(w))
            if d >= min_dice:
                best.append((d, _DF_ALL.get(w, 0), w))
        best.sort(key=lambda x: (-x[0], -x[1]))
        for d, _df, w in best[:max_cand]:
            if w not in out:
                out[w] = d * 0.6
    return out


_RARE_SCAN_MAX = 160   # 只扫描查询开头这些字符（后面通常是附加日志，很长且无意义）


def _nearest_known(s: str, min_dice: float = 0.6):
    """在已知词表里找与 s 等长的最相似词（字符集 Dice ≥ min_dice）。

    要求**等长**：错别字、语序颠倒（祥瑞↔瑞祥）、同音替换都是等长的，
    而「不出来」这类口语后缀与短词的长度不等，能天然被挡在门外，少误报。

    阈值 0.6 是量出来的，不是拍的：3 字词错一个字时 Dice = 2×2/(3+3) = 0.667，
    最早写 0.67 正好把「翔瑞卡 ↔ 祥瑞卡」这种最典型的错写卡在门外。
    """
    ss = set(s)
    cand = set()
    for ch in ss:
        cand |= _CHAR_IDX.get(ch, set())
    best = None
    for w in cand:
        if len(w) != len(s) or _DF_ALL.get(w, 0) < 2:
            continue
        d = _dice(ss, set(w))
        if d >= min_dice and (best is None or d > best[0]):
            best = (d, w)
    return best[1] if best else None


def _rare_rescue(query: str, expanded: dict) -> dict:
    """找「生僻疑似错写词」并顺手救援：把它的最近已知词加进召回的扩展词。

    为什么不靠分词结果找：jieba 遇到没见过的名字会切散 ——「翔瑞卡」被切成
    「翔 / 瑞卡」，整词根本不存在于 token 里。所以这里直接在**原始文本**上滑窗，
    对 2~4 字的每个片段问一句「库里认识它吗 / 长得像谁」。

    返回 {生僻词: 救援词}，同时也是「检索可能跑偏」的证据（见 _judge_weak）。
    """
    out = {}
    for run in re.findall(r"[\u4e00-\u9fff]+", query[:_RARE_SCAN_MAX]):
        n = len(run)
        # 只看 3~4 字片段：2 字片段里「单退/出来」这种巧合太多，误伤率远高于收益，
        # 2 字的错写交给别名表人工登记（那才是可靠的）。
        for L in (3, 4):
            for i in range(0, n - L + 1):
                s = run[i:i + L]
                if s in out or _DF_ALL.get(s, 0) > 1:
                    continue          # 库里见过这个词 → 不是生僻词
                w = _nearest_known(s)
                if not w or w in expanded or s in expanded:
                    continue          # 找不到像的，或已经被别名/模糊扩展救过
                out[s] = w
                expanded.setdefault(w, 0.8)   # 猜测性救援，权重低于人工登记的别名
                for x in (_ALIAS.get(w) or ()):   # 救援词若在别名组里，整组一起带上
                    expanded.setdefault(x, 0.8)
    return out


def _judge_weak(hits: list, expanded: dict, rare: dict, q_tokens: list = None) -> bool:
    """弱命中判据：救援过的生僻词最终在 top 结果里还是找不着落。

    返回 True = 检索大概没对上用户真正问的东西。上层据此插入一条低匹配度提示，
    让模型老实说没找到、而不是拿沾边的片段拼一个流程出来（那是编造，比答不出更糟）。

    ⚠️ 2026-10-21 补了第二条判据（词面覆盖率），原因：
       原判据要求 `rare` 非空 —— 也就是**必须先发生过生僻词救援**才会判弱。
       但实测更常见的一类失败是「中文词 df>0、语义却对不上」：
       `收银台点不动` 的 top1 只有 15.5 分、内容完全无关，`rare` 却是空的
       ⇒ 弱命中从不触发 ⇒ 模型拿沾边片段拼流程。
    """
    if not hits:
        return False
    top_text = "\n".join(
        (h.get("text") or "") + " " + " ".join(h.get("kw") or []) + " " + " ".join(h.get("alias") or [])
        for h in hits[:3]
    ).lower()
    # 判据①（原有）：救援过的生僻词在 top3 里找不到落点
    if rare:
        for s, w in rare.items():
            if s in top_text or (w and w in top_text):
                continue
            return True
    # 判据②（新增）：top3 连提问的中文实词都覆盖不到 _WEAK_COVER_MIN
    #   ⚠️ 只在中文实词 ≥2 个时才判，避免单词提问被误判；df>0 才纳入分母
    #   （df=0 的词已由 unknown_tokens 单独提示，不该重复惩罚）。
    #   ⚠️⚠️ 覆盖判定必须把**别名组**算进来，否则会误伤：
    #      「收银台点不动」的正解是「重启收银机」，文档里当然没有「点不动」这三个字，
    #      但它恰恰是**命中**。实测初版就把它误判成弱命中（top1 已 80.6 分），
    #      而弱命中提示会让模型拒绝回答 —— 这种误伤比漏判更糟。
    zh = [t for t in (q_tokens or [])
          if _ZH_RE.match(t) and len(t) >= 2 and _DF_TOK.get(t, 0) > 0]
    if len(zh) >= 2:
        def _covered(t: str) -> bool:
            if t in top_text:
                return True
            grp = _ALIAS.get(t) or _ALIAS.get(t.lower()) or ()
            return any(m in top_text for m in grp)
        if sum(1 for t in zh if _covered(t)) / len(zh) < _WEAK_COVER_MIN:
            return True
    return False


# 「查无此词」判定的参数
_UNK_MIN_LEN = 3          # 太短的 token（`号`/`的`）没有判别力
_UNK_KEEP = 8             # 最多回几个，别把提示语刷得太长
_WEAK_COVER_MIN = 0.34    # top3 对中文实词的覆盖率低于这个值 ⇒ 判弱命中


def unknown_tokens(query: str) -> list:
    """找出「查询里的关键词，整库 df=0」的那些 —— 即**库里根本没有这个对象**。

    为什么要这个：实测 deep 档最常见的浪费是**反复换写法搜一个库/源码里不存在的东西**。
    典型是错误码：`E-7731` 在知识库全库出现 **0 次**（它是渠道/银行透传的），
    模型却试了 `7731`/`E-7731`/`E-77`/`E-7`/`E7731`/"E-"/`E-` 共 8 种写法、21 次调用。
    每次换个词，BM25 都能返回"新片段"（其实是无关内容）⇒ 收敛判据永远不触发。

    这里直接把「这个词库里没有」摆给模型看，从**源头**掐掉"再换个写法试试"的冲动。

    ⚠️ 返回空列表 ≠ 有问题，也**不能**据此说"答不了" —— 中文近义改写很正常
    （用户问「结算报错」，库里写的是「结算失败」）。所以调用方只把它当**提示**，
     绝不当作"拒绝回答"的依据。

    ⚠️ 2026-10-21 补：**中文实词也纳入本提示**。
       原先只报"像专名"的（数字/错误码/长英文），于是**中文**的
       「库里根本没这东西」从不提示 —— 模型拿到一堆词面沾边的片段就敢拼流程。
       注意两点，避免误伤：
         · 这只是**提示**；硬拦判据仍是 `has_known_content`（它本来就认中文实词）；
         · `_DF_TOK == 0` 只说明"这个切分形式没在库里出现过"，不等于"库里没这个概念"
           （例如「隔日退」被 jieba 切成「隔日/退」，整词 df=0 但概念库里明明有）。
           所以中文词**单独收集、排在专名之后**，措辞上也不与专名混为一谈。
    """
    if not query or not _DF_TOK:
        return []
    seen, spec_miss, zh_miss = set(), [], []
    for t in _tokenize(query):
        if t in seen:
            continue
        is_spec = any(ch.isdigit() for ch in t) or bool(re.search(r"[a-z]{4,}", t))
        is_zh = bool(_ZH_RE.match(t)) and len(t) >= 2
        if not is_spec and not is_zh:
            continue
        if not is_zh and len(t) < _UNK_MIN_LEN:
            continue
        seen.add(t)
        if _DF_TOK.get(t, 0) == 0:
            (spec_miss if is_spec else zh_miss).append(t)
    return (spec_miss + zh_miss)[:_UNK_KEEP]


def _is_object_token(t: str) -> bool:
    """是「对象词」吗 —— 即能指代一个东西的词（错误码 / 编号 / 标识符 / 中文实词）。

    单字母（`e`）、标点（`-`）、单汉字（`号`）判别力太低，全部排除：
    否则 `E-7731` 会因为有 `e` 而"看起来有内容"，拦截立刻失效。
    """
    if len(t) < 2:
        return False
    if _ZH_RE.match(t):                       # 中文：2 字以上才算实词
        return len(t) >= 2
    # 非中文：含数字（错误码/编号）或够长的英文标识符（≥4 字母）
    return any(ch.isdigit() for ch in t) or bool(re.search(r"[a-z]{4,}", t))


def has_known_content(query: str) -> bool:
    """query 里是否**至少有一个**库里认识的对象词。

    用途：`unknown_tokens()` 告诉我们"哪些词库里没有"，但一条 query 常
    「既有未知词、也有正常词」（`E-7731 结算报错`），那还是该正常检索。
    只有**整条 query 全由查无此词的专名构成**（`E-7731` / `7731` / `E-77`）时，
    才说明模型在空转 —— 上层据此从第 2 次起直接拦截、不再返回检索片段
    （给了片段它就会以为"换个词还是有东西"，实测能空转到 8 轮上限）。
    """
    if not query or not _DF_TOK:
        return True                       # 判定不了就别拦
    objs = [t for t in _tokenize(query) if _is_object_token(t)]
    if not objs:
        return True                       # 没有可判别的对象词 → 不拦
    return any(_DF_TOK.get(t, 0) > 0 for t in objs)


# ---------------- 检索（BM25） ----------------

def search_ex(query: str, top_k: int = None):
    """检索并给出质量判据（**并发安全**：整体持 `_lock`）。

    返回 {"hits": [...], "weak": bool, "expanded": {词: 权重}}。
    hits: [{file, idx, text, kw, score}]，按相关度降序。

    ⚠️ 2026-09-25：改为「加锁包一层、原逻辑放内部函数」。
       原因：reload()/load() 会就地改 `_DOCS`/`_INDEX`，检索若不加锁，
       恰好在重建窗口内并发请求会读到长度不一致的两个列表 → IndexError（偶发 500）。
       代价是检索串行化 —— 实测单次仅在毫秒级，当前用户量下无感。
    """
    with _lock:
        return _search_ex_locked(query, top_k)


def _search_ex_locked(query: str, top_k: int = None):
    """实际检索逻辑（调用方必须已持 `_lock`）。"""
    top_k = top_k or config.TOP_K
    if not _DOCS or not query.strip():
        return {"hits": [], "weak": False, "expanded": {}}

    q_tokens = [t for t in _tokenize(query) if len(t) > 1 or any("\u4e00" <= c <= "\u9fff" for c in t)]
    # ⚠️ 2026-10-23：剔掉查询停用词（见 _QUERY_STOP 注释里的实测动机）。
    #   关键：**过滤后若一个词都不剩，就退回未过滤的结果** ——
    #   否则用户只问「怎么办？」这种全停用词的句子会变成空查询、什么都搜不到。
    _qt_nostop = [t for t in q_tokens if t.lower() not in _QUERY_STOP]
    if _qt_nostop:
        q_tokens = _qt_nostop
    if not q_tokens:
        q_tokens = _tokenize(query)

    # ①② 查询扩展：别名组（人工登记，权重 1.0）+ 罕见词模糊 + 生僻错写词字符级救援
    expanded = _alias_expand(q_tokens)
    # ②′ 整串子串匹配（2026-10-21 新增）：解决「口语被 jieba 切碎后命不中别名 key」。
    #    例：一线说「刷不了卡」，jieba 切成 ['刷','不了','卡']，没有一个是别名 key；
    #    但整串里含「刷不了卡」这个登记过的别名 ⇒ 用整串匹配把它接上。
    for w, wt in _alias_substr_expand(query).items():
        expanded.setdefault(w, wt)
    for w, wt in _fuzzy_expand(q_tokens).items():
        expanded.setdefault(w, wt)
    rare = _rare_rescue(query, expanded)

    qf = Counter(q_tokens)
    for w, wt in expanded.items():
        if w not in qf:
            qf[w] = wt  # 扩展词权重 < 1，命中时贡献低于用户原词

    n = len(_DOCS)
    # ⚠️ 2026-09-25 性能修复：df（含该词的文档数）**只依赖 token、与文档无关**，
    #    必须在这里按 token 算一次。原来的写法把它写在 (文档 × 查询词) 的双层循环内，
    #    于是每个文档都要把全部 1520 个 Counter 重新扫一遍 → 每次检索白跑百万级成员判断；
    #    而 search_kb 在一轮 /ask 里可能被调用多次。
    _df_cache = {tok: sum(1 for c in _INDEX if tok in c) for tok in qf}
    # ⚠️ 2026-10-26：区分「用户原词」与「别名扩展词」——扩展词单独限额累加，
    #    防止「别名清单块」靠几十个同义词叠加刷分（见 _EXP_TERM_CAP 的注释）。
    _user_toks = frozenset(q_tokens)
    scores = []
    for di, cnt in enumerate(_INDEX):
        score = 0.0
        exp_contribs = None          # 延迟创建：绝大多数块一个扩展词都不命中
        dl = sum(cnt.values())
        for tok, qn in qf.items():
            tf = cnt.get(tok, 0)
            if tf == 0:
                continue
            df = _df_cache[tok]
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * dl / _AVGDL))
            contrib = idf * tf_norm * (1.2 + qn)   # qn 让多词命中略有加成
            if tok in _user_toks:
                score += contrib
            else:
                if exp_contribs is None:
                    exp_contribs = []
                exp_contribs.append(contrib)
        if exp_contribs:
            if len(exp_contribs) > _EXP_TERM_CAP:
                exp_contribs.sort(reverse=True)
                del exp_contribs[_EXP_TERM_CAP:]
            score += sum(exp_contribs)
        if score > 0:
            if _DOCS[di].get("kw_only"):
                score *= 0.35   # 纯词表块是元数据，别让它把正文块挤下去
            # ⚠️ 2026-10-22 新增：**历史归档文档降权**（`KB_HIST_DOC_DECAY`，默认 0.55）。
            #   动机：给文档加了「🗄️ 历史文档」标记后，它们**照样与现行文档平权参与检索** ——
            #   实测「A8终端」的 top1 被一篇 2020 年的《系统管理员手册》以 28.79:28.33 微弱压过
            #   （而那篇已标历史归档）。标记只有配上打分衰减才真正起作用。
            #   衰减而非剔除：历史文档里仍有个别条目有效（如 A8 培训稿的「撤销必须整单」），
            #   真需要时它照样能进 top-k，只是不再抢 top1。
            if _DOCS[di].get("hist"):
                score *= _HIST_DECAY
            # ⚠️ 2026-10-22 新增：**标题命中加成**（`KB_TITLE_BOOST`，默认 0.5）。
            #   用**用户原词**（q_tokens，不含别名扩展词）去比标题词集合：
            #   命中比例越高，加成越大（全部命中 ⇒ ×1.5）。
            #   只加成、不衰减，所以不会把原本正确的文档压下去；系数上限 2.0 防过头。
            if _TITLE_BOOST > 0:
                _tt = _TITLE_TOKENS.get(_DOCS[di]["file"])
                if _tt:
                    _hit = sum(1 for _t in q_tokens if _t in _tt)
                    if _hit:
                        score *= 1.0 + _TITLE_BOOST * (_hit / len(q_tokens))
            # ⚠️ 2026-10-10 评审 ⑩（**默认关**，KB_API_DOC_DECAY=0.55 生效）：
            #   银行接口规范类文档里「签到/结算/退款」是**接口字段名**（词频高、文档短
            #   → BM25 爆分），实测「收银台不能签到」top3 全是银行接口文档，而
            #   FAQ/手册排 top6 之后 —— 对一线「怎么操作」类问题这是纯噪声。
            #   衰减而非剔除：真要查接口字段的人（问题里含「接口/报文/字段」类词）不受影响
            #   —— 交互式问题才衰减，接口类问题不衰减（下面按查询词动态判）。
            if _API_DOC_DECAY_ON and _API_DOC_RE.search(_DOCS[di]["file"]) and not _api_query(q_tokens):
                score *= _API_DOC_DECAY
            scores.append((score, di))
    scores.sort(key=lambda x: -x[0])
    picked, per_file = [], Counter()
    for s, di in scores:
        f = _DOCS[di]["file"]
        # 四类「不是知识的知识」，都不该进模型上下文：
        #   · 别名表 = 检索配置（它的词已经通过 _ALIAS 生效），本体只会抢答；
        #   · 纯关键词块 = 条目自己的检索标签，没有正文，答不了任何问题；
        #   · 纯截图块 = 只有 ![](assets/…)，模型看不到图，只会白占名额（见 _img_only）。
        if _ALIAS_FILE_RE.search(f) or _DOCS[di].get("kw_only") or _DOCS[di].get("img_only"):
            continue
        # 每篇文档最多占 _MAX_PER_FILE 个名额：字典类文档（《后台配置参数速查表》199 块、
        # 《运维FAQ全量》172 块）里全是短条目，条条高分，会直接把 top5 占满 ——
        # 症状是「明明有《打印配置故障案例》这种整篇讲这个问题的文档，却一条都进不来」。
        if per_file[f] >= _MAX_PER_FILE:
            continue
        per_file[f] += 1
        picked.append((s, di))
        if len(picked) >= top_k:
            break
    # 命中的文件本来就少时，上面那道上限会把名额空着丢掉 → 放开上限补满。
    if len(picked) < top_k:
        have = {di for _s, di in picked}
        for s, di in scores:
            if len(picked) >= top_k:
                break
            if di in have:
                continue
            f = _DOCS[di]["file"]
            if _ALIAS_FILE_RE.search(f) or _DOCS[di].get("kw_only") or _DOCS[di].get("img_only"):
                continue
            picked.append((s, di))
    hits = [
        {**_DOCS[di], "score": round(s, 4)}
        for s, di in picked
    ]
    hits = _expand_docs(hits)
    return {"hits": hits, "weak": _judge_weak(hits, expanded, rare, q_tokens),
            "expanded": expanded, "rare": rare}


_FILL_TOTAL_CHARS = 5200   # 整篇注入的总字数上限（多篇加起来不超过它）
_FILL_DOC_MAX = 4000       # 单篇正文超过这么多字 = 字典/FAQ 集合，不整篇注入
_FILL_MIN_AVG = 60         # 平均块长低于这个 = 字典型条目（《支付方式编码对照表》每块 23 字）
_FILL_MIN_RATIO = 0.7      # 候选文档的最高块分需 ≥ 这个比例 × 全场最高块分
_FILL_MAX_DOCS = 3         # 最多整篇注入几篇
_MAX_PER_FILE = 2          # 初步选块时，每篇文档最多占几个名额
# 「元文档」= 索引 / 导航 / 速查类条目。它们本身就是「指路牌」，不是答案本体，
# 整篇注入只会带进一堆别的功能名（实测问「打印配置在哪改」会把《源码仓库导航》
# 21 块全灌进来）。已命中的那几块仍然保留，只是不整篇展开。
# 2026-09-16 增补：**台账 / 清单类**文档（版本说明、更新迭代、应知应会、对照表）也算元文档。
# 起因（实测，主人当天报障「答非所问」）：报障卡里写着「POS版本：5.7.55」，检索命中
# 《POS更新迭代版本说明》，而它的标题块 `> 来源文档：POS更新迭代版本说明.docx` 字面分虚高，
# 相关度比值 0.98 轻松过闸一 → **整篇 27 段灌进上下文**（4999 字里的 27 段），
# 模型于是把「5.7.55/5.7.56 适配农行新大陆」「非收银台最后一版」「6.2.4 起仅小POS 支持平台收单」
# 「支付单号重用问题」当成这次故障的答案写了出去——而用户问的是一次工行聚合支付的失败。
# 这类文档和「速查表」一样是**查表用的台账**，只能按块命中，不能整篇展开。
# 2026-09-16 二次增补：**备案 / 资产索引**类也要拦。
# 同日踩到：往《编码规范与经验数据资产备案》里加了一行「工行聚合支付交易失败（RSP:02 / PAYERROR）→ 见《…》」的
# 交叉链接，那行关键词密度极高，直接把该文档顶到 78.3 分（top 是 107.6，ratio 0.728 刚过闸一），
# **整篇 15 段灌进上下文**，模型于是从备案里的「运维工具箱 / 腾讯云日志 / 修单工具」清单编出了一串无关步骤。
# 教训：① 索引/备案类文档一律不整篇注入；② 往索引里加交叉链接 = 制造高密度关键词块，能不加以不加
# （条目自己在被问到时已经能排 top1，不需要索引反向引流）。
_META_FILE_RE = re.compile(
    r"(导航|索引|速查|目录|术语别名|常见错写|版本说明|更新迭代|应知应会|版本对照|备案|资产清单)")


def _expand_docs(hits: list) -> list:
    """把「本条目里剩下的正文」补齐 —— 让回答有机会写出完整的步骤/参数/坑。

    背景（2026-09-14，主人报障「助手输出比 WorkBuddy 薄太多」）：
    检索是**块级**的，一段话自成一块。问「瑞祥在后台怎么配支付方式」时命中的是条目的
    「一句话结论」块，而真正写着「后台点击路径」「参数值格式」「⚠️ 常见坑」的十几个块
    因为与问题字面不像而全部落选 —— 于是答案只有一句结论、没有可执行步骤，
    可同一篇文档里其实写得清清楚楚（实测 top5 里还有两块是**纯截图占位**，白占名额）。

    规则（宁缺毋滥，三道闸）：
      · 候选文档 = 已命中块所属的文件，按「该文件最高块分」降序；
      · 闸一（相关度）：最高块分 ≥ _FILL_MIN_RATIO × 全场最高块分，否则**直接停止**
        （分数已按降序排，后面的只会更低）；
      · 闸二（体量）：该文档正文总量 ≤ _FILL_DOC_MAX，否则只保留已命中的块 ——
        《后台配置参数速查表》199 块 / 2.1 万字、《运维FAQ全量》172 块 / 2.2 万字，
        整篇灌进来只会把答案淹掉；
      · 闸三（体裁）：平均块长 ≥ _FILL_MIN_AVG。《支付方式编码对照表》全文才 927 字、
        却有 39 块（每块 23 字，一条一个编码），体量闸拦不住它，但它同样是字典 ——
        「文章」的块长明显更长（《祥瑞卡、杭州市民卡配置》均长 98、《打印配置故障案例》均长 304）。
    输出的顺序：全场最高分块仍是 [1]，其后是同文档的其余块（按原文顺序），再排其他文档 ——
    既有强信号也有完整上下文。
    """
    if not hits:
        return hits
    top_score = hits[0]["score"]
    best: dict = {}
    for h in hits:
        f = h["file"]
        if h["score"] > best.get(f, 0.0):
            best[f] = h["score"]
    order = sorted(best.items(), key=lambda x: -x[1])

    have = {(h["file"], h["idx"]) for h in hits}
    added, budget = [], _FILL_TOTAL_CHARS
    for f, sc in order[:_FILL_MAX_DOCS]:
        if sc < top_score * _FILL_MIN_RATIO:
            break
        if _META_FILE_RE.search(f):
            continue
        siblings = [d for d in _DOCS if d["file"] == f
                    and not d.get("kw_only") and not d.get("img_only")]
        if not siblings:
            continue
        total = sum(len(d["text"]) for d in siblings)
        if total > _FILL_DOC_MAX or total > budget \
                or total / len(siblings) < _FILL_MIN_AVG:
            continue                  # 字典/FAQ 集合：不整篇注入，保留已命中的块即可
        extra = [d for d in siblings if (d["file"], d["idx"]) not in have]
        if not extra:
            continue
        added.extend({**d, "score": round(sc * 0.99, 4), "sibling": True} for d in extra)
        budget -= total
    if not added:
        return hits

    head, rest = hits[0], hits[1:] + added
    by_file: dict = defaultdict(list)
    for h in rest:
        by_file[h["file"]].append(h)
    out = [head]
    for f, _sc in order:
        for h in sorted(by_file.get(f, []), key=lambda x: x["idx"]):
            if h["file"] == head["file"] and h["idx"] == head["idx"]:
                continue              # 最高分块已放在 [1]
            out.append(h)
    return out


def search(query: str, top_k: int = None):
    """兼容旧签名：只要命中列表。"""
    return search_ex(query, top_k)["hits"]

"""源码查询模块（codesearch）：为排障/定位类问题提供真实源码证据。

背景：LLM 在本链路内无法调用工具，因此本模块在服务端完成「代码线索词提取 →
离线代码索引检索 → 文件:行片段」三步，把结果注入 llm 上下文，让 AI 回答
「为什么 / 根因 / 在哪实现」类问题时引用真实源码，而不是只靠知识库文本。

设计：
- 索引 = 全量“有效代码行”（扁平 _LINES）+ 每文件跨度 _FILE_SPANS
        + 倒排 token → {file_idx: [行下标…]}
- build() 全量扫描 → pickle 落盘；load() 启动读盘（秒级）；
  源码变更后调 /reload 触发增量感知或 /code-rebuild 强制重建。
- 检索：由 question + BM25 命中片段提取代码线索词 → 倒排查行 → 行打分
  （命中线索越多、词越稀有得分越高）→ 每文件截断 → 拼进 LLM 上下文。
- 触发策略保守：只有像“排障/定位/查实现”的问题才查源码。

本机注意：源码在主人本机（/Users/mac/工作/示例企业/code），服务同机运行，
全量构建一次性（几十秒级），之后启动只读盘。
"""
import os
import pickle
import re
import threading
import time
from collections import Counter, defaultdict

from . import config

_lock = threading.Lock()

_LINES = []          # [(line_no, text), ...] 扁平有效行
_FILE_NAMES = []     # 文件相对路径列表（对齐 _FILE_SPANS）
_FILE_SPANS = []     # [(start, end_exclusive)] 每文件在 _LINES 中的区间
_POSTING = None      # {token: {file_idx: [行下标…]}}
_STATS = {"files": 0, "lines": 0, "tokens": 0, "built_at": 0, "dirs": []}

_STOPWORDS = {
    "pos", "the", "and", "for", "with", "this", "that", "from", "into",
    "http", "https", "www", "com", "cn", "api", "json", "string",
    "number", "boolean", "null", "undefined", "true", "false", "return",
    "function", "class", "interface", "export", "import", "const", "let", "var",
    "public", "private", "async", "await", "void", "new", "if", "else", "throw",
    "try", "catch", "then", "page", "index", "app", "key", "value", "data",
    "list", "type", "error", "msg", "code", "status", "date", "time", "name",
}

_MIN_LEN, _MAX_LEN = 14, 320
_SKIP_PREFIX = (" *", "/**", "*/", "<!--", "<!DOCTYPE", "// =", "//-", "/* ")

# os.walk 时直接剪枝的目录名：全部是依赖/构建产物，不含项目源码。
# ⚠️ 必须排除 node_modules —— 后台仓库（pos_ht/pos-account 等）里残留了前端工程，
# 一个 node_modules 就能带进 1.7 万个第三方 .js，把索引从 39 万行撑到 184 万行、文件 130MB。
_SKIP_DIRS = {
    "node_modules", ".git", ".idea", ".vscode", ".gradle", ".venv", "venv",
    "target", "build", "dist", "out", "bin", "coverage", "__pycache__",
    ".next", "vendor", "bower_components",
}


# ---------------- 索引构建 ----------------

def _line_ok(line: str) -> bool:
    s = line.strip()
    if not (_MIN_LEN <= len(s) <= _MAX_LEN):
        return False
    if s.startswith(_SKIP_PREFIX):
        return False
    # import / package / #include 等导入声明无检索价值，且会引入大量类名噪声 token
    # （Java 的 org.springframework.* 尤甚），直接丢弃。
    if s.startswith(("import ", "package ", "from ", "#include")):
        return False
    return True


def _tokens_of_line(line: str) -> list:
    """行内检索词：英文标识符(≥4) + 连续中文(2~10字)；去停用词。"""
    toks = set()
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", line):
        w = m.group(0)
        if len(w) >= 4 and w.lower() not in _STOPWORDS and not w.isdigit():
            toks.add(w.lower())
    for m in re.finditer(r"[\u4e00-\u9fff]{2,10}", line):
        toks.add(m.group(0))
    return list(toks)


def _candidate_dirs():
    if config.CODE_DIRS:
        return [d for d in config.CODE_DIRS if os.path.isdir(d)]
    return [os.path.join(config.CODE_ROOT, rel) for rel in config.CODE_AUTO_DIRS
            if os.path.isdir(os.path.join(config.CODE_ROOT, rel))]


def build(progress=None) -> dict:
    """全量扫描候选源码目录 → 构建倒排 → 写盘。"""
    global _LINES, _FILE_NAMES, _FILE_SPANS, _POSTING, _STATS
    dirs = _candidate_dirs()
    lines, fnames, spans = [], [], []
    t0 = time.time()
    for d in dirs:
        for root, subdirs, files in os.walk(d):
            subdirs[:] = [x for x in subdirs if x not in _SKIP_DIRS]   # 剪枝：跳过依赖/构建产物
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in config.CODE_EXTS:
                    continue
                fp = os.path.join(root, fn)
                try:
                    with open(fp, encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                start = len(lines)
                for no, raw in enumerate(text.split("\n"), 1):
                    if _line_ok(raw):
                        lines.append((no, raw.strip()))
                if len(lines) == start:
                    continue
                fnames.append(os.path.relpath(fp, config.CODE_ROOT))
                spans.append((start, len(lines)))
                if progress:
                    progress(f"  {os.path.relpath(fp, config.CODE_ROOT)}")
    posting = defaultdict(lambda: defaultdict(list))
    for li, (_, txt) in enumerate(lines):
        for tok in _tokens_of_line(txt):
            fid = _file_of(li, spans)
            posting[tok][fid].append(li)

    with _lock:
        _LINES[:] = lines
        _FILE_NAMES[:] = fnames
        _FILE_SPANS[:] = spans
        _POSTING = posting
        _STATS.update(files=len(fnames), lines=len(lines), tokens=len(posting),
                      built_at=int(t0),
                      dirs=[os.path.relpath(d, config.CODE_ROOT) for d in dirs])
        try:
            with open(config.CODE_INDEX_FILE, "wb") as f:
                pickle.dump({
                    "meta": dict(_STATS),
                    "file_names": fnames, "file_spans": spans,
                    "lines": lines,
                    "posting": {k: dict(v) for k, v in posting.items()},
                }, f, protocol=4)
        except Exception as e:
            print(f"[codesearch] 索引写盘失败: {e}")
    return dict(_STATS)


def load():
    """启动/重载：读盘优先，缺失则构建。"""
    global _LINES, _FILE_NAMES, _FILE_SPANS, _POSTING, _STATS
    if not config.CODE_ENABLED:
        return
    idx = config.CODE_INDEX_FILE
    if idx.exists():
        try:
            with open(idx, "rb") as f:
                data = pickle.load(f)
            with _lock:
                _FILE_NAMES[:] = data["file_names"]
                _FILE_SPANS[:] = data["file_spans"]
                _LINES[:] = data["lines"]
                _POSTING = {k: {int(fk): vl for fk, vl in v.items()}
                            for k, v in data["posting"].items()}
                _STATS.update(data["meta"])
            print(f"[codesearch] 索引已加载: {_STATS['files']} 文件 / "
                  f"{_STATS['lines']} 行 / {_STATS['tokens']} 词")
            return
        except Exception as e:
            print(f"[codesearch] 索引加载失败({e})，重建…")
    build()


def stats() -> dict:
    return dict(_STATS)


def count_lines() -> int:
    """已索引代码行数（健康检查/统计用）。"""
    return len(_LINES)


def _file_of(line_idx: int, spans: list = None) -> int:
    """行下标 → 文件下标（二分）。"""
    spans = spans if spans is not None else _FILE_SPANS
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][1] <= line_idx:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------- 检索 ----------------

def need_code(question: str, kb_hits: list) -> bool:
    """是否值得查源码：排障/定位/实现原理类问题才查。"""
    if not config.CODE_ENABLED:
        return False
    q = question.lower()
    intent = ("为什么", "原因", "根因", "排查", "定位", "哪里", "怎么", "如何",
              "报错", "异常", "失败", "无效", "修复", "处理", "实现", "逻辑",
              "代码", "源码", "哪段", "触发", "导致", "啥原因", "是什么",
              # 2026-09-14 扩充：门店一线常把问题写成「弹窗提示 X」「被拦截」这类字面描述，
              # 补上这批词，避免「明明有现象却一轮源码都不查」就回头要用户补充信息。
              "提示", "弹窗", "拦截", "校验", "无法", "不能", "卡住", "没反应", "不生效")
    if any(k in q for k in intent):
        return True
    if re.search(r"\d+\.\d+(\.\d+)?", question):  # 带 POS 版本号的报障
        return True
    if any(str(h.get("file", "")).endswith(("pos.md", "源码仓库导航")) for h in kb_hits):
        return True
    # 知识库一无所获时，也先试一轮源码，再考虑回头要用户补充（2026-09-14，「先搜再问」）
    if not kb_hits:
        return True
    return False


def extract_clues(question: str, kb_hits: list, max_clues: int = 10) -> list:
    """从问题 + 命中知识片段提取代码线索词。"""
    cands: Counter = Counter()

    def _feed(s: str, weight: int):
        # 蛇形标识符（printing_number / order_for_backend）
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+){1,}", s):
            w = m.group(0).lower()
            if len(w) >= 4:
                cands[w] += weight
        # 驼峰标识符（buildCopies / orderDetailsRePrint / payList）
        for m in re.finditer(r"[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*){2,}|[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+", s):
            w = m.group(0)
            if len(w) >= 5:
                cands[w.lower()] += weight
        # 点分配置 key（pos.print.printing_number）
        for m in re.finditer(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+){2,}", s):
            cands[m.group(0).lower()] += weight * 2
        # 接口路径尾段（refund/cashierRetry / recv-digital-transaction-upload-result）
        for m in re.finditer(r"(?:refund|pay|order|print|device|shop|store|sale|reprint|upload|search|digital|trans|voucher|batch)[A-Za-z0-9_/-]*",
                             s, re.I):
            w = m.group(0).lower()
            if len(w) >= 4:
                cands[w] += weight

    _feed(question, 3)
    for h in (kb_hits or [])[:5]:
        _feed(h.get("text", ""), 1)

    out, seen = [], set()
    for w, c in cands.most_common():
        if len(out) >= max_clues:
            break
        if len(w) < 4 or w in seen:
            continue
        if re.fullmatch(r"\d+", w):
            continue
        if any(x in w for x in ("example.com", "https", "http://", "admin")):
            continue
        seen.add(w)
        out.append(w)
    return out


_SEP_RE = re.compile(r"[^A-Za-z0-9_\u4e00-\u9fff]+")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _norm_clues(clues: list) -> tuple:
    """把检索词规范化成索引里的真实形态。

    ⚠️ 为什么必须有（2026-10-05 修，实测硬证据）：建索引时标识符被**统一小写**存
    （见 `_tokens_of_line` 的 `toks.add(w.lower())`），而模型会自然地传
    `PayMethodsKeyBoard` / `pos.pay.digital.force` / `checkCode` 这类"看起来对"的写法
    → **直接零命中**，它只能一次次换词试（实测一轮里 search_code 被调了 22 次）。

        实测对比：  PayMethodsKeyBoard → 0 条    paymethodskeyboard → 38 条
                    checkCode          → 0 条    checkcode          → 40 条
                    pos.pay.digital.force → 0 条   pos/pay/digital/force → 14 条

    返回 (primary, expanded)：
      primary  —— 与索引同口径的精确形态（整串小写 + 去分隔符连写），**先只用它**；
      expanded —— 再加「按分隔符切分」与「驼峰拆词」，**仅当 primary 零命中时**才用
                  （保精度优先，避免 pay/force 这类常用词把噪声带进来）。
    """
    primary, expanded, seen = [], [], set()

    def _add(lst, w):
        if w and w not in seen:
            seen.add(w)
            lst.append(w)

    for c in clues or []:
        raw = (c or "").strip()
        if not raw:
            continue
        low = raw.lower()
        _add(primary, low)
        _add(primary, _SEP_RE.sub("", low))          # pos.pay.digital.force → pospaydigitalforce
        for p in _SEP_RE.split(raw):                 # → pos / pay / digital / force
            _add(expanded, p.lower())
        if re.search(r"[a-z][A-Z]", raw):            # 驼峰：PayMethodsKeyBoard → pay/methods/keyboard
            for w in _CAMEL_RE.findall(raw):
                if len(w) >= 4:
                    _add(expanded, w.lower())
    return primary, expanded


def search(clues: list, top_n: int = None, per_file: int = None,
           path: str = None) -> list:
    """按线索词检索，返回 [{file, line, text, score}]。

    `path`：可选，只保留路径里含该子串的命中（用于把范围缩到某个仓库/模块）。
    ⚠️ 2026-10-05：线索词先经 `_norm_clues` 规范化 —— 见那里的实测证据。
    """
    primary, expanded = _norm_clues(clues)
    if not getattr(config, "CODE_NORM", True):     # 评测用开关：模拟旧行为（原样传词）
        primary, expanded = list(clues), []
    out = _search_once(primary, top_n, per_file, path)
    if out:
        return out
    return _search_once(expanded, top_n, per_file, path)


def _search_once(clues: list, top_n: int = None, per_file: int = None,
                 path: str = None) -> list:
    """单轮检索（内部）：不做线索词规范化，由 search() 负责。"""
    if not config.CODE_ENABLED or not _POSTING or not clues:
        return []
    top_n = top_n or config.CODE_MAX_HIT_LINES
    per_file = per_file or config.CODE_MAX_LINES_PER_FILE

    n_files = len(_FILE_NAMES) or 1
    line_score: Counter = Counter()
    for w in set(clues):
        post = _POSTING.get(w)
        if not post:
            continue
        df = sum(len(v) for v in post.values())
        if df > config.CODE_DF_NOISE:      # 太常见的词无区分度
            continue
        idf = 1.0 + 0.5 * n_files / max(df, 1)
        for fid, idxs in post.items():
            for li in idxs:
                line_score[li] += idf
    if not line_score:
        return []

    cand = [li for li, _ in line_score.most_common(top_n * per_file * 2)]
    if path:
        # 2026-10-05：路径过滤（把范围缩到某个仓库/模块，如 a8_new_ui / login-service）
        p = path.lower()
        cand = [li for li in cand if p in _FILE_NAMES[_file_of(li)].lower()]
        if not cand:
            return []
    by_file: dict = defaultdict(list)
    for li in cand:
        by_file[_file_of(li)].append(li)

    out = []
    for fid, lis in by_file.items():
        lis.sort(key=lambda li: -line_score[li])
        for li in lis[:per_file]:
            no, txt = _LINES[li]
            out.append({"file": _FILE_NAMES[fid], "line": no, "text": txt,
                        "score": round(float(line_score[li]), 3)})
    out.sort(key=lambda x: -x["score"])
    # 相对分数门槛：砍掉与最高分差距过大的低分命中（同为「噪声刷屏」的常见形态，
    # 例如同一接口在多个前端仓库的副本）。至少保留 1 条，避免全部被过滤。
    if out and config.CODE_SCORE_FLOOR_RATIO > 0:
        floor = out[0]["score"] * config.CODE_SCORE_FLOOR_RATIO
        kept = [x for x in out if x["score"] >= floor]
        out = kept or out[:1]
    # 同名文件跨仓库去重：同一个文件往往同时存在于 a8_new_ui / a8_5.7.x / a8_55x / elo / elo_cash
    # 等多个仓库副本，内容相近、分数也相近（相对门槛拦不住），会把命中名额占满
    # （典型症状：一次返回 40 行、清一色是 login-service）。每个文件名只保留代表分最高的副本，
    # 该副本内部的多个命中行仍然保留。
    if out:
        best_file: dict = {}          # 文件名 -> (文件路径, 分数)
        for x in out:
            name = os.path.basename(x["file"])
            cur = best_file.get(name)
            # out 已按分数降序，因此「先到者」即最高分；并列时也只留第一个
            if cur is None or x["score"] > cur[1]:
                best_file[name] = (x["file"], x["score"])
        keep = {v[0] for v in best_file.values()}
        out = [x for x in out if x["file"] in keep]
    return out[:top_n]


def _lines_of_file(fname: str) -> dict:
    """该文件在索引里的 {行号: 行文本}（⚠️ 是**过滤后**的有效行，行号可能不连续：
    注释/import/过短过长行在 `_line_ok` 里就被丢掉了 —— 取上下文必须按行号查表，不能做算术）。"""
    try:
        fid = _FILE_NAMES.index(fname)
    except (ValueError, AttributeError):
        return {}
    s, e = _FILE_SPANS[fid]
    return {no: txt for no, txt in _LINES[s:e]}


def merge_blocks(hits: list, gap: int = 3, ctx: int = 1) -> list:
    """把同一文件里相近的命中行合并成「带上下文的连续块」。

    ⚠️ 为什么要合并（2026-10-05）：旧的渲染是逐行 `file:line  text`，模型拿到的是**零散单行**，
    判断分支（if/else、switch）被拆开，只能靠猜 —— 这正是它反复换关键词重查的一部分原因。
    合并后给出连续行段 + 上下各 ctx 行，能直接读到整段逻辑。

    返回 [{file, line, end, score, text}]，text 为多行原文（行号前缀已加好）。
    """
    if not hits:
        return []
    by_file: dict = defaultdict(list)
    for h in hits:
        by_file[h["file"]].append(h)

    blocks = []
    for fname, hs in by_file.items():
        hs.sort(key=lambda x: x["line"])
        cur = None
        for h in hs:
            if cur is not None and h["line"] - cur["end"] <= gap:
                cur["end"] = max(cur["end"], h["line"])
                cur["score"] = max(cur["score"], h["score"])
            else:
                if cur is not None:
                    blocks.append(cur)
                cur = {"file": fname, "line": h["line"], "end": h["line"], "score": h["score"]}
        if cur is not None:
            blocks.append(cur)

    # 补上下文并渲染（按行号查表取原文）
    for b in blocks:
        tbl = _lines_of_file(b["file"])
        lo, hi = b["line"] - ctx, b["end"] + ctx
        seg = [(no, tbl[no]) for no in sorted(tbl) if lo <= no <= hi]
        b["text"] = "\n".join(f"{no:>6}| {txt}" for no, txt in seg) or "（原文不可取）"
    blocks.sort(key=lambda x: -x["score"])
    return blocks

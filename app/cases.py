"""案例审查 & 教练记录。

背景：助手目前只能覆盖一部分报障，多数最终要转人工。用户点「已解决/未解决」、
点「复制给人工」，都是**免费的信号**——但此前这些信号全丢在浏览器里没人看。
本模块把它们沉淀成案例库，让运维能在一个页面上判断：哪些案例可以入库、
哪些需要人工纠正，纠正后再由 AI 复用（见 llm.py 的 coach 注入）。

数据落两个 JSONL（追加写、按 id 去重取最后一条）：
  data/cases.jsonl  案例：一轮问答 + 用户结论 + 审查状态
  data/coach.jsonl  教练记录：人工写的「正确的问题点 / 正确的处理方式 / 给助手的线索」

为什么不用数据库：
  量级很小（一天几条到几十条），文件可直接打开看、可 git 管、坏一行不影响其它行；
  追加写天然免「读-改-写」，多线程下只需一把锁保证整行不交错。
"""
import json
import os
import re
import threading
import time
import uuid

from . import config, intent

_lock = threading.Lock()

_INTENTS = set(intent.INTENTS) - {"unknown"}

_VERDICTS = {"solved", "unsolved", "none"}
_STATUSES = {"new", "ready", "skip"}      # 审查结论：待定 / 建议入库 / 忽略


# ---------------- 有效回答判定（2026-10-09 加）----------------
# ⚠️ 为什么加：案例库的「已解决 / 未解决」是**调优唯一的一手信号**，但它会被"非答案"污染。
#   实测案例库里躺着这么一条：问「查手机号」、答「⚠️ 本机 AI 服务当前不在线（可能关机/断网/未启动）…」、
#   用户标「未解决」—— 那**不是回答质量问题，是服务离线**，用户当场只看到一句故障提示。
#   这种记录混进 solved/unsolved 比例，会把「可用性问题」误判成「回答质量问题」
#   （2026-10-09 我就差点据此去改提示词）。
#   ⇒ 判定为"非答案"的案例打 `no_answer: true`，**不计入有效样本**，但**保留原文**
#     —— 离线/报错的频率本身也是个值得盯的指标。
# ⚠️ 只认「**我们自己发的那几句故障提示**」的原文，**不要**用"超时 / 请求失败 / 服务异常"
#   这类通用词 —— 它们同样出现在**正经答案的正文里**（实测踩到：一条讲"结算超时"的好答案
#   被误判成 no_answer）。判据必须锚定到我们自己的文案，不能是业务词汇。
_NO_ANSWER_PAT = re.compile(
    r"本机 AI 服务当前不在线|服务不在线|AI 生成超过|服务内部错误|请稍后再试，或联系运维")
_MIN_ANSWER_CHARS = 20          # 短于这个长度不可能是有效答复


def is_no_answer(answer: str) -> bool:
    """这条"答案"是不是**根本没有答复**（服务离线 / 报错 / 空）。

    ⚠️ 判定刻意保守：只在**开头 120 字**出现故障提示词、或短得离谱时才判 True ——
       宁可漏判（当成正常案例），也不要误判（把一次真实回答踢出统计）。
    """
    a = (answer or "").strip()
    if len(a) < _MIN_ANSWER_CHARS:
        return True
    return bool(_NO_ANSWER_PAT.search(a[:120]))


# ---------------- 底层读写 ----------------

def _append(path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _load(path, limit: int = 2000) -> list[dict]:
    """读 JSONL：坏行跳过（一行坏不该让整个列表打不开），按 id 去重取最后一条。

    同一 id 允许被追加多次（先上报标记、后回写审查状态），后写的字段合并覆盖前面的，
    未提及的字段保留原值——这样回写状态时不必把整条案例重发一遍。
    """
    if not path.is_file():
        return []
    merged: dict[str, dict] = {}
    order: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                rid = str(rec.get("id") or "")
                if not rid:
                    continue
                if rid in merged:
                    merged[rid].update(rec)
                else:
                    merged[rid] = rec
                    order.append(rid)
    except OSError:
        return []
    out = [merged[r] for r in order]
    # 落盘时可能带毫秒（前端上报），读取时统一回秒，免得排序错乱、时间显示成 5 万年后
    for r in out:
        if r.get("ts"):
            r["ts"] = _to_sec(r["ts"])
            r["time"] = _fmt_time(r["ts"])   # 落盘的 time 可能已是错的（毫秒当秒算），以 ts 为准重算
        # ⚠️ `no_answer` **在读取时也算一遍**，不只在写入时打标：
        #    否则**历史记录永远没这个标记**（2026-10-09 实测：写完只打标新数据，
        #    老库里那条"服务不在线"仍被算进 unsolved，统计照样是错的）。
        #    以读取时的判定为准，写入时的标记只当缓存/留痕。
        # ⚠️ 只对**带 answer 字段**的记录判（coach 记录也走这个 _load，它们没有 answer，
        #    空答案会被判成 no_answer，那就张冠李戴了）。
        if "answer" in r:
            try:
                r["no_answer"] = is_no_answer(r.get("answer"))
            except Exception:                # noqa: BLE001
                r["no_answer"] = False
    out.sort(key=lambda r: r.get("ts") or 0, reverse=True)
    return out[:limit]


def _now() -> int:
    return int(time.time())


def _to_sec(ts) -> int:
    """统一成「秒」。

    ⚠️ 前端 Date.now() 是**毫秒**，不转换会算出 58672 年（曾真实踩到）。
    1e11 是个安全分界：秒级时间戳到 5138 年才会越界，毫秒级现在必然大于它。
    """
    try:
        t = float(ts)
    except (TypeError, ValueError):
        return _now()
    if t > 1e11:
        t /= 1000.0
    return int(t)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _fmt_time(ts: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _clip(s, n: int) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


# ---------------- 案例 ----------------

def upsert_case(rec: dict) -> dict:
    """上报或更新一个案例。

    调用方（前端）负责给稳定 id；只带 id + 少量字段也可（做增量更新），
    例如「点复制转人工」时只回写 verdict/via。
    """
    cid = str(rec.get("id") or "").strip() or _new_id()
    ts = _to_sec(rec["ts"]) if rec.get("ts") else _now()
    out = {"id": cid, "ts": ts, "time": _fmt_time(ts)}

    for k, limit in (("question", 6000), ("answer", 20000)):
        if k in rec:
            out[k] = _clip(rec[k], limit)
    # ⚠️ 非答案（服务离线 / 报错 / 空）单独打标：不进"答得好不好"的统计，但保留原文
    if "answer" in out:
        try:
            if is_no_answer(out.get("answer")):
                out["no_answer"] = True
        except Exception:                       # noqa: BLE001
            pass                                # 判定失败就当正常案例，别因为统计把上报弄丢
    if "verdict" in rec:
        v = str(rec.get("verdict") or "none").lower()
        out["verdict"] = v if v in _VERDICTS else "none"
    if "status" in rec:
        st = str(rec.get("status") or "new").lower()
        out["status"] = st if st in _STATUSES else "new"
    if "via" in rec:
        out["via"] = _clip(rec.get("via"), 24)
    # ⭐ 2026-10-12 使用统计：记录「这次是谁在用」（由 Worker 注入的身份，前端搬运）。
    if "user" in rec:
        out["user"] = _clip(rec.get("user"), 64)
    if "meta" in rec and isinstance(rec.get("meta"), dict):
        out["meta"] = {k: _clip(v, 200) for k, v in rec["meta"].items() if v}
    if "imgs" in rec and isinstance(rec.get("imgs"), list):
        # 只留本机图片 id（32 位 hex.jpg），防止把 base64 写进案例文件
        out["imgs"] = [str(i) for i in rec["imgs"]
                       if re.match(r"^[0-9a-f]{32}\.(jpg|png|webp|gif)$", str(i))]
    if "note" in rec:
        out["note"] = _clip(rec.get("note"), 1000)
    if "tags" in rec and isinstance(rec.get("tags"), list):
        out["tags"] = [_clip(t, 24) for t in rec["tags"] if str(t).strip()][:10]
    # ⭐ 2026-10-29：「仅改状态」的写入如果和现值相同，就**不再落盘**。
    #   实测主人连点 6 次「忽略」⇒ 文件里 6 行一样的 skip，既膨胀文件、也让"点击次数"失真。
    if set(out) <= {"id", "ts", "time", "status", "verdict", "via", "user", "note", "tags"}:
        cur = next((r for r in _load(config.CASES_FILE) if str(r.get("id") or "") == cid), None)
        if cur and all(cur.get(k) == out.get(k) for k in out if k not in ("ts", "time")):
            return cur
    _append(config.CASES_FILE, out)
    return out


def list_cases() -> list[dict]:
    """案例列表（按 id 合并后的）。

    ⭐ 每条附带 `intent`：按问题文本**规则推断**的适用场景，供审查页「适用场景」默认勾选
    （零成本、只在读的时候算，不落盘 —— 问题文本不会变，重算结果稳定）。
    """
    rows = _load(config.CASES_FILE)
    for r in rows:
        try:
            r["intent"] = intent.detect(r.get("question") or "")
        except Exception:                       # noqa: BLE001
            r["intent"] = "unknown"
    return rows


def count_cases() -> dict:
    """案例统计。

    ⚠️ 2026-10-09 两处修正（都是为了别把"信号"读错）：
      ① **非答案不计入结论统计**：服务离线 / 报错 / 空答案应记在 `no_answer` 里，
         不能混进 solved/unsolved —— 否则「离线」会被读成「答得不好」。
      ② **`unsolved` 按来源拆开**：前端点「复制给人工」时会把 verdict 直接设成 unsolved，
         但"我要转给同事看"≠"答案没解决我"。实测 8 条 unsolved 里 8 条都来自 copy ⇒
         这个混淆会让"未解决率"虚高。现在 `unsolved_click`（用户主动判）与
         `unsolved_copy`（点复制连带产生）分开计数，看趋势时以 **click** 为准。
    """
    rows = list_cases()
    stat = {"total": len(rows), "no_answer": 0, "effective": 0,
            "solved": 0, "unsolved": 0, "none": 0,
            "unsolved_click": 0, "unsolved_copy": 0,
            "ready": 0, "skip": 0, "new": 0, "coached": 0}
    for r in rows:
        if r.get("no_answer"):
            stat["no_answer"] += 1
            continue                            # 非答案：不进结论统计（仍留在 total 里）
        stat["effective"] += 1
        v = r.get("verdict") or "none"
        stat[v] = stat.get(v, 0) + 1
        if v == "unsolved":
            stat["unsolved_copy" if (r.get("via") == "copy") else "unsolved_click"] += 1
        st = r.get("status") or "new"
        stat[st] = stat.get(st, 0) + 1
    stat["coached"] = len({c.get("case_id") for c in list_coach() if c.get("case_id")})
    return stat


# ---------------- 使用统计（2026-10-12 主人要求：管理面板看最近 15 天）----------------

_USAGE_DAYS = int(os.getenv("KB_USAGE_DAYS", "15"))


def usage_stats(days: int = 0, users: list | None = None) -> dict:
    """最近 N 天（默认 15）的使用统计，供管理面板展示。

    口径说明（⚠️ 三个指标的含义要分清，否则会把数字读歪）：
      · `sessions`（事件处理）= 案例条数 —— 一条案例 = 一次问答（含追问各算一条）；
      · `users`（多少人用）= 案例里出现过的**不同 user 身份**数；
        ⚠️ 2026-10-12 之前的历史案例没有 user 字段，故 `unattributed` 单独报出来，
           不要把它当成"一个用户"；
      · `solved/unsolved/none` = 用户点了「已解决 / 未解决 / 没点」。与 count_cases 同口径：
        `no_answer`（服务离线/报错）不计入三者，单独报 `no_answer`。
    每日趋势按**本地日期**（Asia/Shanghai）切分。

    :param days: 统计天数；<=0 取 KB_USAGE_DAYS（默认 15）
    :param users: 用户表（供把 uid 映射成姓名）；不传则只返回 uid
    """
    n = int(days) if days and int(days) > 0 else _USAGE_DAYS
    now = _now()
    # 起始点取「今天 0 点往前 (n-1) 天」→ 含今天共 n 天
    lt = time.localtime(now)
    day0 = int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1)))
    start = day0 - (n - 1) * 86400

    uid2name = {}
    uid2staff = {}
    for u in (users or []):
        uid2name[u.get("id") or ""] = u.get("name") or ""
        # 2026-11-04 主人要求：管理页「用户」列改为显示用户自己填的 LDAP 账号（原员工编号），
        # 因为老邀请流程的名字是邀请备注/匿名名，管理员看不出是谁。
        uid2staff[u.get("id") or ""] = u.get("staff_id") or ""

    buckets: dict = {}          # "YYYY-MM-DD" -> 各计数
    for i in range(n):
        d = time.strftime("%Y-%m-%d", time.localtime(start + i * 86400))
        buckets[d] = {"date": d, "total": 0, "solved": 0, "unsolved": 0,
                      "none": 0, "no_answer": 0}

    seen_users: set = set()
    unattributed = 0
    tot = {"total": 0, "solved": 0, "unsolved": 0, "none": 0, "no_answer": 0,
           "effective": 0}
    for r in list_cases():
        ts = int(r.get("ts") or 0)
        if ts < start or ts > now + 86400:      # 窗口外（+1 天容忍时钟偏差）
            continue
        d = time.strftime("%Y-%m-%d", time.localtime(ts))
        b = buckets.get(d)
        if b is None:
            continue
        u = (r.get("user") or "").strip()
        if u:
            seen_users.add(u)
        else:
            unattributed += 1

        b["total"] += 1
        tot["total"] += 1
        if r.get("no_answer"):
            b["no_answer"] += 1
            tot["no_answer"] += 1
            continue
        tot["effective"] += 1
        v = r.get("verdict") or "none"
        if v not in ("solved", "unsolved"):
            v = "none"
        b[v] = b.get(v, 0) + 1
        tot[v] = tot.get(v, 0) + 1

    # 用户明细（谁用得最多）—— 按案例数降序
    per_user: dict = {}
    for r in list_cases():
        ts = int(r.get("ts") or 0)
        if ts < start or ts > now + 86400:
            continue
        u = (r.get("user") or "").strip()
        if not u:
            continue
        e = per_user.setdefault(u, {"uid": u, "name": uid2name.get(u, ""),
                                    "staff_id": uid2staff.get(u, ""), "count": 0,
                                    "solved": 0, "unsolved": 0, "last_ts": 0})
        e["count"] += 1
        if r.get("verdict") == "solved":
            e["solved"] += 1
        elif r.get("verdict") == "unsolved":
            e["unsolved"] += 1
        e["last_ts"] = max(e["last_ts"], ts)
    users_out = sorted(per_user.values(), key=lambda x: -x["count"])
    for e in users_out:
        e["last_used"] = _fmt_time(e["last_ts"]) if e["last_ts"] else ""

    return {
        "days": n,
        "start": time.strftime("%Y-%m-%d", time.localtime(start)),
        "end": time.strftime("%Y-%m-%d", time.localtime(now)),
        "summary": {
            "users": len(seen_users),          # 多少人用（去重身份）
            "sessions": tot["total"],          # 事件处理数（问答条数）
            "solved": tot["solved"],
            "unsolved": tot["unsolved"],
            "none": tot["none"],
            "no_answer": tot["no_answer"],
            "effective": tot["effective"],
            "unattributed": unattributed,      # 无身份的历史案例（不算进 users）
        },
        "daily": [buckets[k] for k in sorted(buckets)],
        "users": users_out,
    }


# ---------------- 教练记录 ----------------

def add_coach(rec: dict) -> dict:
    """人工纠正：这个问题真正的问题点在哪、正确该怎么处理。

    这些内容会（在相似问题再次出现时）**优先于知识库**注入给模型，
    相当于「现场手把手教一次，之后同类问题都照这个来」。
    """
    ts = _now()
    out = {
        "id": _new_id(), "ts": ts, "time": _fmt_time(ts),
        "case_id": _clip(rec.get("case_id"), 40),
        "question": _clip(rec.get("question"), 6000),
        "raw": _clip(rec.get("raw"), 8000),          # ⭐ 单框模式：整段教练内容（2026-10-29）
        "point": _clip(rec.get("point"), 2000),      # 正确的问题点（真正的根因方向）
        "howto": _clip(rec.get("howto"), 6000),      # 正确的处理方式
        "hint": _clip(rec.get("hint"), 2000),        # 给助手的线索：该看什么日志/源码/字段
        "tags": [_clip(t, 24) for t in (rec.get("tags") or []) if str(t).strip()][:10],
        "active": True,
    }
    # ⭐ 2026-10-29「适用场景」：声明了场景的教练按**意图精确匹配**（不再靠字面相似度）；
    #   没传或传空就自动推断（猜不出留 unknown ⇒ 退回原来的字面匹配，不会乱串）。
    intents = [str(i).strip() for i in (rec.get("intents") or [])
               if str(i).strip() in _INTENTS and str(i).strip() != "unknown"]
    if not intents:
        guess = intent.infer_text(out.get("question") or "", out.get("raw") or "")
        if guess != "unknown":
            intents = [guess]
    out["intents"] = intents[:4]
    if not (out["raw"] or out["point"] or out["howto"] or out["hint"]):
        raise ValueError("教练记录至少要有「教练内容」（或旧三段的任一段）")
    _append(config.COACH_FILE, out)
    # 案例侧留痕，审查页一眼能看出这条已经调教过。
    # ⚠️ 只写 note，不动 verdict：用户明确标过「已解决」的话，不该因为事后补了条
    # 教练记录就被翻成「未解决」——结论是用户的判断，系统不做反悔。
    if out["case_id"]:
        upsert_case({"id": out["case_id"],
                     "note": f"已教练（{out['time']}）：{_clip(out['point'] or out['howto'], 120)}"})
    return out


def set_coach_active(cid: str, active: bool) -> dict:
    """停用/启用一条教练记录（停用后不再注入，但保留存档）。

    ⚠️ 不写 ts：_load 按 ts 排序，回写时带上新 ts 会让这条记录跳到列表最前，
    把「最近调教过的」顺序打乱。
    """
    rec = {"id": str(cid), "active": bool(active)}
    _append(config.COACH_FILE, rec)
    return rec


def list_coach(active_only: bool = False) -> list[dict]:
    rows = _load(config.COACH_FILE)
    return [r for r in rows if r.get("active", True)] if active_only else rows


def delete_coach(cid: str) -> dict:
    return set_coach_active(cid, False)


def delete_case(cid: str) -> dict:
    """**物理删除**一条案例（同 id 的多行追加记录一并删），连带它的教练记录。

    ⚠️ 为什么要真删而不是打 `active:false` 墓碑：
       案例库是「调优样本库」，杂七杂八的提问留着会污染统计口径、也碍眼；
       主人明确要求「有些杂七杂八的问题没必要一直留着」。
    ⚠️ 但真删有代价，所以**删之前把原行追加到 `cases_deleted.jsonl`**：
       误删还能捞回来（一条 jsonl 行，成本可忽略）。
    返回 {"removed": 删掉的案例行数, "coach": 删掉的教练行数}。
    """
    cid = str(cid or "").strip()
    if not cid:
        raise ValueError("id 不能为空")

    def _split(path, hit) -> tuple[list[str], list[str]]:
        keep, gone = [], []
        if not path.is_file():
            return keep, gone
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except Exception:                 # 坏行原样保留，绝不因为一个坏行连带删数据
                keep.append(raw)
                continue
            (gone if hit(rec) else keep).append(raw)
        return keep, gone

    with _lock:
        keep, gone = _split(config.CASES_FILE, lambda r: str(r.get("id") or "") == cid)
        ckeep, cgone = _split(config.COACH_FILE, lambda r: str(r.get("case_id") or "") == cid)
        if not gone:
            return {"removed": 0, "coach": 0}
        # ① 先归档（追加到 cases_deleted.jsonl），② 再原子替换，顺序不能反
        # ⚠️ 这里**不能**调 _append：它内部再取 _lock，而当前线程已持锁 ⇒ 自锁死。
        arch = config.CASES_FILE.with_name("cases_deleted.jsonl")
        arch.parent.mkdir(parents=True, exist_ok=True)
        with arch.open("a", encoding="utf-8") as f:
            for raw in gone:
                f.write(json.dumps({"archived": raw}, ensure_ascii=False) + "\n")
            for raw in cgone:
                f.write(json.dumps({"archived_coach": raw}, ensure_ascii=False) + "\n")
        for path, lines in ((config.CASES_FILE, keep), (config.COACH_FILE, ckeep)):
            cur = ([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
                   if path.is_file() else [])
            if lines == cur:
                continue                      # 没变化就别重写
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            os.replace(tmp, path)             # 原子替换：中途崩了也不会留半个文件
    return {"removed": len(gone), "coach": len(cgone)}


# ---------------- 教练匹配（决定「这次提问要不要带上人工纠正」）----------------
# 不引入分词依赖：中文按 2-gram 做 Jaccard，短文本上足够稳；
# 关键词/短语级重合（如「赠券 退货 券资产不足」）能被明显拉高分数。

def _grams(s: str) -> set:
    s = re.sub(r"\s+", "", str(s or ""))
    s = re.sub(r"[，。、；：？！,.;:?!（）()\[\]【】\-—_/\\]+", "", s)
    # 数字归一化：手机号 / 订单号是**标识符**，不是语义。
    # 不归一化时「11 位手机号 + 的所有订单」与「另一个手机号 + 的所有订单」相似度只有 0.160
    # （低于 0.18 门槛 ⇒ 同一条教练换个号码就失效）。归一后两者完全一致。
    s = re.sub(r"\d+", "#", s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _sim(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def match_coach(question: str, top_k: int | None = None,
                min_score: float | None = None,
                intent: str | None = None) -> list[dict]:
    """挑出与本次提问最像的人工纠正记录。

    打分取「原始问题」与「正确问题点」的较大值：用户复述同一个故障时措辞会变，
    但人工写下的问题点往往更接近本质，两个入口任一命中都应该带上。

    ⭐ 2026-10-29 加「场景路由」（intent 参数）：
      - 教练**声明了 intents** 且本次意图命中 → 直接给 1.0（确定性，不押字面）；
      - 教练声明了 intents 但意图**不符** → **直接跳过**，不让它在无关问题上串味；
      - 老教练（没声明 intents）或本次意图 unknown → 仍走原来的 2-gram 字面匹配，行为不变。
    """
    q = str(question or "")
    if len(q.strip()) < 4:
        return []
    top_k = top_k or config.COACH_TOP_K
    min_score = config.COACH_MIN_SCORE if min_score is None else min_score
    scored = []
    for r in list_coach(active_only=True):
        declared = [i for i in (r.get("intents") or []) if i and i != "unknown"]
        if intent and intent != "unknown" and declared:
            if intent in declared:
                scored.append((1.0, r))         # 场景精确命中 ⇒ 满分级
            continue                            # 场景不符 ⇒ 排除
        s = max(_sim(q, r.get("question")), _sim(q, r.get("point")) * 0.9)
        if s >= min_score:
            scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, r in scored[:top_k]:
        out.append({**r, "score": round(s, 3)})
    return out

"""会话存储：把一段报障对话落到本机磁盘，换台电脑凭「会话 ID」继续排查。

⚠️ 轻量定位（2026-09-22 主人定）：这是**轻量级、实时性**助手 ——
   **不做归档、不做冷备**，到 7 天即硬删；`GET /session/{sid}` 拉不到，
   就如实回「会话不存在或已过期」，**不要引导用户去找回**。

⚠️⚠️ 删除是本模块唯一危险的动作，硬约束（改代码时不许放宽）：
  1. **只允许删 `data/sessions/` 与 `data/uploads/` 目录内的文件** ——
     路径 `resolve()` 后必须确认父目录是这两者之一，否则**拒绝 + 记 REFUSE 日志**。
  2. **绝不触碰** `data/kb/`（知识库 185M）、`data/code_index.pkl`、
     `data/diagrams/`、`cases.jsonl`、`coach.jsonl` —— 它们都不在白名单内。
  3. 每次删除**先写 `data/gc.log` 再删**，可追溯。
  4. `KB_GC_DRY_RUN=1` 时只列清单、不真删。
"""
from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path

from . import config

# Crockford Base32：去掉 i / l / o / u —— 电话里念或手抄时不容易看错
_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"
SID_LEN = 14
SID_RE = re.compile(r"^pos-[0-9a-z]{14}$")

NOT_FOUND_MSG = f"会话不存在或已过期（会话仅保留 {config.SESSION_TTL_DAYS} 天）"


def new_sid() -> str:
    return "pos-" + "".join(secrets.choice(_ALPHABET) for _ in range(SID_LEN))


def valid(sid: str) -> bool:
    return bool(SID_RE.match(sid or ""))


def path_of(sid: str) -> Path | None:
    """sid → 文件路径。⚠️ 正则不通过一律返回 None，防目录穿越。"""
    return (config.SESSION_DIR / f"{sid}.json") if valid(sid) else None


# ---------------- 读写 ----------------

def _trim(messages: list, limit_bytes: int) -> tuple[list, int]:
    """超上限时**从最旧的消息开始丢**（用户最近说的才是重点），并留一条截断说明。"""
    kept = list(messages)
    dropped = 0
    while kept and len(json.dumps(kept, ensure_ascii=False).encode("utf-8")) > limit_bytes:
        kept.pop(0)
        dropped += 1
    return kept, dropped


def save(sid: str, payload: dict) -> dict:
    """新建或覆盖写入会话。返回体积与消息条数，便于前端/日志观察。"""
    p = path_of(sid)
    if not p:
        return {"ok": False, "error": "invalid_sid"}
    config.SESSION_DIR.mkdir(parents=True, exist_ok=True)

    old = load(sid) or {}
    now = int(time.time())
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        msgs = old.get("messages") or []
    msgs, dropped = _trim(msgs, config.SESSION_MAX_KB * 1024)
    if dropped:
        msgs.insert(0, {"role": "assistant", "time": "", "ts": 0, "imgs": [],
                        "content": f"（更早的 {dropped} 条内容已因单会话体积上限被截断）"})

    doc = {
        "sid": sid,
        "created_at": old.get("created_at") or now,
        "last_active_at": now,
        "title": (str(payload.get("title") or old.get("title") or ""))[:80],
        "messages": msgs,
    }
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)                      # 原子替换：避免读到半截 JSON
    return {"ok": True, "sid": sid, "created_at": doc["created_at"],
            "bytes": p.stat().st_size, "messages": len(msgs), "trimmed": dropped}


def load(sid: str) -> dict | None:
    p = path_of(sid)
    if not p or not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


# ---------------- 清理 ----------------

def _log(lines: list) -> None:
    if not lines:
        return
    config.SESSION_GC_LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with config.SESSION_GC_LOG.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(f"{stamp} {ln}\n")


def _safe_unlink(p: Path, dry: bool) -> str | None:
    """⚠️ 唯一的删除出口：**只允许删 SESSION_DIR / UPLOAD_DIR / DIAGRAM_DIR 内的文件**。

    ⚠️ 2026-11-02 补上 DIAGRAM_DIR：原先白名单漏了它，而 gc 里又确实在删简图，
       ⇒ 每次都被判「不在白名单目录内，拒绝删除」⇒ **简图一直只增不减**（静默失效）。
    """
    try:
        rp = p.resolve()
    except Exception:
        return None
    allowed = {config.SESSION_DIR.resolve(), config.UPLOAD_DIR.resolve(),
               config.DIAGRAM_DIR.resolve()}
    if rp.parent not in allowed:
        _log([f"REFUSE  不在白名单目录内，拒绝删除: {rp}"])
        return None
    if dry:
        return f"DRY     would delete: {rp.name}"
    try:
        rp.unlink()
        return f"DELETE  {rp}"
    except FileNotFoundError:
        return None
    except Exception as e:
        _log([f"ERROR   删除失败 {rp}: {e}"])
        return None


def _referenced_uploads(alive_sids: list[str]) -> set:
    """收集「仍在保留期内」的会话引用到的图片文件名。

    ⚠️ 必须先算引用再删图：否则会把还能拉回来的会话里的截图删掉，
       用户重新打开会话看到的就是裂图（这比多占点磁盘严重得多）。
    """
    ref = set()
    for sid in alive_sids:
        doc = load(sid) or {}
        for m in doc.get("messages") or []:
            if not isinstance(m, dict):
                continue
            for im in (m.get("imgs") or []):
                if not isinstance(im, dict):
                    continue
                # 图片 id 就是完整文件名（uuid.hex + 扩展名）；url 作兜底
                for k in ("id", "url"):
                    v = im.get(k)
                    if v:
                        ref.add(Path(str(v)).name)
    return ref


def gc(dry_run: bool | None = None) -> dict:
    """清理过期会话与无主图片。默认每天由后台任务调用，也可 `POST /gc` 手动触发。"""
    dry = config.GC_DRY_RUN if dry_run is None else bool(dry_run)
    ttl = config.SESSION_TTL_DAYS * 86400
    # 图片用**独立**保留期（默认 15 天），比会话的 7 天长：
    # 截图常被「复制给人工」带出去，运维几天后才回头核对，会话到期了图还在更实用。
    up_ttl = config.UPLOAD_TTL_DAYS * 86400
    now = time.time()
    lines: list[str] = []
    if dry:
        lines.append("MODE    DRY-RUN（只列清单，不删除）")

    expired, alive = [], []
    if config.SESSION_DIR.is_dir():
        for f in config.SESSION_DIR.glob("*.json"):
            doc = load(f.stem) or {}
            last = doc.get("last_active_at") or f.stat().st_mtime
            try:
                old = (now - float(last)) > ttl
            except Exception:
                old = False
            (expired if old else alive).append(f.stem)

    for sid in expired:
        p = path_of(sid)
        if p:
            r = _safe_unlink(p, dry)
            if r:
                lines.append(f"SESSION {r}")

    ref = _referenced_uploads(alive)

    n_up = 0
    if config.UPLOAD_DIR.is_dir():
        for f in config.UPLOAD_DIR.iterdir():
            if not f.is_file():
                continue
            if f.name in ref:
                continue                       # 还有会话在用，绝不删
            try:
                if (now - f.stat().st_mtime) <= up_ttl:
                    continue
            except Exception:
                continue
            r = _safe_unlink(f, dry)
            if r:
                lines.append(f"UPLOAD  {r}")
                n_up += 1

    # ── 简图（data/diagrams/*.svg）──────────────────────────────────────────
    # ⭐ 2026-10-12（第五轮评审）：原先**只增不减** —— 上传图片与日志都有清理，
    #    唯独简图没有。虽然它是**内容寻址**（sha1 前 16 位）⇒ 相同图只存一份、
    #    增长不会很猛（线上 31 个文件 / 128K），但毕竟是**无界增长**。
    #    ⚠️ 安全性：简图是**纯自包含 SVG**（本地模板渲染，无外链、无引用计数需求），
    #       与「会话还在用」无关 —— 删了之后前端 <img> 只是裂图，且用户重新提问会
    #       按同样内容重新生成同名文件（内容寻址 ⇒ 名字相同 ⇒ 自然"复活"）。
    #       所以这里**不需要**像 uploads 那样先算引用，只看 TTL 即可。
    n_dg = 0
    try:
        if config.DIAGRAM_DIR.is_dir():
            for f in config.DIAGRAM_DIR.iterdir():
                if not f.is_file():
                    continue
                try:
                    if (now - f.stat().st_mtime) <= ttl:
                        continue
                except Exception:                  # noqa: BLE001
                    continue
                r = _safe_unlink(f, dry)
                if r:
                    lines.append(f"DIAGRAM {r}")
                    n_dg += 1
    except Exception as e:                         # noqa: BLE001
        lines.append(f"DIAGRAM 清理异常（忽略）：{type(e).__name__}")

    _log(lines)
    return {"ok": True, "dry_run": dry,
            "sessions_total": len(expired) + len(alive),
            "sessions_expired": len(expired),
            "sessions_alive": len(alive),
            "uploads_expired": n_up,
            "diagrams_expired": n_dg,
            "referenced_uploads": len(ref),
            "ttl_days": config.SESSION_TTL_DAYS,
            "upload_ttl_days": config.UPLOAD_TTL_DAYS,
            "log": lines}

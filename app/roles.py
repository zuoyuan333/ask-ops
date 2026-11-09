"""角色定义（按角色决定回答形态）—— 配置化模块，2026-11-09

为什么要单独抽出来：
    原来三套角色骨架**硬编码在 llm.py**，auth.py 又单独抄了一份角色常量 ——
    改一个角色要动两处、十几行文案；更麻烦的是**开源分发时会连私有部署的业务角色一起带出去**
    （零售业的「门店 / 收银 / 项目IT」对别人毫无意义，还泄露了组织形态）。

现在：
    · 角色元信息 + 骨架文案全部在 `source/data/roles/`（`roles.json` + `<key>.md`）
    · 代码里只留**一份实现**，llm.py 与 auth.py 都从这里取，不再各抄一份
    · **同一套代码、两份数据**：开源版放通用角色，私有版放自己的角色

目录约定（`source/data/roles/`）：
    roles.json   {"default": "<key>", "roles": [{key, label, aliases, file}, ...]}
    <key>.md     该角色的回答骨架，纯文本，原样拼进 user message 的最前面

⚠️ 读不到 / 格式错 → 回落内置的最小默认（**保证服务一定起得来**，不因配置问题白屏）
⚠️ 启动时加载一次并缓存；改完要重启（或调用 `reload()`）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# roles 目录：默认 source/data/roles（BASE_DIR = source/），可用环境变量覆盖
_DIR = Path(os.getenv(
    "KB_ROLES_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "roles"),
))

# 内置兜底：**只有一个通用角色、无任何骨架文案** ⇒ 等同于「没做角色分级」的老行为。
# ⚠️ 刻意不带任何行业词：万一配置丢了，开源用户也不能被塞进某种业务语境。
_FALLBACK = {
    "default": "general",
    "roles": [
        {"key": "general", "label": "通用", "aliases": ["通用", "general"], "file": ""},
    ],
}

_cache: dict | None = None


def _read_json(p: Path) -> dict | None:
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    roles = doc.get("roles")
    if not isinstance(roles, list) or not roles:
        return None
    # 只保留有 key 的条目；缺 label/file 的补默认值，不让一条脏数据毁掉整份配置
    clean = []
    for r in roles:
        if not isinstance(r, dict):
            continue
        k = str(r.get("key") or "").strip()
        if not k:
            continue
        clean.append({
            "key": k,
            "label": str(r.get("label") or k),
            "aliases": [str(a).strip().lower() for a in (r.get("aliases") or []) if str(a).strip()],
            "file": str(r.get("file") or "").strip(),
        })
    if not clean:
        return None
    default = str(doc.get("default") or "").strip()
    if default not in {c["key"] for c in clean}:
        default = clean[-1]["key"]          # 配错就把最后一个当默认（通常是「一线」）
    return {"default": default, "roles": clean}


def _load() -> dict:
    """加载配置（带缓存）。任何异常都回落 _FALLBACK。"""
    global _cache
    if _cache is not None:
        return _cache
    doc = _read_json(_DIR / "roles.json")
    _cache = doc or _FALLBACK
    return _cache


def reload() -> dict:
    """丢掉缓存重新读（改完 roles/ 后调用；不调就沿用进程启动时的快照）。"""
    global _cache
    _cache = None
    return _load()


# ---------------- 对外 API ----------------

def keys() -> list[str]:
    return [r["key"] for r in _load()["roles"]]


def default_key() -> str:
    return _load()["default"]


def labels() -> dict[str, str]:
    return {r["key"]: r["label"] for r in _load()["roles"]}


def label(key: str | None) -> str:
    return labels().get(norm(key), "")


def norm(raw: str | None) -> str:
    """把任意输入归一成配置里的某个 key；不认识的一律落到 default。

    支持三种写法：key 本身（hq_ops）、label（总部运维）、配置里的 aliases（总部/ops）。
    """
    doc = _load()
    valid = {r["key"] for r in doc["roles"]}
    v = (raw or "").strip()
    if v in valid:
        return v
    low = v.lower()
    for r in doc["roles"]:
        if r["label"].lower() == low or low in r["aliases"]:
            return r["key"]
    return doc["default"]


def prompt(key: str | None) -> str:
    """取该角色的回答骨架。找不到文件/没配 file → 返回空串（= 不加角色约束）。"""
    k = norm(key)
    for r in _load()["roles"]:
        if r["key"] != k:
            continue
        f = r.get("file") or ""
        if not f:
            return ""
        try:
            return (_DIR / f).read_text(encoding="utf-8").strip()
        except Exception:
            # 配置了文件但读不到：宁可退回「无骨架」，也不让整个问答 500
            print(f"[roles] 读不到骨架文件：{_DIR / f}", flush=True)
            return ""
    return ""

"""「👎 未解决」反馈的自动摘要 + webhook 推送（2026-09-27 主人要求）。

背景：用户对助手的回答不满意、点「👎 未解决」时，前端本来就会把这条案例
POST /feedback（含 question / answer / verdict / meta）。这里在案例落盘之后
再补一步：**生成一段对话摘要 → 推送到 rwork 机器人**，让主人第一时间知道
「助手这次没答好、用户问的是什么、助手答的是什么」。

四条纪律（延续项目一贯做法，别改）：
1. **不阻塞用户**：摘要 + 推送全在守护线程里做，/feedback 立刻返回。
   用户点按钮这件事本身，不能被推送耗时拖慢、更不能被推送失败影响。
2. **同一案例只推一次**：按 case id 去重（用户取消再点不该重复打扰）。
   进程内记录，重启丢失 —— 可接受（最坏是偶尔多推一条）。
3. **摘要 LLM 生成、失败必降级**：llm.summarize 返回空串（模型挂了 / echo 模式）时，
   退回「用户问 X；助手答 Y」的规则摘要。摘要只是辅助信息，
   案例本身已经落盘，**绝不因为摘要失败就不推**。
4. **webhook 复用 queue 的卡片通道**：同一个 rwork 机器人、同一条
   `trust_env=False` 纪律（环境代理坑见 queue.py 注释）。
"""
import threading

from . import llm, queue

CARD_TITLE = "👎 助手未解决反馈"

_pushed: set = set()          # 已推送过的 case id（进程内去重）
_lock = threading.Lock()


def notify_unsolved(case) -> None:
    """未解决案例落盘后调用。异步执行，任何异常都不向上抛。"""
    cid = str(getattr(case, "id", "") or "").strip()
    if not cid:
        return
    with _lock:
        if cid in _pushed:
            return
        _pushed.add(cid)

    question = str(getattr(case, "question", "") or "").strip()
    answer = str(getattr(case, "answer", "") or "").strip()
    via = str(getattr(case, "via", "") or "").strip()
    meta = getattr(case, "meta", None) or {}

    def _run():
        try:
            lines = ["**📋 对话摘要**\n" + _brief(question, answer)]
            if question:
                lines.append("**🙋 用户问题**\n" + _clip(question, 240))
            if answer:
                lines.append("**🤖 助手回答（节选）**\n" + _clip(answer, 260))
            if isinstance(meta, dict) and meta:
                pairs = " · ".join(f"{k}：{v}" for k, v in meta.items() if v)
                if pairs:
                    lines.append("**🏷 报障信息**\n" + pairs)
            from_cn = {"click": "用户点「未解决」", "copy": "用户点了「复制给人工」"}.get(via, via or "用户反馈")
            lines.append(f"{from_cn} · 案例 `{cid}` —— 已进入待处置清单，可在「案例审查」页处理")
            queue._post_card(CARD_TITLE, lines, "orange")
        except Exception as e:                        # noqa: BLE001
            print(f"[alerts] 未解决推送失败：{type(e).__name__}: {e}", flush=True)

    threading.Thread(target=_run, daemon=True).start()
    print(f"[alerts] 已排入未解决推送队列：case={cid} via={via or '-'}", flush=True)


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _brief(question: str, answer: str) -> str:
    """LLM 摘要（转人工交接简报口径）；失败时降级为规则摘要。"""
    txt = ""
    try:
        # llm.summarize 内部已按 provider 分支（echo 模式直接返回空串），
        # 这里只管「拿到就用、拿不到就降级」。
        txt = llm.summarize([
            {"role": "user", "content": question[:3000]},
            {"role": "assistant", "content": answer[:6000]},
        ])
    except Exception:                                 # noqa: BLE001
        txt = ""
    if txt:
        return _clip(txt, 700)
    # 降级：用户问什么 + 助手答的第一句
    q1 = _clip(question, 140)
    a1 = _clip(answer.split("。")[0] if answer else "", 140)
    return f"用户问：{q1}" + (f"\n助手答：{a1}。" if a1 else "")

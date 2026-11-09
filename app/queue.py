"""排队控制 + 积压/异常告警（2026-09-25 主人提）

背景：
- `/ask` 是**同步**函数，FastAPI 会放进线程池跑（默认 40 并发），所以 5 个人同时用
  **不会排队** —— 但上游大模型有频率限制（实测撞过 429：整点限流）。
  一旦超限，所有人一起失败，比"排队等一下"体验差得多。
- 所以需要：① **限制并发**（超出就排队，而不是一起打爆上游）；
  ② **告诉用户前面还有几位**（不然白等会以为卡死）；
  ③ **积压/异常时告警到主人**（rwork 机器人）。

设计取舍：
- 并发数用**信号量**控制，排队的是线程（同步接口天然支持阻塞等待），不引入额外队列服务。
- 「前面还有几位」用**递增票号**实现：票号比我小且还没完成的，就是排在我前面的。
- 告警**异步发送**（守护线程），绝不能阻塞用户请求；且带**冷却时间**，避免刷屏。
"""

import os
import time
import threading
import collections

import httpx

# ── 可配项（env.sh 里改）──
MAX_CONCURRENT = int(os.getenv("KB_MAX_CONCURRENT_ASK", "4"))     # 同时处理几路 /ask
ALERT_WAITING = int(os.getenv("KB_QUEUE_ALERT_WAITING", "2"))     # 积压到几位就告警
ALERT_BURST = int(os.getenv("KB_QUEUE_ALERT_BURST", "20"))        # 60 秒内多少请求算异常
ALERT_COOLDOWN = int(os.getenv("KB_QUEUE_ALERT_COOLDOWN", "300")) # 同类告警冷却（秒）
WEBHOOK = os.getenv(
    "KB_RWORK_WEBHOOK",
    "https://open.rwork.example.com/open-apis/bot/v2/hook/47b0596d-eab1-4e19-b314-903aa3619c7c",
)

_lock = threading.Lock()
_seq = 0                      # 递增票号
_outstanding: dict = {}       # ticket -> {"rid", "at"}
_rid2ticket: dict = {}        # 前端生成的 rid -> 票号（用于反查"我排第几"）
_active = 0                   # 真正在跑（拿到信号量）的数量
_sem = threading.Semaphore(MAX_CONCURRENT)
_recent = collections.deque()  # 最近请求时间戳（判断异常流量）
_last_alert: dict = {}         # 告警类型 -> 上次时间


def enter(rid: str = "") -> int:
    """领取票号（请求进入时调用）。"""
    global _seq
    with _lock:
        _seq += 1
        t = _seq
        _outstanding[t] = {"rid": rid or "", "at": time.time()}
        if rid:
            _rid2ticket[rid] = t
            # 防止长期运行后无限增长：只保留最近 500 个
            if len(_rid2ticket) > 500:
                for k in list(_rid2ticket)[:200]:
                    _rid2ticket.pop(k, None)
        _recent.append(time.time())
        # 只留 5 分钟窗口
        while _recent and time.time() - _recent[0] > 300:
            _recent.popleft()
    _check_burst()
    return t


def leave(ticket: int) -> None:
    with _lock:
        _outstanding.pop(ticket, None)


def acquire(timeout: float = 0) -> bool:
    """占位（阻塞直到有空位）。

    ⚠️ 2026-10-10 修（评审 ⑧）：原来**无限期阻塞** —— 排队第 3 位要等 3×60s=180s，
    而前端 150s 就先爆 → 用户看到「假超时」，后端还在跑、结果永远到不了他手里
    （还占着并发位）。现在支持等待上限：等不到就返回 False，由调用方**快速回一条
    带位次的人话**（而不是让用户白等到超时）。
    timeout ≤ 0 = 不限（兼容旧行为；调用方传 0 走这条，保持显式）。
    """
    global _active
    if timeout and timeout > 0:
        end = time.time() + timeout
        while not _sem.acquire(timeout=0.5):
            if time.time() >= end:
                return False
    else:
        _sem.acquire()
    with _lock:
        _active += 1
    _check_backlog()
    return True


def release() -> None:
    global _active
    with _lock:
        _active = max(0, _active - 1)
    _sem.release()


def ahead_of(ticket: int) -> int:
    """我前面还有几位（票号更小且尚未完成）。"""
    with _lock:
        return sum(1 for t in _outstanding if t < ticket)


def ticket_of(rid: str) -> int:
    """用前端传来的 rid 反查票号（查不到返回 0）。"""
    with _lock:
        return _rid2ticket.get(rid or "", 0)


def snapshot_by_rid(rid: str) -> dict:
    return snapshot(ticket_of(rid) or None)


def snapshot(ticket: int | None = None) -> dict:
    """给前端的排队快照。"""
    with _lock:
        waiting = max(0, len(_outstanding) - _active)
        active = _active
        ahead = sum(1 for t in _outstanding if t < ticket) if ticket else 0
    return {
        "ahead": ahead,
        "active": active,
        "waiting": waiting,
        "limit": MAX_CONCURRENT,
        "ticket": ticket or 0,
    }


# ────────────────────────── 告警 ──────────────────────────

def _should_alert(kind: str) -> bool:
    now = time.time()
    if now - _last_alert.get(kind, 0) < ALERT_COOLDOWN:
        return False
    _last_alert[kind] = now
    return True


def _send_async(title: str, lines: list, template: str = "red") -> None:
    """异步发卡片（守护线程，失败只记日志 —— 绝不影响用户请求）。"""
    def _run():
        try:
            _post_card(title, lines, template)
        except Exception as e:                       # noqa: BLE001
            print(f"[queue] 告警发送失败：{type(e).__name__}: {e}", flush=True)
    threading.Thread(target=_run, daemon=True).start()


def _post_card(title: str, lines: list, template: str = "red") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    elements = [{"tag": "div", "text": {"tag": "lark_md", "content": str(x)}}
                for x in lines if x]
    elements.append({"tag": "hr"})
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": f"更新时间 {ts}"}})
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": title}, "template": template},
            "elements": elements,
        },
    }
    # ⚠️ trust_env=False 必需：自动化环境会注入 http_proxy=127.0.0.1:<死端口>，
    #    urllib/httpx 默认继承它 → 表现为"连不上"，而同一时刻 curl 是通的（项目里踩过两次）。
    with httpx.Client(timeout=12, trust_env=False) as cli:
        r = cli.post(WEBHOOK, json=payload)
        print(f"[queue] 告警已发送 HTTP {r.status_code}", flush=True)


def _check_backlog() -> None:
    """排队积压告警。"""
    with _lock:
        waiting = max(0, len(_outstanding) - _active)
        active = _active
    if waiting >= ALERT_WAITING and _should_alert("backlog"):
        _send_async(
            "⚠️ POS 助手：请求排队积压",
            [
                f"**排队等待**：{waiting} 位（阈值 {ALERT_WAITING}）",
                f"**正在处理**：{active} / 并发上限 {MAX_CONCURRENT}",
                "",
                "可能是同事同时在用；若持续出现，考虑上调 `KB_MAX_CONCURRENT_ASK`。",
            ],
            "orange",
        )


def _check_burst() -> None:
    """短时间大量请求（疑似攻击/误用）。"""
    now = time.time()
    with _lock:
        n = sum(1 for t in _recent if now - t <= 60)
    if n >= ALERT_BURST and _should_alert("burst"):
        _send_async(
            "🚨 POS 助手：请求量异常",
            [
                f"**近 60 秒请求数**：{n}（阈值 {ALERT_BURST}）",
                f"**并发上限**：{MAX_CONCURRENT}",
                "",
                "若非同事集中使用，可能是被刷。必要时上调限流或封禁来源。",
            ],
            "red",
        )
